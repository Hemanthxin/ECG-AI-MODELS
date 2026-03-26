"""
Hugging Face Space — ECG Digitizer API
Full self-contained app.py with Advanced Colab Signal Extraction & Memory Management
"""

import io, base64, shutil, os, math, gc
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from scipy import signal as sp_signal
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# =========================================================================
# PATHS & CONFIG
# =========================================================================
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
REFERENCE_IMAGE  = os.path.join(BASE_DIR, "assets", "reference.png")
CHECKPOINTS      = {
    "II":    os.path.join(BASE_DIR, "models", "best_nnunet_init_lead_II_113.pth"),
    "V":     os.path.join(BASE_DIR, "models", "best_nnunet(2).pth"),
    "OTHER": os.path.join(BASE_DIR, "models", "best_nnunet_init_lead_other_59.pth"),
}
YOLO_WAVE_PATH   = os.path.join(BASE_DIR, "models", "yolo_wave.pt")
YOLO_LEADS_PATH  = os.path.join(BASE_DIR, "models", "yolo_13Leads.pt")

LEADS        = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
V_LEADS      = {"V1","V2","V3","V4","V5","V6"}
ALL_13_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6", "II_short"]

pixel_per_volt  = 80
THRESHOLD       = 0.5
min_valid_ratio = 0.05
REF_SIZE        = (2200, 1700)
device          = "cpu"

LEAD_CROP_COORDS = {
    "I":        [120, 553, 607, 867],
    "aVR":      [613, 553, 1099, 867],
    "V1":       [1105, 553, 1591, 867],
    "V4":       [1597, 553, 2088, 867],
    "II_short": [120, 836, 607, 1150],
    "aVL":      [613, 836, 1099, 1150],
    "V2":       [1105, 836, 1591, 1150],
    "V5":       [1597, 836, 2088, 1150],
    "III":      [120, 1118, 607, 1432],
    "aVF":      [613, 1118, 1099, 1432],
    "V3":       [1105, 1118, 1591, 1432],
    "V6":       [1597, 1118, 2088, 1432],
    "II":       [120, 1378, 2084, 1692]
}

def get_target_len(record_id, lead):
    return 10000 if lead == "II" else 2500

# =========================================================================
# YOLO LAZY-LOAD
# =========================================================================
_row_detector  = None
_lead_detector = None

def get_yolo_models():
    global _row_detector, _lead_detector
    if _row_detector is None:
        print("[YOLO] Loading row detector...")
        _row_detector  = YOLO(YOLO_WAVE_PATH)
    if _lead_detector is None:
        print("[YOLO] Loading lead detector...")
        _lead_detector = YOLO(YOLO_LEADS_PATH)
    return _row_detector, _lead_detector

# =========================================================================
# COLAB NNUNET ARCHITECTURE
# =========================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_instance_norm=True):
        super().__init__()
        norm = nn.InstanceNorm2d(out_channels, affine=True) if use_instance_norm else nn.BatchNorm2d(out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm, nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
    def forward(self, x): return self.conv(x)

class StackedConvBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, use_instance_norm=True):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, out_channels, use_instance_norm),
            *[ConvBlock(out_channels, out_channels, use_instance_norm) for _ in range(num_convs - 1)]
        )
    def forward(self, x): return self.blocks(x)

class nnUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32, num_pool=4, conv_per_stage=2, use_instance_norm=True, deep_supervision=True):
        super().__init__()
        self.num_pool, self.deep_supervision = num_pool, deep_supervision
        features = [base_filters * (2 ** i) for i in range(num_pool + 1)]
        
        self.encoders, self.pools = nn.ModuleList(), nn.ModuleList()
        for i in range(num_pool):
            in_ch = in_channels if i == 0 else features[i-1]
            self.encoders.append(StackedConvBlocks(in_ch, features[i], conv_per_stage, use_instance_norm))
            self.pools.append(nn.Conv2d(features[i], features[i], kernel_size=2, stride=2))
            
        self.bottleneck = StackedConvBlocks(features[num_pool-1], features[num_pool], conv_per_stage, use_instance_norm)
        
        self.upconvs, self.decoders = nn.ModuleList(), nn.ModuleList()
        for i in range(num_pool):
            self.upconvs.append(nn.ConvTranspose2d(features[num_pool-i], features[num_pool-i-1], kernel_size=2, stride=2))
            self.decoders.append(StackedConvBlocks(features[num_pool-i], features[num_pool-i-1], conv_per_stage, use_instance_norm))
            
        if deep_supervision:
            self.seg_outputs = nn.ModuleList([nn.Conv2d(features[num_pool-i-1], out_channels, kernel_size=1) for i in range(num_pool)])
        else:
            self.seg_outputs = nn.ModuleList([nn.Conv2d(features[0], out_channels, kernel_size=1)])

    def forward(self, x):
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)
            
        x = self.bottleneck(x)
        seg_outputs = []
        
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = skip_connections[-(i+1)]
            if x.shape != skip.shape: 
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)
            if self.deep_supervision: 
                seg_outputs.append(self.seg_outputs[i](x))
                
        if not self.deep_supervision: 
            seg_outputs = [self.seg_outputs[0](x)]
        return seg_outputs

# =========================================================================
# MEMORY-SAFE LAZY-LOAD MODEL & PREDICT MASK
# =========================================================================
_models = {}
_current_model_name = None

def get_model(name):
    global _current_model_name
    
    # If a different model is requested, wipe the old one from RAM completely
    if _current_model_name is not None and _current_model_name != name:
        print(f"[Memory] Unloading {_current_model_name} to free RAM...")
        _models.clear()
        gc.collect() # Force Python to clear RAM
        
    if name not in _models:
        print(f"[nnUNet] Loading model: {name}")
        model = nnUNet(base_filters=32, num_pool=4, conv_per_stage=2, use_instance_norm=True, deep_supervision=True).to(device)
        model.load_state_dict(torch.load(CHECKPOINTS[name], map_location=device))
        model.eval()
        _models[name] = model
        _current_model_name = name
        
    return _models[name]

def predict_mask(gray_img, lead):
    model = get_model("II" if lead == "II" else ("V" if lead in V_LEADS else "OTHER"))
    img_norm = gray_img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        pred_mask = (torch.sigmoid(outputs[-1])[0, 0] > THRESHOLD).cpu().numpy()
    return img_as_ubyte(skeletonize(pred_mask))

# =========================================================================
# ADVANCED SIGNAL EXTRACTION (COLAB LOGIC)
# =========================================================================
def detect_baseline_from_mask(skeleton_mask):
    H, W = skeleton_mask.shape
    col_medians = []
    for x in range(W):
        ys = np.where(skeleton_mask[:, x] > 0)[0]
        if len(ys) > 0: col_medians.append(float(np.median(ys)))
    if not col_medians: return H // 2
    return int(np.clip(round(np.median(col_medians)), 0, H - 1))

def extract_signal_robust(waveform_binary, target_len):
    H, W = waveform_binary.shape
    if W <= 1: return np.zeros(target_len, dtype=np.float32)

    waveform_binary[:5, :] = 0
    waveform_binary[-35:, :] = 0
    waveform_binary[:, :int(W * 0.015)] = 0
    waveform_binary[:, int(W * 0.975):] = 0

    initial_baseline = detect_baseline_from_mask(waveform_binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(waveform_binary, connectivity=8)
    clean_binary = np.zeros_like(waveform_binary)
    
    for i in range(1, num_labels):
        min_y = stats[i, cv2.CC_STAT_TOP]
        max_y = min_y + stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if min_y > initial_baseline + 30: continue
        if max_y < initial_baseline - 40: continue
        if area > 5: clean_binary[labels == i] = 255

    waveform_binary = clean_binary
    baseline_y = detect_baseline_from_mask(waveform_binary)

    volt_signal = np.full(W, np.nan, dtype=np.float32)
    prev_y = baseline_y
    valid_count = 0

    for x in range(W):
        ys = np.where(waveform_binary[:, x] > 0)[0]
        if len(ys) == 0: continue
        y_min, y_max = ys[0], ys[-1]

        if (y_max - y_min) > 4:
            dist_top = abs(y_min - baseline_y)
            dist_bottom = abs(y_max - baseline_y)
            chosen_y = y_min if dist_top > dist_bottom else y_max
        else:
            chosen_y = ys[np.argmin(np.abs(ys - prev_y))]

        volt_signal[x] = (baseline_y - chosen_y) / float(pixel_per_volt)
        prev_y = chosen_y
        valid_count += 1

    if valid_count < int(W * min_valid_ratio): return np.zeros(target_len, dtype=np.float32)

    valid_xs = np.where(~np.isnan(volt_signal))[0]
    valid_vals = volt_signal[valid_xs]
    full_x = np.arange(W)
    volt_filled = np.interp(full_x, valid_xs, valid_vals).astype(np.float32)

    max_gap = W // 50
    gap_dist = np.full(W, W, dtype=np.int32)
    for vx in valid_xs: gap_dist[max(0, vx - max_gap) : min(W, vx + max_gap + 1)] = 0
    volt_filled[gap_dist > 0] = 0.0

    volt_filled -= np.median(volt_filled)
    volt_smooth = sp_signal.savgol_filter(volt_filled, window_length=11, polyorder=3)

    src_x = np.linspace(0, W - 1, num=W)
    dst_x = np.linspace(0, W - 1, num=target_len)
    return np.interp(dst_x, src_x, volt_smooth).astype(np.float32)

# =========================================================================
# ADVANCED DETECTIONS
# =========================================================================
def get_binary_robust(roi):
    is_color = False
    if len(roi.shape) == 3:
        b, g, r = cv2.split(roi)
        if np.mean(cv2.absdiff(r, g)) > 5: is_color = True

    gray = roi[:, :, 2] if is_color else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_v)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_clean)

def find_side_by_walls(th, h, w):
    margin = int(w * 0.15)
    def get_max_wall_strength(zone):
        if zone.size == 0: return 0
        v_proj = np.sum(zone > 0, axis=0)
        return len(v_proj[v_proj > (h * 0.4)])
    if get_max_wall_strength(th[:, -margin:]) > get_max_wall_strength(th[:, :margin]): return 'right'
    return 'left'

def find_origin_coords(roi, forced_side=None):
    h, w = roi.shape[:2]
    th = get_binary_robust(roi)
    side = forced_side if forced_side else find_side_by_walls(th, h, w)

    best_y, max_cross = h // 2, -1
    for y in range(int(h * 0.45), int(h * 0.85)):
        crossings = np.sum(np.abs(np.diff(th[y, :] > 0)))
        if crossings > max_cross: max_cross, best_y = crossings, y

    margin = int(w * 0.15)
    search_th = th[best_y-5:best_y+5, -margin:] if side == 'right' else th[best_y-5:best_y+5, :margin]
    idx = np.where(np.sum(search_th > 0, axis=0) > 0)[0]
    ox = ((w - margin) + idx[-1] if len(idx)>0 else 0) if side == 'right' else (idx[0] if len(idx)>0 else 0)
    return ox, best_y, side

# =========================================================================
# PROCESSORS (RIGHT 13-LEAD & LEFT ORB)
# =========================================================================
def process_right_calib_13leads(raw_img, image_path, record_id):
    _, lead_detector = get_yolo_models()
    extracted_signals = {lead: np.zeros(get_target_len(record_id, lead), dtype=np.float32) for lead in LEADS + ['II_short']}
    extracted_masks = {}

    res = lead_detector.predict(source=str(image_path), conf=0.25, verbose=False)
    boxes = [[int(c) for c in box.xyxy[0].tolist()] for r in res for box in r.boxes]

    if not boxes: return extracted_signals, extracted_masks

    boxes.sort(key=lambda b: (b[1]+b[3])/2)
    rows = []
    current_row = [boxes[0]]
    img_h = raw_img.shape[0]
    y_thresh = img_h * 0.05

    for b in boxes[1:]:
        mean_y = sum((cb[1]+cb[3])/2 for cb in current_row) / len(current_row)
        if abs((b[1]+b[3])/2 - mean_y) < y_thresh: current_row.append(b)
        else:
            rows.append(current_row)
            current_row = [b]
    rows.append(current_row)

    for r in rows: r.sort(key=lambda b: (b[0]+b[2])/2)

    row_map = [
        ["I", "aVR", "V1", "V4"],
        ["II_short", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
        ["II"]
    ]

    for r_idx, r_boxes in enumerate(rows):
        if r_idx >= 4: break
        for c_idx, box in enumerate(r_boxes):
            if c_idx >= len(row_map[r_idx]): break
            lead_name = row_map[r_idx][c_idx]

            x1, y1, x2, y2 = box
            pad_y = int((y2 - y1) * 0.1)
            roi = raw_img[max(0, y1 - pad_y) : min(img_h, y2 + pad_y), x1:x2]

            if len(roi.shape) == 3:
                b, g, r = cv2.split(roi)
                roi_gray = r if np.mean(cv2.absdiff(r, g)) > 5 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else: roi_gray = roi.copy()

            if c_idx == len(r_boxes) - 1:
                ox, _, _ = find_origin_coords(roi, forced_side='right')
                valid_w = ox if ox > 0 else roi_gray.shape[1]
                roi_gray = roi_gray[:, :valid_w]

            target_len = get_target_len(record_id, lead_name)

            if lead_name == "II":
                row_resized = cv2.resize(roi_gray, (1964, 314))
                model_type = 'II'
            else:
                row_resized = cv2.resize(roi_gray, (491, 314))
                model_type = 'V' if lead_name in V_LEADS else 'OTHER'

            row_mask = predict_mask(row_resized, model_type)
            sig = extract_signal_robust(row_mask, target_len)

            extracted_signals[lead_name] = sig
            extracted_masks[lead_name] = row_mask

    return extracted_signals, extracted_masks


def process_left_calib_orb(img_path, record_id):
    sigs, masks = {}, {}
    img = cv2.imread(str(img_path))
    ref_img = cv2.imread(REFERENCE_IMAGE)

    if ref_img is None or img is None: 
        aligned = cv2.resize(cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE), REF_SIZE)
    else:
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if ref_img.ndim == 3 else ref_img.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

        if not (len(img.shape) == 3 and np.mean(cv2.absdiff(img[:,:,2], img[:,:,1])) > 5):
            th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
            img_for_orb = cv2.bitwise_not(cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))))
        else: img_for_orb = img[:,:,2]

        orb = cv2.ORB_create(5000)
        kp_ref, des_ref = orb.detectAndCompute(ref_img[:,:,2] if ref_img.ndim==3 else ref_gray, None)
        kp, des = orb.detectAndCompute(img_for_orb, None)

        if des is not None and des_ref is not None:
            matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des, des_ref)
            if len(matches) >= 10:
                src_pts = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                H_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                aligned = cv2.warpPerspective(img_gray, H_matrix, REF_SIZE) if H_matrix is not None else cv2.resize(img_gray, REF_SIZE)
            else: aligned = cv2.resize(img_gray, REF_SIZE)
        else: aligned = cv2.resize(img_gray, REF_SIZE)

    for lead, (x1, y1, x2, y2) in LEAD_CROP_COORDS.items():
        raw_crop = aligned[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

        h, w = raw_crop.shape[:2]
        target_w = 1964 if lead == "II" else 491
        pad_h, pad_w = max(0, 314 - h), max(0, target_w - w)
        crop = np.pad(raw_crop, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode="constant", constant_values=255)

        mask = predict_mask(crop, 'II' if lead == 'II' else ('V' if lead in V_LEADS else 'OTHER'))
        sigs[lead] = extract_signal_robust(mask, get_target_len(record_id, lead))
        masks[lead] = mask

    return sigs, masks

# =========================================================================
# PLOT HELPER — checks if a signal is essentially flat/failed
# =========================================================================
def _is_flat(sig, min_peak_to_peak=0.05):
    """Returns True if signal has no meaningful variation (failed extraction)."""
    if sig is None or len(sig) == 0:
        return True
    arr = np.asarray(sig, dtype=np.float32)
    return float(np.max(arr) - np.min(arr)) < min_peak_to_peak


# =========================================================================
# AUTO-ROTATE — fix portrait-scanned ECG images before any processing
# =========================================================================
def _auto_rotate_ecg(img):
    """
    Standard 12-lead ECG paper is always landscape (w > h).
    If the scan is portrait (h > w), detect the best rotation candidate
    and return the corrected image + the rotation code applied.
    """
    h, w = img.shape[:2]
    if w >= h:
        return img, None  # already landscape

    candidates = [
        (cv2.ROTATE_90_CLOCKWISE,        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        (cv2.ROTATE_180,                 cv2.rotate(img, cv2.ROTATE_180)),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]
    target_ratio = 1.4  # typical landscape ECG w/h
    best_code, best_img, best_score = None, img, float('inf')
    for code, rotated in candidates:
        rh, rw = rotated.shape[:2]
        if rh == 0:
            continue
        ratio = rw / rh
        score = abs(ratio - target_ratio) if ratio >= 1.0 else abs(ratio - target_ratio) + 10
        if score < best_score:
            best_score, best_code, best_img = score, code, rotated

    print(f"[AutoRotate] portrait h={h} w={w} -> applied rotation code={best_code}")
    return best_img, best_code


# =========================================================================
# MAIN FASTAPI PIPELINE
# =========================================================================
def main_pipeline_api(image_path):
    record_id = ''.join([c for c in Path(image_path).stem if c.isdigit()])
    raw       = cv2.imread(str(image_path))
    if raw is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Auto-correct portrait/rotated scans to landscape BEFORE any detector
    raw, _rot = _auto_rotate_ecg(raw)
    if _rot is not None:
        rotated_path = str(image_path).rsplit('.', 1)
        rotated_path = rotated_path[0] + '_rotated.' + (rotated_path[1] if len(rotated_path) > 1 else 'png')
        cv2.imwrite(rotated_path, raw)
        image_path = rotated_path
        print(f"[AutoRotate] saved corrected image -> {rotated_path}")

    ih, iw       = raw.shape[:2]
    side_detected = 'left'
    boxes         = []
    
    try:
        row_detector, _ = get_yolo_models()
        res   = row_detector.predict(source=str(image_path), conf=0.25, verbose=False)
        boxes = [
            [int(c) for c in box.xyxy[0].tolist()]
            for r in res for box in r.boxes
            if (box.xyxy[0][2] - box.xyxy[0][0]) >= iw * 0.4
        ]
        if boxes:
            boxes.sort(key=lambda b: b[1])
            _, _, side_detected = find_origin_coords(
                raw[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2]]
            )
    except Exception as e:
        print(f"[YOLO-Row] Detection failed: {e}")
        
    print(f"[Pipeline] record={record_id} side={side_detected} boxes={len(boxes)}")
    
    if side_detected == 'right' and len(boxes) == 4:
        sigs, masks = process_right_calib_13leads(raw, image_path, record_id)
    else:
        sigs, masks = process_left_calib_orb(image_path, record_id)

    # =====================================================================
    # OUTPUT PLOT — Pink Medical Grid
    # BUG FIX: row2 uses 'II_short', not 'II'
    # IMPROVEMENT: flat/zero leads show "No Signal" instead of flat line
    # =====================================================================
    def get_sig(lead, length):
        return sigs.get(lead, np.zeros(length, dtype=np.float32))

    # Per-lead flat detection before concatenation
    flat_map = {}
    for lead in list(sigs.keys()) + ['II_short']:
        flat_map[lead] = _is_flat(sigs.get(lead))

    # Row layout — note II_short (not II) for row 2
    row_lead_map = [
        ['I',        'aVR', 'V1', 'V4'],
        ['II_short', 'aVL', 'V2', 'V5'],   # was wrongly labelled 'II' before
        ['III',      'aVF', 'V3', 'V6'],
        ['II'],
    ]

    # Display labels (II_short shows as "II" on chart since it's the short rhythm strip)
    row_display_labels = [
        ['I',   'aVR', 'V1', 'V4'],
        ['II',  'aVL', 'V2', 'V5'],
        ['III', 'aVF', 'V3', 'V6'],
        ['II'],
    ]

    row4      = get_sig('II', 10000)
    time_axis = np.linspace(0, 10, len(row4))

    fig, axes = plt.subplots(4, 1, figsize=(20, 11))
    fig.patch.set_facecolor('#fafafa')
    fig.subplots_adjust(hspace=0, wspace=0)

    for i, ax in enumerate(axes):
        ax.set_facecolor('#fff5f5')

        # ── Medical grid ──────────────────────────────────────────────
        ax.set_xlim(0, 10)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xticks(np.arange(0, 10.01, 0.04), minor=True)
        ax.set_yticks(np.arange(-2.5, 2.51, 0.1), minor=True)
        ax.grid(which='minor', color='#ffcccc', linestyle='-', linewidth=0.5, zorder=0)
        ax.set_xticks(np.arange(0, 10.01, 0.2))
        ax.set_yticks(np.arange(-2.5, 2.51, 0.5))
        ax.grid(which='major', color='#ff8080', linestyle='-', linewidth=1.0, zorder=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # ── Draw signals ──────────────────────────────────────────────
        if i < 3:
            for j, lead_name in enumerate(row_lead_map[i]):
                display_label = row_display_labels[i][j]
                x_start = j * 2.5
                x_end   = x_start + 2.5

                seg = get_sig(lead_name, 2500)

                if flat_map.get(lead_name, True):
                    # Dashed baseline + "No Signal" text
                    ax.plot([x_start, x_end], [0, 0],
                            color='#ddbbbb', linewidth=0.9, linestyle='--', zorder=4)
                    ax.text(x_start + 1.25, 0.0, 'No Signal',
                            ha='center', va='center', fontsize=7.5,
                            color='#b06060', fontstyle='italic',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5),
                            zorder=10)
                else:
                    t_seg = np.linspace(x_start, x_end, len(seg))
                    ax.plot(t_seg, seg, color='#111111', linewidth=1.0, zorder=5)

                # Lead label (top-left of each column)
                ax.text(x_start + 0.08, 1.9, display_label,
                        fontsize=13, fontweight='bold', color='#111111',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1),
                        zorder=10)

                # Vertical divider between leads
                if j > 0:
                    ax.axvline(x_start, color='#222222', lw=1.2, zorder=6)

        else:
            # Row 4 — full II rhythm strip
            if flat_map.get('II', True):
                ax.plot([0, 10], [0, 0],
                        color='#ddbbbb', linewidth=0.9, linestyle='--', zorder=4)
                ax.text(5.0, 0.0, 'No Signal — II (Rhythm Strip)',
                        ha='center', va='center', fontsize=9,
                        color='#b06060', fontstyle='italic',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2),
                        zorder=10)
            else:
                ax.plot(time_axis, row4, color='#111111', linewidth=1.0, zorder=5)

            ax.text(0.08, 1.9, 'II',
                    fontsize=13, fontweight='bold', color='#111111',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1),
                    zorder=10)

    # Title with extraction summary
    valid_leads  = [l for l in sigs if not flat_map.get(l, True)]
    total_leads  = len([l for l in sigs if l != 'II_short'])  # don't double-count II
    subtitle     = f"{len(valid_leads)}/{len(sigs)} leads extracted"

    plt.suptitle(
        f"Medical Grid ECG  |  Record: {record_id}  |  {subtitle}",
        fontsize=16, fontweight='bold', y=0.998, color='#1a1a1a'
    )

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Process json response
    json_sigs = {}
    for lead, arr in sigs.items():
        arr_list = arr.tolist() if hasattr(arr, 'tolist') else list(arr)
        json_sigs[lead] = [
            None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)
            for x in arr_list
        ]
        
    # Final cleanup to ensure memory is cleared after request
    gc.collect()
        
    return {
        "plot_image":    f"data:image/png;base64,{img_b64}",
        "signals":       json_sigs,
        "lead_list":     list(sigs.keys()),
        "side_detected": side_detected
    }

# =========================================================================
# FASTAPI APP
# =========================================================================
app = FastAPI(title="ECG Digitizer API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"status": "ECG Digitizer Space running", "endpoint": "POST /analyze-ecg"}

@app.post("/analyze-ecg")
async def analyze_ecg(file: UploadFile = File(...)):
    temp_path = f"/tmp/ecg_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        data = main_pipeline_api(temp_path)
        return {"status": "success", "data": data}
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[API] ERROR:\n{tb}")
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e), "traceback": tb})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        # Clean up rotated temp file if it was created
        rotated_parts = temp_path.rsplit('.', 1)
        rotated_tmp = rotated_parts[0] + '_rotated.' + (rotated_parts[1] if len(rotated_parts) > 1 else 'png')
        if os.path.exists(rotated_tmp):
            os.remove(rotated_tmp)
