import argparse
import math
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np

# ──────────────────────────────────────────────
# Global configuration
# ──────────────────────────────────────────────
CLASSES = ["CuRB", "N-CuRB"]   # Binary classification labels
SEED = 42                        # Random seed for reproducibility
IMG_SIZE = 224                   # Standard input image size (pixels)


def set_seed(seed: int = SEED):
    """Fix random seeds for Python, NumPy, and PyTorch to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ══════════════════════════════════════════════
# STAGE 1 — Image Processing (All_photo.ipynb)
# ══════════════════════════════════════════════

# ── Image I/O ──────────────────────────────────
def imread_gray(path: Path) -> np.ndarray:
    """Read an image as grayscale. Supports paths with non-ASCII (e.g. Chinese) characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv.imdecode(data, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def imread_color(path: Path) -> np.ndarray:
    """Read an image in BGR color. Supports paths with non-ASCII characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv.imdecode(data, cv.IMREAD_COLOR)


def imwrite(path: Path, img: np.ndarray):
    """Write an image to disk. Supports paths with non-ASCII characters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Cannot write: {path}")
    buf.tofile(str(path))


# ── Core image utilities ────────────────────────
def rotate(img: np.ndarray, deg: float, expand: bool = True, binary: bool = False) -> np.ndarray:
    """
    Rotate an image by `deg` degrees.
    If expand=True, the canvas is enlarged so no content is clipped.
    If binary=True, INTER_NEAREST is used to preserve binary pixel values.
    """
    h, w = img.shape[:2]
    c = (w / 2, h / 2)
    M = cv.getRotationMatrix2D(c, deg, 1.0)

    if not expand:
        flags = cv.INTER_NEAREST if binary else cv.INTER_LINEAR
        return cv.warpAffine(img, M, (w, h), flags=flags,
                             borderMode=cv.BORDER_CONSTANT, borderValue=0)

    # Compute expanded canvas dimensions to avoid clipping after rotation
    cos_v, sin_v = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin_v + w * cos_v)
    new_h = int(h * cos_v + w * sin_v)
    M[0, 2] += (new_w / 2) - c[0]
    M[1, 2] += (new_h / 2) - c[1]

    flags = cv.INTER_NEAREST if binary else cv.INTER_LINEAR
    return cv.warpAffine(img, M, (new_w, new_h), flags=flags,
                         borderMode=cv.BORDER_CONSTANT, borderValue=0)


def preprocess(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a grayscale X-ray image.
    Steps: high-pass filter -> Otsu thresholding -> horizontal erosion.
    Returns:
      hp   -- high-pass filtered image
      bw01 -- binary mask (0/1) after erosion
    """
    blur = cv.GaussianBlur(gray, (31, 31), 0)
    hp = cv.addWeighted(gray, 1.5, blur, -0.5, 0)   # High-pass = original - low-pass
    _, th = cv.threshold(hp, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    bin_img = 25 * th                                 # Scale to a visible range
    k = cv.getStructuringElement(cv.MORPH_RECT, (31, 1))
    eroded = cv.erode(bin_img, k, iterations=1)       # Horizontal erosion suppresses vertical noise
    bw01 = (eroded > 0).astype(np.uint8)
    return hp, bw01


def trough_from_band(bw01: np.ndarray, band_ratio: float = 0.3) -> Tuple[int, int, int, float]:
    """
    Find the horizontal row with the minimum projection value within the central band.
    Used to locate the gap between upper and lower jaws.
    Returns: (band_top, band_bottom, split_y, min_val)
    """
    h, _ = bw01.shape
    band_h = int(h * band_ratio)
    t = (h - band_h) // 2
    b = t + band_h
    proj = bw01[t:b].sum(axis=1)        # Horizontal projection within central band
    y_local = int(np.argmin(proj))
    return t, b, t + y_local, float(proj[y_local])


def draw_line(img: np.ndarray, y: int, color=(0, 0, 255), thick: int = 2) -> np.ndarray:
    """Draw a horizontal line at row `y` on a copy of the image (BGR output)."""
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    h, w = vis.shape[:2]
    cv.line(vis, (0, y), (w - 1, y), color, thick)
    return vis


# ── Step 1: Preprocessing ──────────────────────
def step1_preprocess(input_dir: Path, out_dir: Path):
    """
    Step 1: Apply high-pass filtering and Otsu thresholding to all input images.
    Saves: <name>_a_hp.png (high-pass) and <name>_b_bw.png (binary mask).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_list = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    print(f"[Step1] Found {len(img_list)} images. Starting preprocessing...")
    for img_path in img_list:
        name = img_path.stem
        gray = imread_gray(img_path)
        hp, bw = preprocess(gray)
        imwrite(out_dir / f"{name}_a_hp.png", hp)
        imwrite(out_dir / f"{name}_b_bw.png", bw * 255)
    print(f"[Step1] Done -> {out_dir}")


# ── Step 2: Angle scanning ─────────────────────
def find_best_angle(gray: np.ndarray) -> Tuple[float, int]:
    """
    Find the best rotation angle using a two-stage scan:
      - Coarse scan: -15 to +15 degrees in steps of 5
      - Fine scan:   best_coarse +/- ~12 degrees in steps of 1
    Returns: (best_deg, best_y) where best_y is the jaw split row at that angle.
    """
    _, bw0 = preprocess(gray)
    best = {"deg": 0, "val": 1e18, "y": 0}

    # Coarse scan: +-15 degrees, step 5 degrees
    for deg in range(-15, 16, 5):
        rot = rotate(bw0, deg)
        _, _, y, val = trough_from_band(rot)
        if val < best["val"]:
            best = {"deg": deg, "val": val, "y": y}

    # Fine scan around the coarse best: step 1 degree
    deg0 = best["deg"]
    for deg in range(deg0 - 5, deg0 + 13):
        rot = rotate(bw0, deg)
        _, _, y, val = trough_from_band(rot)
        if val < best["val"]:
            best = {"deg": deg, "val": val, "y": y}

    return float(best["deg"]), int(best["y"])


def step2_scan(input_dir: Path, out_dir: Path):
    """
    Step 2: Find the best rotation angle for each image and save the rotated
    result with the detected jaw split line drawn on it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_list = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    print(f"[Step2] Scanning angles for {len(img_list)} images...")
    for img_path in img_list:
        name = img_path.stem
        gray = imread_gray(img_path)
        best_deg, best_y = find_best_angle(gray)
        rot = rotate(gray, best_deg)
        vis = draw_line(rot, best_y)
        imwrite(out_dir / f"{name}_rot.png", vis)
        print(f"  OK {name} | best angle = {best_deg} deg")
    print(f"[Step2] Done -> {out_dir}")


# ── Step 3: Upper / lower jaw split ────────────
def split_upper_lower(gray: np.ndarray, deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Rotate the image by `deg` degrees, then split it horizontally into
    upper and lower jaw regions at the detected trough row.
    Returns: (annotated_full, upper_crop, lower_crop, split_y)
    """
    rot = rotate(gray, deg)
    _, bw01 = preprocess(rot)
    _, _, y, _ = trough_from_band(bw01)
    upper = rot[:y, :]
    lower = rot[y:, :]
    vis = draw_line(rot, y)
    return vis, upper, lower, y


def step3_split_ul(input_dir: Path, out_dir: Path):
    """
    Step 3: For each image, detect the best rotation angle and split into
    upper/lower jaw. Saves annotated full image, upper crop, and lower crop.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_list = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    print(f"[Step3] Splitting upper/lower jaw for {len(img_list)} images...")
    for img_path in img_list:
        name = img_path.stem
        gray = imread_gray(img_path)
        best_deg, _ = find_best_angle(gray)
        vis, upper, lower, _ = split_upper_lower(gray, best_deg)
        imwrite(out_dir / f"{name}_{best_deg:+.0f}_full.png", vis)
        imwrite(out_dir / f"{name}_{best_deg:+.0f}_upper.png", upper)
        imwrite(out_dir / f"{name}_{best_deg:+.0f}_lower.png", lower)
        print(f"  OK {name}")
    print(f"[Step3] Done -> {out_dir}")


# ── Step 4: Vertical tooth segmentation ────────
KSIZE_HP = 31       # Kernel size for high-pass Gaussian blur
ERODE_K = (1, 41)   # Vertical erosion kernel to suppress horizontal structures
SMOOTH_W = 35       # Moving-average window width for smoothing the column projection
MIN_DIST_F = 0.55   # Minimum inter-valley distance factor (relative to image width)
BORDER_GUARD = 15   # Minimum pixel distance from image border for a valid valley


def vertical_teeth_segmentation(gray: np.ndarray, tag: str, out_dir: Path) -> List[int]:
    """
    Detect vertical valleys in the column-wise projection of a jaw image.
    Each valley corresponds to a gap between adjacent teeth.
    Saves an annotated image with red vertical lines at each valley position.
    Returns: list of valley x-coordinates.
    """
    try:
        from scipy.signal import find_peaks
    except ImportError:
        raise ImportError("scipy is required. Install with: pip install scipy")

    H, W = gray.shape

    # Enhanced high-pass + black-hat morphology to emphasize tooth boundaries
    blur = cv.GaussianBlur(gray, (KSIZE_HP, KSIZE_HP), 0)
    hp = cv.addWeighted(gray, 1.5, blur, -0.5, 0)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    blackhat = cv.morphologyEx(hp, cv.MORPH_BLACKHAT, kernel)
    hp = cv.addWeighted(hp, 1.0, blackhat, 1.2, 0)

    # Binarize and apply vertical erosion to isolate inter-dental gaps
    _, th = cv.threshold(hp, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    bin_inv = 25 * th
    kv = cv.getStructuringElement(cv.MORPH_RECT, ERODE_K)
    eroded = cv.erode(bin_inv, kv, iterations=1)

    # Compute and normalize the column-wise projection
    proj = eroded.sum(axis=0).astype(np.float32)
    proj_s = np.convolve(proj, np.ones(SMOOTH_W) / SMOOTH_W, mode="same")
    proj_s = (proj_s - proj_s.min()) / (proj_s.max() - proj_s.min() + 1e-6)

    # Find valleys as peaks in the inverted projection
    inv_proj = 1.0 - proj_s
    min_dist = int(W * MIN_DIST_F / 5)
    peaks, _ = find_peaks(inv_proj, distance=min_dist, prominence=0.05, height=0.05)
    valleys = [x for x in peaks if BORDER_GUARD < x < W - BORDER_GUARD]

    # Draw a red vertical line at each detected valley
    vis = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    for x in valleys:
        cv.line(vis, (x, 0), (x, H - 1), (0, 0, 255), 2)

    out_dir.mkdir(parents=True, exist_ok=True)
    imwrite(out_dir / f"{tag}_valley_lines.png", vis)
    print(f"  {tag}: {len(valleys)} valleys detected -> ~{len(valleys)+1} teeth")
    return valleys


def step4_vertical(ul_dir: Path, out_dir: Path):
    """
    Step 4: Run vertical tooth segmentation on all upper/lower jaw images
    produced in Step 3.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    upper_files = sorted(ul_dir.glob("*_upper.png"))
    lower_files = sorted(ul_dir.glob("*_lower.png"))
    all_files = upper_files + lower_files
    print(f"[Step4] Vertical segmentation: {len(upper_files)} upper + {len(lower_files)} lower...")
    for f in all_files:
        gray = imread_gray(f)
        tag = f.stem
        vertical_teeth_segmentation(gray, tag, out_dir)
    print(f"[Step4] Done -> {out_dir}")


# ── Step 5: Individual tooth cropping ──────────
def detect_vertical_lines(img_bgr: np.ndarray, thr: int = 180, min_len: int = 20) -> List[int]:
    """
    Detect the x-coordinates of red vertical lines in a BGR image.
    These lines mark inter-dental gaps drawn in Step 4.
    Returns: sorted list of x-coordinates (one per detected gap).
    """
    red_mask = (img_bgr[:, :, 2] > thr) & (img_bgr[:, :, 1] < 100) & (img_bgr[:, :, 0] < 100)
    x_sum = np.sum(red_mask, axis=0)
    x_vals = np.where(x_sum > min_len)[0]

    lines = []
    if len(x_vals) > 0:
        group = [x_vals[0]]
        for x in x_vals[1:]:
            if x - group[-1] > 3:   # Start a new group if gap exceeds 3 pixels
                lines.append(int(np.mean(group)))
                group = [x]
            else:
                group.append(x)
        lines.append(int(np.mean(group)))
    return lines


def split_teeth(img_gray: np.ndarray, valleys: List[int], tag: str, out_dir: Path) -> int:
    """
    Crop individual teeth from a jaw image using valley x-coordinates as boundaries.
    Segments narrower than 40 pixels are skipped (likely artifacts).
    Returns: number of tooth images saved.
    """
    H, W = img_gray.shape
    xs = [0] + valleys + [W - 1]
    count = 0
    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i + 1]
        if x2 - x1 < 40:    # Skip segments that are too narrow
            continue
        tooth = img_gray[:, x1:x2]
        imwrite(out_dir / f"{tag}_tooth_{i+1:02d}.png", tooth)
        count += 1
    print(f"  OK {tag}: {count} teeth cropped")
    return count


def step5_split_teeth(ver_dir: Path, out_dir: Path):
    """
    Step 5: For each valley-line image from Step 4, detect the red boundary lines
    and crop out individual teeth into separate files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    valley_files = sorted(ver_dir.glob("*_valley_lines.png"))
    print(f"[Step5] Splitting individual teeth from {len(valley_files)} images...")
    for f in valley_files:
        color = imread_color(f)
        gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
        valleys = detect_vertical_lines(color)
        tag = f.stem
        split_teeth(gray, valleys, tag, out_dir)
    print(f"[Step5] Done -> {out_dir}")


# ── Step 6: Half-tooth splitting ───────────────
def step6_split_half(teeth_dir: Path, out_dir: Path):
    """
    Step 6: Split each individual tooth image vertically at the midpoint
    into left (_L) and right (_R) halves.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(teeth_dir.glob("*.png"))
    print(f"[Step6] Splitting {len(files)} teeth into left/right halves...")
    for f in files:
        img = cv.imread(str(f), cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  WARNING: Cannot read: {f}")
            continue
        H, W = img.shape
        mid = W // 2
        imwrite(out_dir / f"{f.stem}_L.png", img[:, :mid])
        imwrite(out_dir / f"{f.stem}_R.png", img[:, mid:])
    print(f"[Step6] Done -> {out_dir}")


# ── Stage 1 entry point ────────────────────────
def run_image_pipeline(input_dir: Path, base_out: Path):
    """Run all 6 image processing steps in sequence."""
    step1_preprocess(input_dir, base_out / "step1_preprocess")
    step2_scan(input_dir, base_out / "step2_scan")
    step3_split_ul(input_dir, base_out / "step3_upper_lower")
    step4_vertical(base_out / "step3_upper_lower", base_out / "step4_vertical")
    step5_split_teeth(base_out / "step4_vertical", base_out / "step5_teeth")
    step6_split_half(base_out / "step5_teeth", base_out / "step6_half_teeth")
    print("\nStage 1 complete!")


# ══════════════════════════════════════════════
# STAGE 2 — Classification Model Training (teachermodel.ipynb)
# ══════════════════════════════════════════════
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DEVICE = "cuda" if (TORCH_AVAILABLE and __import__("torch").cuda.is_available()) else "cpu"


def _require_torch():
    """Raise an ImportError if PyTorch is not installed."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch torchvision")


# ── Shared utilities ───────────────────────────
def ensure_dir(p: Path):
    """Create a directory and all its parents if they do not exist."""
    p.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> List[Path]:
    """Recursively list all image files under a folder."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


def check_dataset_structure(root: Path, splits=("train", "val", "test")):
    """Verify that the expected ImageFolder-style directory structure exists."""
    for sp in splits:
        for cls in CLASSES:
            p = root / sp / cls
            if not p.exists():
                raise FileNotFoundError(f"Missing folder: {p}")


# ── Metrics (no sklearn dependency) ───────────
def auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute ROC-AUC using the Mann-Whitney U rank statistic.
    Handles tied scores by averaging ranks.
    Returns nan if only one class is present in y_true.
    """
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    order = s.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks for tied scores
    uniq, inv, cnts = np.unique(s, return_inverse=True, return_counts=True)
    for k, cnt in enumerate(cnts):
        if cnt > 1:
            idx = np.where(inv == k)[0]
            ranks[idx] = ranks[idx].mean()
    sum_ranks_pos = ranks[lab == 1].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute Average Precision (area under the precision-recall curve)."""
    order = np.argsort(-scores)
    y = y_true[order]
    tp = (y == 1).astype(np.float32)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum((y == 0).astype(np.float32))
    prec = tp_cum / np.maximum(1.0, tp_cum + fp_cum)
    rec = tp_cum / np.maximum(1.0, tp.sum())
    ap, prev_rec = 0.0, 0.0
    for p, r in zip(prec, rec):
        ap += p * (r - prev_rec)
        prev_rec = r
    return float(ap)


def confusion_from_probs(y_true: np.ndarray, prob_pos: np.ndarray, t: float = 0.5):
    """Compute TP, TN, FP, FN at a given classification threshold t."""
    y_pred = (prob_pos >= t).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, tn, fp, fn


def compute_metrics(y_true: np.ndarray, prob_pos: np.ndarray, t: float = 0.5) -> Dict:
    """
    Compute a full set of classification metrics at threshold t:
    Accuracy, Sensitivity (Recall), Specificity, Precision, F1,
    MCC (Matthews Correlation Coefficient), AUC, and Average Precision.
    """
    tp, tn, fp, fn = confusion_from_probs(y_true, prob_pos, t)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    sens = tp / max(1, tp + fn)     # Sensitivity / Recall / True Positive Rate
    spec = tn / max(1, tn + fp)     # Specificity / True Negative Rate
    prec = tp / max(1, tp + fp)     # Precision / Positive Predictive Value
    f1 = 0.0 if (prec + sens) == 0 else (2 * prec * sens) / (prec + sens)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = 0.0 if denom == 0 else ((tp * tn - fp * fn) / denom)
    auc = auc_rank(y_true, prob_pos)
    ap = average_precision(y_true, prob_pos)
    return dict(acc=acc, sens=sens, spec=spec, prec=prec, f1=f1, mcc=mcc,
                auc=auc, ap=ap, tp=tp, tn=tn, fp=fp, fn=fn)


def tune_threshold(y_val: np.ndarray, p_val: np.ndarray,
                   mode: str = "youden", min_sens: float = 0.85) -> Dict:
    """
    Search all unique predicted probabilities to find the optimal decision threshold.
    Supported modes:
      youden        -- maximize Youden index (sensitivity + specificity - 1)
      f1            -- maximize F1 score
      sens_at_least -- maximize specificity subject to sensitivity >= min_sens
    Returns the best threshold dict including all metrics and a '_score' key.
    """
    cand = np.concatenate([[0.0], np.unique(p_val), [1.0]])
    best = None
    for t in cand:
        m = compute_metrics(y_val, p_val, float(t))
        if mode == "youden":
            score = m["sens"] + m["spec"] - 1.0
        elif mode == "f1":
            score = m["f1"]
        elif mode == "sens_at_least":
            if m["sens"] < min_sens:
                continue
            score = m["spec"]
        else:
            raise ValueError("mode must be one of: youden / f1 / sens_at_least")
        if best is None or score > best["_score"]:
            best = {**m, "t": float(t), "_score": score}
    # Fallback to t=0.5 if no candidate satisfies the constraint
    return best or {**compute_metrics(y_val, p_val, 0.5), "t": 0.5, "_score": -float("inf")}


def bootstrap_ci(y_true: np.ndarray, prob_pos: np.ndarray,
                 t: float = 0.5, n: int = 500, seed: int = SEED) -> Dict:
    """
    Estimate 95% confidence intervals for all metrics using bootstrap resampling.
    Returns a dict mapping metric name -> (lower_2.5%, upper_97.5%).
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    stats = {k: [] for k in ["acc", "sens", "spec", "prec", "f1", "mcc", "auc"]}
    for _ in range(n):
        sample = rng.choice(idx, size=len(idx), replace=True)
        m = compute_metrics(y_true[sample], prob_pos[sample], t)
        for k in stats:
            stats[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
            for k, v in stats.items()}


def fmt_metrics(m: Dict) -> str:
    """Format a metrics dict into a single human-readable string."""
    return (f"Acc {m['acc']:.3f} | Sens {m['sens']:.3f} | Spec {m['spec']:.3f} | "
            f"Prec {m['prec']:.3f} | F1 {m['f1']:.3f} | MCC {m['mcc']:.3f} | "
            f"AUC {m['auc']:.3f} | mAP {m['ap']:.3f} | "
            f"TP {m['tp']} TN {m['tn']} FP {m['fp']} FN {m['fn']}")


def fmt_ci(ci: Dict) -> str:
    """Format a confidence interval dict into a single human-readable string."""
    keys = ["acc", "sens", "spec", "prec", "f1", "mcc", "auc"]
    parts = [f"{k.upper()} [{ci[k][0]:.3f},{ci[k][1]:.3f}]" for k in keys]
    return " | ".join(parts)


# ── DataLoader utilities ───────────────────────
def make_transforms(img_size: int):
    """
    Create training and evaluation torchvision transform pipelines.
    Training: resize, random horizontal flip, random rotation, normalize.
    Evaluation: resize and normalize only (no augmentation).
    """
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def make_loaders(root: Path, img_size: int = IMG_SIZE, batch: int = 32, num_workers: int = 2):
    """
    Build train / val / test DataLoaders from an ImageFolder-style dataset.
    Uses WeightedRandomSampler on training data to handle class imbalance.
    Returns: (train_ds, val_ds, test_ds, train_loader, val_loader, test_loader)
    """
    train_tf, eval_tf = make_transforms(img_size)
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(root / "val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(root / "test",  transform=eval_tf)

    # Verify that all expected class folders are present
    for c in CLASSES:
        for ds, sp in [(train_ds, "train"), (val_ds, "val"), (test_ds, "test")]:
            if c not in ds.class_to_idx:
                raise ValueError(f"[{root}] Class '{c}' not found in {sp}/")

    # Compute per-sample weights as inverse class frequency
    targets = np.array(train_ds.targets)
    class_count = np.bincount(targets, minlength=len(train_ds.class_to_idx))
    sample_weights = 1.0 / np.maximum(1, class_count[targets])
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(train_ds), replacement=True
    )
    kw = dict(num_workers=num_workers, pin_memory=True)
    return (train_ds, val_ds, test_ds,
            DataLoader(train_ds, batch_size=batch, sampler=sampler, **kw),
            DataLoader(val_ds,   batch_size=batch, shuffle=False,   **kw),
            DataLoader(test_ds,  batch_size=batch, shuffle=False,   **kw))


@torch.no_grad()
def collect_probs(model, loader, pos_idx: int):
    """
    Run inference on a DataLoader and collect predicted probabilities for the positive class.
    Returns: (y_binary, prob_positive) as numpy arrays, where y_binary = 1 iff label == pos_idx.
    """
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out   # Handle InceptionV3 named tuple
        prob = torch.softmax(logits, dim=1)[:, pos_idx].cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)
    y = np.concatenate(ys).astype(int)
    p = np.concatenate(ps).astype(float)
    return (y == pos_idx).astype(int), p


# ── Model builder ──────────────────────────────
def build_model(name: str, num_classes: int = 2) -> Tuple[nn.Module, int]:
    """
    Instantiate a pretrained model with its final classification layer replaced
    to output `num_classes` logits.
    Returns: (model, expected_input_size_in_pixels)

    Supported model names:
      resnet18, resnet50, alexnet, mobilenetv2,
      inceptionv3, densenet121, deit_small
    """
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m, 224
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m, 224
    if name == "alexnet":
        m = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
        return m, 224
    if name == "mobilenetv2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m, 224
    if name == "inceptionv3":
        # Disable the auxiliary classifier; only the main head is used
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        m.aux_logits = False
        m.AuxLogits = None
        return m, 299
    if name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m, 224
    if name == "deit_small":
        try:
            import timm
        except ImportError:
            raise ImportError("DeiT-small requires the timm library: pip install timm")
        m = timm.create_model("deit_small_patch16_224", pretrained=True, num_classes=num_classes)
        return m, 224
    raise ValueError(
        f"Unknown model name: '{name}'. "
        "Choose from: resnet18, resnet50, alexnet, mobilenetv2, "
        "inceptionv3, densenet121, deit_small"
    )


# ── Teacher model + Grad-CAM ───────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Registers forward and backward hooks on a target convolutional layer
    to compute a class-discriminative spatial heatmap.

    Reference: Selvaraju et al. (2017), https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._acts = None    # Feature maps captured by the forward hook
        self._grads = None   # Gradients captured by the backward hook
        self._h1 = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_acts", o))
        self._h2 = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grads", go[0]))

    def close(self):
        """Remove hooks to release memory."""
        self._h1.remove()
        self._h2.remove()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a single image tensor and a target class.
        Args:
          x         -- input tensor of shape (1, 3, H, W)
          class_idx -- index of the class to explain
        Returns: numpy array of shape (IMG_SIZE, IMG_SIZE), values in [0, 1].
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[0, class_idx].backward()          # Backpropagate through the target class score
        acts = self._acts.detach()               # Shape: (1, C, h, w)
        grads = self._grads.detach()             # Shape: (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)   # Global average pooling of gradients
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() + 1e-8)    # Normalize to [0, 1]
        return cam.cpu().numpy()


def heatmap_to_bbox(hm: np.ndarray, percentile: int = 85,
                    min_area_frac: float = 0.02, pad_frac: float = 0.08):
    """
    Convert a Grad-CAM heatmap into a bounding box by thresholding the hottest pixels.
    Returns None if the activation region is too small (likely noise).
    pad_frac controls how much extra context is added around the box.
    """
    thr = np.percentile(hm, percentile)
    mask = (hm >= thr).astype(np.uint8)
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask > 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    # Reject boxes that cover too small a fraction of the image
    if (x2 - x1 + 1) * (y2 - y1 + 1) < min_area_frac * IMG_SIZE * IMG_SIZE:
        return None
    # Expand the bounding box by pad_frac for additional context
    pad_x = int(pad_frac * (x2 - x1 + 1))
    pad_y = int(pad_frac * (y2 - y1 + 1))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(IMG_SIZE - 1, x2 + pad_x)
    y2 = min(IMG_SIZE - 1, y2 + pad_y)
    return x1, y1, x2 + 1, y2 + 1   # x2/y2 are exclusive (PIL convention)


# ── Teacher training ───────────────────────────
def run_teacher(src_root: Path, roi_root: Path,
                backbone: str = "resnet18", epochs: int = 15,
                batch: int = 32, lr: float = 1e-4):
    """
    Train a ResNet teacher classifier on the full half-tooth dataset,
    then use Grad-CAM on the best checkpoint to generate ROI crops for every image.
    Saves:
      - Best model weights: roi_root/teacher_best.pth
      - ROI dataset: roi_root/{train,val,test}/{CuRB,N-CuRB}/*.png
    """
    _require_torch()
    check_dataset_structure(src_root)
    ensure_dir(roi_root)

    # Build model and identify the Grad-CAM target layer (last conv in layer4)
    model_obj, _ = build_model(backbone)
    if backbone == "resnet18":
        target_layer = model_obj.layer4[-1].conv2
    elif backbone == "resnet50":
        target_layer = model_obj.layer4[-1].conv3
    else:
        raise ValueError("Teacher backbone must be resnet18 or resnet50")

    model_obj = model_obj.to(DEVICE)

    train_ds, val_ds, _, train_loader, val_loader, _ = make_loaders(src_root, IMG_SIZE, batch)
    class_to_idx = train_ds.class_to_idx
    curb_idx = class_to_idx["CuRB"]

    # Class-weighted cross-entropy to handle imbalance
    counts = {c: len(list_images(src_root / "train" / c)) for c in CLASSES}
    total = sum(counts.values())
    w = torch.zeros(len(class_to_idx), dtype=torch.float32)
    for c in CLASSES:
        w[class_to_idx[c]] = total / (len(CLASSES) * max(1, counts[c]))
    w = w.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.AdamW(model_obj.parameters(), lr=lr, weight_decay=1e-4)

    best_val, best_path = -1.0, roi_root / "teacher_best.pth"

    print(f"\n[Teacher] Training ({backbone}) on {DEVICE}")
    for ep in range(1, epochs + 1):
        # Training step
        model_obj.train()
        loss_sum, n, correct = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model_obj(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
            correct += (logits.argmax(1) == y).sum().item()

        # Validation step: overall accuracy + per-class sensitivity/specificity
        model_obj.eval()
        v_correct, v_total, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model_obj(x).argmax(1)
                v_correct += (pred == y).sum().item()
                v_total += y.size(0)
                tp += ((pred == curb_idx) & (y == curb_idx)).sum().item()
                fn += ((pred != curb_idx) & (y == curb_idx)).sum().item()
                fp += ((pred == curb_idx) & (y != curb_idx)).sum().item()
                tn += ((pred != curb_idx) & (y != curb_idx)).sum().item()

        val_acc = v_correct / max(1, v_total)
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        print(f"  ep {ep:02d}/{epochs} | loss {loss_sum/max(1,n):.4f} | "
              f"train {correct/max(1,n)*100:.2f}% | val {val_acc*100:.2f}% | "
              f"CuRB_sens {sens*100:.2f}% | spec {spec*100:.2f}%")

        # Save checkpoint whenever validation accuracy improves
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"state_dict": model_obj.state_dict(),
                        "class_to_idx": class_to_idx,
                        "backbone": backbone}, best_path)
            print(f"  -> Saved best ({best_val*100:.2f}%): {best_path}")

    # ── Generate ROI dataset with Grad-CAM ──────
    print("\n[Teacher] Generating ROI dataset with Grad-CAM...")
    ckpt = torch.load(best_path, map_location="cpu")
    model_reload, _ = build_model(ckpt["backbone"])
    model_reload.load_state_dict(ckpt["state_dict"])
    model_reload = model_reload.to(DEVICE).eval()

    tl = (model_reload.layer4[-1].conv2 if ckpt["backbone"] == "resnet18"
          else model_reload.layer4[-1].conv3)
    cam = GradCAM(model_reload, tl)

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    from PIL import Image
    for sp in ["train", "val", "test"]:
        for cls in CLASSES:
            ensure_dir(roi_root / sp / cls)
            for p in list_images(src_root / sp / cls):
                img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                x = tf(img).unsqueeze(0).to(DEVICE)
                hm = cam(x, class_idx=curb_idx)
                # Fall back to a fixed center crop if the heatmap is too diffuse
                bbox = heatmap_to_bbox(hm) or (
                    (IMG_SIZE - 96) // 2, (IMG_SIZE - 96) // 2,
                    (IMG_SIZE + 96) // 2, (IMG_SIZE + 96) // 2)
                roi = img.crop(bbox).resize((IMG_SIZE, IMG_SIZE))
                roi.save(roi_root / sp / cls / p.name)
        print(f"  [ROI] {sp} done")

    cam.close()
    print(f"\nROI dataset saved to: {roi_root}")
    return best_path


# ── Full vs ROI ablation ───────────────────────
def train_one(root: Path, out_path: Path,
              backbone: str = "resnet18", epochs: int = 20,
              batch: int = 32, lr: float = 1e-4, patience: int = 6):
    """
    Train a single ResNet model on the dataset at `root`.
    Applies early stopping when validation AUC does not improve for `patience` epochs.
    The best checkpoint (by val AUC) is saved to `out_path`.
    Returns a result dict with raw (t=0.5) and tuned-threshold metrics for val and test.
    """
    _require_torch()
    model_obj, img_size = build_model(backbone)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = \
        make_loaders(root, img_size, batch)
    class_to_idx = train_ds.class_to_idx
    pos_idx = class_to_idx["CuRB"]

    # Class-weighted CE from training distribution
    targets = np.array(train_ds.targets)
    counts = np.bincount(targets, minlength=len(class_to_idx))
    total = counts.sum()
    w = np.zeros(len(class_to_idx), dtype=np.float32)
    for n, idx in class_to_idx.items():
        w[idx] = total / (len(class_to_idx) * max(1, counts[idx]))
    w = torch.tensor(w, device=DEVICE)

    model_obj = model_obj.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.AdamW(model_obj.parameters(), lr=lr, weight_decay=1e-4)

    best_auc, bad_epochs = -1.0, 0

    for ep in range(1, epochs + 1):
        model_obj.train()
        loss_sum, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            out = model_obj(x)
            logits = out.logits if hasattr(out, "logits") else out
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)

        yv, pv = collect_probs(model_obj, val_loader, pos_idx)
        val_auc = auc_rank(yv, pv)
        val_m = compute_metrics(yv, pv, 0.5)
        print(f"  [{root.name}] ep {ep:02d}/{epochs} | loss {loss_sum/max(1,n):.4f} | "
              f"VAL AUC {val_auc:.3f} | sens {val_m['sens']:.3f} | "
              f"spec {val_m['spec']:.3f} | f1 {val_m['f1']:.3f}")

        if not np.isnan(val_auc) and val_auc > best_auc + 1e-4:
            best_auc = val_auc
            torch.save(model_obj.state_dict(), out_path)
            print(f"  -> [SAVE] VAL AUC={best_auc:.3f} -> {out_path}")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  -> Early stopping (best VAL AUC={best_auc:.3f})")
                break

    # Reload the best checkpoint for final evaluation
    if out_path.exists():
        model_obj.load_state_dict(torch.load(out_path, map_location="cpu"))

    yv, pv = collect_probs(model_obj, val_loader, pos_idx)
    yt, pt = collect_probs(model_obj, test_loader, pos_idx)

    best_t_info = tune_threshold(yv, pv)
    best_t = best_t_info["t"]

    return {
        "val_prob":   (yv, pv), "test_prob": (yt, pt),
        "val_auc":    auc_rank(yv, pv), "test_auc": auc_rank(yt, pt),
        "val_m05":    compute_metrics(yv, pv, 0.5),       # At default threshold
        "test_m05":   compute_metrics(yt, pt, 0.5),
        "val_tuned":  compute_metrics(yv, pv, best_t),    # At tuned threshold
        "test_tuned": compute_metrics(yt, pt, best_t),
        "best_t":     best_t,
    }


def run_ablation(full_root: Path, roi_root: Path, out_dir: Path,
                 backbone: str = "resnet18", epochs: int = 20, batch: int = 32):
    """
    Full vs ROI ablation study:
      1. Train on the full half-tooth dataset.
      2. Train on the Grad-CAM ROI dataset.
      3. Print a side-by-side comparison of val and test metrics.
    """
    _require_torch()
    check_dataset_structure(full_root)
    check_dataset_structure(roi_root)
    ensure_dir(out_dir)

    print("\n==================== TRAIN: FULL ====================")
    full_res = train_one(full_root, out_dir / "full_best.pth", backbone, epochs, batch)

    print("\n==================== TRAIN: ROI  ====================")
    roi_res = train_one(roi_root, out_dir / "roi_best.pth", backbone, epochs, batch)

    print("\n" + "=" * 70)
    print("RESULTS (VAL, t=0.5)")
    print(f"FULL: {fmt_metrics(full_res['val_m05'])}")
    print(f"ROI : {fmt_metrics(roi_res['val_m05'])}")
    print("\nRESULTS (TEST, t=0.5)")
    print(f"FULL: {fmt_metrics(full_res['test_m05'])}")
    print(f"ROI : {fmt_metrics(roi_res['test_m05'])}")
    print("\nRESULTS (VAL, tuned threshold)")
    print(f"FULL (t={full_res['best_t']:.4f}): {fmt_metrics(full_res['val_tuned'])}")
    print(f"ROI  (t={roi_res['best_t']:.4f}): {fmt_metrics(roi_res['val_tuned'])}")
    print("\nRESULTS (TEST, tuned threshold)")
    print(f"FULL: {fmt_metrics(full_res['test_tuned'])}")
    print(f"ROI : {fmt_metrics(roi_res['test_tuned'])}")
    print("=" * 70)


# ── 5-model training ───────────────────────────
MODEL_LIST = [
    ("AlexNet",     "alexnet"),
    ("MobileNetV2", "mobilenetv2"),
    ("InceptionV3", "inceptionv3"),
    ("DenseNet121", "densenet121"),
    ("DeiT-small",  "deit_small"),
]


def train_one_model_full(label: str, name: str, data_root: Path, out_dir: Path,
                          epochs: int = 20, batch: int = 16, lr: float = 1e-4,
                          lr_drop_period: int = 10, lr_drop_factor: float = 0.1,
                          bootstrap_n: int = 500) -> Dict:
    """
    Train a single model for the full number of epochs (no early stopping).
    The best checkpoint by validation AUC is saved automatically each epoch.
    After training:
      - Evaluates with the tuned decision threshold.
      - Computes 95% bootstrap CI for all metrics.
      - Saves a validation ROC curve plot.
    Returns a result dict with metrics, CI, training time, and best threshold.
    """
    _require_torch()
    model_obj, img_size = build_model(name)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = \
        make_loaders(data_root, img_size, batch)
    class_to_idx = train_ds.class_to_idx
    pos_idx = class_to_idx["CuRB"]

    model_obj = model_obj.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model_obj.parameters(), lr=lr)
    # Step decay: multiply LR by lr_drop_factor every lr_drop_period epochs
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_drop_period, gamma=lr_drop_factor)

    best_auc, best_path = -1.0, out_dir / f"{label}_best.pth"
    start = time.time()

    print(f"\n[{label}] Training | img={img_size} | epochs={epochs} | batch={batch}")
    for ep in range(1, epochs + 1):
        model_obj.train()
        loss_sum, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            out = model_obj(x)
            logits = out.logits if hasattr(out, "logits") else out
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()

        yv, pv = collect_probs(model_obj, val_loader, pos_idx)
        val_auc = auc_rank(yv, pv)
        val_m = compute_metrics(yv, pv, 0.5)
        print(f"  ep {ep:02d}/{epochs} | loss {loss_sum/max(1,n):.4f} | "
              f"VAL AUC {val_auc:.3f} | sens {val_m['sens']:.3f} | "
              f"spec {val_m['spec']:.3f} | f1 {val_m['f1']:.3f}")

        if not np.isnan(val_auc) and val_auc > best_auc + 1e-4:
            best_auc = val_auc
            torch.save(model_obj.state_dict(), best_path)
            print(f"  -> [SAVE] VAL AUC={best_auc:.3f}")

    # Load best checkpoint for final evaluation
    if best_path.exists():
        model_obj.load_state_dict(torch.load(best_path, map_location="cpu"))

    yv, pv = collect_probs(model_obj, val_loader, pos_idx)
    yt, pt = collect_probs(model_obj, test_loader, pos_idx)

    # Tune decision threshold on the validation set
    best_t_info = tune_threshold(yv, pv)
    best_t = best_t_info["t"]

    val_m  = compute_metrics(yv, pv, best_t)
    test_m = compute_metrics(yt, pt, best_t)
    val_ci  = bootstrap_ci(yv, pv, best_t, bootstrap_n)
    test_ci = bootstrap_ci(yt, pt, best_t, bootstrap_n)

    # Save validation ROC curve (non-fatal if matplotlib is unavailable)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _roc(y_true, scores):
            """Compute FPR / TPR arrays for the ROC curve."""
            thrs = np.concatenate([[1.0], np.unique(scores), [0.0]])
            fprs, tprs = [], []
            for t in thrs:
                tp, tn, fp, fn = confusion_from_probs(y_true, scores, t)
                tprs.append(tp / max(1, tp + fn))
                fprs.append(fp / max(1, fp + tn))
            return np.array(fprs), np.array(tprs)

        fpr, tpr = _roc(yv, pv)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(fpr, tpr, label=f"AUC={auc_rank(yv, pv):.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")   # Random classifier diagonal
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"{label} VAL ROC"); ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{label}_val_roc.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass   # Continue silently if plotting fails

    elapsed = time.time() - start
    return {
        "label":    label,
        "best_t":   best_t,
        "val":      val_m,
        "test":     test_m,
        "val_ci":   val_ci,
        "test_ci":  test_ci,
        "time_sec": elapsed,
    }


def run_train5(data_root: Path, out_dir: Path,
               epochs: int = 20, batch: int = 16, bootstrap_n: int = 500):
    """
    Train all 5 models sequentially on the same dataset.
    Prints a summary table showing tuned-threshold metrics and 95% CI
    for both validation and test sets.
    """
    _require_torch()
    check_dataset_structure(data_root)
    ensure_dir(out_dir)

    print(f"[DEVICE] {DEVICE} | epochs={epochs} | batch={batch}")
    results = []
    for label, name in MODEL_LIST:
        try:
            res = train_one_model_full(
                label, name, data_root, out_dir,
                epochs=epochs, batch=batch, bootstrap_n=bootstrap_n)
            results.append(res)
        except ImportError as e:
            print(f"  [SKIP] {label}: {e}")

    print("\n================ SUMMARY (VAL, tuned threshold) ================")
    for r in results:
        print(f"{r['label']} (t={r['best_t']:.4f}): {fmt_metrics(r['val'])}")
        print(f"  95% CI: {fmt_ci(r['val_ci'])}")

    print("\n================ SUMMARY (TEST, tuned threshold) ===============")
    for r in results:
        print(f"{r['label']}: {fmt_metrics(r['test'])}")
        print(f"  95% CI: {fmt_ci(r['test_ci'])}")

    print(f"\nDone! Output saved to: {out_dir}")


# ══════════════════════════════════════════════
# CLI argument parser
# ══════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CuRB X-ray Classification Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── image ──────────────────────────────────
    p_img = sub.add_parser("image", help="Stage 1: X-ray image processing (6 steps)")
    p_img.add_argument("--input-dir", required=True, type=Path,
                       help="Folder containing raw X-ray images (.jpg / .png)")
    p_img.add_argument("--out-dir", default=Path("output_pipeline"), type=Path,
                       help="Root folder for all step output sub-directories")

    # ── teacher ────────────────────────────────
    p_t = sub.add_parser(
        "teacher",
        help="Train ResNet teacher model and generate Grad-CAM ROI dataset")
    p_t.add_argument("--src-root", required=True, type=Path,
                     help="Half-tooth dataset (train/val/test x CuRB/N-CuRB)")
    p_t.add_argument("--roi-root", required=True, type=Path,
                     help="Output directory for the Grad-CAM ROI dataset")
    p_t.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    p_t.add_argument("--epochs", default=15, type=int)
    p_t.add_argument("--batch",  default=32, type=int)
    p_t.add_argument("--lr",     default=1e-4, type=float)

    # ── ablation ───────────────────────────────
    p_ab = sub.add_parser("ablation", help="Full vs ROI ablation study")
    p_ab.add_argument("--full-root", required=True, type=Path,
                      help="Full half-tooth dataset")
    p_ab.add_argument("--roi-root",  required=True, type=Path,
                      help="Grad-CAM ROI dataset")
    p_ab.add_argument("--out-dir",   default=Path("ablation_out"), type=Path)
    p_ab.add_argument("--backbone",  default="resnet18", choices=["resnet18", "resnet50"])
    p_ab.add_argument("--epochs",    default=20, type=int)
    p_ab.add_argument("--batch",     default=32, type=int)

    # ── train5 ─────────────────────────────────
    p_t5 = sub.add_parser(
        "train5",
        help="Train 5 models: AlexNet / MobileNetV2 / InceptionV3 / DenseNet121 / DeiT-small")
    p_t5.add_argument("--data-root", required=True, type=Path,
                      help="Dataset root (train/val/test x CuRB/N-CuRB)")
    p_t5.add_argument("--out-dir",   default=Path("model5_out"), type=Path)
    p_t5.add_argument("--epochs",    default=20,  type=int)
    p_t5.add_argument("--batch",     default=16,  type=int)
    p_t5.add_argument("--bootstrap-n", default=500, type=int,
                      help="Number of bootstrap iterations for 95% CI estimation")

    # ── all ────────────────────────────────────
    p_all = sub.add_parser(
        "all",
        help="Full pipeline: Stage 1 image processing + teacher training + ablation study")
    p_all.add_argument("--input-dir", required=True, type=Path,
                       help="Folder containing raw X-ray images")
    p_all.add_argument("--dataset-root", type=Path,
                       help="Pre-built half-tooth dataset; if omitted, Stage 2 is skipped")
    p_all.add_argument("--out-dir",  default=Path("pipeline_out"), type=Path)
    p_all.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])

    return parser


def main():
    set_seed()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "image":
        run_image_pipeline(args.input_dir, args.out_dir)

    elif args.command == "teacher":
        run_teacher(args.src_root, args.roi_root,
                    args.backbone, args.epochs, args.batch, args.lr)

    elif args.command == "ablation":
        run_ablation(args.full_root, args.roi_root, args.out_dir,
                     args.backbone, args.epochs, args.batch)

    elif args.command == "train5":
        run_train5(args.data_root, args.out_dir,
                   args.epochs, args.batch, args.bootstrap_n)

    elif args.command == "all":
        img_out = args.out_dir / "image_pipeline"
        run_image_pipeline(args.input_dir, img_out)

        if args.dataset_root:
            roi_root = args.out_dir / "roi_dataset"
            run_teacher(args.dataset_root, roi_root, args.backbone)
            run_ablation(args.dataset_root, roi_root, args.out_dir / "ablation")
        else:
            print("\n[INFO] --dataset-root not provided. Skipping Stage 2 training.")

    print("\nDone.")


if __name__ == "__main__":
    main()

