"""
features.py — Signal extraction for screen-vs-real photo detection.  (v2)

"""

import numpy as np
import cv2
from scipy import stats

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("[features] WARNING: PyWavelets not found. Wavelet features degraded. "
          "Run: pip install PyWavelets", flush=True)

try:
    from skimage.feature import local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[features] WARNING: scikit-image not found. LBP features degraded. "
          "Run: pip install scikit-image", flush=True)

#Processing constants
FFT_CROP     = 512  
PROC_SIZE    = 512   
TEXTURE_SIZE = 192   

FEATURE_DIM = 68   

#Preprocessing

def preprocess(img_bgr: np.ndarray) -> dict:
 
    h, w = img_bgr.shape[:2]

    #FFT crop
    cy, cx = h // 2, w // 2
    half = FFT_CROP // 2
    y1, y2 = max(0, cy - half), min(h, cy + half)
    x1, x2 = max(0, cx - half), min(w, cx + half)
    crop = img_bgr[y1:y2, x1:x2]
    ph, pw = FFT_CROP - crop.shape[0], FFT_CROP - crop.shape[1]
    if ph > 0 or pw > 0:
        crop = cv2.copyMakeBorder(crop, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    #PROC_SIZE
    bgr_small   = cv2.resize(img_bgr, (PROC_SIZE, PROC_SIZE), interpolation=cv2.INTER_LINEAR)
    gray_small  = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)
    hsv_small   = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    ycrcb_small = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2YCrCb)

    #TEXTURE_SIZE
    bgr_tex   = cv2.resize(img_bgr, (TEXTURE_SIZE, TEXTURE_SIZE), interpolation=cv2.INTER_LINEAR)
    gray_tex  = cv2.cvtColor(bgr_tex, cv2.COLOR_BGR2GRAY)
    hsv_tex   = cv2.cvtColor(bgr_tex, cv2.COLOR_BGR2HSV)
    ycrcb_tex = cv2.cvtColor(bgr_tex, cv2.COLOR_BGR2YCrCb)

    return dict(
        gray_crop=gray_crop,
        gray_small=gray_small, bgr_small=bgr_small,
        hsv_small=hsv_small, ycrcb_small=ycrcb_small,
        gray_tex=gray_tex, hsv_tex=hsv_tex, ycrcb_tex=ycrcb_tex,
    )


def fast_corr(a: np.ndarray, b: np.ndarray) -> float:

    a = a.ravel()
    b = b.ravel()
    if len(a) < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-8
    val = np.dot(a, b) / den
    return float(val) if np.isfinite(val) else 0.0


#Shared helper

def _texture_stats(channel: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:

    if not HAS_SKIMAGE:
        ch = channel.astype(np.float64)
        gx = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        return np.array([
            float(np.std(mag)),
            float(np.sum((mag / (mag.sum() + 1e-8)) ** 2)),
            float(stats.entropy(np.histogram(mag, bins=10)[0] + 1e-12)),
            float(np.percentile(mag, 95)),
        ], dtype=np.float32)

    lbp = local_binary_pattern(channel, P, R, method='uniform')
    n_bins = P + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    return np.array([
        float(np.std(hist)),
        float(np.sum(hist ** 2)),
        float(stats.entropy(hist + 1e-12)),
        float(np.max(hist)),
    ], dtype=np.float32)


#Feature Group 1: FFT / Frequency Analysis (luminance)

_H = _W = FFT_CROP
_HANN_WIN = np.outer(np.hanning(_H), np.hanning(_W)).astype(np.float32)
_CY, _CX = _H // 2, _W // 2
_Y_, _X_ = np.ogrid[:_H, :_W]
_DIST = np.sqrt((_X_ - _CX) ** 2 + (_Y_ - _CY) ** 2)
_R_DC, _R_MID = min(_H, _W) * 0.04, min(_H, _W) * 0.45
_DC_MASK  = _DIST < _R_DC
_MID_MASK = (_DIST >= _R_DC) & (_DIST <= _R_MID)
_HI_MASK  = _DIST > _R_MID
_VERT_BAND  = _MID_MASK & (np.abs(_X_ - _CX) < _W * 0.03)
_HORIZ_BAND = _MID_MASK & (np.abs(_Y_ - _CY) < _H * 0.03)


def fft_features(gray_crop: np.ndarray) -> np.ndarray:

    img = gray_crop.astype(np.float32)
    img_w = img * _HANN_WIN

    f = np.fft.fft2(img_w)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    mid_vals = magnitude[_MID_MASK]
    mean_m   = mid_vals.mean()
    std_m    = mid_vals.std()

    peak_ratio   = mid_vals.max() / (mean_m + 1e-8)
    n_peaks_frac = (mid_vals > mean_m + 3.0 * std_m).mean()

    total    = magnitude.sum() + 1e-8
    mid_frac = mid_vals.sum() / total
    hi_frac  = magnitude[_HI_MASK].sum() / total

    cov = std_m / (mean_m + 1e-8)

    vp = magnitude[_VERT_BAND].max()  if _VERT_BAND.any()  else 0.0
    hp = magnitude[_HORIZ_BAND].max() if _HORIZ_BAND.any() else 0.0
    aniso = abs(vp - hp) / (vp + hp + 1e-8)

    return np.array([peak_ratio, n_peaks_frac, mid_frac, hi_frac, cov, aniso],
                    dtype=np.float32)


#Feature Group 2: Luminance Noise Correlation (Wang 2017)

def noise_correlation_features(gray_small: np.ndarray) -> np.ndarray:

    img = gray_small.astype(np.float32)

    diffs = {
        'dx':  img[:, 1:] - img[:, :-1],
        'dy':  img[1:, :] - img[:-1, :],
        'dxy': img[1:, 1:] - img[:-1, :-1],
        'dyx': img[1:, :-1] - img[:-1, 1:],
    }

    feats = []
    for d in diffs.values():
        flat = d.flatten()
        feats.append(fast_corr(flat[:-1], flat[1:]) if len(flat) > 2 else 0.0)

    dx = diffs['dx']
    if dx.shape[0] > 1:
        feats.append(fast_corr(dx[:-1, :], dx[1:, :]))
    else:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32)


#Feature Group 3: Wavelet Subband Statistics (luminance, trimmed)

def wavelet_features(gray_small: np.ndarray) -> np.ndarray:

    img = gray_small.astype(np.float32)

    if not HAS_PYWT:
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        hp = cv2.filter2D(img, -1, kernel)
        abs_hp = np.abs(hp)
        core = np.array([float(np.std(hp)), float(np.mean(abs_hp))], dtype=np.float32)
        return np.tile(core, 3)  # pad to 6

    cA, (cH, cV, cD) = pywt.dwt2(img, 'db4')
    feats = []
    for band in [cH, cV, cD]:
        abs_b = np.abs(band)
        feats.extend([float(np.std(band)), float(np.mean(abs_b))])
    return np.array(feats, dtype=np.float32)


#Feature Group 4: Luminance LBP Texture

def lbp_features(gray_tex: np.ndarray) -> np.ndarray:
  
    return _texture_stats(gray_tex, P=16, R=2)


#Feature Group 5: Edge Sharpness Analysis

def sharpness_features(gray_small: np.ndarray) -> np.ndarray:
 
    img = gray_small.astype(np.float64)

    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap_var = float(np.var(lap))

    blurred = cv2.GaussianBlur(img.astype(np.float32), (15, 15), 0).astype(np.float64)
    hf = img - blurred
    hf_ratio = float(np.var(hf) / (np.var(img) + 1e-8))

    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.Canny(gray_small, 50, 150)
    edge_mask = edges > 0

    if edge_mask.any():
        gx_abs = np.abs(gx[edge_mask])
        gy_abs = np.abs(gy[edge_mask])
        axis_aligned = ((gx_abs > 3 * gy_abs) | (gy_abs > 3 * gx_abs)).mean()
    else:
        axis_aligned = 0.0

    tenengrad = float(np.mean(gx**2 + gy**2))

    return np.array([lap_var, hf_ratio, float(axis_aligned), tenengrad],
                    dtype=np.float32)


#Feature Group 6: RGB Color Channel Statistics

def color_features(bgr_small: np.ndarray) -> np.ndarray:

    b = bgr_small[:, :, 0].astype(np.float32)
    g = bgr_small[:, :, 1].astype(np.float32)
    r = bgr_small[:, :, 2].astype(np.float32)

    rg, rb, gb = fast_corr(r, g), fast_corr(r, b), fast_corr(g, b)

    hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    rm, gm, bm = r.mean(), g.mean(), b.mean()
    total = rm + gm + bm + 1e-8

    return np.array([
        rg, rb, gb,
        float(sat.mean()), float(sat.std()), float(val.std()),
        rm / total, gm / total, bm / total,
    ], dtype=np.float32)


#Feature Group 7: JPEG Block Artifact

def jpeg_artifact_features(gray_small: np.ndarray) -> np.ndarray:
    """Double-JPEG 8×8 block boundary signature. Kept from v1. Returns 2."""
    img = gray_small.astype(np.float32)
    ratios = []
    for axis in [0, 1]:
        grad = np.abs(np.diff(img, axis=axis))
        size = grad.shape[axis]
        boundaries     = np.arange(7, size - 1, 8)
        non_boundaries = np.setdiff1d(np.arange(size - 1), boundaries)
        if len(boundaries) == 0 or len(non_boundaries) == 0:
            ratios.append(1.0)
            continue
        if axis == 0:
            bnd_mean, non_bnd_mean = grad[boundaries, :].mean(), grad[non_boundaries, :].mean()
        else:
            bnd_mean, non_bnd_mean = grad[:, boundaries].mean(), grad[:, non_boundaries].mean()
        ratios.append(float(bnd_mean / (non_bnd_mean + 1e-8)))
    return np.array(ratios, dtype=np.float32)


#Feature Group 8: Luminance Distribution

def luminance_features(gray_small: np.ndarray) -> np.ndarray:

    flat = gray_small.flatten().astype(np.float32)
    dark_frac   = float((flat < 20).mean())
    bright_frac = float((flat > 235).mean())
    q25, q75    = np.percentile(flat, [25, 75])
    iqr_norm    = float((q75 - q25) / 255.0)
    return np.array([dark_frac, bright_frac, iqr_norm], dtype=np.float32)




#Feature Group 9: Color-Channel LBP Texture

def color_lbp_features(hsv_tex: np.ndarray, ycrcb_tex: np.ndarray) -> np.ndarray:

    H = hsv_tex[:, :, 0]
    S = hsv_tex[:, :, 1]
    Cr = ycrcb_tex[:, :, 1]
    Cb = ycrcb_tex[:, :, 2]

    feats = [_texture_stats(ch, P=8, R=1) for ch in [H, S, Cb, Cr]]
    return np.concatenate(feats).astype(np.float32)


#Feature Group 10: Chroma Noise Correlation

def chroma_noise_correlation_features(ycrcb_small: np.ndarray) -> np.ndarray:

    feats = []
    for ch in [ycrcb_small[:, :, 1].astype(np.float32),   # Cr
              ycrcb_small[:, :, 2].astype(np.float32)]:  # Cb
        dx = ch[:, 1:] - ch[:, :-1]
        dy = ch[1:, :] - ch[:-1, :]
        for d in [dx, dy]:
            flat = d.flatten()
            feats.append(fast_corr(flat[:-1], flat[1:]) if len(flat) > 2 else 0.0)
    return np.array(feats, dtype=np.float32)


#Feature Group 11: Specular Highlight Geometry

def specular_features(bgr_small: np.ndarray, hsv_small: np.ndarray) -> np.ndarray:

    V = hsv_small[:, :, 2].astype(np.float32)
    S = hsv_small[:, :, 1].astype(np.float32)

    mask = ((V > 0.80 * 255) & (S < 0.30 * 255)).astype(np.uint8)
    h, w = mask.shape
    total_px = h * w

    highlight_frac = float(mask.sum()) / total_px

    if mask.sum() < 10:   
        return np.array([highlight_frac, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    n_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    areas = stats_cc[1:, cv2.CC_STAT_AREA]
    areas = areas[areas >= 5]

    if len(areas) == 0:
        return np.array([highlight_frac, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    n_blobs_norm = float(np.log1p(len(areas)))
    largest_idx  = np.argmax(areas)
    largest_area = float(areas[largest_idx])
    concentration = largest_area / (areas.sum() + 1e-8)

    largest_label = 1 + largest_idx
    blob_mask = (labels == largest_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        compactness = float(4 * np.pi * largest_area / (perimeter ** 2 + 1e-8))
        compactness = min(compactness, 1.0)
    else:
        compactness = 0.0

    ys, xs = np.where(labels == largest_label)
    touches_edge = float(
        (ys.min() <= 1) or (ys.max() >= h - 2) or
        (xs.min() <= 1) or (xs.max() >= w - 2)
    )

    return np.array([highlight_frac, n_blobs_norm, concentration, compactness,
                     touches_edge], dtype=np.float32)


#Feature Group 12: Chromaticity / White-Point Consistency

def chromaticity_features(bgr_small: np.ndarray) -> np.ndarray:

    img = bgr_small.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    h, w = r.shape
    bs_h, bs_w = h // 4, w // 4
    chroma_r, chroma_g = [], []
    for i in range(4):
        for j in range(4):
            br = r[i*bs_h:(i+1)*bs_h, j*bs_w:(j+1)*bs_w].mean()
            bg = g[i*bs_h:(i+1)*bs_h, j*bs_w:(j+1)*bs_w].mean()
            bb = b[i*bs_h:(i+1)*bs_h, j*bs_w:(j+1)*bs_w].mean()
            tot = br + bg + bb + 1e-6
            chroma_r.append(br / tot)
            chroma_g.append(bg / tot)
    std_chroma = float(np.std(chroma_r) + np.std(chroma_g))

    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(
        np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) +
        0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )

    rm, gm, bm = r.mean(), g.mean(), b.mean()
    br_extremity = float(abs(bm / (rm + 1e-6) - 1.0))

    return np.array([std_chroma, colorfulness, br_extremity, 0.0], dtype=np.float32)

FEATURE_GROUPS = [
    ("FFT / moiré (luma)",        0,  6),
    ("Noise correlation (luma)",  6,  11),
    ("Wavelet (luma)",            11, 17),
    ("LBP texture (luma)",        17, 21),
    ("Sharpness / edges",         21, 25),
    ("RGB color stats",           25, 34),
    ("JPEG block artifact",       34, 36),
    ("Luminance distribution",    36, 39),
    ("Color-LBP (H/S/Cb/Cr)",     39, 55),
    ("Chroma noise correlation",  55, 59),
    ("Specular geometry",         59, 64),
    ("Chromaticity / white-pt",   64, 68),
]

FEATURE_NAMES = [
    "FFT peak ratio", "FFT peak fraction", "FFT mid-freq energy",
    "FFT high-freq energy", "FFT coeff of variation", "FFT anisotropy",
    "Noise corr (dx)", "Noise corr (dy)", "Noise corr (dxy)",
    "Noise corr (dyx)", "Noise corr (2D rows)",
    "Wavelet H std", "Wavelet H mean", "Wavelet V std",
    "Wavelet V mean", "Wavelet D std", "Wavelet D mean",
    "Luma-LBP std", "Luma-LBP energy", "Luma-LBP entropy", "Luma-LBP max",
    "Laplacian variance", "HF energy ratio", "Axis-aligned edges", "Tenengrad",
    "Color corr R-G", "Color corr R-B", "Color corr G-B",
    "Saturation mean", "Saturation std", "Value std",
    "R fraction", "G fraction", "B fraction",
    "JPEG block ratio (vert)", "JPEG block ratio (horiz)",
    "Dark pixel fraction", "Bright pixel fraction", "Luminance IQR",
    "Hue-LBP std", "Hue-LBP energy", "Hue-LBP entropy", "Hue-LBP max",
    "Sat-LBP std", "Sat-LBP energy", "Sat-LBP entropy", "Sat-LBP max",
    "Cb-LBP std", "Cb-LBP energy", "Cb-LBP entropy", "Cb-LBP max",
    "Cr-LBP std", "Cr-LBP energy", "Cr-LBP entropy", "Cr-LBP max",
    "Cr noise corr (dx)", "Cr noise corr (dy)",
    "Cb noise corr (dx)", "Cb noise corr (dy)",
    "Highlight area frac", "Highlight blob count", "Highlight concentration",
    "Highlight compactness", "Highlight touches edge",
    "Block chromaticity std", "Colorfulness", "B/R ratio extremity", "(reserved)",
]

assert len(FEATURE_NAMES) == FEATURE_DIM, \
    f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries, expected {FEATURE_DIM}"


#Master extractor

def extract_features(img_bgr: np.ndarray) -> np.ndarray:

    p = preprocess(img_bgr)

    parts = [
        fft_features(p['gray_crop']),                                   # 6
        noise_correlation_features(p['gray_small']),                    # 5
        wavelet_features(p['gray_small']),                               # 6
        lbp_features(p['gray_tex']),                                     # 4
        sharpness_features(p['gray_small']),                             # 4
        color_features(p['bgr_small']),                                  # 9
        jpeg_artifact_features(p['gray_small']),                         # 2
        luminance_features(p['gray_small']),                             # 3
        color_lbp_features(p['hsv_tex'], p['ycrcb_tex']),                # 16  ← new
        chroma_noise_correlation_features(p['ycrcb_small']),             # 4   ← new
        specular_features(p['bgr_small'], p['hsv_small']),               # 5   ← new
        chromaticity_features(p['bgr_small']),                           # 4   ← new
    ]                                                                    # = 68

    feats = np.concatenate(parts).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)

    assert len(feats) == FEATURE_DIM, \
        f"Feature dim mismatch: got {len(feats)}, expected {FEATURE_DIM}"

    return feats