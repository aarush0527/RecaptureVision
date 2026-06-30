#!/usr/bin/env python3
"""
predict.py — SalesCode recapture detector.

Usage:
    python predict.py <image_path>
    python predict.py <image_path> --verbose

Output (stdout):
    0.93          ← single float, 0 = real photo, 1 = photo of a screen

Stderr:
    # SCREEN (21.4ms)   ← label + latency (won't interfere with pipelines)

Two modes:
  ① Trained model (model.pkl exists) — highest accuracy, recommended.
     Run `python train.py` first with your real/ and screen/ folders.
  ② Classical fallback (no model.pkl) — works immediately, lower accuracy.
     Uses FFT peak analysis only. Expect ~75–85% accuracy vs ~95% with model.
"""

import sys
import os
import time
import pickle
import argparse
import textwrap
import numpy as np
import cv2

from features import extract_features, HAS_PYWT, HAS_SKIMAGE, FEATURE_DIM

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')

#Model loading

def load_model():
    """Load and validate the trained model bundle."""
    if not os.path.exists(MODEL_PATH):
        return None

    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)

    if bundle.get('has_pywt') != HAS_PYWT or bundle.get('has_skimage') != HAS_SKIMAGE:
        print(
            "WARNING: Library availability differs between training and inference.\n"
            f"  Training: pywt={bundle.get('has_pywt')}  skimage={bundle.get('has_skimage')}\n"
            f"  Now:      pywt={HAS_PYWT}  skimage={HAS_SKIMAGE}\n"
            "  Feature vectors may not match. Retrain: python train.py",
            file=sys.stderr
        )

    if bundle.get('feature_dim') != FEATURE_DIM:
        print(
            f"WARNING: Model expects {bundle.get('feature_dim')} features, "
            f"but current code extracts {FEATURE_DIM}. Retrain the model.",
            file=sys.stderr
        )

    return bundle


#Classical fallback (no trained model)

def classical_score(features: np.ndarray) -> float:

    fft_peak_ratio = float(features[0])
    fft_peak_frac  = float(features[1])   
    fft_cov        = float(features[4])   
    noise_dx       = float(features[6])   

    s_peak  = np.clip((fft_peak_ratio - 4.0)  / 12.0, 0.0, 1.0)
    s_pfrac = np.clip((fft_peak_frac  - 0.0)  /  0.005, 0.0, 1.0)
    s_cov   = np.clip((fft_cov        - 0.4)  /  0.8,  0.0, 1.0)

    s_noise = np.clip((0.5 - noise_dx) / 0.3,           0.0, 1.0)

    score = 0.45 * s_peak + 0.25 * s_pfrac + 0.20 * s_cov + 0.10 * s_noise
    return float(np.clip(score, 0.01, 0.99))


#Main predictor

def predict(image_path: str, model_bundle=None):
  
    t_start = time.perf_counter()

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path!r}")

    features = extract_features(img)
    t_feat = time.perf_counter()

    if model_bundle is not None:
        scaler = model_bundle['scaler']
        model  = model_bundle['model']
        X_s    = scaler.transform(features.reshape(1, -1))
        score  = float(model.predict_proba(X_s)[0][1])
        method = 'trained_model'
    else:
        score  = classical_score(features)
        method = 'classical_fallback'

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000
    feat_ms  = (t_feat - t_start) * 1000

    return score, total_ms, feat_ms, method, features


#Verbose breakdow

FEATURE_NAMES = [
    # FFT (0–5)
    "FFT peak ratio",
    "FFT peak fraction",
    "FFT mid-freq energy",
    "FFT high-freq energy",
    "FFT coeff of variation",
    "FFT anisotropy",
    # Noise correlation (6–10)
    "Noise corr (dx)",
    "Noise corr (dy)",
    "Noise corr (dxy)",
    "Noise corr (dyx)",
    "Noise corr (2D rows)",
    # Wavelet (11–28)  — abbreviated
    "Wavelet L1-H std", "Wavelet L1-H mean", "Wavelet L1-H p95",
    "Wavelet L1-V std", "Wavelet L1-V mean", "Wavelet L1-V p95",
    "Wavelet L1-D std", "Wavelet L1-D mean", "Wavelet L1-D p95",
    "Wavelet L2-H std", "Wavelet L2-H mean", "Wavelet L2-H p95",
    "Wavelet L2-V std", "Wavelet L2-V mean", "Wavelet L2-V p95",
    "Wavelet L2-D std", "Wavelet L2-D mean", "Wavelet L2-D p95",
    # LBP (29–32)
    "LBP hist mean", "LBP hist std", "LBP hist max", "LBP entropy",
    # Sharpness (33–36)
    "Laplacian variance", "HF energy ratio", "Axis-aligned edges", "Tenengrad",
    # Color (37–45)
    "Color corr R-G", "Color corr R-B", "Color corr G-B",
    "Saturation mean", "Saturation std", "Value std",
    "R fraction", "G fraction", "B fraction",
    # JPEG artifacts (46–47)
    "JPEG block ratio (vert)", "JPEG block ratio (horiz)",
    # Luminance (48–50)
    "Dark pixel fraction", "Bright pixel fraction", "Luminance IQR",
]


def print_verbose(image_path, score, total_ms, feat_ms, method, features, model_bundle):
    label      = 'SCREEN' if score > 0.5 else 'REAL'
    confidence = abs(score - 0.5) * 2

    print(f"\n{'─'*60}")
    print(f"  Image   : {image_path}")
    print(f"  Score   : {score:.4f}  →  {label}  (confidence {confidence:.0%})")
    print(f"  Method  : {method}")
    print(f"  Timing  : {total_ms:.1f}ms total  |  {feat_ms:.1f}ms features")

    if model_bundle:
        cv_acc = model_bundle.get('cv_accuracy_mean', 0)
        cv_std = model_bundle.get('cv_accuracy_std',  0)
        print(f"  Model   : CV accuracy {cv_acc:.1%} ± {cv_std:.1%}  "
              f"(trained on {model_bundle.get('n_train_real', '?')} real + "
              f"{model_bundle.get('n_train_screen', '?')} screen images)")

    print(f"\n  Feature breakdown (key signals):")
    print(f"  {'Feature':<28}  {'Value':>10}  Signal direction")
    print(f"  {'─'*28}  {'─'*10}  {'─'*20}")

    directions = {
        0: "↑ screen",  1: "↑ screen",  2: "↑ screen",  3: "↓ screen",
        4: "↑ screen",  5: "? varies",
        6: "↓ screen",  7: "↓ screen",  8: "? varies",  9: "? varies",  10: "↓ screen",
    }
    for i, (name, val) in enumerate(zip(FEATURE_NAMES, features)):
        if i < 11 or i in [29, 30, 31, 32, 33, 34, 35, 46, 47, 48, 49, 50]:
            hint = directions.get(i, "")
            print(f"  {name:<28}  {val:>10.4f}  {hint}")

    print(f"{'─'*60}\n")


#CLI

def main():
    parser = argparse.ArgumentParser(
        description='Predict: real photo (0.0) or photo of a screen (1.0)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python predict.py shelf_photo.jpg
          python predict.py upload_0042.jpg --verbose
          python predict.py *.jpg           (batch: one score per line)
        """)
    )
    parser.add_argument('images', nargs='+', help='Image file(s) to classify')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show feature breakdown and latency details')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold (default 0.5; lower = flag more as screen)')
    args = parser.parse_args()

    model_bundle = load_model()

    if model_bundle is None:
        print(
            "WARNING: model.pkl not found. Using classical FFT fallback (~75–85% accuracy).\n"
            "  → Train a model: python train.py --real real/ --screen screen/",
            file=sys.stderr
        )

    exit_code = 0

    for image_path in args.images:
        if not os.path.exists(image_path):
            print(f"ERROR: {image_path!r} not found", file=sys.stderr)
            exit_code = 1
            continue

        try:
            score, total_ms, feat_ms, method, features = predict(
                image_path, model_bundle
            )
        except Exception as e:
            print(f"ERROR processing {image_path}: {e}", file=sys.stderr)
            exit_code = 1
            continue

        label = 'SCREEN' if score >= args.threshold else 'REAL'

        if len(args.images) == 1:
            print(f"{score:.4f}")
        else:
            print(f"{score:.4f}  {image_path}")

        print(f"# {label} ({total_ms:.1f}ms)", file=sys.stderr)

        if args.verbose:
            print_verbose(image_path, score, total_ms, feat_ms,
                          method, features, model_bundle)

    return exit_code


if __name__ == '__main__':
    sys.exit(main())