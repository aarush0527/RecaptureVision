"""
evaluate.py — Evaluate the trained model on a held-out test set.

Usage:
    python evaluate.py --real test_real/ --screen test_screen/
    python evaluate.py --real test_real/ --screen test_screen/ --threshold 0.4

"""

import os
import sys
import glob
import argparse
import time
import pickle
import numpy as np
import cv2
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score,
    precision_recall_curve, roc_curve
)
from predict import predict, load_model


IMG_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')


def collect_paths(directory):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description='Evaluate recapture detector on held-out images')
    parser.add_argument('--real',      default='test_real',   help='Folder of real test photos')
    parser.add_argument('--screen',    default='test_screen', help='Folder of screen test photos')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold (default 0.5)')
    args = parser.parse_args()

    model_bundle = load_model()
    if model_bundle is None:
        print("WARNING: No model.pkl found. Evaluation will use classical fallback.",
              file=sys.stderr)

    all_scores  = []
    all_labels  = []
    all_latencies = []
    errors = []

    for label, directory in [(0, args.real), (1, args.screen)]:
        cls_name = 'real' if label == 0 else 'screen'
        paths = collect_paths(directory)

        if not paths:
            print(f"ERROR: No images in '{directory}'")
            sys.exit(1)

        print(f"\n[{cls_name}] {len(paths)} images  ({directory})")

        for i, path in enumerate(paths):
            try:
                score, ms, _, _, _ = predict(path, model_bundle)
                all_scores.append(score)
                all_labels.append(label)
                all_latencies.append(ms)
                status = '✓' if (score >= args.threshold) == label else '✗'
                print(f"  {status}  {score:.3f}  {os.path.basename(path)}")
            except Exception as e:
                errors.append((path, str(e)))
                print(f"  ERR  {os.path.basename(path)}: {e}")

    if not all_scores:
        print("No images processed.")
        return

    all_scores  = np.array(all_scores)
    all_labels  = np.array(all_labels)
    all_preds   = (all_scores >= args.threshold).astype(int)

    accuracy    = accuracy_score(all_labels, all_preds)
    auc         = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) == 2 else float('nan')
    cm          = confusion_matrix(all_labels, all_preds)

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    precision   = tp / (tp + fp + 1e-8)
    recall      = tp / (tp + fn + 1e-8)

    lat_mean = np.mean(all_latencies)
    lat_p95  = np.percentile(all_latencies, 95)

    print(f"\n{'=' * 55}")
    print(f"  EVALUATION RESULTS  (threshold = {args.threshold})")
    print(f"{'=' * 55}")
    print(f"  Accuracy   : {accuracy:.1%}   ({int(accuracy * len(all_scores))}/{len(all_scores)} correct)")
    print(f"  ROC AUC    : {auc:.4f}")
    print(f"  Precision  : {precision:.1%}  (of flagged, how many are real screens)")
    print(f"  Recall     : {recall:.1%}  (of real screens, how many caught)")
    print(f"\n  Confusion matrix:           Pred REAL   Pred SCREEN")
    print(f"    True REAL (honest rep)  :    {tn:4d}          {fp:4d}  ← false positives")
    print(f"    True SCREEN (cheater)   :    {fn:4d}          {tp:4d}  ← false negatives")
    print(f"\n  Latency    : {lat_mean:.1f}ms mean  |  {lat_p95:.1f}ms p95")
    print(f"  Errors     : {len(errors)}")

    if errors:
        for path, msg in errors:
            print(f"    {path}: {msg}")

    #Score distribution
    real_scores   = all_scores[all_labels == 0]
    screen_scores = all_scores[all_labels == 1]
    print(f"\n  Score distribution:")
    print(f"    Real photos  : mean={real_scores.mean():.3f}  std={real_scores.std():.3f}  "
          f"max={real_scores.max():.3f}")
    print(f"    Screen photos: mean={screen_scores.mean():.3f}  std={screen_scores.std():.3f}  "
          f"min={screen_scores.min():.3f}")

    real_fp   = [(s, p) for s, p, l in zip(all_scores, collect_paths(args.real)  + collect_paths(args.screen), all_labels) if l == 0 and s >= args.threshold]
    screen_fn = [(s, p) for s, p, l in zip(all_scores, collect_paths(args.real)  + collect_paths(args.screen), all_labels) if l == 1 and s < args.threshold]
    print(f"{'=' * 55}\n")


if __name__ == '__main__':
    main()