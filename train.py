"""
train.py — Train the recapture detector.

"""

import os
import sys
import glob
import pickle
import time
import argparse
import numpy as np
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score
)

from features import extract_features, HAS_PYWT, HAS_SKIMAGE, FEATURE_DIM


#Data loading

IMG_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')


def collect_paths(directory: str) -> list:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)


def augment_image(img: np.ndarray) -> list:

    variants = [img]
    variants.append(cv2.flip(img, 1))
    for factor in [0.75, 1.25]:
        bright = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        variants.append(bright)
    return variants


def load_dataset(real_dir: str, screen_dir: str, augment: bool = True):
 
    X, y = [], []

    for label, directory in [(0, real_dir), (1, screen_dir)]:
        cls_name = 'real' if label == 0 else 'screen'
        paths = collect_paths(directory)

        if not paths:
            print(f"  ERROR: No images found in '{directory}'")
            sys.exit(1)

        print(f"  [{cls_name}] {len(paths)} images found in '{directory}'")

        for i, path in enumerate(paths):
            img = cv2.imread(path)
            if img is None:
                print(f"    SKIP (unreadable): {path}")
                continue

            imgs_to_process = augment_image(img) if augment else [img]

            for aug_img in imgs_to_process:
                feats = extract_features(aug_img)
                X.append(feats)
                y.append(label)

            if (i + 1) % 10 == 0 or (i + 1) == len(paths):
                print(f"    processed {i + 1}/{len(paths)} ...", end='\r', flush=True)

        print()  

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


#Model definition

def build_ensemble():
    """
    Soft-voting ensemble. SVM is the workhorse on this feature space.
    """
    svm = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',     
        probability=True,       
        class_weight='balanced',
        random_state=42,
    )
    lr = LogisticRegression(
        C=0.5,
        solver='lbfgs',
        max_iter=3000,
        class_weight='balanced',
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    return VotingClassifier(
        estimators=[('svm', svm), ('lr', lr), ('rf', rf)],
        voting='soft',
        weights=[3, 1, 1],   
        n_jobs=-1,
    )


#Training

def main():
    parser = argparse.ArgumentParser(
        description='Train the SalesCode recapture detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --real real/ --screen screen/
  python train.py --real real/ --screen screen/ --no-augment
  python train.py --real real/ --screen screen/ --output models/v2.pkl
        """
    )
    parser.add_argument('--real',       default='real',      help='Folder of real photos (label 0)')
    parser.add_argument('--screen',     default='screen',    help='Folder of screen photos (label 1)')
    parser.add_argument('--output',     default='model.pkl', help='Output model file')
    parser.add_argument('--no-augment', action='store_true', help='Disable data augmentation')
    parser.add_argument('--cv-folds',   type=int, default=5, help='Cross-validation folds')
    args = parser.parse_args()

    print("=" * 60)
    print("  SalesCode Recapture Detector — Training")
    print("=" * 60)
    print(f"\n  Libraries:  pywt={'OK' if HAS_PYWT else 'MISSING (install PyWavelets)'}  "
          f"skimage={'OK' if HAS_SKIMAGE else 'MISSING (install scikit-image)'}")
    print(f"  Feature dim: {FEATURE_DIM}")

    #Load data
    print(f"\n[1/5] Loading and extracting features ...")
    print(f"      Augmentation: {'OFF' if args.no_augment else 'ON (4× per image)'}")
    t0 = time.time()
    X, y = load_dataset(args.real, args.screen, augment=not args.no_augment)
    t1 = time.time()

    n_real   = (y == 0).sum()
    n_screen = (y == 1).sum()
    print(f"\n  Total samples : {len(X)}  ({n_real} real, {n_screen} screen)")
    print(f"  Extraction    : {t1 - t0:.1f}s  ({(t1-t0)/len(X)*1000:.0f}ms per image)")

    if len(X) < 20:
        print("\n  WARNING: Very few samples. Collect more photos for reliable results.")

    #Scale features
    print(f"\n[2/5] Scaling features ...")
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Cross-validate
    print(f"\n[3/5] {args.cv_folds}-fold stratified cross-validation ...")
    model_cv = build_ensemble()
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

    t_cv = time.time()
    cv_scores = cross_val_score(model_cv, X_scaled, y, cv=cv,
                                scoring='accuracy', n_jobs=-1)

    oof_preds = cross_val_predict(build_ensemble(), X_scaled, y,
                                  cv=cv, method='predict', n_jobs=-1)
    t_cv_end = time.time()

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}          │")
    print(f"  │  Per-fold:    {' '.join(f'{s:.3f}' for s in cv_scores)}     │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"  (Time: {t_cv_end - t_cv:.1f}s)")

    cm = confusion_matrix(y, oof_preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\n  Out-of-fold confusion matrix:")
    print(f"              Pred REAL   Pred SCREEN")
    print(f"  True REAL     {tn:5d}       {fp:5d}     (false positives — honest rep flagged)")
    print(f"  True SCREEN   {fn:5d}       {tp:5d}     (true positives — cheat caught)")
    print(f"\n  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")


    print(f"\n[4/5] Training final model on full dataset ...")
    model_final = build_ensemble()
    t_train = time.time()
    model_final.fit(X_scaled, y)
    t_train_end = time.time()
    print(f"  Training time: {t_train_end - t_train:.1f}s")

    train_acc = accuracy_score(y, model_final.predict(X_scaled))
    print(f"  Training accuracy (optimistic upper bound): {train_acc:.3f}")

    #Save
    print(f"\n[5/5] Saving model to '{args.output}' ...")
    bundle = {
        'model':             model_final,
        'scaler':            scaler,
        'feature_dim':       FEATURE_DIM,
        'has_pywt':          HAS_PYWT,
        'has_skimage':       HAS_SKIMAGE,
        'cv_accuracy_mean':  float(cv_scores.mean()),
        'cv_accuracy_std':   float(cv_scores.std()),
        'n_train_real':      int(n_real),
        'n_train_screen':    int(n_screen),
    }
    with open(args.output, 'wb') as f:
        pickle.dump(bundle, f, protocol=4)

    size_kb = os.path.getsize(args.output) / 1024
    print(f"  Saved ({size_kb:.0f} KB)")

    print(f"\n{'=' * 60}")
    print(f"  REPORTED ACCURACY:")
    print(f"    {cv_scores.mean():.1%} ± {cv_scores.std():.1%}  ({args.cv_folds}-fold CV)")
    print(f"\n  Next step:")
    print(f"    python predict.py <image.jpg>")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()