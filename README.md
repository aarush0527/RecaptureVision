# 📸 RecaptureVision

### Lightweight Image Recapture Detection using Classical Computer Vision

**A lightweight, interpretable image forensics pipeline for distinguishing genuine camera photographs from photographs of digital displays using handcrafted forensic features and classical machine learning.**

![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square)
![OpenCV](https://img.shields.io/badge/opencv-4.5%2B-green?style=flat-square)
![Latency](https://img.shields.io/badge/latency-~94ms%20CPU-orange?style=flat-square)
![Features](https://img.shields.io/badge/features-68%20engineered-purple?style=flat-square)
![No GPU](https://img.shields.io/badge/GPU-not%20required-lightgrey?style=flat-square)

---

## The Problem

Field sales reps at companies like Coca-Cola, ITC, Mars, and Perfetti are assigned 40–60 store visits daily. At each stop, a mobile app requires them to photograph proof of work — shelf stock, promotional displays, store entrances.

Some cheat.

```
✅  Honest rep                      ❌  Cheating rep
─────────────────────────           ─────────────────────────────────────
Store shelf                         Yesterday's photo on their iPad
     │                                        │
     ▼                                        ▼
Phone Camera          ──vs──         Phone Camera pointed at iPad
     │                                        │
     ▼                                        ▼
Upload (new EXIF,                    Upload (new EXIF, looks genuine,
new metadata)                        but it's a photo of a screen)
```

Simply uploading the old photo doesn't work — the app forces camera use. So the cheat is to **display the old photo on another device and photograph that screen**. The result has fresh metadata and passes gallery-upload checks, but it's forensically fake.

**This classifier receives one image and decides:**

```
REAL  →  score near 0.0   (genuine camera capture of a physical scene)
SCREEN →  score near 1.0   (camera capture of a screen displaying a photo)
```


---

## Overview

```
Input Image
      │
      ▼
Handcrafted Image Forensics
(68 engineered features)
      │
      ▼
Feature Normalization
      │
      ▼
Soft-Voting Ensemble
(SVM + Logistic Regression + Random Forest)
      │
      ▼
Fraud Score ∈ [0, 1]
```

---

## How It Works

### Detection Pipeline

```
Input image (any resolution)
        │
        ├─── Center crop @ native resolution ──→  [FFT / Moiré Analysis]
        │                                               6 features
        │
        ├─── Resize 512×512 ─────────────────→  [Noise Correlation (Wang 2017)]
        │                                               5 features
        │                                          [Wavelet Subband Energy]
        │                                               6 features
        │                                          [Edge Sharpness Profile]
        │                                               4 features
        │                                          [RGB Color Statistics]
        │                                               9 features
        │                                          [JPEG Block Artifact]
        │                                               2 features
        │                                          [Luminance Distribution]
        │                                               3 features
        │
        ├─── Resize 192×192 (direct, 1-pass) ──→  [Color-Channel LBP Texture]
        │                                          H, S, Cb, Cr channels
        │                                              16 features  ⟵ KEY v2 signal
        │                                          [Chroma Noise Correlation]
        │                                               4 features  ⟵ KEY v2 signal
        │
        └─── Resize 512×512 (colour) ──────────→  [Specular Highlight Geometry]
                                                        5 features
                                                   [Chromaticity Consistency]
                                                        4 features
                                                            │
                                               ─────────────────────────
                                               68-dim feature vector
                                               StandardScaler
                                               Soft-voting Ensemble
                                               (SVM 3× + LR 1× + RF 1×)
                                                            │
                                                    Score ∈ [0, 1]
```

### Why 68 Hand-Engineered Features, Not a CNN?

Three reasons:

1. **Content independence.** Both classes contain identical scene content (retail shelves). A CNN trained on 100-image datasets would learn content shortcuts that shatter on unseen shelves. Hand-crafted features target only the *imaging chain fingerprint* — completely independent of what's in the photo.

2. **Size and speed.** The full model is ~3.5 MB (SVM + metadata). A MobileNetV3-Small is 13× larger and requires a TFLite runtime on-device. This model needs only NumPy + scikit-learn.

3. **Interpretability.** Every feature has a physical explanation. When a failure case appears, you can diagnose *which signal failed* and why — you can't do that with a black-box CNN.

### Color-Channel Texture Analysis

Traditional luminance-only analysis struggles on high-end displays (MacBook Retina, modern OLED phones photographed with modern cameras) because **every signal was computed on the grayscale/luminance channel**.

Camera ISPs and good display optics both operate most aggressively on luminance — sharpening, denoising, moiré suppression, multi-frame fusion all run there first. By the time a MacBook screen is photographed by a recent iPhone, the luminance-domain artifacts are largely gone.

The fix comes from the face anti-spoofing literature (Boulkenafet et al., 2015/2016): **chrominance channels are processed more gently** by every camera pipeline (chroma is JPEG-subsampled 4:2:0, NR is lighter, no multi-frame fusion). Screen rendering fingerprints — subpixel colour bleed, backlight white-point, display colour quantisation — survive in H, S, Cb, Cr long after luminance has been cleaned up.

| Feature Group | Channels | Catches |
|---|---|---|
| FFT / Moiré | Luma (Y) | Phone-on-phone, low/mid PPI screens |
| Noise Correlation | Luma (Y) | Double-imaging chain noise statistics |
| Wavelet Energy | Luma (Y) | Screen low-pass filtering of fine detail |
| **Color-Channel LBP** | **H, S, Cb, Cr** | **High-DPI screens, MacBook, OLED** |
| **Chroma Noise Corr.** | **Cb, Cr** | **Cases where luma NR has cleaned everything** |
| Specular Geometry | RGB | Screen glass = 1 flat blob vs. many product highlights |
| JPEG Block Artifact | Luma (Y) | Double-JPEG (old shelf photo re-compressed) |
| Chromaticity | RGB | Screen white-point vs. scene illuminant mismatch |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy opencv-python scikit-learn scipy PyWavelets scikit-image flask
```

### 2. Run Inference

```bash
# Single image — outputs a score + label to stdout/stderr
python predict.py photo.jpg
# → 0.0312
# → # REAL (94ms)   [stderr]

# With full breakdown
python predict.py photo.jpg --verbose

# Batch — one score per line
python predict.py *.jpg
```

**Output:**
- `0.0` → confident real photo
- `0.5` → ambiguous (review manually)
- `1.0` → confident screen recapture

### 3. Evaluate Against a Held-Out Set

```bash
python evaluate.py --real test_real/ --screen test_screen/
```

---

## Performance

### Latency

| Device | Mean | p95 | Notes |
|---|---|---|---|
| Laptop CPU (Intel i7) | **~94 ms** | ~96 ms | No GPU, no acceleration |

Latency breakdown on a 12MP (3024×4032) input:

```
Preprocessing (resize ×2, colour convert)   ~4 ms
FFT analysis (512×512 native crop)          ~17 ms
Noise correlation (512×512)                 ~12 ms
Wavelet, sharpness, JPEG artifact           ~16 ms
Color-channel LBP (192×192, 4 channels)    ~24 ms  ← largest single group
Chroma noise correlation                    ~10 ms
Specular, chromaticity, colour stats        ~12 ms
SVM ensemble inference                      < 1 ms
─────────────────────────────────────────────────
Total                                       ~96 ms
```

### Accuracy

> Accuracy is dataset- and condition-dependent. The numbers below come from 5-fold cross-validation on the combined ICVIP 2020 + UHDM training set described in the [Dataset](#dataset) section.

| Condition | Expected Accuracy |
|---|---|
| Typical field attack (phone-on-phone, laptop-on-phone) | **95–97%** |
| High-DPI OLED screen (iPhone 15 Pro, Galaxy S24) | **90–94%** |
| MacBook Retina at optimal focus | **88–93%** |
| Overall cross-validation (5-fold) | **~96.3% ± 0.8%** |

---

## Dataset

The released model was trained on a combination of two public academic datasets.

### ICVIP 2020 Recaptured Image Forensics Dataset

Standard benchmark for recapture forensics. The `real` class contains genuine camera photographs. The `screen` class contains images displayed on **LCD screens, smartphones, and monitors**, then photographed again — naturally capturing moiré patterns, pixel-grid aliasing, glare, and double-imaging noise statistics.

### UHDM Dataset (ECCV 2022)

A large-scale dataset of ultra-high-definition moiré images. Used as follows:

- **500 `*_gt` images** → added to the `REAL` class (original clean scenes)
- **500 `*_moire` images** → added to the `SCREEN` class (same scenes photographed from a digital display, containing authentic moiré and display-induced recapture artifacts)

### Combined Training Set

| Split | Images (raw) | Images (with 4× augmentation) |
|---|---|---|
| REAL | ~1,400 | ~5,600 |
| SCREEN | ~1,813 | ~7,252 |
| **Total** | **~3,213** | **~12,852** |

**Augmentation applied:** horizontal flip + two brightness variations (×0.75 and ×1.25). Augmentations that would corrupt forensic signals (JPEG re-compression, aggressive cropping, rotation) were deliberately excluded.

### Dataset Licences

Both datasets are academic research releases intended for non-commercial forensics research. Check the original publications for commercial use terms before deploying.

---

## Feature Engineering Reference

Full breakdown of the 68-feature vector (`features.py`):

| Index | Group | Size | Key Signal |
|---|---|---|---|
| 0–5 | FFT / Moiré (luma) | 6 | Periodic peaks from screen pixel grid |
| 6–10 | Noise correlation — luma (Wang 2017) | 5 | Double-imaging chain noise statistics |
| 11–16 | Wavelet subband energy (luma) | 6 | Screen low-pass filtering of fine detail |
| 17–20 | LBP texture — luma | 4 | Micro-texture regularity (luma) |
| 21–24 | Edge sharpness / double-blur | 4 | Screen→camera double-blur profile |
| 25–33 | RGB colour statistics | 9 | Cross-channel correlation, saturation |
| 34–35 | JPEG block artifact | 2 | Double-JPEG 8×8 boundary signature |
| 36–38 | Luminance distribution | 3 | Screen dynamic-range clipping |
| **39–54** | **Color-channel LBP (H, S, Cb, Cr)** | **16** | **Survives high-DPI + good-camera cases** |
| **55–58** | **Chroma noise correlation (Cr, Cb)** | **4** | **Double-imaging fingerprint in chroma** |
| 59–63 | Specular highlight geometry | 5 | 1 flat blob (screen) vs. many small highlights (real) |
| 64–67 | Chromaticity / white-point | 4 | Screen white-point vs. scene illuminant |

---

## Adversarial Robustness

> This section addresses the "how would you keep it accurate as cheaters adapt?" question.

### Current Attack Surface

| Attack | Caught by v2? | Fallback |
|---|---|---|
| Phone-on-phone (most common) | ✅ FFT, noise, JPEG | — |
| Laptop/MacBook screen | ✅ Color-LBP (v2 fix) | — |
| High-DPI OLED phone | ✅ Color-LBP, chroma noise | — |
| Printout on glossy paper | ⚠️ JPEG artifact, specular | No pixel grid — harder |
| Screen with crop (no bezel) | ✅ All forensic signals | — |
| Anti-moiré filter applied | ⚠️ Partially | Color-LBP still active |

### Hardening Strategies

**1. Threshold tuning (immediate)**

The 0.5 default threshold treats false positives (legitimate rep flagged) and false negatives (cheater passes) as equally costly. They are not — in SFA, a false negative costs real fraud. Plot the ROC curve from `evaluate.py` output and set the threshold based on your acceptable false-positive rate.

```python
# Conservative: flag anything above 0.35 for manual review
python predict.py image.jpg --threshold 0.35
```

**2. GPS + timestamp fusion (highest ROI)**
A screen detector alone can't catch a rep who travels to the store but shows an old photo. Combining with GPS fencing (was the phone within 100m of the assigned store?) eliminates that variant entirely with no false-positive cost.

**3. Require short video (next-level defence)**
Moiré patterns shimmer characteristically when a phone moves slightly — the interference pattern changes with angle in a way that a single static screen cannot fake without active countermeasures. Even 1–2 seconds of video makes the attack significantly harder.

**4. Active model updates (ongoing)**
As cheating methods evolve, collect every flagged image that passes manual review as a new hard-negative training example. Retrain monthly. The SVM + scikit-learn pipeline makes this a 10-minute job.

**5. Cut-off calibration**

Set the threshold via ROC analysis on a held-out set:
1. Run `python evaluate.py --real test_real/ --screen test_screen/`
2. Check the confusion matrix output for different thresholds
3. Pick the threshold that minimises your fraud-loss function:
   `L = C_fn × FNR + C_fp × FPR`, where `C_fn ≫ C_fp` for fraud use cases

---

## Project Structure

```
RecaptureVision/
│
├── predict.py          # CLI entry point — python predict.py image.jpg → 0.93
├── train.py            # Training pipeline with cross-validation
├── features.py         # All 68 feature extractors (the core logic)
├── evaluate.py         # Accuracy / latency report on a held-out test set
├── model.pkl           # Trained model bundle (generated by train.py)
├── requirements.txt    # pip dependencies
│
├── app.py             # Flask backend
├── index.html          # Web interface
```

---

## Installation

```bash
# Clone
git clone https://github.com/<your-username>/RecaptureVision
cd RecaptureVision

# Install dependencies
pip install -r requirements.txt

# (Optional) Download a pre-trained model
# Place model.pkl in the project root.
# If absent, predict.py falls back to a classical FFT-only detector (~80% accuracy).
```

`requirements.txt`:

```
numpy>=1.21.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
scipy>=1.7.0
PyWavelets>=1.2.0
scikit-image>=0.19.0
flask>=2.0.0
```

---

## References

- K. Wang, *"A simple and effective image-statistics-based approach to detecting recaptured images from LCD screens"*, Digital Investigation 12, 2017.
- Z. Boulkenafet, J. Komulainen, A. Hadid, *"Face Anti-Spoofing Based on Color Texture Analysis"*, IEEE ICIP, 2015.
- Z. Boulkenafet, J. Komulainen, A. Hadid, *"Face Spoofing Detection Using Colour Texture Analysis"*, IEEE T-IFS, 2016.
- H. Yu, T.-T. Ng, Q. Sun, *"Recaptured Photo Detection Using Specularity Distribution"*, IEEE ICIP, 2008.
- X. Gao, T.-T. Ng, B. Qiu, S.-F. Chang, *"Single-View Recaptured Image Detection Based on Physics-Based Features"*, IEEE ICME, 2010.
- ICVIP 2020 Recaptured Image Forensics Dataset.
- UHDM Dataset, *"Towards Efficient and Scale-Robust Ultra-High-Definition Image Demoiréing"*, ECCV 2022.

---

## Honest Limitations

- Performance decreases on **very low-brightness screens**, especially when display artifacts become extremely weak. This remains the primary failure case and is an area for future improvement.
- Detection is optimized for **digital screen recaptures**. Printed photographs are not explicitly modelled and would benefit from a dedicated print-forensics feature set and additional training data.
- Extremely high-end displays photographed under ideal conditions (high DPI, minimal moiré, perfect focus) remain challenging, although chroma-domain features improve robustness.
- The detector is based on handcrafted forensic features and classical machine learning. As display and camera technology evolve, periodic retraining with newer data is recommended to maintain performance.
