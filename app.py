"""
demo/app.py — Flask web server for live recapture detection demo.

Usage:
    cd ..
    pip install flask
    python app.py

Then open http://localhost:5000 in your browser.
The page uses your camera, captures an image on button press, and shows
the real/screen score live.
"""

import os
import sys
import io
import time
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import predict, load_model

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))

_model_bundle = None


def get_model():
    global _model_bundle
    if _model_bundle is None:
        _model_bundle = load_model()
    return _model_bundle


def decode_image(data_url: str) -> np.ndarray:

    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]
    img_bytes = base64.b64decode(data_url)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@app.route('/')
def index():
    return send_from_directory(
        os.path.dirname(os.path.abspath(__file__)),
        'index.html'
    )


@app.route('/predict', methods=['POST'])
def predict_endpoint():

    t0 = time.perf_counter()

    data = request.get_json(force=True)
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing "image" field'}), 400

    try:
        img = decode_image(data['image'])
    except Exception as e:
        return jsonify({'error': f'Image decode failed: {e}'}), 400

    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    tmp_path = '/tmp/demo_capture.jpg'
    cv2.imwrite(tmp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    try:
        score, total_ms, feat_ms, method, features = predict(tmp_path, get_model())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    label = 'SCREEN' if score >= 0.5 else 'REAL'
    confidence = abs(score - 0.5) * 2

    return jsonify({
        'score':      round(float(score), 4),
        'label':      label,
        'confidence': round(float(confidence), 3),
        'ms':         round(float(total_ms), 1),
        'method':     method,

        'signals': {
            'fft_peak_ratio':  round(float(features[0]), 2),
            'noise_corr_dx':   round(float(features[6]),  3),
            'jpeg_ratio_vert': round(float(features[46]), 3),
            'lbp_entropy':     round(float(features[32]), 3),
        }
    })


@app.route('/health')
def health():
    model = get_model()
    return jsonify({
        'status': 'ok',
        'model': 'loaded' if model else 'classical_fallback',
        'cv_accuracy': model.get('cv_accuracy_mean') if model else None,
    })


if __name__ == '__main__':
    print("\n  SalesCode Recapture Detector — Live Demo")
    print("  Open: http://localhost:5000\n")
    model = get_model()
    if model:
        print(f"  Model loaded (CV accuracy: {model.get('cv_accuracy_mean', 0):.1%})")
    else:
        print("  No model.pkl found — using classical fallback")
    app.run(host='0.0.0.0', port=5000, debug=False)