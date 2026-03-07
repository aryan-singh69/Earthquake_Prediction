import os
import sys
import tempfile
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_from_directory

# Temp folder D: pe set karo
tempfile.tempdir = 'D:\\Earthquake_Prediction\\uploads'

# Model import
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from models import MultiTaskCNN  # SimpleCNN hata diya — MultiTaskCNN hi use hoga

app = Flask(__name__)

UPLOAD_FOLDER        = 'D:\\Earthquake_Prediction\\uploads'
SAMPLE_FOLDER        = 'D:\\Earthquake_Prediction\\samples'
MULTITASK_MODEL_PATH = 'D:\\Earthquake_Prediction\\models\\multitask_model.pth'

app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_FOLDER, exist_ok=True)
os.makedirs('notebooks',   exist_ok=True)

# Normalization constants — train.py se same
P_S_MAX   = 6000.0
MAG_MAX   = 9.0
LAT_MAX   = 90.0
LON_MAX   = 180.0
DEPTH_MAX = 700.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Single Model — MultiTaskCNN (Detection + Phase + Mag + Loc)
multitask_model = MultiTaskCNN().to(device)

if os.path.exists(MULTITASK_MODEL_PATH):
    multitask_model.load_state_dict(
        torch.load(MULTITASK_MODEL_PATH, map_location=device, weights_only=True)
    )
    print("✅ MultiTask Model (multitask_model.pth) loaded!")
else:
    print("⚠️  multitask_model.pth nahi mila — untrained model use ho raha hai!")

multitask_model.eval()


def normalize_waveform(data: np.ndarray) -> np.ndarray:
    """Per-channel zero-mean, unit-std normalization. data shape: (3, 6000)"""
    mean = data.mean(axis=1, keepdims=True)
    std  = data.std(axis=1,  keepdims=True) + 1e-8
    return (data - mean) / std


def normalize_shape(data: np.ndarray) -> np.ndarray:
    """Any input shape → (6000, 3)"""
    if data.ndim == 1:
        data = np.stack([data, data, data], axis=1)
    if data.shape == (3, 6000):
        data = data.T
    if data.shape[0] == 3:
        data = data.T
    return data  # (6000, 3)


def run_prediction(data: np.ndarray) -> dict:
    """data: numpy (6000, 3) → predict sab kuch"""
    waveform = data.T                        # (3, 6000)
    waveform = normalize_waveform(waveform)  # normalize — zaroori hai
    tensor   = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = multitask_model(tensor)

    # Detection
    prob       = torch.sigmoid(outputs['detection']).item()
    prediction = "Earthquake" if prob >= 0.5 else "Noise"
    confidence = round(prob * 100, 2) if prob >= 0.5 else round((1 - prob) * 100, 2)

    # Phase picking — denormalize
    phase    = outputs['phase'].squeeze().cpu().numpy()
    p_sample = float(phase[0]) * P_S_MAX
    s_sample = float(phase[1]) * P_S_MAX
    p_sec    = round(p_sample / 100, 2)
    s_sec    = round(s_sample / 100, 2)

    # Magnitude — denormalize
    magnitude = float(outputs['magnitude'].squeeze().cpu().numpy()) * MAG_MAX
    magnitude = round(max(0.0, min(9.0, magnitude)), 2)

    # Location — denormalize
    loc       = outputs['location'].squeeze().cpu().numpy()
    latitude  = round(max(-90.0,  min(90.0,  float(loc[0]) * LAT_MAX)),   4)
    longitude = round(max(-180.0, min(180.0, float(loc[1]) * LON_MAX)),   4)
    depth     = round(max(0.0,    min(700.0, float(loc[2]) * DEPTH_MAX)), 2)

    print(f"  [MultiTaskCNN] {prediction} ({confidence}%) | "
          f"P:{p_sec}s S:{s_sec}s | Mag:{magnitude} | "
          f"Lat:{latitude} Lon:{longitude} Depth:{depth}km")

    # Noise hone pe phase/mag/loc None karo
    if prediction == "Noise":
        p_sec = s_sec = magnitude = latitude = longitude = depth = None

    return {
        "prediction": prediction,
        "confidence": confidence,
        "p_arrival":  p_sec,
        "s_arrival":  s_sec,
        "magnitude":  magnitude,
        "latitude":   latitude,
        "longitude":  longitude,
        "depth":      depth
    }


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    fname    = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, fname)
    file.save(filepath)
    print(f"File saved: {filepath}")

    try:
        if fname.endswith('.npy'):
            data = np.load(filepath, allow_pickle=True)
        elif fname.endswith('.csv'):
            data = pd.read_csv(filepath, header=None).values
        elif fname.endswith('.hdf5') or fname.endswith('.h5'):
            with h5py.File(filepath, 'r') as f:
                first_key = list(f['data'].keys())[0]
                data      = f['data'][first_key][()]
        else:
            return jsonify({"status": "error", "message": "Unsupported format. Use .npy, .csv, .hdf5"}), 400

        print(f"Loaded shape: {data.shape}")
        data   = normalize_shape(data)
        result = run_prediction(data)

        return jsonify({
            "status":    "success",
            "waveform":  data.T.tolist(),
            "file_type": fname.split('.')[-1].upper(),
            **result
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/load-sample/<sample_type>')
def load_sample(sample_type):
    prefix = 'earthquake_local' if sample_type == 'earthquake' else 'noise'

    try:
        files = sorted([
            f for f in os.listdir(SAMPLE_FOLDER)
            if f.startswith(prefix) and f.endswith('.npy')
        ])

        if not files:
            return jsonify({"status": "error", "message": f"No sample found: {sample_type}"}), 404

        data   = np.load(os.path.join(SAMPLE_FOLDER, files[0]), allow_pickle=True)
        data   = normalize_shape(data)
        result = run_prediction(data)

        return jsonify({
            "status":    "success",
            "waveform":  data.T.tolist(),
            "file_type": "NPY (Sample)",
            **result
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/notebooks/<path:filename>')
def serve_notebooks(filename):
    return send_from_directory('notebooks', filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)