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
from models import MultiTaskCNN

app = Flask(__name__)

UPLOAD_FOLDER = 'D:\\Earthquake_Prediction\\uploads'
SAMPLE_FOLDER = 'D:\\Earthquake_Prediction\\samples'
MODEL_PATH    = 'D:\\Earthquake_Prediction\\models\\multitask_model.pth'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_FOLDER, exist_ok=True)
os.makedirs('notebooks',   exist_ok=True)

# ── Normalization constants — train.py se same ────────────────
P_S_MAX   = 6000.0
MAG_MAX   = 9.0
LAT_MAX   = 90.0
LON_MAX   = 180.0
DEPTH_MAX = 700.0

# ── Model Load ────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = MultiTaskCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print("✅ MultiTask Model loaded!")

# ── Helper — Shape Normalize ──────────────────────────────────
def normalize_shape(data):
    if data.ndim == 1:
        data = np.stack([data, data, data], axis=1)
    if data.shape == (3, 6000):
        data = data.T
    if data.shape[0] == 3:
        data = data.T
    return data  # (6000, 3)

# ── Helper — Real Prediction ──────────────────────────────────
def run_prediction(data):
    tensor = torch.tensor(data.T, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    # Detection
    prob       = torch.sigmoid(outputs['detection']).item()
    prediction = "Earthquake" if prob >= 0.5 else "Noise"
    confidence = round(prob * 100, 2) if prob >= 0.5 else round((1 - prob) * 100, 2)

    # Phase — denormalize
    phase    = outputs['phase'].squeeze().cpu().numpy()
    p_sample = float(phase[0]) * P_S_MAX
    s_sample = float(phase[1]) * P_S_MAX
    p_sec    = round(p_sample / 100, 2)   # 100Hz
    s_sec    = round(s_sample / 100, 2)

    # Magnitude — denormalize
    magnitude = round(float(outputs['magnitude'].squeeze().cpu().numpy()) * MAG_MAX, 2)
    magnitude = max(0.0, min(9.0, magnitude))  # clamp 0-9

    # Location — denormalize
    location  = outputs['location'].squeeze().cpu().numpy()
    latitude  = round(float(location[0]) * LAT_MAX,   4)
    longitude = round(float(location[1]) * LON_MAX,   4)
    depth     = round(float(location[2]) * DEPTH_MAX, 2)

    # Clamp to valid ranges
    latitude  = max(-90.0,  min(90.0,  latitude))
    longitude = max(-180.0, min(180.0, longitude))
    depth     = max(0.0,    min(700.0, depth))

    print(f"  Raw phase: {phase}")
    print(f"  P: {p_sec}s | S: {s_sec}s | Mag: {magnitude} | Lat: {latitude} | Lon: {longitude} | Depth: {depth}km")

    # Noise hone pe None karo
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

# ── Routes ────────────────────────────────────────────────────
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
            print(f"NPY loaded, shape: {data.shape}")

        elif fname.endswith('.csv'):
            data = pd.read_csv(filepath, header=None).values
            print(f"CSV loaded, shape: {data.shape}")

        elif fname.endswith('.hdf5') or fname.endswith('.h5'):
            print("Loading HDF5 first trace...")
            with h5py.File(filepath, 'r') as f:
                first_key = list(f['data'].keys())[0]
                data      = f['data'][first_key][()]
            print(f"HDF5 loaded, shape: {data.shape}")

        else:
            return jsonify({"status": "error", "message": "Unsupported format"}), 400

        data = normalize_shape(data)
        print(f"Final shape: {data.shape}")

        result = run_prediction(data)
        print(f"Prediction: {result['prediction']} ({result['confidence']}%)")

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

        filepath = os.path.join(SAMPLE_FOLDER, files[0])
        data     = np.load(filepath, allow_pickle=True)
        print(f"Sample: {files[0]}, shape: {data.shape}")

        data   = normalize_shape(data)
        result = run_prediction(data)
        print(f"Sample Prediction: {result['prediction']} ({result['confidence']}%)")

        return jsonify({
            "status":    "success",
            "waveform":  data.T.tolist(),
            "file_type": "NPY (Sample)",
            **result
        })

    except Exception as e:
        print(f"Sample error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/notebooks/<path:filename>')
def serve_notebooks(filename):
    return send_from_directory('notebooks', filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)