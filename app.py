import os
import sys
import tempfile
import numpy as np
import pandas as pd
import h5py
from flask import Flask, render_template, request, jsonify, send_from_directory

# Temp folder D: pe set karo
tempfile.tempdir = 'D:\\Earthquake_Prediction\\uploads'

# Model + predictor import
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from inference.predictor import Predictor

app = Flask(__name__)

UPLOAD_FOLDER        = 'D:\\Earthquake_Prediction\\uploads'
SAMPLE_FOLDER        = 'D:\\Earthquake_Prediction\\samples'
MULTITASK_MODEL_PATH = 'D:\\Earthquake_Prediction\\models\\checkpoints\\multitask_model.pth'

def validate_waveform_shape(data: np.ndarray) -> np.ndarray:
    """Ensure input is (3, 6000) or (6000, 3), return (3, 6000)."""
    if data.ndim != 2:
        raise ValueError("Waveform must be 2D with shape (3, 6000) or (6000, 3)")
    if data.shape not in [(3, 6000), (6000, 3)]:
        raise ValueError("Invalid shape; expected (3, 6000) or (6000, 3)")
    if data.shape == (6000, 3):
        data = data.T
    return data

app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_FOLDER, exist_ok=True)
os.makedirs('notebooks',   exist_ok=True)

predictor = Predictor(model_path=MULTITASK_MODEL_PATH)


def prepare_waveform_for_plot(data: np.ndarray) -> dict:
    """Downsample waveform for visualization without affecting inference."""
    step = int(os.getenv("VIS_DOWNSAMPLE", "5"))
    step = max(1, step)
    x = np.arange(0, data.shape[1], step)
    # Display normalization only (per-channel max-abs scaling)
    display = data.astype(np.float32).copy()
    for i in range(display.shape[0]):
        max_abs = float(np.max(np.abs(display[i])))
        if max_abs > 0:
            display[i] = display[i] / max_abs
    return {
        "waveform_x": x.tolist(),
        "waveform_e": display[0, ::step].tolist(),
        "waveform_n": display[1, ::step].tolist(),
        "waveform_z": display[2, ::step].tolist(),
        "waveform_normalized": True,
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
    if not os.path.exists(MULTITASK_MODEL_PATH):
        return jsonify({"status": "error", "message": "Model not found"}), 500
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
        data   = validate_waveform_shape(data)
        plot_payload = prepare_waveform_for_plot(data)
        result = predictor.predict(data)
        p_sec = result.get("p_arrival_sec")
        s_sec = result.get("s_arrival_sec")
        alert = result.get("alert")
        prediction = result.get("prediction")
        early_warning = False
        warning_time_sec = None
        emergency_message = "P/S timing unavailable or unreliable."
        if (
            prediction == "Earthquake"
            and alert is True
            and p_sec is not None
            and s_sec is not None
            and s_sec > p_sec
        ):
            warning_time_sec = round(float(s_sec - p_sec), 2)
            early_warning = warning_time_sec > 0
            if early_warning:
                emergency_message = (
                    "Strong earthquake shaking may begin in approximately "
                    f"{warning_time_sec:.2f} seconds.\n\n"
                    "Take immediate action:\n"
                    "• Move to a safe place\n"
                    "• Drop, Cover, and Hold On\n\n"
                    f"Estimated time remaining: {warning_time_sec:.2f} seconds"
                )

        return jsonify({
            "status":    "success",
            **plot_payload,
            "file_type": fname.split('.')[-1].upper(),
            **result,
            "early_warning": early_warning,
            "warning_time_sec": warning_time_sec,
            "emergency_message": emergency_message
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/load-sample/<sample_type>')
def load_sample(sample_type):
    if not os.path.exists(MULTITASK_MODEL_PATH):
        return jsonify({"status": "error", "message": "Model not found"}), 500
    prefix = 'earthquake_local' if sample_type == 'earthquake' else 'noise'

    try:
        files = sorted([
            f for f in os.listdir(SAMPLE_FOLDER)
            if f.startswith(prefix) and f.endswith('.npy')
        ])

        if not files:
            return jsonify({"status": "error", "message": f"No sample found: {sample_type}"}), 404

        data   = np.load(os.path.join(SAMPLE_FOLDER, files[0]), allow_pickle=True)
        data   = validate_waveform_shape(data)
        plot_payload = prepare_waveform_for_plot(data)
        result = predictor.predict(data)
        p_sec = result.get("p_arrival_sec")
        s_sec = result.get("s_arrival_sec")
        alert = result.get("alert")
        prediction = result.get("prediction")
        early_warning = False
        warning_time_sec = None
        emergency_message = "P/S timing unavailable or unreliable."
        if (
            prediction == "Earthquake"
            and alert is True
            and p_sec is not None
            and s_sec is not None
            and s_sec > p_sec
        ):
            warning_time_sec = round(float(s_sec - p_sec), 2)
            early_warning = warning_time_sec > 0
            if early_warning:
                emergency_message = (
                    "Strong earthquake shaking may begin in approximately "
                    f"{warning_time_sec:.2f} seconds.\n\n"
                    "Take immediate action:\n"
                    "• Move to a safe place\n"
                    "• Drop, Cover, and Hold On\n\n"
                    f"Estimated time remaining: {warning_time_sec:.2f} seconds"
                )

        return jsonify({
            "status":    "success",
            **plot_payload,
            "file_type": "NPY (Sample)",
            **result,
            "early_warning": early_warning,
            "warning_time_sec": warning_time_sec,
            "emergency_message": emergency_message
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/notebooks/<path:filename>')
def serve_notebooks(filename):
    return send_from_directory('notebooks', filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)