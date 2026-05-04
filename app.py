import os
import sys
import tempfile
import json
import numpy as np
import pandas as pd
import h5py
from flask import Flask, render_template, request, jsonify, send_from_directory

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'uploads')
SAMPLE_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'samples')
NOTEBOOKS_FOLDER = os.path.join(PROJECT_ROOT, 'notebooks')
MULTITASK_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'checkpoints', 'multitask_model.pth')
SAMPLE_FILES = {
    'earthquake': 'earthquake_demo.npy',
    'noise': 'noise_demo.npy',
}

tempfile.tempdir = UPLOAD_FOLDER

# Model + predictor import
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
from inference.predictor import Predictor
from inference.postprocess import resolve_threshold

app = Flask(__name__)

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
os.makedirs(NOTEBOOKS_FOLDER, exist_ok=True)

predictor = Predictor(model_path=MULTITASK_MODEL_PATH)


def format_percent(value):
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def format_metric(value, digits=3):
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def load_json_file(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dashboard_context():
    final_summary = load_json_file(os.path.join(PROJECT_ROOT, "models", "metrics", "final_summary.json"))
    multitask_metrics = load_json_file(os.path.join(PROJECT_ROOT, "models", "metrics", "multitask_metrics.json"))
    threshold_used, _ = resolve_threshold(predictor.recommended_threshold)

    return {
        "evaluation": {
            "accuracy": format_percent(final_summary.get("accuracy")),
            "precision": format_percent(final_summary.get("precision")),
            "recall": format_percent(final_summary.get("recall")),
            "f1": format_percent(final_summary.get("f1") or final_summary.get("f1_score")),
            "false_positives": format_metric(final_summary.get("false_positives"), 0),
            "false_negatives": format_metric(final_summary.get("false_negatives"), 0),
            "p_wave_mae_sec": format_metric(final_summary.get("p_wave_mae_sec"), 2),
            "magnitude_mae": format_metric(final_summary.get("magnitude_mae"), 2),
        },
        "model_settings": {
            "threshold": format_metric(threshold_used, 3),
            "model_path": os.path.relpath(MULTITASK_MODEL_PATH, PROJECT_ROOT),
            "model_status": "Active" if os.path.exists(MULTITASK_MODEL_PATH) else "Missing",
            "sample_rate": "100 Hz",
            "input_shape": "(3, 6000) or (6000, 3)",
            "p_wave_rmse_sec": format_metric(multitask_metrics.get("p_wave_rmse_sec"), 2),
            "s_wave_rmse_sec": format_metric(multitask_metrics.get("s_wave_rmse_sec"), 2),
        },
    }


def render_dashboard():
    return render_template('upload.html', **get_dashboard_context())


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


def build_prediction_payload(data: np.ndarray, file_type: str, filename: str | None = None) -> dict:
    data = validate_waveform_shape(data)
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
                f"{warning_time_sec:.2f} seconds.\n"
                "Take immediate action: Move to a safe place. Drop, Cover, and Hold On."
            )

    payload = {
        "status": "success",
        **plot_payload,
        "file_type": file_type,
        **result,
        "early_warning": early_warning,
        "warning_time_sec": warning_time_sec,
        "emergency_message": emergency_message,
    }
    if filename:
        payload["filename"] = filename
    return payload


# Routes
@app.route('/')
def index():
    return render_dashboard()

@app.route('/upload')
def upload():
    return render_dashboard()

@app.route('/dashboard')
def dashboard():
    return render_dashboard()

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
        payload = build_prediction_payload(
            data,
            file_type=fname.split('.')[-1].upper(),
            filename=fname,
        )
        return jsonify(payload)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/sample/<sample_type>')
@app.route('/load-sample/<sample_type>')
def load_sample(sample_type):
    if not os.path.exists(MULTITASK_MODEL_PATH):
        return jsonify({"status": "error", "message": "Model not found"}), 500

    if sample_type not in SAMPLE_FILES:
        return jsonify({
            "status": "error",
            "message": "Unknown sample type. Use earthquake or noise.",
        }), 400

    sample_filename = SAMPLE_FILES[sample_type]
    sample_path = os.path.join(SAMPLE_FOLDER, sample_filename)
    rel_sample_path = os.path.relpath(sample_path, PROJECT_ROOT)

    try:
        if not os.path.exists(sample_path):
            return jsonify({
                "status": "error",
                "message": f"Demo sample missing: {rel_sample_path}",
            }), 404

        data = np.load(sample_path, allow_pickle=True)
        payload = build_prediction_payload(
            data,
            file_type="NPY (Sample)",
            filename=sample_filename,
        )
        payload["sample_type"] = sample_type
        return jsonify(payload)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/notebooks/<path:filename>')
def serve_notebooks(filename):
    return send_from_directory(NOTEBOOKS_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
