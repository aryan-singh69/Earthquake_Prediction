# Earthquake Prediction — STEAD-Based Seismic ML System

A production-ready machine learning pipeline for earthquake detection, phase picking, magnitude estimation, and location prediction using the **Stanford Earthquake Dataset (STEAD)**.

## Project Structure

```
Earthquake_Prediction/
├── app.py
├── data/
│   └── raw/                     # merge.csv, merge.hdf5
├── docs/
├── models/
│   ├── checkpoints/             # Trained .pth files
│   └── metrics/                 # Evaluation reports
├── notebooks/                   # EDA scripts + outputs
├── scripts/
│   └── evaluate_model.py
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── inspect_data.py
│   └── utils/
│       └── metrics.py
├── static/
├── templates/
├── uploads/
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

### 1. Create virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Ensure PyTorch matches your CUDA version if using GPU. See [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Place data files

Place the STEAD dataset files in `data/raw/`:

```
data/raw/merge.csv      # ~370 MB metadata
data/raw/merge.hdf5     # ~97 GB waveforms
```

## Training

### Multitask (MultiTaskCNN)

```bash
python src/train.py
```

Trains detection + phase picking + magnitude + location simultaneously. Checkpoint saved to `models/multitask_model.pth`.

> Note: `src/train.py` expects `merge.csv` and `merge.hdf5` in the current working directory. Run it from the folder that contains those files (for example, `data/raw`).

## Evaluation

```bash
python scripts/evaluate_model.py
```

Outputs:
- `models/metrics/evaluation_report.json` — accuracy, precision, recall, F1, ROC-AUC
- `models/metrics/confusion_matrix.png` — visual confusion matrix

Override defaults via environment variables:

```bash
$env:MODEL_PATH="models/checkpoints/multitask_model.pth"
python scripts/evaluate_model.py
```

## Running the App

```bash
python app.py
```

### Endpoints

| Method | Endpoint     | Description                  |
|--------|--------------|------------------------------|
| GET    | `/`          | Home page                    |
| GET    | `/upload`    | Upload UI                    |
| GET    | `/dashboard` | Dashboard view               |
| POST   | `/predict`   | Predict from uploaded file   |

### Example prediction

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@data/samples/earthquake_local_0.npy"
```

Response:

```json
{
  "prediction": "Earthquake",
  "confidence": 97.5,
  "p_arrival_sec": 4.23,
  "s_arrival_sec": 7.81,
  "magnitude": 3.42,
  "latitude": 35.1234,
  "longitude": -117.5678,
  "depth_km": 12.5,
  "location_status": "experimental"
}
```

## Testing

```bash
python -m pytest tests/ -v
```

Tests are lightweight and don't require the full 97GB HDF5 file.

## Technologies

- **PyTorch** — Model architecture and training
- **Flask** — Web app
- **H5Py** — HDF5 efficient chunk parsing
- **Pandas / NumPy / Scikit-Learn** — Data processing and metrics
- **Matplotlib / Seaborn** — Visualization
