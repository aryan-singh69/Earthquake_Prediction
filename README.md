# 🌍 Seismic AI Early Warning System

An end-to-end AI-powered seismic analysis system that detects earthquakes from raw waveform data and simulates an early warning alert pipeline.

---

## 🚀 Overview

This project goes beyond a traditional ML model and implements a **complete AI system pipeline**:

- 📊 High-accuracy earthquake detection (~99.9%)
- 📡 Multi-task seismic analysis (P-wave, S-wave, magnitude)
- ⚠️ Alert decision system with early warning simulation
- 🌐 Interactive web dashboard with waveform visualization
- 🧠 Modular architecture for future real-world deployment

---

## 📘 Detailed Walkthrough

For a complete technical breakdown of the system:

👉 [Project Walkthrough](./WALKTHROUGH.md)

---

## 🛠️ Tech Stack

- Language: Python
- ML Framework: PyTorch
- Data Processing: Pandas, NumPy, h5py
- Backend: Flask / FastAPI
- Frontend: HTML, CSS, JavaScript
- Visualization: Matplotlib
- Model Deployment: TorchScript / ONNX (planned)

---

## 🎯 Problem Statement

In many environments (remote areas, disaster zones, mines, forests), traditional communication and seismic monitoring systems are unreliable or unavailable.

This project explores a **low-cost AI-based seismic monitoring approach** that can:

- Detect seismic activity from waveform signals
- Estimate important seismic characteristics
- Simulate early warning alerts for safety response

---

## 🧠 Model Architecture

Multi-task Conv1D neural network:

- Input: `(3, 6000)` seismic waveform (E, N, Z channels)
- Shared CNN encoder
- Task-specific heads:

| Task | Output |
|------|--------|
| Detection | Earthquake / Noise |
| Phase Picking | P-wave, S-wave |
| Magnitude | Scalar value |
| Location | Experimental |

---

## ⚙️ System Pipeline

```text
Raw Waveform (.npy / HDF5)
↓
Preprocessing
↓
MultiTaskCNN Model
↓
Post-processing Logic
↓
Decision Engine
↓
Alert System + Dashboard
```

---

## 📊 Results

### Detection Performance

- Accuracy: **99.91%**
- Precision: **99.90%**
- Recall: **99.92%**
- F1 Score: **99.91%**
- ROC-AUC: **1.00**

✔ Detection model achieves near-perfect classification performance on the dataset, forming the reliable core of the system.

### Confusion Matrix

| | Pred Noise | Pred EQ |
|---|---:|---:|
| Actual Noise | 23584 | 24 |
| Actual EQ | 18 | 23460 |

---

## ⚠️ Multi-task Performance (Experimental)

| Metric | Value |
|--------|-------|
| P-wave MAE | ~6.6 sec ❌ |
| S-wave MAE | ~13.4 sec ❌ |
| Magnitude MAE | ~1.37 ❌ |

> These outputs are currently experimental and under improvement. Detection is highly reliable, but phase timing and magnitude estimation are not yet production-ready.

---

## 🚨 Alert System

The system includes a simulated early warning mechanism:

- Detects seismic event
- Estimates warning window using model-derived signal characteristics (simulated)
- Triggers emergency alert in UI

### Example Alert

```text
EMERGENCY WARNING

Strong earthquake shaking may begin in ~9.8 seconds.

Take immediate action:
• Move to a safe place
• Drop, Cover, and Hold On
```

---

## 🖥️ Dashboard Features

- Upload seismic waveform
- Real-time prediction output
- 3-channel waveform visualization
- Alert popup system
- Signal metrics (energy, variance)
- Clean UI with decision explanation

---

## 🎥 Demo

- Waveform visualization
- Alert popup system
- Prediction output

(Add screenshots or demo video here)

---

## 📁 Project Structure

```text
Earthquake_Prediction/
│
├── app/                 # Backend (Flask / FastAPI logic)
├── configs/             # Training configs
├── data/raw/            # Dataset (not included)
├── models/
│   ├── checkpoints/     # Trained models (not tracked)
│   └── metrics/         # Evaluation outputs
├── scripts/             # Training & evaluation scripts
├── src/
│   ├── models/          # Model architecture
│   ├── training/        # Training utilities
│   ├── inference/       # Prediction logic
│   └── utils/           # Metrics & helpers
├── static/              # Frontend assets
├── templates/           # UI templates
└── app.py               # Main dashboard entry
```

---

## 🛠️ Setup

```bash
git clone <repo-url>
cd Earthquake_Prediction

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Run Dashboard

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

## 🧪 Training (Improved Model)

```bash
python scripts/train_improved_multitask.py
```

---

## 📊 Evaluation

```bash
python scripts/evaluate_model.py
```

---

## ⚠️ Disclaimer

- This is a research / prototype system.
- Not connected to real seismic stations.
- Uses single-station data only.
- Early warning is simulated, not real-world reliable.
- Not intended for real emergency use.

---

## 🔮 Future Improvements

- Multi-station validation system
- Real-time streaming pipeline
- Hardware integration (geophone sensors)
- Model pruning + edge deployment
- Improved phase detection accuracy
- Better magnitude estimation

---

## 🧑‍💻 Author

Aryan Singh  
B.Tech CSE (AI & DS)

GitHub: https://github.com/aryan-singh69  
LinkedIn: https://www.linkedin.com/in/aryan-singh-4b1992323/

---

## ⭐ Key Highlights

- Built a production-style AI inference system, not just a model
- Integrated ML + backend + frontend + decision logic
- Designed a robust alert pipeline beyond raw predictions
- Demonstrates real-world system thinking under uncertainty
