# 🌍 Seismic AI Early Warning System

A deployment-oriented, end-to-end AI system for real-time earthquake detection, seismic phase estimation, and early warning alert simulation — built around a multi-task deep learning pipeline with inference-ready architecture.

---

## 🚀 Overview

This project implements a **complete AI inference and alert pipeline**, not just a trained model:

- 🎯 Near-perfect earthquake detection (99.92% accuracy)
- 📡 Multi-task seismic analysis (detection, P/S-wave timing, magnitude)
- ⚙️ Post-processing and decision engine with configurable alert logic
- ⚠️ Simulated early warning system with countdown estimation
- 🌐 Interactive web dashboard with real-time waveform visualization
- 🧠 Modular, deployment-ready architecture with GPU-optimized training

---

## 📘 Detailed Walkthrough

For a complete technical breakdown of the system architecture, training pipeline, and design decisions:

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| Language | Python |
| ML Framework | PyTorch (mixed precision, cached tensors) |
| Data Processing | Pandas, NumPy, h5py |
| Backend | Flask / FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Visualization | Matplotlib |
| Deployment | TorchScript / ONNX (planned) |

---

## 🎯 Problem Statement

In many environments — remote areas, disaster zones, mines, forests — traditional seismic monitoring infrastructure is unreliable or unavailable.

This system explores a **low-cost, AI-driven seismic monitoring approach** capable of:

- Detecting seismic events from raw 3-channel waveform signals
- Estimating seismic phase arrivals for warning-time calculation
- Simulating early warning alerts for immediate safety response

---

## 🧠 Model Architecture

Multi-task Conv1D neural network with shared feature extraction and task-specific prediction heads:

- **Input:** `(3, 6000)` — 60-second, 3-channel seismic waveform (E, N, Z)
- **Encoder:** Shared 1D convolutional backbone with batch normalization
- **Heads:** Independent output layers per task

| Task | Output | Status |
|------|--------|--------|
| Detection | Earthquake / Noise | ✅ Production-ready |
| P-wave Timing | Arrival sample index | ✅ Strong performance |
| S-wave Timing | Arrival sample index | ✅ Strong performance |
| Magnitude | Scalar estimate | 🧪 Experimental |

> **Note:** Magnitude estimation is included as an **auxiliary multi-task output**. It is not used in the final alert decision logic and remains experimental. The primary system value is driven by detection accuracy and phase timing.

---

## ⚙️ System Pipeline

```text
Raw Waveform (.npy / HDF5)
       ↓
  Preprocessing & Normalization
       ↓
  MultiTaskCNN Inference
       ↓
  Post-processing Logic
  (threshold filtering, phase conversion, confidence scoring)
       ↓
  Decision Engine
  (alert criteria evaluation)
       ↓
  Alert System + Web Dashboard
```

---

## 📊 Results

### Detection Performance

| Metric | Value |
|--------|-------|
| Accuracy | **99.92%** |
| Precision | **99.94%** |
| Recall | **99.91%** |
| F1 Score | **99.92%** |
| ROC-AUC | **1.00** |

✅ Detection achieves near-perfect classification, forming the **reliable core** of the alert pipeline.

### Confusion Matrix

| | Pred Noise | Pred EQ |
|---|---:|---:|
| Actual Noise | 23,584 | 24 |
| Actual EQ | 18 | 23,460 |

---

## 📐 Multi-task Performance

### Phase Timing (Regression)

| Metric | Value |
|--------|-------|
| P-wave MAE | **0.45 sec** |
| S-wave MAE | **0.59 sec** |

✅ Phase timing has been **significantly improved** through normalization corrections, proper masking of non-earthquake samples, and evaluation pipeline fixes. Sub-second accuracy enables meaningful warning-time estimation.

### Magnitude Estimation (Experimental)

| Metric | Value |
|--------|-------|
| Magnitude MAE | **~1.37** |

🧪 Magnitude estimation remains **experimental** — included as an auxiliary multi-task objective for research purposes. It is **not factored into alert decisions** and is not considered production-ready.

---

## ⚡ Training Optimization

The training pipeline is engineered for throughput and GPU utilization:

- **Cached Tensor Pipeline** — Pre-processed waveforms are cached as tensors to eliminate repeated HDF5 I/O during training
- **GPU Acceleration** — Full CUDA-accelerated training with automatic device management
- **Mixed Precision Training** — `torch.cuda.amp` for reduced memory footprint and faster forward/backward passes
- **Optimized DataLoader** — Multi-worker data loading with pinned memory and prefetching
- **Freeze/Unfreeze Encoder Strategy** — Staged training: freeze shared encoder while warming up task heads, then unfreeze for joint fine-tuning

---

## 🔧 Key Engineering Challenges

| Challenge | Approach |
|-----------|----------|
| **HDF5 I/O bottleneck** | Cached tensor pipeline to bypass repeated disk reads |
| **GPU utilization** | Mixed precision, pinned memory, optimized batch sizes |
| **Multi-task loss balancing** | Weighted loss combination with task-specific scaling |
| **Regression target normalization** | Sample-index to seconds conversion aligned between training and evaluation |
| **Evaluation consistency** | Centralized conversion helpers, proper masking for non-earthquake samples |

---

## 🚨 Alert System

The system includes a simulated early warning mechanism:

1. Detects seismic event via classification head
2. Estimates P–S arrival differential for warning window calculation
3. Applies confidence thresholding and decision logic
4. Triggers visual alert in the dashboard UI

### Example Alert

```text
⚠️ EMERGENCY WARNING

Strong earthquake shaking may begin in ~9.8 seconds.

Take immediate action:
  • Move to a safe place
  • Drop, Cover, and Hold On
```

---

## 🖥️ Dashboard Features

- Upload raw seismic waveform for inference
- Real-time multi-task prediction output
- 3-channel waveform visualization (E, N, Z)
- Alert popup with countdown estimation
- Signal quality metrics (energy, variance, SNR)
- Decision explanation with confidence scores

---

## 🎥 Demo
https://earthquake-prediction-42ti.onrender.com/

---

## 📁 Project Structure

```text
Earthquake_Prediction/
│
├── app/                 # Backend (Flask / FastAPI inference server)
├── configs/             # Training and evaluation configs
├── data/raw/            # Dataset (not included in repo)
├── models/
│   ├── checkpoints/     # Trained model weights (not tracked)
│   └── metrics/         # Evaluation outputs and plots
├── scripts/             # Training & evaluation entry points
├── src/
│   ├── models/          # Model architecture definitions
│   ├── training/        # Training loop, loss functions, schedulers
│   ├── inference/       # Inference pipeline and post-processing
│   └── utils/           # Metrics, conversion helpers, I/O utilities
├── static/              # Frontend assets (CSS, JS)
├── templates/           # Dashboard UI templates
└── app.py               # Main application entry point
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

```text
http://127.0.0.1:5000
```

---

## 🧪 Training

```bash
python scripts/train_improved_multitask.py
```

---

## 📊 Evaluation

```bash
python scripts/evaluate_model.py
```

---

## 🧠 Engineering Focus

This project emphasizes **practical ML systems engineering** beyond model accuracy:

- **End-to-end system design** — from raw data ingestion to user-facing alerts
- **Inference pipeline architecture** — post-processing, decision logic, and alert simulation
- **Uncertainty handling** — confidence thresholds, experimental task flagging, graceful degradation
- **Deployment thinking** — modular codebase, config-driven training, TorchScript export path
- **Beyond benchmark accuracy** — real-world considerations like I/O bottlenecks, GPU utilization, and evaluation integrity

---

## ⚠️ Disclaimer

- This is a **research prototype** and applied ML system demonstration.
- Not connected to real seismic station networks.
- Uses single-station waveform data only.
- Early warning timing is simulated, not field-validated.
- Not intended for operational emergency use.

---

## 🔮 Future Improvements

- Multi-station cross-validation pipeline
- Real-time streaming inference with WebSocket integration
- Hardware integration (geophone / accelerometer sensors)
- Model pruning and quantization for edge deployment
- Transformer-based encoder exploration
- Improved magnitude estimation with station metadata features

---

## 🧑‍💻 Author

**Aryan Singh**
B.Tech CSE (AI & DS)

GitHub: [aryan-singh69](https://github.com/aryan-singh69)
LinkedIn: [Aryan Singh](https://www.linkedin.com/in/aryan-singh-4b1992323/)

---

## ⭐ Key Highlights

- Built a **production-style AI inference system**, not just a trained model
- Integrated ML pipeline + backend + frontend + decision engine
- Designed a robust alert pipeline with post-processing beyond raw predictions
- Engineered GPU-optimized training with cached tensor acceleration
- Demonstrates real-world systems thinking under uncertainty
