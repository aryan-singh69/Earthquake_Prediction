# 🔍 Project Walkthrough — Seismic AI Early Warning System

This walkthrough is intended for technical reviewers and explains both implementation and design reasoning.

This document provides a detailed step-by-step walkthrough of how the system works — from raw seismic data to final alert generation.

---

## 🧠 1. Problem Understanding

The goal is to build an AI system that can:

- Detect earthquakes from seismic waveforms
- Estimate key parameters (P-wave, S-wave, magnitude)
- Simulate an early warning alert system

---

## 📦 2. Data Source

Dataset used: **STEAD (Stanford Earthquake Dataset)**

### Key properties:

- ~1.2M waveform samples
- Each waveform:
  - Length: **6000 samples**
  - Sampling rate: **100 Hz**
  - Channels: **3 (E, N, Z)**

---

## 🔄 3. Data Flow Overview

```text
CSV Metadata + HDF5 Waveforms
↓
Custom DataLoader (STEADDataset)
↓
Batch Input (3 × 6000)
↓
Model (MultiTaskCNN)
↓
Predictions (multi-task)
↓
Post-processing
↓
Alert Decision
↓
Dashboard Output
```

---

## 🧩 4. Data Loading

File: `src/dataset.py`

### Process:

1. CSV (`merge.csv`) provides metadata:
   - trace name
   - label (noise / earthquake)
   - magnitude
   - P/S arrival times

2. HDF5 (`merge.hdf5`) stores waveform data

3. DataLoader:
   - Dynamically loads waveform using `trace_name`
   - Converts into tensor shape `(3, 6000)`

---

## 🧠 5. Model Architecture

File: `src/models/multitask_cnn.py`

### Shared Encoder

- Conv1D layers extract temporal features
- MaxPooling reduces dimensionality
- Fully connected layer produces shared representation

### Task Heads

| Head | Purpose |
|------|---------|
| Detection | Earthquake vs Noise |
| Phase | Predict P and S arrival |
| Magnitude | Estimate magnitude |
| Location | Experimental |

---

## ⚙️ 6. Training Strategy

File: `scripts/train_improved_multitask.py`

### Key Concepts:

#### Multi-task Loss

```text
Total Loss =
1.0 * detection_loss +
5.0 * phase_loss +
3.0 * magnitude_loss
```

- Phase and magnitude losses applied only to earthquake samples
- Noise samples ignored for regression tasks

### Training Phases

#### Phase 1: Feature learning

- Encoder frozen
- Only heads trained

#### Phase 2: Fine-tuning

- Encoder partially unfrozen
- Lower learning rate applied

---

## 🧠 6.1 Design Decisions (Why this approach?)

Multi-task learning is used because seismic analysis is not a single-output problem. Earthquake detection, phase picking, and magnitude estimation all depend on shared waveform patterns, so a shared encoder can learn useful signal representations once and reuse them across related tasks.

Detection is prioritized because it is the most safety-critical and most reliable output in the current system. Before estimating timing or magnitude, the pipeline must first answer the most important question: whether the signal likely represents an earthquake or noise.

Phase and magnitude prediction are treated as secondary tasks because they are more sensitive to waveform quality, labeling noise, and single-station ambiguity. These regression outputs add useful context, but they are not trusted as strongly as the detection head when making final decisions.

A post-processing layer is necessary because raw neural network outputs can be overconfident or unstable on signals that differ from the training distribution. Combining model confidence with signal-level checks creates a more robust decision pipeline than using logits or probabilities alone.

The design favors robustness over accuracy alone. In an early warning-style system, a technically high model score is less useful if the surrounding pipeline cannot handle uncertainty, noisy inputs, or ambiguous cases.

---

## 📊 7. Evaluation Pipeline

File: `scripts/evaluate_model.py`

### Metrics computed:

- Accuracy
- Precision / Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

### Multi-task Metrics

- P-wave MAE (seconds)
- S-wave MAE (seconds)
- Magnitude MAE

---

## 🔎 8. Post-processing Logic

File: `src/inference/postprocess.py`

Model predictions are not used directly. A rule-based decision layer is applied on top of model outputs to improve robustness.

This mimics real-world systems where AI predictions are combined with domain heuristics before triggering user-facing decisions. It helps reduce false positives and unstable predictions when waveform quality or model confidence is uncertain.

### Additional features computed:

- Signal energy
- Signal variance

### Decision logic:

```text
if high confidence + strong signal:
    Earthquake
elif medium confidence:
    Possible Earthquake
else:
    Noise
```

---

## 🚨 9. Alert System

The system simulates early warning:

### Logic:

1. Detect earthquake
2. Estimate warning window based on model-derived signal characteristics
3. Generate warning message

This is a simulated approximation and not a physically accurate seismic prediction.

### Example:

```text
Strong shaking expected in ~10 seconds
```

---

## 🖥️ 10. Frontend System

Files:

- `templates/upload.html`
- `static/js/main.js`

### Features:

- File upload (.npy waveform)
- Real-time prediction
- Waveform visualization (3 channels)
- Alert popup display
- Metrics display

---

## 🔄 11. End-to-End Flow

```text
User uploads waveform
↓
Backend receives file
↓
Preprocessing
↓
Model inference
↓
Post-processing
↓
Decision logic
↓
Alert generation
↓
Frontend display
```

---

## ⚠️ 12. Current Limitations

- Single-station system, which limits the ability to validate events spatially or triangulate source location.
- No real-time streaming pipeline; current inference is based on uploaded waveform files rather than continuous sensor ingestion.
- P/S predictions are noisy and not yet accurate enough for reliable operational timing estimates.
- Magnitude estimation is weak for small events and can be sensitive to waveform scaling, station effects, and dataset labeling noise.
- Dataset vs real-world distribution gap: STEAD is valuable for training, but field deployment would face different sensors, noise patterns, geologies, and event types.
- Not suitable for real emergency deployment because the alert logic is simulated and has not been validated against operational seismic standards.

The system is a prototype focused on architecture and pipeline design, not production safety.

---

## 🔮 13. Future Improvements

- Multi-station validation
- Real-time streaming ingestion
- Improved phase detection models
- Hardware integration (geophones)
- Edge deployment optimization

---

## 🎯 Conclusion

This project demonstrates end-to-end AI system design rather than isolated model training. It connects waveform ingestion, deep learning inference, post-processing logic, alert generation, and a user-facing dashboard into one coherent pipeline.

The system also shows how practical AI applications must handle uncertainty. Instead of treating model output as final truth, the pipeline combines ML predictions with signal metrics and decision rules to produce more stable, explainable behavior.

By integrating machine learning, domain-inspired logic, and UI feedback, the project reflects real-world deployment thinking beyond notebooks and benchmark scores.

> The focus is on building a complete intelligent system, not just achieving high accuracy.
