# 🔍 Project Walkthrough — Seismic AI Early Warning System

This walkthrough is intended for technical reviewers and explains both implementation details and design reasoning behind the system.

It covers the full pipeline — from raw seismic data ingestion through multi-task inference to alert generation and dashboard output.

---

## 🧠 1. Problem Understanding

The goal is to build an end-to-end AI system that can:

- Detect earthquakes from raw seismic waveforms with high reliability
- Estimate seismic phase arrivals (P-wave, S-wave) for warning-time calculation
- Provide auxiliary magnitude estimation as a multi-task output
- Simulate an early warning alert pipeline with post-processing and decision logic

---

## 📦 2. Data Source

Dataset used: **STEAD (Stanford Earthquake Dataset)**

### Key properties:

- ~1.2M waveform samples
- Each waveform:
  - Length: **6000 samples**
  - Sampling rate: **100 Hz** (60 seconds of data)
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
Multi-task Predictions
       ↓
Post-processing & Confidence Scoring
       ↓
Decision Engine
       ↓
Alert System + Dashboard Output
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
   - Applies normalization and label encoding

---

## 🔬 5. Why Multi-task Learning?

Seismic analysis is inherently a multi-output problem. Detection, phase picking, and magnitude estimation all operate on the same underlying waveform signal. Multi-task learning exploits this by:

- **Shared waveform representations** — A single encoder learns temporal and spectral features once, then reuses them across all tasks. This is more efficient than training separate models per task.
- **Encoder reuse** — The shared convolutional backbone extracts low-level seismic features (onset patterns, frequency content, amplitude envelopes) that are relevant to multiple downstream predictions.
- **Correlated seismic tasks** — P-wave arrival, S-wave arrival, magnitude, and event classification are physically correlated. Joint training allows the model to implicitly capture these relationships, improving generalization on the primary detection task.

The trade-off is that auxiliary tasks (magnitude, phase timing) may not individually reach the performance ceiling of dedicated single-task models — but they provide useful additional context at negligible marginal inference cost.

---

## 🧠 6. Model Architecture

File: `src/models/multitask_cnn.py`

### Shared Encoder

- Conv1D layers extract temporal features from raw waveforms
- MaxPooling reduces temporal dimensionality
- Batch normalization stabilizes training
- Fully connected layer produces shared representation vector

### Task Heads

| Head | Purpose | Status |
|------|---------|--------|
| Detection | Earthquake vs Noise classification | ✅ Production-ready |
| Phase | Predict P and S arrival indices | ✅ Strong performance |
| Magnitude | Estimate event magnitude | 🧪 Experimental |

> **Note:** Magnitude estimation is an **auxiliary multi-task output**. It is included for research purposes but is **not used in the final alert decision logic**. The system's reliability is anchored on detection accuracy and phase timing.

---

## ⚙️ 7. Training Strategy

File: `scripts/train_improved_multitask.py`

### Multi-task Loss

```text
Total Loss =
    1.0 × detection_loss +
    5.0 × phase_loss +
    3.0 × magnitude_loss
```

- Phase and magnitude losses are applied **only to earthquake samples**
- Noise samples are masked out for regression tasks
- Higher phase weight reflects its importance for warning-time estimation

### Training Phases

#### Phase 1: Head warm-up

- Shared encoder **frozen**
- Only task-specific heads are trained
- Prevents early gradient noise from corrupting learned representations

#### Phase 2: Joint fine-tuning

- Encoder **partially unfrozen**
- Lower learning rate applied to encoder layers
- Allows end-to-end optimization without catastrophic forgetting

---

## ⚡ 8. Training Optimization Pipeline

The training pipeline is engineered for throughput and GPU utilization:

### Cached Tensor Pipeline

Raw waveforms are read from HDF5 once, preprocessed, and cached as tensors on disk. Subsequent training epochs load directly from cached tensors, completely bypassing the HDF5 I/O bottleneck. This dramatically reduces per-epoch wall time, especially on datasets with millions of traces.

### GPU Acceleration

- Full CUDA-accelerated training with automatic device management
- `torch.cuda.amp` mixed precision training for reduced memory footprint and faster forward/backward passes
- Gradient scaling to maintain numerical stability under FP16

### Optimized DataLoader

- Multi-worker data loading with `num_workers > 0`
- Pinned memory (`pin_memory=True`) for faster CPU-to-GPU transfers
- Prefetching to overlap data loading with computation

### Freeze/Unfreeze Strategy

Staged training prevents task heads from receiving noisy gradients through an untrained encoder during early epochs. The encoder is frozen during head warm-up, then selectively unfrozen with a reduced learning rate for joint fine-tuning. This improves convergence stability and final task performance.

---

## 🧠 9. Design Decisions

### Why detection is the primary task

Detection is the most safety-critical output. Before estimating timing or magnitude, the pipeline must first answer the most important question: **is this signal an earthquake or noise?** Detection is also the most reliable output — classification on well-separated waveform classes achieves near-perfect accuracy, giving the system a trustworthy foundation.

### Why regression tasks are auxiliary

Phase timing and magnitude estimation are inherently harder than binary classification. They are more sensitive to waveform quality, labeling noise, single-station ambiguity, and dataset-specific characteristics. These outputs provide useful context but are not trusted as strongly as the detection head in final alert decisions.

### Why post-processing is necessary

Raw neural network outputs can be overconfident or unstable on signals that differ from the training distribution. Combining model confidence with signal-level heuristics (energy, variance, SNR) creates a more robust decision pipeline than using raw logits or probabilities alone. This mirrors real-world seismic systems where AI predictions are combined with domain checks before triggering alerts.

### Why real-world uncertainty matters

In an early warning-style system, a technically high benchmark score is less useful if the surrounding pipeline cannot handle uncertainty, noisy inputs, or ambiguous edge cases. The system design prioritizes robustness and explainability over raw accuracy alone.

---

## 📊 10. Evaluation Pipeline

File: `scripts/evaluate_model.py`

### Detection Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **99.92%** |
| Precision | **99.94%** |
| Recall | **99.91%** |
| F1 Score | **99.92%** |
| ROC-AUC | **1.00** |

### Multi-task Metrics

| Metric | Value |
|--------|-------|
| P-wave MAE | **0.45 sec** |
| S-wave MAE | **0.59 sec** |
| Magnitude MAE | **~1.37** |

Phase timing has been **significantly improved** after multi-task optimization — proper normalization, evaluation masking, and conversion handling now yield **sub-second average timing error** on the evaluation split.

Magnitude estimation remains **experimental** and is not factored into alert decisions.

---

## 🧪 11. Evaluation Improvements

The evaluation pipeline has been refined to ensure reported metrics are accurate and meaningful:

### Architecture-aware evaluation

The evaluation script detects whether it is running against the baseline or improved multi-task model and adjusts its metric computation accordingly. This prevents mismatched output parsing between model versions.

### Correct phase conversion handling

A centralized `convert_phase_to_seconds` helper handles the conversion between STEAD ground-truth sample indices and normalized model outputs. This eliminates the scaling errors that previously inflated P/S MAE values by orders of magnitude.

### Proper validation masking

Regression metrics (phase timing, magnitude) are computed **only on earthquake samples**. Non-earthquake/noise samples are masked out to prevent diluting or distorting error statistics.

### Threshold analysis

Detection performance is evaluated across multiple confidence thresholds, not just the default 0.5 cutoff. This provides a more complete picture of the precision-recall trade-off.

### Extended error analysis

Per-task error distributions are logged to identify systematic biases (e.g., consistent over/under-prediction on specific magnitude ranges or phase positions).

---

## 🔎 12. Post-processing Logic

File: `src/inference/postprocess.py`

Model predictions are not used directly. A rule-based decision layer is applied on top of model outputs to improve robustness.

This mirrors real-world systems where AI predictions are combined with domain heuristics before triggering user-facing decisions. It reduces false positives and handles unstable predictions when waveform quality or model confidence is uncertain.

### Additional features computed:

- Signal energy
- Signal variance
- SNR estimation

### Decision logic:

```text
if high confidence + strong signal:
    → Earthquake
elif medium confidence:
    → Possible Earthquake
else:
    → Noise
```

---

## 🚨 13. Alert System

The system simulates early warning:

### Logic:

1. Detect earthquake via classification head
2. Estimate warning window using P–S arrival differential
3. Apply confidence thresholding
4. Generate warning message with countdown

This is a simulated approximation and not a physically accurate seismic prediction.

### Example:

```text
⚠️ Strong shaking expected in ~10 seconds
```

---

## 🖥️ 14. Frontend System

Files:

- `templates/upload.html`
- `static/js/main.js`

### Features:

- File upload (.npy waveform)
- Real-time multi-task prediction output
- 3-channel waveform visualization (E, N, Z)
- Alert popup with countdown display
- Signal quality metrics
- Decision explanation with confidence scores

---

## 🔄 15. End-to-End Flow

```text
User uploads waveform
       ↓
Backend receives file
       ↓
Preprocessing & normalization
       ↓
Multi-task model inference
       ↓
Post-processing & confidence scoring
       ↓
Decision engine (alert criteria)
       ↓
Alert generation
       ↓
Frontend display with explanation
```

---

## ⚠️ 16. Current Limitations

- **Single-station system** — limits ability to spatially validate events or triangulate source location.
- **No real-time streaming** — current inference operates on uploaded waveform files rather than continuous sensor ingestion.
- **Magnitude estimation is experimental** — sensitive to waveform scaling, station effects, and dataset labeling noise. Not used in alert logic.
- **Dataset vs. real-world gap** — STEAD is valuable for training, but field deployment would face different sensors, noise patterns, geologies, and event types.
- **Simulated alert logic** — the early warning pipeline has not been validated against operational seismic standards and is not suitable for real emergency deployment.

The system is a research prototype focused on architecture, pipeline design, and inference reliability — not production safety certification.

---

## 🔮 17. Future Improvements

- Multi-station cross-validation pipeline
- Real-time streaming ingestion with WebSocket integration
- Transformer-based encoder exploration
- Hardware integration (geophone / accelerometer sensors)
- Model pruning and quantization for edge deployment
- Improved magnitude estimation with station metadata features

---

## 🎯 Conclusion

This project demonstrates **end-to-end AI system engineering** rather than isolated model training. It connects waveform ingestion, GPU-optimized training, deep learning inference, post-processing logic, alert generation, and a user-facing dashboard into one coherent pipeline.

The system shows how practical AI applications must **handle uncertainty at every layer**. Instead of treating model output as final truth, the pipeline combines ML predictions with signal-level metrics and rule-based decision logic to produce more stable, explainable behavior.

The inference architecture is designed with **deployment in mind** — modular components, config-driven training, cached data pipelines, and clear separation between model outputs and system decisions.

> The focus is on building a complete, reliable intelligent system — not just achieving high accuracy on a benchmark.
