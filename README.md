# STEAD-Based Seismic ML Project

## 1. Project Goal
This project aims to build a Machine Learning pipeline that uses the **Stanford Earthquake Dataset (STEAD)** to detect earthquakes, pick P/S wave arrivals, and classify seismic signals (earthquake vs. noise). The dataset includes approximately 1.2 million global seismic waveforms recorded at various stations.

## 2. Project Architecture
The overarching goal is to parse 60 seconds of 3-component seismic waveform recordings at a sampling rate of 100 Hz (shape: `3 x 6000`), extract distinguishing features, and classify/regress key metrics.

### Phase Framework
- **Phase 1: Setup & Data Acquisition**: Downloaded the ~97GB STEAD HDF5 format and CSV metadata files. Set up the development pipeline.
- **Phase 2: EDA & Preprocessing**: Analyze distribution attributes (class imbalance, magnitude, depth, distance). Construct a bespoke PyTorch DataLoader that parses chunks of the HDF5 dataset dynamically without causing system OOM (Out-of-Memory) crashes.
- **Phase 3: Task Selection & Modeling**: Currently starting with a **Simple 1D CNN Base Model** for binary Earthquake Detection. Future iterations will consider U-Net styles (PhaseNet) and Transformer architectures (EQTransformer).
- **Phase 4: Training & Evaluation**: PyTorch training scripts configured to process batched arrays gracefully, optimizing for BCEWithLogitsLoss.
- **Phase 5 & 6**: Post-analysis augmentation, bias correction, and final API/Deployment.

## 3. Directory Structure
```text
Earthquake_Prediction/
├── data/
│   └── raw/                    # Store sliced subsets or data splits here.
├── docs/                       # Project documentation and literature.
├── models/                     # Saved PyTorch (.pth) checkpoints.
├── notebooks/
│   ├── 01_eda.py               # Generates Matplotlib distribution plots.
│   └── *.png                   # Resulting chart images (class count, magnitude, etc).
├── src/
│   ├── dataset.py              # Custom STEADDataset DataLoader logic handling h5py dynamically.
│   ├── models.py               # SimpleCNN architectural definition.
│   ├── train.py                # PyTorch Training loops & evaluation workflow.
│   └── inspect_data.py         # Testing script for safely traversing HDF5 keys.
├── merge.csv                   # STEAD metadata (1.2M rows x 35 columns)
├── merge.hdf5                  # STEAD Waveform data (97GB root data block containing the time-series arrays)
└── requirements.txt            # Python environment constraints.
```

## 4. Setup Instructions

**1. Create & Activate a virtual environment (Recommended)**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# or
source venv/bin/activate      # Unix
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

*(Ensure PyTorch is correctly matched to your CUDA architecture if utilizing GPU acceleration).*

## 5. Usage Guide

### Generating Exploratory Data Analysis (EDA)
Navigate to the `notebooks` directory and run the EDA script to produce statistical distributions of the dataset.
```bash
cd notebooks
python 01_eda.py
```
*Outputs: `class_distribution.png`, `depth_distribution.png`, `distance_distribution.png`, `magnitude_distribution.png`*

### Training the Model
To initiate a training run of the baseline `SimpleCNN` model on a subset of the dataset:
```bash
python src/train.py
```
This iterates across pseudo-splits, batches the dataloaders, computes gradients, prints epoch losses, evaluates raw accuracy, and saves the final checkpoint into `models/baseline_cnn.pth`.

---

### Technologies
* **Python**
* **PyTorch & TorchAudio**
* **H5Py** (HDF5 efficient chunk parsing)
* **Pandas / NumPy / Scikit-Learn**
* **Matplotlib / Seaborn**
* **Weights & Biases (W&B)** (Ready/Configurable experiment tracking)
