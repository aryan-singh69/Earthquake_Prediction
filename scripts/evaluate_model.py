"""
Model Evaluation Script
=======================
Loads a trained MultiTaskCNN checkpoint and evaluates detection performance
on the test split of the STEAD dataset.

Outputs:
  - models/metrics/evaluation_report.json  (full metrics)
  - models/metrics/confusion_matrix.png    (heatmap image)

Usage:
  python scripts/evaluate_model.py

  Optional arguments (via environment variables):
    CSV_PATH    — metadata CSV   (default: data/raw/merge.csv)
    HDF5_PATH   — waveform HDF5  (default: data/raw/merge.hdf5)
    MODEL_PATH  — checkpoint path (default: models/checkpoints/multitask_model.pth)
    BATCH_SIZE  — eval batch size (default: 512)
"""

import os
import sys
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# ── Resolve project root (scripts/ is one level deep) ────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

# ── Import project modules ───────────────────────────────────
from dataset import STEADDataset
from models import MultiTaskCNN
from src.utils.metrics import (
    compute_all_metrics,
    save_metrics_to_json,
    plot_confusion_matrix,
    print_metrics_summary,
)


def main():
    # ── Configuration (relative paths, overridable via env) ───
    csv_path    = os.getenv("CSV_PATH",   "data/raw/merge.csv")
    hdf5_path   = os.getenv("HDF5_PATH",  "data/raw/merge.hdf5")
    model_path  = os.getenv("MODEL_PATH", "models/checkpoints/multitask_model.pth")
    batch_size  = int(os.getenv("BATCH_SIZE", "512"))

    # Resolve relative paths against project root
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, csv_path)
    if not os.path.isabs(hdf5_path):
        hdf5_path = os.path.join(PROJECT_ROOT, hdf5_path)
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)

    metrics_dir = os.path.join(PROJECT_ROOT, "models", "metrics")

    report_path = os.path.join(metrics_dir, "evaluation_report.json")
    cm_path     = os.path.join(metrics_dir, "confusion_matrix.png")
    errors_path = os.path.join(metrics_dir, "error_samples.csv")
    thresh_path = os.path.join(metrics_dir, "threshold_analysis.csv")

    os.makedirs(metrics_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  EARTHQUAKE DETECTION — MODEL EVALUATION")
    print("=" * 60)
    print(f"  Model     : {model_path}")
    print(f"  CSV       : {csv_path}")
    print(f"  HDF5      : {hdf5_path}")
    print(f"  Device    : {device}")
    print(f"  Batch Size: {batch_size}")
    print()

    # ── Validate files exist ──────────────────────────────────
    for fpath, label in [(model_path, "Model"), (csv_path, "CSV"), (hdf5_path, "HDF5")]:
        if not os.path.exists(fpath):
            print(f"  ERROR: {label} file not found → {fpath}")
            sys.exit(1)

    # ── Load & balance dataset (same logic as train.py) ───────
    print("  Loading and balancing dataset...")
    df = pd.read_csv(csv_path, low_memory=False).reset_index(drop=True)

    eq_df    = df[df["trace_category"] == "earthquake_local"].sample(n=235426, random_state=42)
    noise_df = df[df["trace_category"] == "noise"]
    balanced_df = pd.concat([eq_df, noise_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Balanced dataset: {len(balanced_df)} samples")

    # ── Reproduce same train/val/test split as train.py ───────
    indices = list(range(len(balanced_df)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.20, random_state=42)
    val_idx,   test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42)
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"  Evaluating on TEST split ({len(test_idx)} samples)...\n")

    # ── Create dataset & loader ───────────────────────────────
    full_dataset = STEADDataset(
        csv_file=csv_path,
        hdf5_file=hdf5_path,
        task="multitask",
        dataframe=balanced_df,
        preload_indices=None,
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ── Load model ────────────────────────────────────────────
    print("  Loading model checkpoint...")
    model = MultiTaskCNN().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    print("  Model loaded successfully!\n")

    # ── Run inference on test set ─────────────────────────────
    all_labels = []
    all_preds  = []
    all_probs  = []
    all_traces = []
    all_p_true = []
    all_s_true = []
    all_p_pred = []
    all_s_pred = []
    all_mag_true = []
    all_mag_pred = []

    start_time = time.time()

    # Run inference, but HDF5 handles may fail in multi-worker mode — catch and retry single-threaded
    def run_inference(loader):
        local_all_labels = []
        local_all_preds  = []
        local_all_probs  = []
        local_all_traces = []
        local_all_p_true = []
        local_all_s_true = []
        local_all_p_pred = []
        local_all_s_pred = []
        local_all_mag_true = []
        local_all_mag_pred = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                features  = batch["features"].to(device, non_blocking=True)
                labels    = batch["label"].numpy()
                # multitask ground-truth fields (on CPU tensors)
                p_true    = batch.get("p_arrival")
                s_true    = batch.get("s_arrival")
                mag_true  = batch.get("magnitude")
                trace_ids = batch["trace_name"]

                outputs   = model(features)
                probs     = torch.sigmoid(outputs["detection"]).cpu().numpy().squeeze(1)
                preds     = (probs >= 0.5).astype(int)

                # Phase and magnitude predictions (model outputs are raw regression values)
                phase_out = outputs["phase"].cpu().numpy()  # (B, 2)
                mag_out   = outputs["magnitude"].cpu().numpy().squeeze(1)

                # Append per-sample multitask fields (ensure numpy arrays)
                if p_true is not None:
                    local_all_p_true.extend(p_true.numpy().tolist())
                else:
                    local_all_p_true.extend([np.nan] * len(preds))

                if s_true is not None:
                    local_all_s_true.extend(s_true.numpy().tolist())
                else:
                    local_all_s_true.extend([np.nan] * len(preds))

                local_all_p_pred.extend(phase_out[:, 0].tolist())
                local_all_s_pred.extend(phase_out[:, 1].tolist())
                if mag_true is not None:
                    local_all_mag_true.extend(mag_true.numpy().tolist())
                else:
                    local_all_mag_true.extend([np.nan] * len(preds))
                local_all_mag_pred.extend(mag_out.tolist())

                local_all_labels.extend(labels.tolist())
                local_all_preds.extend(preds.tolist())
                local_all_probs.extend(probs.tolist())
                local_all_traces.extend(list(trace_ids))

                if (batch_idx + 1) % 20 == 0:
                    print(f"    Processed {(batch_idx + 1) * batch_size} / {len(test_idx)} samples...")

        return (local_all_labels, local_all_preds, local_all_probs, local_all_traces,
                local_all_p_true, local_all_s_true, local_all_p_pred, local_all_s_pred,
                local_all_mag_true, local_all_mag_pred)

    try:
        (all_labels, all_preds, all_probs, all_traces,
         all_p_true, all_s_true, all_p_pred, all_s_pred,
         all_mag_true, all_mag_pred) = run_inference(test_loader)
    except KeyError as e:
        print("\n  Warning: DataLoader worker KeyError (HDF5 access). Retrying single-threaded...\n")
        single_loader = DataLoader(
            Subset(full_dataset, test_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        (all_labels, all_preds, all_probs, all_traces,
         all_p_true, all_s_true, all_p_pred, all_s_pred,
         all_mag_true, all_mag_pred) = run_inference(single_loader)

    elapsed = time.time() - start_time
    print(f"  Inference complete! ({elapsed:.1f}s for {len(test_idx)} samples)\n")

    # ── Compute metrics ───────────────────────────────────────
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = compute_all_metrics(y_true, y_pred, y_prob=y_prob)
    metrics["model_path"]     = model_path
    metrics["num_test_samples"] = len(test_idx)
    metrics["inference_time_sec"] = round(elapsed, 2)

    # ── Save outputs ──────────────────────────────────────────
    save_metrics_to_json(metrics, report_path)
    plot_confusion_matrix(y_true, y_pred, cm_path)

    # ── Save error samples (FP/FN) ───────────────────────────
    test_meta = balanced_df.iloc[test_idx].reset_index(drop=True)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    error_mask = fp_mask | fn_mask

    # Compute absolute errors for P/S (in seconds) and magnitude
    sampling_rate = 100.0  # Hz — convert sample indices -> seconds
    p_true_arr = np.array(all_p_true, dtype=float)
    s_true_arr = np.array(all_s_true, dtype=float)
    p_pred_arr = np.array(all_p_pred, dtype=float)
    s_pred_arr = np.array(all_s_pred, dtype=float)
    mag_true_arr = np.array(all_mag_true, dtype=float)
    mag_pred_arr = np.array(all_mag_pred, dtype=float)

    # Absolute errors (seconds for P/S). Use NaN where ground-truth is missing or non-positive.
    valid_p_mask = p_true_arr >= 0
    valid_s_mask = s_true_arr >= 0
    abs_error_p_sec = np.full_like(p_true_arr, np.nan, dtype=float)
    abs_error_s_sec = np.full_like(s_true_arr, np.nan, dtype=float)
    abs_error_p_sec[valid_p_mask] = np.abs(p_pred_arr[valid_p_mask] - p_true_arr[valid_p_mask]) / sampling_rate
    abs_error_s_sec[valid_s_mask] = np.abs(s_pred_arr[valid_s_mask] - s_true_arr[valid_s_mask]) / sampling_rate

    # Magnitude absolute error
    mag_error = np.full_like(mag_true_arr, np.nan, dtype=float)
    mag_valid_mask = ~np.isnan(mag_true_arr)
    mag_error[mag_valid_mask] = np.abs(mag_pred_arr[mag_valid_mask] - mag_true_arr[mag_valid_mask])

    error_rows = {
        "trace_name":     np.array(all_traces, dtype=object)[error_mask],
        "true_label":     y_true[error_mask],
        "predicted_label": y_pred[error_mask],
        "probability":    y_prob[error_mask],
    }

    if "trace_category" in test_meta.columns:
        error_rows["trace_category"] = test_meta.loc[error_mask, "trace_category"].values
    else:
        error_rows["trace_category"] = [None] * int(error_mask.sum())

    if "source_id" in test_meta.columns:
        error_rows["source_id"] = test_meta.loc[error_mask, "source_id"].values
    else:
        error_rows["source_id"] = [None] * int(error_mask.sum())

    if "station_id" in test_meta.columns:
        error_rows["station_id"] = test_meta.loc[error_mask, "station_id"].values
    else:
        error_rows["station_id"] = [None] * int(error_mask.sum())

    if "source_magnitude" in test_meta.columns:
        error_rows["magnitude"] = test_meta.loc[error_mask, "source_magnitude"].values
    elif "magnitude" in test_meta.columns:
        error_rows["magnitude"] = test_meta.loc[error_mask, "magnitude"].values
    else:
        error_rows["magnitude"] = [None] * int(error_mask.sum())

    pd.DataFrame(error_rows).to_csv(errors_path, index=False)
    print(f"  Error samples saved → {errors_path}")

    # Add new columns to the saved error CSV: error_type and absolute errors
    err_df = pd.read_csv(errors_path)
    # error_mask relative to full test set; extract indices
    ep = np.where(error_mask)[0]
    # Build arrays for CSV rows
    err_df["error_type"] = ["false_positive" if fp else "false_negative" for fp in fp_mask[error_mask]]
    # Map absolute error arrays to the filtered rows
    err_df["absolute_error_p_sec"] = abs_error_p_sec[ep]
    err_df["absolute_error_s_sec"] = abs_error_s_sec[ep]
    # magnitude_error (keep NaN where not available)
    err_df["magnitude_error"] = mag_error[ep]
    err_df.to_csv(errors_path, index=False)
    print(f"  Extended error samples saved → {errors_path}")

    # ── Threshold analysis ──────────────────────────────────
    thresholds = [0.3, 0.5, 0.7, 0.9]
    rows = []
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        rows.append({
            "threshold": thr,
            "accuracy":  float(accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall":    float(recall_score(y_true, pred, zero_division=0)),
            "f1":        float(f1_score(y_true, pred, zero_division=0)),
            "false_positives": fp,
            "false_negatives": fn,
            "false_positive_count": fp,
            "false_negative_count": fn,
        })

    pd.DataFrame(rows).to_csv(thresh_path, index=False)
    print(f"  Threshold analysis saved → {thresh_path}")

    # ── Multi-task metrics: P/S arrival and magnitude errors ─────────────────
    multitask_metrics = {}

    # P-wave metrics (seconds)
    if valid_p_mask.any():
        p_wave_mae_sec = float(np.nanmean(abs_error_p_sec[valid_p_mask]))
        p_wave_rmse_sec = float(np.sqrt(np.nanmean(abs_error_p_sec[valid_p_mask] ** 2)))
    else:
        p_wave_mae_sec = None
        p_wave_rmse_sec = None

    # S-wave metrics (seconds)
    if valid_s_mask.any():
        s_wave_mae_sec = float(np.nanmean(abs_error_s_sec[valid_s_mask]))
        s_wave_rmse_sec = float(np.sqrt(np.nanmean(abs_error_s_sec[valid_s_mask] ** 2)))
    else:
        s_wave_mae_sec = None
        s_wave_rmse_sec = None

    # Magnitude metrics (only earthquake samples)
    eq_mask = (y_true == 1)
    eq_mag_mask = eq_mask & (~np.isnan(mag_true_arr))
    if eq_mag_mask.any():
        magnitude_mae = float(np.nanmean(mag_error[eq_mag_mask]))
        magnitude_rmse = float(np.sqrt(np.nanmean((mag_error[eq_mag_mask]) ** 2)))
    else:
        magnitude_mae = None
        magnitude_rmse = None

    multitask_metrics["p_wave_mae_sec"] = p_wave_mae_sec
    multitask_metrics["p_wave_rmse_sec"] = p_wave_rmse_sec
    multitask_metrics["s_wave_mae_sec"] = s_wave_mae_sec
    multitask_metrics["s_wave_rmse_sec"] = s_wave_rmse_sec
    multitask_metrics["magnitude_mae"] = magnitude_mae
    multitask_metrics["magnitude_rmse"] = magnitude_rmse

    multitask_path = os.path.join(metrics_dir, "multitask_metrics.json")
    with open(multitask_path, "w") as f:
        json.dump(multitask_metrics, f, indent=2)
    print(f"  Multi-task metrics saved → {multitask_path}")

    # ── Final summary JSON (combined) ───────────────────────
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    final_summary = {
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1_score"),
        "roc_auc": metrics.get("roc_auc"),
        "p_wave_mae_sec": p_wave_mae_sec,
        "magnitude_mae": magnitude_mae,
        "false_positives": int(cm[0][1]) if cm is not None else None,
        "false_negatives": int(cm[1][0]) if cm is not None else None,
    }

    final_path = os.path.join(metrics_dir, "final_summary.json")
    with open(final_path, "w") as f:
        json.dump(final_summary, f, indent=2)
    print(f"  Final summary saved → {final_path}")

    # ── Print concise multi-task summary ─────────────────────
    print("\nMULTI-TASK PERFORMANCE")
    print(f"P-wave MAE: {p_wave_mae_sec if p_wave_mae_sec is not None else 'N/A'} sec")
    print(f"S-wave MAE: {s_wave_mae_sec if s_wave_mae_sec is not None else 'N/A'} sec")
    print(f"Magnitude MAE: {magnitude_mae if magnitude_mae is not None else 'N/A'}")

    # ── Print summary ─────────────────────────────────────────
    print_metrics_summary(metrics)

    print(f"  Output files:")
    print(f"    → {report_path}")
    print(f"    → {cm_path}")
    print("\n  Evaluation complete! ✅\n")


if __name__ == "__main__":
    main()
