"""
Post-processing utilities for detection decisions and signal checks.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Optional, Tuple

import numpy as np


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """
    Per-channel zero-mean, unit-std normalization.
    Expects (3, N) input and returns (3, N).
    """
    mean = waveform.mean(axis=1, keepdims=True)
    std = waveform.std(axis=1, keepdims=True) + 1e-8
    return (waveform - mean) / std


def ensure_channel_first(waveform: np.ndarray) -> np.ndarray:
    """Ensure waveform shape is (3, N)."""
    if waveform.ndim != 2:
        raise ValueError("Waveform must be 2D")
    if waveform.shape[0] == 3:
        return waveform
    if waveform.shape[1] == 3:
        return waveform.T
    raise ValueError("Waveform must have 3 channels")


def compute_signal_energy(waveform: np.ndarray) -> float:
    """Mean squared amplitude over all channels."""
    return float(np.mean(np.square(waveform)))


def compute_signal_variance(waveform: np.ndarray) -> float:
    """Variance over all channels."""
    return float(np.var(waveform))


def compute_peak_to_peak(waveform: np.ndarray) -> float:
    """Peak-to-peak amplitude across all channels."""
    return float(np.max(waveform) - np.min(waveform))


def detect_spikes(waveform: np.ndarray, spike_threshold: float) -> bool:
    """Detect spikes based on max absolute amplitude."""
    return float(np.max(np.abs(waveform))) >= spike_threshold


def load_threshold_analysis(path: str) -> Tuple[Optional[float], Dict[str, float]]:
    """
    Read threshold analysis CSV and return recommended thresholds.

    Returns:
        (recommended_threshold, summary_dict)
    """
    if not os.path.exists(path):
        return None, {}

    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "threshold": float(row["threshold"]),
                    "accuracy": float(row["accuracy"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                })
            except (KeyError, ValueError):
                continue

    if not rows:
        return None, {}

    best_f1 = max(rows, key=lambda r: r["f1"])

    precision_min = float(os.getenv("PRECISION_ACCEPTABLE_MIN", "0.7"))
    recall_min = float(os.getenv("RECALL_PREFERRED_MIN", "0.9"))
    recall_candidates = [r for r in rows if r["precision"] >= precision_min]
    best_recall = max(recall_candidates, key=lambda r: r["recall"], default=None)

    if best_recall and best_recall["recall"] >= recall_min:
        recommended = best_recall["threshold"]
        summary = {
            "recommended": recommended,
            "basis": "high_recall",
            "precision": best_recall["precision"],
            "recall": best_recall["recall"],
            "f1": best_recall["f1"],
        }
    else:
        recommended = best_f1["threshold"]
        summary = {
            "recommended": recommended,
            "basis": "best_f1",
            "precision": best_f1["precision"],
            "recall": best_f1["recall"],
            "f1": best_f1["f1"],
        }

    return recommended, summary


def resolve_threshold(recommended: Optional[float]) -> Tuple[float, Dict[str, float]]:
    """
    Resolve threshold from env vars or recommendation.
    Returns (threshold_used, thresholds_dict).
    """
    strict_thr = float(os.getenv("STRICT_THRESHOLD", "0.8"))
    medium_thr = float(os.getenv("MEDIUM_THRESHOLD", "0.6"))
    low_thr = float(os.getenv("LOW_THRESHOLD", "0.3"))
    default_thr = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))

    override = os.getenv("DETECTION_THRESHOLD")
    if override is not None:
        threshold = float(override)
        return threshold, {
            "strict": strict_thr,
            "medium": medium_thr,
            "low": low_thr,
            "default": default_thr,
        }

    mode = os.getenv("DETECTION_THRESHOLD_MODE", "default").strip().lower()
    if mode == "strict":
        threshold = strict_thr
    elif mode == "medium":
        threshold = medium_thr
    elif mode == "low":
        threshold = low_thr
    elif mode == "auto" and recommended is not None:
        threshold = recommended
    else:
        threshold = default_thr

    return threshold, {
        "strict": strict_thr,
        "medium": medium_thr,
        "low": low_thr,
        "default": default_thr,
    }


def postprocess_prediction(
    prob: float,
    waveform: np.ndarray,
    threshold_used: float,
    strict_threshold: float,
    energy_min: float,
    variance_min: float,
    flatness_min: float,
    weak_min_prob: float,
    weak_energy_min: float,
    spike_threshold: float,
) -> Dict[str, object]:
    """
    Apply thresholding + signal checks to produce a final label.
    Returns a dict with prediction, confidence, decision_reason, alert,
    signal_energy, and signal_variance.
    """
    energy = compute_signal_energy(waveform)
    variance = compute_signal_variance(waveform)
    peak_to_peak = compute_peak_to_peak(waveform)
    spike = detect_spikes(waveform, spike_threshold)

    prediction = "Earthquake" if prob >= threshold_used else "Noise"
    decision_reason = "high_confidence" if prob >= strict_threshold else "below_threshold"

    if prediction == "Earthquake":
        if (energy < energy_min) or (variance < variance_min) or (peak_to_peak < flatness_min):
            prediction = "Noise"
            decision_reason = "low_energy_filtered"
    else:
        if (weak_min_prob <= prob < threshold_used) and (energy >= weak_energy_min or spike):
            prediction = "Possible Earthquake"
            decision_reason = "possible_event_low_confidence"

    if prediction == "Noise":
        confidence = round((1.0 - prob) * 100.0, 2)
        alert = False
    else:
        confidence = round(prob * 100.0, 2)
        alert = True

    return {
        "prediction": prediction,
        "confidence": confidence,
        "decision_reason": decision_reason,
        "alert": alert,
        "signal_energy": round(energy, 6),
        "signal_variance": round(variance, 6),
    }


def validate_phase(p_sec: Optional[float], s_sec: Optional[float],
                   min_gap: float = 0.5, max_gap: float = 40.0) -> Tuple[str, Optional[float]]:
    """Validate P/S ordering and gap bounds."""
    if p_sec is None or s_sec is None:
        return "missing", None
    gap = s_sec - p_sec
    if gap <= 0 or gap < min_gap or gap > max_gap:
        return "invalid", None
    return "valid", round(gap, 3)
