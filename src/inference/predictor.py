"""
Predictor wrapper that handles model inference and post-processing.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import numpy as np
import torch

from .postprocess import (
    normalize_waveform,
    ensure_channel_first,
    load_threshold_analysis,
    resolve_threshold,
    postprocess_prediction,
    validate_phase,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from models import MultiTaskCNN

# Normalization constants (same as training)
P_S_MAX = 6000.0
MAG_MAX = 9.0
LAT_MAX = 90.0
LON_MAX = 180.0
DEPTH_MAX = 700.0


def normalize_shape(data: np.ndarray) -> np.ndarray:
    """Any input shape -> (6000, 3)."""
    if data.ndim == 1:
        data = np.stack([data, data, data], axis=1)
    if data.shape == (3, 6000):
        data = data.T
    if data.shape[0] == 3:
        data = data.T
    return data


class Predictor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        metrics_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.model_path = model_path or os.getenv(
            "MODEL_PATH",
            os.path.join(PROJECT_ROOT, "models", "multitask_model.pth"),
        )
        self.metrics_dir = metrics_dir or os.getenv(
            "METRICS_DIR",
            os.path.join(PROJECT_ROOT, "models", "metrics"),
        )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MultiTaskCNN().to(self.device)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device, weights_only=True)
            )
            print(f" MultiTask Model loaded: {self.model_path}")
        else:
            print(f" Warning: model not found -> {self.model_path}")

        self.model.eval()

        analysis_path = os.path.join(self.metrics_dir, "threshold_analysis.csv")
        recommended, summary = load_threshold_analysis(analysis_path)
        self.recommended_threshold = recommended
        if summary:
            print(
                " Threshold recommendation -> "
                f"{summary['recommended']} ({summary['basis']}, "
                f"prec={summary['precision']:.3f}, "
                f"rec={summary['recall']:.3f}, "
                f"f1={summary['f1']:.3f})"
            )
        else:
            print(" Threshold recommendation unavailable (missing analysis file)")

    def predict(self, data: np.ndarray) -> Dict[str, object]:
        data = normalize_shape(data)  # (6000, 3)
        waveform = ensure_channel_first(data.T)  # (3, 6000)
        waveform_norm = normalize_waveform(waveform)

        tensor = torch.tensor(waveform_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)

        prob = torch.sigmoid(outputs["detection"]).item()
        phase = outputs["phase"].squeeze().cpu().numpy()
        magnitude = float(outputs["magnitude"].squeeze().cpu().numpy()) * MAG_MAX
        loc = outputs["location"].squeeze().cpu().numpy()

        # Denormalize
        p_sec = round(float(phase[0]) * P_S_MAX / 100.0, 2)
        s_sec = round(float(phase[1]) * P_S_MAX / 100.0, 2)
        magnitude = round(max(0.0, min(9.0, magnitude)), 2)
        latitude = round(max(-90.0, min(90.0, float(loc[0]) * LAT_MAX)), 4)
        longitude = round(max(-180.0, min(180.0, float(loc[1]) * LON_MAX)), 4)
        depth = round(max(0.0, min(700.0, float(loc[2]) * DEPTH_MAX)), 2)

        threshold_used, thresholds = resolve_threshold(self.recommended_threshold)

        # Postprocess thresholds + signal checks
        energy_min = float(os.getenv("MIN_SIGNAL_ENERGY", "0.2"))
        variance_min = float(os.getenv("MIN_SIGNAL_VARIANCE", "0.2"))
        flatness_min = float(os.getenv("MIN_SIGNAL_FLATNESS", "0.3"))
        weak_min_prob = float(os.getenv("WEAK_EVENT_MIN_PROB", "0.2"))
        weak_energy_min = float(os.getenv("WEAK_ENERGY_MIN", "0.8"))
        spike_threshold = float(os.getenv("SPIKE_THRESHOLD", "3.0"))

        post = postprocess_prediction(
            prob=prob,
            waveform=waveform_norm,
            threshold_used=threshold_used,
            strict_threshold=thresholds["strict"],
            energy_min=energy_min,
            variance_min=variance_min,
            flatness_min=flatness_min,
            weak_min_prob=weak_min_prob,
            weak_energy_min=weak_energy_min,
            spike_threshold=spike_threshold,
        )

        prediction = post["prediction"]
        decision_reason = post["decision_reason"]

        # Phase sanity
        phase_status, s_p_gap = validate_phase(p_sec, s_sec)
        if prediction != "Noise" and phase_status == "invalid":
            if decision_reason not in {"low_energy_filtered", "possible_event_low_confidence"}:
                decision_reason = "invalid_phase"
            p_sec = None
            s_sec = None
            s_p_gap = None

        if prediction == "Noise":
            p_sec = None
            s_sec = None
            s_p_gap = None
            magnitude = None
            latitude = None
            longitude = None
            depth = None

        return {
            "prediction": prediction,
            "confidence": post["confidence"],
            "threshold_used": round(float(threshold_used), 3),
            "p_arrival_sec": p_sec,
            "s_arrival_sec": s_sec,
            "s_p_gap_sec": s_p_gap,
            "magnitude": magnitude,
            "alert": post["alert"],
            "decision_reason": decision_reason,
            "signal_energy": post["signal_energy"],
            "signal_variance": post["signal_variance"],
            "latitude": latitude,
            "longitude": longitude,
            "depth": depth,
            "phase_status": phase_status,
            "location_status": "experimental",
            # Backward-compatible keys
            "p_arrival": p_sec,
            "s_arrival": s_sec,
        }
