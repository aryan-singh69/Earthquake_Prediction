"""
FastAPI inference utilities.
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from inference.predictor import Predictor


_predictor = None


def get_predictor() -> Predictor:
    """Create or return a cached Predictor instance."""
    global _predictor
    if _predictor is None:
        model_path = os.getenv(
            "MODEL_PATH",
            os.path.join(PROJECT_ROOT, "models", "checkpoints", "multitask_model.pth"),
        )
        print(f"Using model path: {model_path}")
        _predictor = Predictor(model_path=model_path)
    return _predictor


def run_inference(waveform: np.ndarray) -> Dict[str, object]:
    """Run model inference + postprocess and return response dict."""
    predictor = get_predictor()
    return predictor.predict(waveform)
