"""Helpers for FastAPI request validation."""

from __future__ import annotations

from typing import Tuple

import numpy as np


ALLOWED_SHAPES = {(3, 6000), (6000, 3)}


def validate_waveform_shape(data: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Validate input waveform shape and return (3, 6000) array.
    """
    if data.ndim != 2:
        raise ValueError("Waveform must be 2D with shape (3, 6000) or (6000, 3)")
    if data.shape not in ALLOWED_SHAPES:
        raise ValueError("Invalid shape; expected (3, 6000) or (6000, 3)")

    if data.shape == (6000, 3):
        data = data.T

    return data, data.shape
