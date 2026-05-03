import numpy as np

from src.inference.postprocess import (
    compute_signal_energy,
    compute_signal_variance,
    postprocess_prediction,
)


def test_energy_and_variance_basic():
    wave = np.ones((3, 6000), dtype=float)
    energy = compute_signal_energy(wave)
    variance = compute_signal_variance(wave)
    assert energy == 1.0
    assert variance == 0.0


def test_threshold_logic_earthquake():
    wave = np.ones((3, 6000), dtype=float)
    result = postprocess_prediction(
        prob=0.65,
        waveform=wave,
        threshold_used=0.6,
        strict_threshold=0.8,
        energy_min=0.1,
        variance_min=0.0,
        flatness_min=0.0,
        weak_min_prob=0.2,
        weak_energy_min=0.8,
        spike_threshold=3.0,
    )
    assert result["prediction"] == "Earthquake"
    assert result["decision_reason"] in {"high_confidence", "below_threshold"}


def test_possible_event_classification():
    wave = np.zeros((3, 6000), dtype=float)
    wave[0, 3000] = 5.0
    result = postprocess_prediction(
        prob=0.25,
        waveform=wave,
        threshold_used=0.6,
        strict_threshold=0.8,
        energy_min=0.1,
        variance_min=0.1,
        flatness_min=0.1,
        weak_min_prob=0.2,
        weak_energy_min=0.8,
        spike_threshold=3.0,
    )
    assert result["prediction"] == "Possible Earthquake"
    assert result["decision_reason"] == "possible_event_low_confidence"
