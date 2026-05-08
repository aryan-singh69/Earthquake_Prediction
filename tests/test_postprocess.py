import numpy as np
import pytest

from src.inference.postprocess import (
    compute_signal_energy,
    compute_signal_variance,
    ensure_channel_first,
    load_threshold_analysis,
    normalize_waveform,
    postprocess_prediction,
    resolve_threshold,
    validate_phase,
)


def test_energy_and_variance_basic():
    wave = np.ones((3, 6000), dtype=float)
    energy = compute_signal_energy(wave)
    variance = compute_signal_variance(wave)
    assert energy == 1.0
    assert variance == 0.0


def test_normalize_waveform_channelwise():
    wave = np.array([[1.0, 2.0, 3.0], [10.0, 11.0, 12.0], [0.0, 0.0, 0.0]])
    normalized = normalize_waveform(wave)
    assert np.allclose(normalized.mean(axis=1), 0.0, atol=1e-7)
    assert np.allclose(normalized.std(axis=1)[:2], 1.0, atol=1e-7)
    assert np.allclose(normalized[2], 0.0)


def test_ensure_channel_first_accepts_or_transposes():
    channel_first = np.ones((3, 4))
    channel_last = np.ones((4, 3))
    assert ensure_channel_first(channel_first).shape == (3, 4)
    assert ensure_channel_first(channel_last).shape == (3, 4)


def test_ensure_channel_first_raises_for_invalid_shapes():
    with pytest.raises(ValueError, match="2D"):
        ensure_channel_first(np.ones(5))
    with pytest.raises(ValueError, match="3 channels"):
        ensure_channel_first(np.ones((4, 4)))


def test_load_threshold_analysis_missing_file(tmp_path):
    recommended, summary = load_threshold_analysis(str(tmp_path / "missing.csv"))
    assert recommended is None
    assert summary == {}


def test_load_threshold_analysis_prefers_high_recall(monkeypatch, tmp_path):
    csv_path = tmp_path / "threshold_analysis.csv"
    csv_path.write_text(
        "threshold,accuracy,precision,recall,f1\n"
        "0.4,0.9,0.72,0.91,0.80\n"
        "0.5,0.92,0.75,0.95,0.83\n"
        "0.6,0.94,0.90,0.89,0.89\n"
    )
    monkeypatch.setenv("PRECISION_ACCEPTABLE_MIN", "0.7")
    monkeypatch.setenv("RECALL_PREFERRED_MIN", "0.9")

    recommended, summary = load_threshold_analysis(str(csv_path))

    assert recommended == 0.5
    assert summary["basis"] == "high_recall"
    assert summary["recall"] == 0.95


def test_load_threshold_analysis_falls_back_to_best_f1(monkeypatch, tmp_path):
    csv_path = tmp_path / "threshold_analysis.csv"
    csv_path.write_text(
        "threshold,accuracy,precision,recall,f1\n"
        "0.3,0.85,0.65,0.80,0.72\n"
        "0.5,0.90,0.78,0.88,0.84\n"
        "0.6,0.91,0.81,0.87,0.86\n"
        "bad,row,that,should,skip\n"
    )
    monkeypatch.setenv("PRECISION_ACCEPTABLE_MIN", "0.9")
    monkeypatch.setenv("RECALL_PREFERRED_MIN", "0.95")

    recommended, summary = load_threshold_analysis(str(csv_path))

    assert recommended == 0.6
    assert summary["basis"] == "best_f1"
    assert summary["f1"] == 0.86


def test_resolve_threshold_override(monkeypatch):
    monkeypatch.setenv("DETECTION_THRESHOLD", "0.73")
    threshold, thresholds = resolve_threshold(recommended=0.41)
    assert threshold == 0.73
    assert thresholds["default"] == 0.5


@pytest.mark.parametrize(
    ("mode", "recommended", "expected"),
    [
        ("strict", 0.42, 0.8),
        ("medium", 0.42, 0.6),
        ("low", 0.42, 0.3),
        ("auto", 0.42, 0.42),
        ("unknown", 0.42, 0.5),
        ("auto", None, 0.5),
    ],
)
def test_resolve_threshold_modes(monkeypatch, mode, recommended, expected):
    monkeypatch.delenv("DETECTION_THRESHOLD", raising=False)
    monkeypatch.setenv("DETECTION_THRESHOLD_MODE", mode)
    threshold, _ = resolve_threshold(recommended=recommended)
    assert threshold == expected


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


def test_high_confidence_can_be_filtered_to_noise():
    wave = np.full((3, 6000), 0.01, dtype=float)
    result = postprocess_prediction(
        prob=0.92,
        waveform=wave,
        threshold_used=0.6,
        strict_threshold=0.8,
        energy_min=0.2,
        variance_min=0.01,
        flatness_min=0.1,
        weak_min_prob=0.2,
        weak_energy_min=0.8,
        spike_threshold=3.0,
    )
    assert result["prediction"] == "Noise"
    assert result["decision_reason"] == "low_energy_filtered"
    assert result["alert"] is False
    assert result["confidence"] == 8.0


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


def test_below_weak_threshold_remains_noise():
    wave = np.ones((3, 6000), dtype=float)
    result = postprocess_prediction(
        prob=0.1,
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
    assert result["prediction"] == "Noise"
    assert result["decision_reason"] == "below_threshold"
    assert result["alert"] is False
    assert result["confidence"] == 90.0


def test_validate_phase_states():
    assert validate_phase(None, 2.0) == ("missing", None)
    assert validate_phase(3.0, 2.0) == ("invalid", None)
    assert validate_phase(1.0, 1.1, min_gap=0.5) == ("invalid", None)
    assert validate_phase(1.0, 60.0, max_gap=40.0) == ("invalid", None)
    assert validate_phase(1.0, 2.234) == ("valid", 1.234)
