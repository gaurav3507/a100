"""Per-window feature extraction for MePhy, one function per modality node
(tasks.md 3.1: ECG/EDA/EMG/blink as four separate GNN nodes, matching
FatigueNet). Kept to the same standard, simple formulas as
common/features_bio.py -- no CWT/fractal HRV, no cvxEDA -- for the same
reason: this is a reimplementation sanity check, not a novel-feature paper.
"""
from __future__ import annotations

import numpy as np

from mephy_repro.windowing import Signal

EDA_TONIC_WINDOW_SAMPLES = 200  # ~0.2s moving average at EDA's 1000Hz rate


def ecg_features(ecg: Signal) -> dict[str, float]:
    """hr column: beats/min, already computed by the Polar H10. rr_ms:
    inter-beat interval in milliseconds, same signal common/features_bio.py
    calls IBI, just in ms instead of seconds here."""
    if len(ecg.times) == 0:
        return {"hr_mean": float("nan"), "hr_std": float("nan"),
                "hrv_rmssd_ms": float("nan"), "hrv_sdnn_ms": float("nan")}
    hr = ecg.values[:, 0]
    rr_ms = ecg.values[:, 1]
    out = {"hr_mean": float(np.mean(hr)), "hr_std": float(np.std(hr))}
    if len(rr_ms) < 2:
        out["hrv_rmssd_ms"] = float("nan")
        out["hrv_sdnn_ms"] = float("nan")
    else:
        diffs = np.diff(rr_ms)
        out["hrv_rmssd_ms"] = float(np.sqrt(np.mean(diffs ** 2)))
        out["hrv_sdnn_ms"] = float(np.std(rr_ms))
    return out


def eda_features(eda: Signal) -> dict[str, float]:
    if len(eda.times) == 0:
        return {"eda_tonic_mean": float("nan"), "eda_phasic_mean": float("nan"),
                "eda_phasic_std": float("nan"), "eda_phasic_max": float("nan")}
    v = eda.values[:, 0]
    k = min(EDA_TONIC_WINDOW_SAMPLES, len(v))
    tonic = np.convolve(v, np.ones(k) / k, mode="same")
    phasic = v - tonic
    return {
        "eda_tonic_mean": float(np.mean(tonic)),
        "eda_phasic_mean": float(np.mean(phasic)),
        "eda_phasic_std": float(np.std(phasic)),
        "eda_phasic_max": float(np.max(phasic)),
    }


def emg_features(emg: Signal) -> dict[str, float]:
    """Standard time-domain EMG features used for muscle-fatigue detection:
    RMS, MAV (mean absolute value), ZC (zero-crossing rate around the
    signal's own mean, since raw ADC counts aren't zero-centered), and
    waveform length (cumulative sample-to-sample variation)."""
    if len(emg.times) == 0:
        return {"emg_rms": float("nan"), "emg_mav": float("nan"),
                "emg_zero_crossing_rate": float("nan"), "emg_waveform_length": float("nan")}
    v = emg.values[:, 0]
    centered = v - np.mean(v)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    mav = float(np.mean(np.abs(centered)))
    zc = int(np.sum(np.diff(np.sign(centered)) != 0))
    duration = emg.times[-1] - emg.times[0] if len(emg.times) > 1 else 1.0
    wl = float(np.sum(np.abs(np.diff(v))))
    return {
        "emg_rms": rms,
        "emg_mav": mav,
        "emg_zero_crossing_rate": zc / duration if duration > 0 else float("nan"),
        "emg_waveform_length": wl,
    }


def blink_features(blink: Signal) -> dict[str, float]:
    """eye_closed is already a boolean per-frame signal (1=closed), so
    PERCLOS here is exactly its mean; blink count = number of closed runs."""
    if len(blink.times) == 0:
        return {"perclos": float("nan"), "blink_count": float("nan"), "blink_rate_per_min": float("nan")}
    closed = blink.values[:, 0] > 0.5
    perclos = float(np.mean(closed))
    edges = np.diff(closed.astype(int))
    count = int(np.sum(edges == 1)) + (1 if len(closed) and closed[0] else 0)
    duration_min = (blink.times[-1] - blink.times[0]) / 60.0 if len(blink.times) > 1 else float("nan")
    return {
        "perclos": perclos,
        "blink_count": float(count),
        "blink_rate_per_min": count / duration_min if duration_min else float("nan"),
    }


def extract_mephy_features(ecg: Signal, eda: Signal, emg: Signal, blink: Signal) -> dict[str, float]:
    out = {}
    out.update(ecg_features(ecg))
    out.update(eda_features(eda))
    out.update(emg_features(emg))
    out.update(blink_features(blink))
    return out
