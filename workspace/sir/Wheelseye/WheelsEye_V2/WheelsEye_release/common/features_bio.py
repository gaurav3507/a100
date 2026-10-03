"""Biometric features from the Empatica E4-class wristband + Checkme O2 Max
CSVs (tasks.md Task 2.2). Deliberately simple, standard formulas -- no
CWT/fractal HRV features (FatigueNet-specific, out of scope here) and no
full cvxEDA solver for tonic/phasic decomposition.

IBI coverage is sparse within a session (wristband loses PPG lock for long
stretches -- see common/windowing.py docstring), so HRV features are NaN for
any window with fewer than 2 detected beats. That must be handled downstream
as a missing value (e.g. per-subject median imputation), not treated as 0.
"""
from __future__ import annotations

import numpy as np

from common.windowing import Signal

EDA_TONIC_WINDOW_SAMPLES = 8  # ~2s moving average at EDA's 4Hz rate


def hr_features(hr: Signal) -> dict[str, float]:
    if len(hr.times) == 0:
        return {"hr_mean": float("nan"), "hr_std": float("nan")}
    v = hr.values[:, 0]
    return {"hr_mean": float(np.mean(v)), "hr_std": float(np.std(v))}


def hrv_features(ibi: Signal) -> dict[str, float]:
    """RMSSD and SDNN from inter-beat intervals (seconds) falling in this window."""
    if len(ibi.times) < 2:
        return {"hrv_rmssd": float("nan"), "hrv_sdnn": float("nan")}
    v = ibi.values[:, 0]
    diffs = np.diff(v)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    sdnn = float(np.std(v))
    return {"hrv_rmssd": rmssd, "hrv_sdnn": sdnn}


def eda_features(eda: Signal) -> dict[str, float]:
    """Tonic (slow-moving baseline) / phasic (raw - tonic) decomposition via
    a simple moving average -- sufficient for windowed summary stats, not a
    full cvxEDA solver."""
    if len(eda.times) == 0:
        return {
            "eda_tonic_mean": float("nan"), "eda_phasic_mean": float("nan"),
            "eda_phasic_std": float("nan"), "eda_phasic_max": float("nan"),
        }
    v = eda.values[:, 0]
    k = min(EDA_TONIC_WINDOW_SAMPLES, len(v))
    kernel = np.ones(k) / k
    tonic = np.convolve(v, kernel, mode="same")
    phasic = v - tonic
    return {
        "eda_tonic_mean": float(np.mean(tonic)),
        "eda_phasic_mean": float(np.mean(phasic)),
        "eda_phasic_std": float(np.std(phasic)),
        "eda_phasic_max": float(np.max(phasic)),
    }


def spo2_features(o2m: Signal) -> dict[str, float]:
    """O2M columns: spo2, pulse_rate, motion. A few sessions have dropped
    readings encoded as NaN (see common/loaders.py) -- use NaN-aware
    reductions, and fall back to all-NaN if a window's readings are all
    dropped rather than raising on an empty nan* reduction.

    v2 (plan §4): the motion column -- the oximeter's own body-movement
    channel, loaded and discarded in v1 -- is now summarized too."""
    empty = {
        "spo2_mean": float("nan"), "spo2_std": float("nan"), "spo2_min": float("nan"),
        "pulse_rate_mean": float("nan"), "pulse_rate_std": float("nan"),
        "motion_mean": float("nan"), "motion_std": float("nan"),
    }
    if len(o2m.times) == 0:
        return empty
    spo2 = o2m.values[:, 0]
    pulse = o2m.values[:, 1]
    motion = o2m.values[:, 2]
    if np.all(np.isnan(spo2)) and np.all(np.isnan(pulse)):
        out = dict(empty)
    else:
        with np.errstate(invalid="ignore"):
            out = {
                "spo2_mean": float(np.nanmean(spo2)),
                "spo2_std": float(np.nanstd(spo2)),
                "spo2_min": float(np.nanmin(spo2)),
                "pulse_rate_mean": float(np.nanmean(pulse)),
                "pulse_rate_std": float(np.nanstd(pulse)),
            }
    if not np.all(np.isnan(motion)):
        with np.errstate(invalid="ignore"):
            out["motion_mean"] = float(np.nanmean(motion))
            out["motion_std"] = float(np.nanstd(motion))
    else:
        out["motion_mean"] = float("nan")
        out["motion_std"] = float("nan")
    return out


def temp_features(temp: Signal) -> dict[str, float]:
    """Mean and slope (deg C / s) -- captures the warming trend at higher
    drowsiness reported in the UL-DD technical validation."""
    if len(temp.times) == 0:
        return {"temp_mean": float("nan"), "temp_slope": float("nan")}
    v = temp.values[:, 0]
    if len(temp.times) < 2:
        slope = 0.0
    else:
        slope = float(np.polyfit(temp.times, v, 1)[0])
    return {"temp_mean": float(np.mean(v)), "temp_slope": slope}


def extract_bio_features(
    hr: Signal, ibi: Signal, eda: Signal, o2m: Signal, temp: Signal
) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(hr_features(hr))
    out.update(hrv_features(ibi))
    out.update(eda_features(eda))
    out.update(spo2_features(o2m))
    out.update(temp_features(temp))
    return out
