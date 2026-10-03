"""Spectral cardiac features from the 64 Hz BVP stream (plan §4 "Bio --
spectral HRV": the LF/HF autonomic markers the capstone proposal's Step 4
promised, computed the honest way).

Why not Welch over the 10 s window directly: LF is 0.04-0.15 Hz -- a 10 s
slice cannot resolve a 25 s-period oscillation, so "LF power over 10 s"
would be pseudo-science a sharp panelist catches. The correct method is
pulse-rate variability (PRV): detect systolic peaks in a longer trailing
BVP context, build the inter-beat-interval series, resample it uniformly,
and take Welch band power. common/windowing.py's `context_slices` parameter
feeds this extractor a trailing BVP_CONTEXT_SECONDS slice ending at the
window's end (only past signal -- causal, deployment-realistic).

Honest limitation, stated where it's made: the 120 s default context gives
~4.8 cycles of the lowest LF frequency; clinical short-term HRV standards
prefer 2-5 min. 120 s is the accepted floor for short-term LF/HF trends and
matches our 10 s / 5 s window cadence; the report should say so.

Beat detection: scipy find_peaks with a 0.33 s minimum separation (caps
plausible HR at ~180 bpm) and prominence at half the context's BVP standard
deviation -- scale-relative, so it survives the E4's per-session amplitude
differences without a tuned absolute threshold.
"""
from __future__ import annotations

import numpy as np

from common.windowing import Signal

BVP_CONTEXT_SECONDS = 120.0  # trailing context handed to this extractor
BVP_RATE_HZ = 64.0
MIN_CONTEXT_SECONDS = 60.0  # below this, LF is unresolvable -> all-NaN
MIN_BEATS = 30  # fewer detected beats than this -> unreliable spectrum -> NaN
RESAMPLE_HZ = 4.0  # uniform IBI-series rate for Welch (standard for PRV)
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)

CARDIAC_SPECTRAL_FEATURES = (
    "bvp_lf_power", "bvp_hf_power", "bvp_lf_hf_ratio", "bvp_total_power",
    "bvp_beat_rate", "bvp_amp_std",
)

_NAN = {k: float("nan") for k in CARDIAC_SPECTRAL_FEATURES}


def detect_beats(times: np.ndarray, bvp: np.ndarray) -> np.ndarray:
    """Systolic peak times (seconds) from a raw BVP trace."""
    from scipy.signal import find_peaks

    prominence = 0.5 * float(np.std(bvp))
    if prominence <= 0:
        return np.empty(0)
    peaks, _ = find_peaks(bvp, distance=int(0.33 * BVP_RATE_HZ), prominence=prominence)
    return times[peaks]


def extract_bvp_spectral_features(bvp: Signal) -> dict[str, float]:
    """PRV spectral features from a trailing BVP context slice.

    bvp_lf_power / bvp_hf_power / bvp_total_power: band power of the
    detrended, uniformly-resampled IBI series (s^2, Welch). bvp_lf_hf_ratio:
    the sympathovagal-balance marker. bvp_beat_rate: detected beats/min --
    doubles as a cross-check against the E4's own HR channel. bvp_amp_std:
    raw BVP amplitude variability over the context.
    """
    if len(bvp.times) < 2 or (bvp.times[-1] - bvp.times[0]) < MIN_CONTEXT_SECONDS:
        return dict(_NAN)

    signal = bvp.values[:, 0]
    span_min = (bvp.times[-1] - bvp.times[0]) / 60.0
    beat_times = detect_beats(bvp.times, signal)

    out = dict(_NAN)
    out["bvp_amp_std"] = float(np.std(signal))
    if len(beat_times) < MIN_BEATS:
        return out
    out["bvp_beat_rate"] = float(len(beat_times) / span_min)

    ibi = np.diff(beat_times)  # seconds, one per beat pair
    ibi_times = beat_times[1:]
    # uniform resample of the IBI series over the beat-covered span
    grid = np.arange(ibi_times[0], ibi_times[-1], 1.0 / RESAMPLE_HZ)
    if len(grid) < int(RESAMPLE_HZ * MIN_CONTEXT_SECONDS / 2):
        return out
    ibi_uniform = np.interp(grid, ibi_times, ibi)
    ibi_uniform = ibi_uniform - np.mean(ibi_uniform)  # detrend (mean removal)

    from scipy.signal import welch

    nperseg = min(len(ibi_uniform), 256)
    freqs, psd = welch(ibi_uniform, fs=RESAMPLE_HZ, nperseg=nperseg)

    def band_power(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            return float("nan")
        return float(np.trapezoid(psd[mask], freqs[mask]))

    lf = band_power(*LF_BAND)
    hf = band_power(*HF_BAND)
    out["bvp_lf_power"] = lf
    out["bvp_hf_power"] = hf
    out["bvp_lf_hf_ratio"] = lf / hf if (hf and not np.isnan(hf) and hf > 0) else float("nan")
    out["bvp_total_power"] = band_power(LF_BAND[0], HF_BAND[1])
    return out
