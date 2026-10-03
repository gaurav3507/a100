"""Wrist-accelerometer features from the Empatica E4 ACC stream (plan §4
"Bio -- wrist movement": 32 Hz, 3-axis, verified 35/35 files, unused in v1).

Rationale: a disengaging driver's wrist goes still -- movement magnitude and
its short-term variation ("jerk") complement grip pressure as behavioral
disengagement cues. Values are raw E4 units (1/64 g per count); features are
either unit-free (fractions) or reported in raw units consistently, so the
downstream per-fold standardization handles scale.
"""
from __future__ import annotations

import numpy as np

from common.windowing import Signal

# |Δmagnitude| per sample below this raw-unit threshold counts as "still".
# 3 counts = 3/64 g ~= 0.047 g between consecutive 32 Hz samples -- small
# deliberate movements exceed this; sensor noise on a resting wrist doesn't.
ACC_STILL_JERK_THRESHOLD = 3.0

ACC_FEATURES = ("acc_mag_mean", "acc_mag_std", "acc_jerk_mean", "acc_still_fraction")

_NAN = {k: float("nan") for k in ACC_FEATURES}


def extract_acc_features(acc: Signal) -> dict[str, float]:
    if len(acc.times) < 2:
        return dict(_NAN)
    mag = np.sqrt(np.sum(acc.values[:, :3] ** 2, axis=1))
    jerk = np.abs(np.diff(mag))
    return {
        "acc_mag_mean": float(np.mean(mag)),
        "acc_mag_std": float(np.std(mag)),
        "acc_jerk_mean": float(np.mean(jerk)),
        "acc_still_fraction": float(np.mean(jerk < ACC_STILL_JERK_THRESHOLD)),
    }
