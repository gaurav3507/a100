"""Session-context features (plan §4 "Context"): time-on-task and each key
feature's drift from the session's own opening minutes.

Drowsiness trends within a 40-minute session, and "how far has this signal
moved since this drive started" is subject-relative information available at
inference time with zero leakage: five minutes into a real drive, the system
has genuinely observed that drive's start. (Distinct from the Tier-3 alert
-baseline calibration in common/calibration.py, which uses a separate
enrollment session -- drift uses the CURRENT drive's own opening minutes.)

Applied to the assembled feature table by scripts/build_feature_table.py,
after per-window extraction and before fold assignment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE_SECONDS = 300.0  # "opening minutes" = first 5 min of the session

# Key features that get a drift twin -- slow physiological/behavioral levels
# where the within-drive trend is the signal (not noisy count features).
DRIFT_FEATURES = (
    "hr_mean", "eda_tonic_mean", "temp_mean", "spo2_mean", "pulse_rate_mean",
    "ear_mean", "perclos", "mar_mean",
    "grip_left_mean", "grip_right_mean", "grip_asymmetry",
    "head_pitch_mean",
)

CONTEXT_FEATURES = ("minutes_elapsed",) + tuple(f"{c}_drift" for c in DRIFT_FEATURES)


def add_context_features(
    df: pd.DataFrame, drift_features: tuple[str, ...] = DRIFT_FEATURES
) -> pd.DataFrame:
    """Return a copy of the feature table with `minutes_elapsed` and one
    `{feature}_drift` column per drift feature (value minus the mean of that
    feature over the same session's windows starting inside the first
    BASELINE_SECONDS). NaN-aware: a session whose opening windows are all
    NaN for a feature yields NaN drift for that feature (missing, not 0)."""
    out = df.copy()
    out["minutes_elapsed"] = (out["t_start"] + out["t_end"]) / 2.0 / 60.0

    opening = out[out["t_start"] < BASELINE_SECONDS]
    base = opening.groupby(["subject", "session"])[list(drift_features)].mean()

    idx = pd.MultiIndex.from_frame(out[["subject", "session"]])
    base_rows = base.reindex(idx).to_numpy()
    drifted = out[list(drift_features)].to_numpy() - base_rows
    for j, c in enumerate(drift_features):
        out[f"{c}_drift"] = drifted[:, j]
    return out
