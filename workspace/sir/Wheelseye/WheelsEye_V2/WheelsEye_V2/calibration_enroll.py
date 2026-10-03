"""Strict enrollment calibration: per-driver baseline from the first k windows
of the Awake session only (default k=48 = 4 minutes at 5 s step = one KSS
interval), applied identically to every window of that driver.

Why this exists: V2's `apply_alert_baseline` standardizes against the WHOLE
Awake session, so the enrollment data and the scored data are the same drive
(label-free, but transductive). This variant is the deployment-faithful
version: what a device knows after a short enrollment, nothing more.

Guarantees (tested): statistics use features only, never labels; only the
first k Awake windows (by window_id) contribute; the transform is applied to
Awake and Drowsy rows alike; an `is_enrollment` mask is returned so the
caller can choose to score or exclude those k windows.

Wire-in (scripts/run_experiment.py): where `apply_alert_baseline(df, feats)`
is called, use
    from common.calibration_enroll import apply_enrollment_baseline
    df, enroll_mask = apply_enrollment_baseline(df, feats, n_enroll=args.n_enroll)
and add `--n-enroll` (int, default 0 = whole-session behaviour) to argparse,
appending f"_enroll{args.n_enroll}" to the result-file name when set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def enrollment_stats(df: pd.DataFrame, features: list[str], n_enroll: int = 48,
                     awake_session: str = "A") -> dict[str, tuple[pd.Series, pd.Series]]:
    """Per-subject (mean, std) from the first n_enroll Awake windows. Std of 0
    or NaN falls back to 1.0 so the transform never divides by zero."""
    if n_enroll <= 1:
        raise ValueError("n_enroll must be >= 2 to estimate a standard deviation")
    stats = {}
    awake = df[df["session"] == awake_session]
    for subj, g in awake.groupby("subject"):
        head = g.sort_values("window_id").head(n_enroll)
        mean = head[features].mean()
        std = head[features].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        stats[subj] = (mean.fillna(0.0), std)
    missing = sorted(set(df["subject"]) - set(stats))
    if missing:
        raise ValueError(f"Subjects with no Awake session to enroll from: {missing}")
    return stats


def apply_enrollment_baseline(df: pd.DataFrame, features: list[str], n_enroll: int = 48,
                              awake_session: str = "A", mean_only: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (calibrated df, is_enrollment mask aligned to df rows)."""
    stats = enrollment_stats(df, features, n_enroll, awake_session)
    out = df.copy()
    out[features] = out[features].astype("float64")   # calibrated deltas are floats regardless of storage dtype
    is_enroll = np.zeros(len(df), dtype=bool)
    for subj, (mean, std) in stats.items():
        if mean_only:
            std = pd.Series(1.0, index=features)   # offset removal only; keep a common scale across drivers
        m = (out["subject"] == subj).to_numpy()
        cal = ((out.loc[m, features] - mean) / std).to_numpy()
        out.loc[m, features] = cal
        for j, c in enumerate(features):          # mirror columns for *_cal consumers
            out.loc[m, f"{c}_cal"] = cal[:, j]
        # mark the k enrollment windows (Awake, lowest window_ids)
        sub_awake = out[m & (out["session"] == awake_session).to_numpy()]
        enroll_idx = sub_awake.sort_values("window_id").index[:n_enroll]
        is_enroll[out.index.get_indexer(enroll_idx)] = True
    return out, is_enroll
