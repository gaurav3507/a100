"""Per-driver alert-baseline calibration (plan §5) -- the direct answer to
the measured root failure: per-subject physiological offsets dominate the
drowsiness effect (UL-DD Table 6: HR between-subject variance 54.1 vs
residual ~6.6; v1's champion predicted all-High for subjects O and S).

Mechanism: z-score each subject's features using statistics computed from
that subject's OWN Awake ("A") session only. Deployment equivalent: a
one-time enrollment drive while alert. This is personalization, not
leakage -- Drowsy-session windows (where every High-class test label lives)
never contribute to the statistics, which `alert_baseline_stats` enforces
structurally by construction and by assertion.

Precedent: the UL-DD authors' own physiological validation standardizes
per subject to make Figure 5's box plots comparable across people.
"""
from __future__ import annotations

import pandas as pd

ALERT_SESSION = "A"


def alert_baseline_stats(df: pd.DataFrame, features: list[str]) -> dict[str, pd.DataFrame]:
    """Per-subject mean/std of `features`, computed from Awake-session rows
    only. Returns {'mean': DataFrame, 'std': DataFrame} indexed by subject.
    NaN-aware (sparse features like HRV can be missing); a zero or all-NaN
    std falls back to 1.0 so calibration never divides by zero."""
    alert = df[df["session"] == ALERT_SESSION]
    assert (alert["session"] == ALERT_SESSION).all()  # structural guarantee
    missing = set(df["subject"].unique()) - set(alert["subject"].unique())
    if missing:
        # Every UL-DD subject has an Awake session (C/F/L have ONLY that one),
        # so this indicates a broken table, not a legitimate state.
        raise ValueError(f"Subjects with no Awake session to calibrate from: {sorted(missing)}")

    grouped = alert.groupby("subject")[features]
    mean = grouped.mean()  # nan-aware by default
    std = grouped.std(ddof=0)
    std = std.where(std.notna() & (std > 0), 1.0)
    mean = mean.fillna(0.0)
    return {"mean": mean, "std": std}


def apply_alert_baseline(
    df: pd.DataFrame, features: list[str], mode: str = "replace"
) -> pd.DataFrame:
    """Calibrate `features` per subject against their alert baseline.

    mode='replace': each feature becomes (x - alert_mean) / alert_std --
    "how far from YOUR normal", the Tier-3 calibrated-LOSO transform.
    mode='append': adds '{feature}_cal' columns instead, leaving the raw
    values available (delta-features usable in any tier)."""
    if mode not in ("replace", "append"):
        raise ValueError(f"mode must be 'replace' or 'append', got {mode!r}")
    stats = alert_baseline_stats(df, features)
    mean = stats["mean"].loc[df["subject"]].to_numpy()
    std = stats["std"].loc[df["subject"]].to_numpy()

    out = df.copy()
    calibrated = (df[features].to_numpy() - mean) / std
    if mode == "replace":
        out[features] = calibrated
    else:
        for j, c in enumerate(features):
            out[f"{c}_cal"] = calibrated[:, j]
    return out
