"""Temporal smoothing of window-level predictions (plan §6): the deployed
system's alert logic already specifies debounce/hysteresis (tasks.md 6.3:
require consecutive High classifications before alerting), so evaluating the
smoothed prediction stream is evaluating exactly what ships -- reported
ALONGSIDE raw-window metrics, never instead of them.

Both operations are CAUSAL (trailing context only): at window t the smoother
sees predictions for t, t-1, ... -- never the future, because the real
system doesn't either.

Only meaningful where a test set contains contiguous session stretches
(Tier 2 blocks, Tier 3 whole sessions). Tier 1's window-scattered test sets
have no contiguous stream to smooth; the runner refuses rather than
producing a nonsense number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def causal_majority_smooth(preds: np.ndarray, k: int = 3) -> np.ndarray:
    """Majority vote over the trailing k predictions (including the current
    one) of a single contiguous stream, ties resolved toward the current raw
    prediction. k=1 is the identity."""
    preds = np.asarray(preds)
    if k <= 1 or len(preds) == 0:
        return preds.copy()
    out = np.empty_like(preds)
    for i in range(len(preds)):
        window = preds[max(0, i - k + 1): i + 1]
        vals, counts = np.unique(window, return_counts=True)
        best = counts.max()
        winners = set(vals[counts == best])
        out[i] = preds[i] if preds[i] in winners else vals[counts.argmax()]
    return out


def smooth_by_session(
    meta: pd.DataFrame, preds: np.ndarray, k: int = 3
) -> np.ndarray:
    """Apply causal_majority_smooth independently within each (subject,
    session) stream of a test set, in window_id order. `meta` must carry
    subject/session/window_id rows aligned 1:1 with `preds`; returns the
    smoothed predictions in the original row order."""
    if len(meta) != len(preds):
        raise ValueError(f"meta ({len(meta)}) and preds ({len(preds)}) misaligned")
    out = np.asarray(preds).copy()
    frame = meta.reset_index(drop=True)
    for _, group in frame.groupby(["subject", "session"], sort=False):
        order = group.sort_values("window_id").index.to_numpy()
        out[order] = causal_majority_smooth(out[order], k)
    return out


def hysteresis_alert_stream(preds: np.ndarray, high_class: int = 2, consecutive: int = 2) -> np.ndarray:
    """The deployment alert rule itself (tasks.md 6.3): alert fires at window
    t iff the last `consecutive` predictions are all `high_class`. Returns a
    boolean array -- used for the alert-behavior analysis (false-alarm rate,
    detection latency), not for accuracy metrics."""
    preds = np.asarray(preds)
    alert = np.zeros(len(preds), dtype=bool)
    run = 0
    for i, p in enumerate(preds):
        run = run + 1 if p == high_class else 0
        alert[i] = run >= consecutive
    return alert
