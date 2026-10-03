from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata


def _matched_abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    corr = np.corrcoef(a.T, b.T)[:a.shape[1], a.shape[1]:]
    rows, cols = linear_sum_assignment(-np.abs(corr))
    return float(np.mean(np.abs(corr[rows, cols])))


def standard_mcc(reference: np.ndarray, transformed: np.ndarray) -> float:
    return float(np.mean([_matched_abs_corr(reference[:, g, :], transformed[:, g, :]) for g in range(reference.shape[1])]))


def rank_aware_mcc(reference: np.ndarray, transformed: np.ndarray) -> float:
    scores = []
    for g in range(reference.shape[1]):
        a = np.apply_along_axis(rankdata, 0, reference[:, g, :])
        b = np.apply_along_axis(rankdata, 0, transformed[:, g, :])
        scores.append(_matched_abs_corr(a, b))
    return float(np.mean(scores))

