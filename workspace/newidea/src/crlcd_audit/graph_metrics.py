from __future__ import annotations

import numpy as np


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def directed_metrics(truth: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    t = (truth != 0) & mask
    e = (estimate != 0) & mask
    tp = int(np.sum(t & e))
    fp = int(np.sum(~t & e))
    fn = int(np.sum(t & ~e))
    metrics = _prf(tp, fp, fn)
    shd = 0
    for i in range(t.shape[0]):
        for j in range(i + 1, t.shape[0]):
            if not (mask[i, j] or mask[j, i]):
                continue
            tv = (bool(t[i, j]), bool(t[j, i]))
            ev = (bool(e[i, j]), bool(e[j, i]))
            if tv != ev:
                shd += 1
    return {**metrics, "shd": shd, "tp": tp, "fp": fp, "fn": fn}


def skeleton_from_directed(adjacency: np.ndarray) -> np.ndarray:
    skel = (adjacency != 0) | (adjacency.T != 0)
    np.fill_diagonal(skel, False)
    return skel


def skeleton_metrics(truth: np.ndarray, estimate_skeleton: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    t = skeleton_from_directed(truth)
    e = np.asarray(estimate_skeleton, dtype=bool)
    upper = np.triu(mask, 1)
    tp = int(np.sum(t & e & upper))
    fp = int(np.sum(~t & e & upper))
    fn = int(np.sum(t & ~e & upper))
    return {**_prf(tp, fp, fn), "shd": fp + fn, "tp": tp, "fp": fp, "fn": fn}


def orientation_metrics(truth: np.ndarray, directed: np.ndarray, skeleton: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    """Score directions only for true edges present in the estimated skeleton.

    A wrong directed orientation contributes one FP and one FN. An undirected
    estimate contributes one FN. Directed false-positive skeleton edges count FP.
    """
    t = (truth != 0) & mask
    d = (directed != 0) & mask
    s = np.asarray(skeleton, dtype=bool) & mask
    tp = fp = fn = 0
    for i in range(t.shape[0]):
        for j in range(i + 1, t.shape[0]):
            if not mask[i, j]:
                continue
            true_dir = 1 if t[i, j] else (-1 if t[j, i] else 0)
            est_dir = 1 if d[i, j] else (-1 if d[j, i] else 0)
            present = bool(s[i, j] or s[j, i])
            if true_dir and present:
                if est_dir == true_dir:
                    tp += 1
                else:
                    fn += 1
                    if est_dir:
                        fp += 1
            elif not true_dir and est_dir:
                fp += 1
    return {**_prf(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}

