from __future__ import annotations

from itertools import combinations

import numpy as np


def inter_group_adjacency(lam1: np.ndarray, num_group: int) -> np.ndarray:
    """Convert G-CaRL lam1 to parent-row, child-column adjacency."""
    dim = lam1.shape[0]
    out = np.zeros((num_group * dim, num_group * dim), dtype=float)
    for c, (a, b) in enumerate(combinations(range(num_group), 2)):
        out[a * dim:(a + 1) * dim, b * dim:(b + 1) * dim] = lam1[:, :, 0, c]
        out[b * dim:(b + 1) * dim, a * dim:(a + 1) * dim] = lam1[:, :, 1, c]
    return out


def intra_group_adjacency(lamin1: np.ndarray) -> np.ndarray:
    """Convert G-CaRL lamin1 to a block-diagonal parent-row adjacency."""
    dim, _, num_group = lamin1.shape
    out = np.zeros((num_group * dim, num_group * dim), dtype=float)
    for group in range(num_group):
        sl = slice(group * dim, (group + 1) * dim)
        out[sl, sl] = lamin1[:, :, group]
    return out


def inter_group_mask(num_group: int, num_dim: int) -> np.ndarray:
    groups = np.repeat(np.arange(num_group), num_dim)
    return groups[:, None] != groups[None, :]


def intra_group_mask(num_group: int, num_dim: int) -> np.ndarray:
    groups = np.repeat(np.arange(num_group), num_dim)
    return (groups[:, None] == groups[None, :]) & ~np.eye(num_group * num_dim, dtype=bool)


def remap_truth_for_permutation(adjacency: np.ndarray, permutations: list[list[int]], num_dim: int) -> np.ndarray:
    order = np.concatenate([g * num_dim + np.asarray(p, dtype=int) for g, p in enumerate(permutations)])
    return adjacency[np.ix_(order, order)]

