from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransformResult:
    values: np.ndarray
    metadata: dict
    permutations: list[list[int]] | None = None


def within_group_permutation(z: np.ndarray, seed: int) -> TransformResult:
    rng = np.random.default_rng(seed)
    out = np.empty_like(z)
    permutations = []
    for g in range(z.shape[1]):
        p = rng.permutation(z.shape[2])
        out[:, g, :] = z[:, g, p]
        permutations.append(p.tolist())
    return TransformResult(out, {"seed": seed, "permutations": permutations}, permutations)


def component_scaling(z: np.ndarray, severity: float, seed: int) -> TransformResult:
    if severity == 0:
        raise ValueError("Scaling severity must be nonzero")
    rng = np.random.default_rng(seed)
    count = z.shape[1] * z.shape[2]
    values = np.resize(np.array([severity, 1.0 / severity], dtype=float), count)
    values = values[rng.permutation(count)].reshape(z.shape[1], z.shape[2])
    return TransformResult(z * values[None, :, :], {"seed": seed, "severity": severity, "multipliers": values.tolist()})


def cubic_warp(z: np.ndarray, lam: float) -> TransformResult:
    if lam < 0:
        raise ValueError("Cubic lambda must be nonnegative")
    return TransformResult(z + lam * z**3, {"lambda": lam, "formula": "z + lambda * z^3"})


def within_group_entanglement(z: np.ndarray, lam: float, seed: int) -> TransformResult:
    rng = np.random.default_rng(seed)
    out = z.copy()
    pairs = []
    for g in range(z.shape[1]):
        i, j = rng.choice(z.shape[2], size=2, replace=False)
        out[:, g, i] = z[:, g, i] + lam * z[:, g, j]
        pairs.append({"group": g, "target": int(i), "source": int(j)})
    return TransformResult(out, {"seed": seed, "lambda": lam, "pairs": pairs})


def additive_measurement_noise(z: np.ndarray, lam: float, seed: int) -> TransformResult:
    rng = np.random.default_rng(seed)
    std = np.std(z, axis=0, ddof=0)
    noise = rng.standard_normal(z.shape)
    out = z + lam * std[None, :, :] * noise
    return TransformResult(out, {"seed": seed, "lambda": lam, "distribution": "standard_normal", "per_variable_sd": std.tolist()})


def apply_transform(z: np.ndarray, family: str, level: float | str, seed: int) -> TransformResult:
    if family == "A1":
        return within_group_permutation(z, seed)
    if family == "B":
        return component_scaling(z, float(level), seed)
    if family == "C":
        return cubic_warp(z, float(level))
    if family == "D":
        return within_group_entanglement(z, float(level), seed)
    if family == "E":
        return additive_measurement_noise(z, float(level), seed)
    raise ValueError(f"Unknown transform family: {family}")

