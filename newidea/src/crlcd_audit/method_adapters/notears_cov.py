"""Faithful linear NOTEARS with an exactly cached least-squares objective.

The dual-ascent and acyclicity code follows xunzheng/notears linear.py at the
pinned vendor commit. For l2 loss, X.T @ X / n is cached once. This is exactly
the same centered objective and gradient as repeatedly forming X @ W.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt


def notears_linear_covariance(x: np.ndarray, lambda1: float, max_iter: int = 100, h_tol: float = 1e-8, rho_max: float = 1e16, w_threshold: float = 0.3) -> np.ndarray:
    x = x - np.mean(x, axis=0, keepdims=True)
    d = x.shape[1]
    covariance = x.T @ x / x.shape[0]

    def loss(weight: np.ndarray) -> tuple[float, np.ndarray]:
        residual_map = np.eye(d) - weight
        value = 0.5 * np.sum(residual_map * (covariance @ residual_map))
        gradient = covariance @ (weight - np.eye(d))
        return float(value), gradient

    def acyclicity(weight: np.ndarray) -> tuple[float, np.ndarray]:
        exponential = slin.expm(weight * weight)
        value = np.trace(exponential) - d
        return float(value), exponential.T * weight * 2

    def adjacency(doubled: np.ndarray) -> np.ndarray:
        return (doubled[:d * d] - doubled[d * d:]).reshape((d, d))

    estimate = np.zeros(2 * d * d)
    rho = 1.0
    alpha = 0.0
    h_value = np.inf
    bounds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        new_estimate = None
        new_h = None
        while rho < rho_max:
            def objective(doubled: np.ndarray) -> tuple[float, np.ndarray]:
                weight = adjacency(doubled)
                loss_value, loss_gradient = loss(weight)
                h_local, h_gradient = acyclicity(weight)
                value = loss_value + 0.5 * rho * h_local * h_local + alpha * h_local + lambda1 * doubled.sum()
                smooth_gradient = loss_gradient + (rho * h_local + alpha) * h_gradient
                gradient = np.concatenate((smooth_gradient + lambda1, -smooth_gradient + lambda1), axis=None)
                return float(value), gradient

            solution = sopt.minimize(objective, estimate, method="L-BFGS-B", jac=True, bounds=bounds)
            new_estimate = solution.x
            new_h, _ = acyclicity(adjacency(new_estimate))
            if new_h > 0.25 * h_value:
                rho *= 10
            else:
                break
        if new_estimate is None or new_h is None:
            break
        estimate, h_value = new_estimate, new_h
        alpha += rho * h_value
        if h_value <= h_tol or rho >= rho_max:
            break
    weight = adjacency(estimate)
    weight[np.abs(weight) < w_threshold] = 0
    return weight
