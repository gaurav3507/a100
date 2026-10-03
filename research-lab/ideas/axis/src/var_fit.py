"""Fit a first-order vector autoregression, VAR(1), by least squares.

Model:  x_t = A x_{t-1} + e_t

Given a T x d series, return the d x d coefficient matrix A and the residual
covariance. A bootstrap option gives a per-coefficient standard error (SE) for
A, using a moving-block bootstrap so the series' own autocorrelation is
respected.

numpy / scipy only. Imports nothing outside ideas/axis/.
"""

import numpy as np


def _design(X):
    """Turn a T x d series into (Z, Y) predictor/target pairs for VAR(1).

    Z[i] = x_{t-1}, Y[i] = x_t, each of shape (T-1, d).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (T x d), got shape {X.shape}")
    T, d = X.shape
    if T < 3:
        raise ValueError(f"need at least 3 time points, got T={T}")
    Z = X[:-1]
    Y = X[1:]
    return Z, Y


def _fit_coeffs(Z, Y):
    """OLS fit of Y = Z @ B; returns A = B.T and residuals E = Y - Z @ B."""
    # lstsq is used instead of a normal-equation inverse for numerical safety.
    B, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)  # B is (d, d), B = A.T
    E = Y - Z @ B
    A = B.T
    return A, E


def fit_var1(X, bootstrap=False, n_boot=500, block=None, seed=None):
    """Fit VAR(1) to a T x d series.

    Parameters
    ----------
    X : array (T, d)
        The time series, rows are time points.
    bootstrap : bool
        If True, also estimate a per-coefficient SE for A.
    n_boot : int
        Number of bootstrap replicates.
    block : int or None
        Moving-block length. None picks a data-driven default (~n**(1/3)).
        block=1 is an ordinary (iid) pairs bootstrap.
    seed : int or None
        Seed for the bootstrap resampling.

    Returns
    -------
    dict with keys:
        A      : (d, d) coefficient matrix.
        Sigma  : (d, d) residual covariance.
        A_se   : (d, d) bootstrap SE of A, or None if bootstrap is False.
        n      : number of (predictor, target) pairs used.
    """
    Z, Y = _design(X)
    n, d = Z.shape
    A, E = _fit_coeffs(Z, Y)

    dof = max(n - d, 1)
    Sigma = (E.T @ E) / dof

    A_se = None
    if bootstrap:
        A_se = _bootstrap_se(Z, Y, n_boot=n_boot, block=block, seed=seed)

    return {"A": A, "Sigma": Sigma, "A_se": A_se, "n": n}


def _bootstrap_se(Z, Y, n_boot=500, block=None, seed=None):
    """Moving-block bootstrap SE for each entry of A.

    Blocks of consecutive (Z, Y) rows are resampled with replacement and the
    VAR(1) coefficients are refit on each replicate. The SE is the standard
    deviation of each coefficient across replicates.
    """
    rng = np.random.default_rng(seed)
    n, d = Z.shape
    if block is None:
        block = max(1, int(round(n ** (1.0 / 3.0))))
    block = min(block, n)

    n_starts = int(np.ceil(n / block))
    max_start = n - block  # inclusive upper bound for a block start

    estimates = np.empty((n_boot, d, d), dtype=float)
    for b in range(n_boot):
        if max_start <= 0:
            idx = np.arange(n)
        else:
            starts = rng.integers(0, max_start + 1, size=n_starts)
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        Ab, _ = _fit_coeffs(Z[idx], Y[idx])
        estimates[b] = Ab

    return estimates.std(axis=0, ddof=1)


if __name__ == "__main__":
    # Smoke test on a tiny known-ish system.
    rng = np.random.default_rng(0)
    d, T = 3, 400
    A_true = np.array([[0.5, 0.1, 0.0],
                       [0.0, 0.4, 0.2],
                       [0.1, 0.0, 0.3]])
    x = np.zeros((T, d))
    for t in range(1, T):
        x[t] = A_true @ x[t - 1] + rng.normal(scale=0.5, size=d)

    out = fit_var1(x, bootstrap=True, n_boot=100, seed=1)
    print("smoke test: fit_var1")
    print("  n pairs        :", out["n"])
    print("  A shape        :", out["A"].shape)
    print("  Sigma shape    :", out["Sigma"].shape)
    print("  A_se shape     :", out["A_se"].shape)
    print("  max |A - A_true|:", float(np.max(np.abs(out["A"] - A_true))))
    print("  mean A_se      :", float(out["A_se"].mean()))
    assert out["A"].shape == (d, d)
    assert out["Sigma"].shape == (d, d)
    assert out["A_se"].shape == (d, d)
    print("OK")
