"""Reconstruction algorithms.

`mlem_fonf` implements Algorithm 1 of the paper.  The spatial-prior variants
(TV / AD / FuzzyAD) swap the spectral relaxation for a spatial denoiser in
the *same* MLEM loop (the paper's prior-substitution ablation), and
`pnp_admm_tv` is the training-free splitting baseline.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve
from skimage.restoration import denoise_tv_chambolle

from .fonf import h_fonf, apply_filter

_EPS = 1e-10


def _mlem_step(f, g, A, AT1):
    ratio = g / np.maximum(A @ f, _EPS)
    return f * (A.T @ ratio) / AT1


def mlem(g, A, n, n_iter=1000, callback=None):
    AT1 = np.maximum(A.T @ np.ones_like(g), _EPS)
    f = np.ones(n * n)
    for k in range(n_iter):
        f = np.maximum(_mlem_step(f, g, A, AT1), 0.0)
        if callback:
            callback(k, f.reshape(n, n))
    return f.reshape(n, n)


def mlem_fonf(g, A, n, n_iter=1000, alpha=0.5, lam_dt=0.9, notches=(),
              tol=0.0, callback=None):
    """Algorithm 1: MLEM data fidelity -> FONF filtering -> relaxation ->
    non-negativity -> convergence test."""
    AT1 = np.maximum(A.T @ np.ones_like(g), _EPS)
    H = h_fonf(n, alpha=alpha, notches=notches)
    f = np.ones(n * n)
    for k in range(n_iter):
        f_prev = f
        f_ml = _mlem_step(f, g, A, AT1)                       # step 1
        f_hat = apply_filter(f_ml.reshape(n, n), H).ravel()   # step 2, Eq. (4)
        f = (1.0 - lam_dt) * f_ml + lam_dt * f_hat            # step 3, Eq. (5)
        f = np.maximum(f, 0.0)                                # step 4
        if callback:
            callback(k, f.reshape(n, n))
        if tol > 0 and np.linalg.norm(f - f_prev) / max(np.linalg.norm(f_prev), _EPS) < tol:
            break                                             # step 5
    return f.reshape(n, n)


# ---------------------------------------------------------------- spatial priors
def _perona_malik(u, n_sub=3, kappa=0.05, dt=0.15, fuzzy=False):
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)
    for _ in range(n_sub):
        gN = np.roll(u, -1, 0) - u
        gS = np.roll(u, 1, 0) - u
        gE = np.roll(u, -1, 1) - u
        gW = np.roll(u, 1, 1) - u
        if fuzzy:  # Gaussian fuzzy membership as the edge-stopping weight
            c = lambda x: np.exp(-(np.abs(x) / kappa) ** 2)
        else:      # classical PM diffusivity
            c = lambda x: 1.0 / (1.0 + (x / kappa) ** 2)
        u = u + dt * (c(gN) * gN + c(gS) * gS + c(gE) * gE + c(gW) * gW)
    return u


def mlem_spatial(g, A, n, prior="tv", n_iter=1000, lam_dt=0.9, tv_weight=0.05,
                 kappa=0.25, dt=0.15, n_sub=2, callback=None):
    """Spatial priors in the same MLEM loop (prior-substitution baselines).

    TV uses the relaxation slot of Eq. (5); AD / FuzzyAD apply diffusion to
    the iterate before the multiplicative update (the stable ordering for
    these nonlinear denoisers; the FONF step needs no such care because it is
    linear and 1-Lipschitz)."""
    AT1 = np.maximum(A.T @ np.ones_like(g), _EPS)
    f = np.ones(n * n)
    for k in range(n_iter):
        if prior == "tv":
            f_ml = _mlem_step(f, g, A, AT1)
            den = denoise_tv_chambolle(f_ml.reshape(n, n), weight=tv_weight)
            f = (1 - lam_dt) * f_ml + lam_dt * den.ravel()
        elif prior in ("ad", "fuzzyad"):
            f = _perona_malik(f.reshape(n, n), n_sub=n_sub, kappa=kappa, dt=dt,
                              fuzzy=(prior == "fuzzyad")).ravel()
            f = _mlem_step(f, g, A, AT1)
        else:
            raise ValueError(prior)
        f = np.maximum(f, 0.0)
        if callback:
            callback(k, f.reshape(n, n))
    return f.reshape(n, n)


def pnp_admm_tv(g, A, n, n_iter=1000, rho=1.0, tv_weight=0.05, ml_sub=2,
                callback=None):
    """Training-free PnP-ADMM with a classical TV denoiser (no training data).

    x-update: a few MLEM sub-iterations pulled toward (z - u) by a quadratic
    coupling; z-update: TV denoising; u: dual ascent.
    """
    AT1 = np.maximum(A.T @ np.ones_like(g), _EPS)
    x = np.ones(n * n)
    z = x.copy()
    u = np.zeros(n * n)
    for k in range(n_iter):
        for _ in range(ml_sub):
            x = _mlem_step(x, g, A, AT1)
            x = (x + rho * (z - u)) / (1.0 + rho)     # quadratic coupling
            x = np.maximum(x, 0.0)
        z = denoise_tv_chambolle((x + u).reshape(n, n), weight=tv_weight).ravel()
        u = u + x - z
        if callback:
            callback(k, np.maximum(z, 0).reshape(n, n))
    return np.maximum(z, 0.0).reshape(n, n)


def mlem_fonf_damped(g, A, n, n_iter=1000, alpha=0.5, lam_dt=0.9, gamma=1.0,
                     notches=(), f0=None, callback=None):
    """Damped Algorithm 1: the multiplicative MLEM correction is applied with
    exponent gamma in (0, 1] (mirror-descent step size on the Poisson KL).
    gamma = 1 recovers the exact letter algorithm; gamma < 1 is the variant
    covered by the Bregman-PnP convergence template."""
    AT1 = np.maximum(A.T @ np.ones_like(g), _EPS)
    H = h_fonf(n, alpha=alpha, notches=notches)
    f = np.ones(n * n) if f0 is None else np.asarray(f0, dtype=float).ravel().copy()
    for k in range(n_iter):
        ratio = (A.T @ (g / np.maximum(A @ f, _EPS))) / AT1
        f_ml = f * np.power(np.maximum(ratio, _EPS), gamma)
        f_hat = apply_filter(f_ml.reshape(n, n), H).ravel()
        f = np.maximum((1.0 - lam_dt) * f_ml + lam_dt * f_hat, 0.0)
        if callback:
            callback(k, f.reshape(n, n))
    return f.reshape(n, n)


def spectral_potential(f, H):
    """psi(f) = 0.5 * f^T (I - A_s) f, computed spectrally (Parseval)."""
    F = np.fft.fft2(f)
    return float(0.5 * np.sum((1.0 - H) * np.abs(F) ** 2) / f.size)


def kl_data(g, A, f):
    """Poisson negative log-likelihood (up to constants): sum(Af) - g log(Af)."""
    af = np.maximum(A @ f.ravel(), _EPS)
    return float(np.sum(af - g * np.log(af)))
