"""A-FONF: automatic, training-free selection of the fractional order alpha.

Principle (P2 of the JBHI roadmap): the FONF roll-off should attenuate only
mildly at the frequency where the object's power-law spectrum meets the
reconstruction noise floor. Both quantities are MEASURED from a short MLEM
burn-in of the data at hand; the rule keeps a single universal constant
delta (per-iteration attenuation budget at the crossover), calibrated once
and frozen across datasets, doses, and geometries.

    alpha_rule = clip( 4 * delta * w_N^2 / rho_x^2 ,  ALPHA_MIN, ALPHA_MAX )

where rho_x is the signal/noise crossover radius of the radially averaged
power spectrum of the burn-in reconstruction.
"""
from __future__ import annotations

import numpy as np

DELTA = 0.045          # universal constant (calibrated once; see THEORY_LOG)
ALPHA_MIN, ALPHA_MAX = 0.2, 2.0


def radial_psd(img: np.ndarray):
    n = img.shape[0]
    F = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    cy = cx = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - cy, xx - cx).astype(int)
    psd = np.bincount(r.ravel(), F.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    rho = np.arange(len(psd)) * (np.pi / (n / 2.0))          # radians/pixel
    return rho, psd


def spectral_diagnostics(recon: np.ndarray):
    """Fit log-log signal slope in the mid band and the high-rho noise
    plateau; return (beta, log_intercept, plateau, rho_cross)."""
    n = recon.shape[0]
    rho, psd = radial_psd(recon)
    w_n = np.pi
    mid = (rho > 0.06 * w_n) & (rho < 0.45 * w_n) & (psd > 0)
    x, y = np.log(rho[mid]), np.log(psd[mid])
    slope, icpt = np.polyfit(x, y, 1)                        # y = icpt + slope x
    beta = -slope / 2.0
    hi = (rho > 0.82 * w_n) & (rho < 0.97 * w_n) & (psd > 0)
    plateau = float(np.median(psd[hi]))
    # crossover of the fitted signal line with the plateau
    log_rx = (np.log(plateau) - icpt) / slope
    rho_x = float(np.exp(log_rx))
    rho_x = float(np.clip(rho_x, 0.05 * w_n, 0.95 * w_n))
    return float(beta), float(icpt), plateau, rho_x


def alpha_rule(g, A, n: int, burn_in: int = 20, delta: float = DELTA,
               return_diag: bool = False):
    """Data-driven fractional order from a short MLEM burn-in."""
    from .algorithms import mlem
    f0 = mlem(g, A, n, burn_in)
    beta, icpt, plateau, rho_x = spectral_diagnostics(f0)
    w_n = np.pi
    alpha = 4.0 * delta * (w_n ** 2) / (rho_x ** 2)
    alpha = float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))
    if return_diag:
        return alpha, dict(beta=beta, plateau=plateau, rho_x=rho_x,
                           rho_x_frac=rho_x / w_n)
    return alpha


# ---------------------------------------------------------- counts-based rule
ALPHA_HI, ALPHA_LO = 2.0, 0.8
_LOGC0, _LOGW = None, None      # set by calibrate_counts_rule / defaults below


def noise_level_from_counts(counts: np.ndarray) -> float:
    """Known-physics noise proxy: relative std of the median-count ray,
    eta = 1/sqrt(median positive counts). No reconstruction, no estimation."""
    pos = counts[counts > 0]
    return float(1.0 / np.sqrt(np.median(pos))) if pos.size else 1.0


def alpha_from_counts(counts: np.ndarray, logc0: float | None = None,
                      logw: float | None = None) -> float:
    """A-FONF order from measured photon statistics (two-anchor logistic in
    log eta): photon-starved data -> ALPHA_HI (strong roll-off), photon-rich
    -> ALPHA_LO (flat-objective regime). Deterministic, training-free."""
    c0 = _LOGC0 if logc0 is None else logc0
    w = _LOGW if logw is None else logw
    eta = noise_level_from_counts(counts)
    z = (np.log(eta) - c0) / w
    return float(np.clip(ALPHA_LO + (ALPHA_HI - ALPHA_LO) / (1 + np.exp(-z)),
                         0.2, 2.0))


def poisson_deviance_per_ray(counts: np.ndarray, mu_counts: np.ndarray) -> float:
    """Mean Poisson deviance 2[c ln(c/mu) - (c - mu)] per ray."""
    mu = np.maximum(mu_counts, 1e-9)
    t = np.where(counts > 0, counts * np.log(np.maximum(counts, 1e-12) / mu), 0.0)
    return float(2.0 * np.mean(t - (counts - mu)))


ALPHA_LADDER = tuple(0.2 * 2 ** k for k in range(12))  # 0.2 ... 409.6


def alpha_grid_for(n: int = 256):
    """Order ladder for the discrepancy search.

    h_fonf attenuates by exp(-alpha |w|^2 / (4 w_Nyq^2)) with w in
    radians/pixel, so alpha is defined RELATIVE TO THE SAMPLING GRID and is
    resolution-invariant: a controlled test (n = 128 vs 256 with the sampling
    scaled proportionally and the dose per ray held fixed) gives the same
    optimum, 1.0, at both. What DOES move the optimum is dose: at n = 256 the
    optimum rises from 1.0 at 300 photons/ray to 4.0 at 75 — beyond the
    0.2-2.0 grid used until v0.9.29, which therefore capped the method in
    every photon-starved setting (0.55 dB measured at 75 photons/ray, and the
    "alpha saturates at 2.0" observation logged earlier was this cap, not
    physics). The ladder below is unbounded in practice and the discrepancy
    criterion decides where to stop; alpha > 2 is admissible (H stays in
    [0, 1], symmetric and nonexpansive for any alpha > 0).

    The ladder reaches 409.6 so that the CRITERION, not the ladder end,
    terminates the search: at 60 photons per ray in the sparse view geometry
    the deviance only crosses tau between 25.6 and 51.2, and the selected
    order lands within 0.02 dB of the PSNR optimum, whereas the 2.0 ceiling
    used until v0.9.29 cost 4 dB there. Early stopping keeps the added
    entries free in every other regime."""
    return ALPHA_LADDER


def alpha_morozov(counts: np.ndarray, count_scale: float, A, n: int,
                  grid=None, tau: float = 1.0,
                  n_iter: int = 800, lam_dt: float = 0.95,
                  return_diag: bool = False):
    """A-FONF (final rule): Poisson-discrepancy (Morozov) selection.

    Run the reconstruction at each candidate order and pick the LARGEST alpha
    whose converged fit remains noise-consistent (per-ray deviance <= tau).
    Content-adaptive by construction, training-free, classical. One-time cost
    ~ len(grid) reconstructions per acquisition (~40 s CPU at 256^2).
    Validated (v0.5.0): worst mean oracle gap 0.142 +/- 0.009 dB over seven
    content x dose configurations incl. four held-out points; see THEORY_LOG.
    `count_scale` converts the line-integral sinogram to expected counts:
    mu_counts = (A f) / count_scale.
    """
    from .algorithms import mlem_fonf
    ladder = grid is None
    grid = alpha_grid_for(n) if ladder else grid
    g = counts * count_scale
    devs, out = {}, {}
    for a in sorted(grid):
        f = mlem_fonf(g, A, n, n_iter, alpha=a, lam_dt=lam_dt)
        devs[a] = poisson_deviance_per_ray(counts, (A @ f.ravel()) / count_scale)
        out[a] = f
        if ladder and devs[a] > tau:
            break            # deviance is monotone in alpha: stop at the first
                             # noise-inconsistent order (saves reconstructions)
    ok = [a for a in devs if devs[a] <= tau]
    a_sel = max(ok) if ok else min(devs, key=devs.get)
    if return_diag:
        return a_sel, dict(deviances=devs, reconstruction=out[a_sel])
    return a_sel
