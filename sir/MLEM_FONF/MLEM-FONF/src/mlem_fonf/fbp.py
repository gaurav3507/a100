"""Filtered backprojection (analytic reference baseline).

Parallel beam: ramp (Ram-Lak, optional Hann apodization) filtering of each
view's detector row, then backprojection via the certified A^T. Fan beam:
standard cosine pre-weighting of detector samples, same ramp, same A^T
(flat-detector, isocenter-equispaced bins). A single global scale constant is
calibrated once on a noiseless phantom and frozen (documented; FBP is the
analytic reference, not a tuned competitor).
"""
from __future__ import annotations

import numpy as np

_SCALE_CACHE: dict = {}


def _ramp(n_bins: int, hann: bool = True) -> np.ndarray:
    f = np.fft.fftfreq(2 * n_bins)                 # zero-padded x2
    filt = 2.0 * np.abs(f)
    if hann:
        filt *= 0.5 * (1 + np.cos(2 * np.pi * f / (2 * 0.5)))
    return filt


def _filter_sino(g: np.ndarray, n_bins: int, n_angles: int,
                 hann: bool = True) -> np.ndarray:
    sino = g.reshape(n_angles, n_bins)
    filt = _ramp(n_bins, hann)
    G = np.fft.fft(sino, n=2 * n_bins, axis=1)
    out = np.real(np.fft.ifft(G * filt[None, :], axis=1))[:, :n_bins]
    return out.ravel()


def fbp(g: np.ndarray, A, n: int, n_bins: int, n_angles: int,
        geometry: str = "parallel", sod: float = 600.0, odd: float = 600.0,
        hann: bool = True, ref_for_scale=None) -> np.ndarray:
    """FBP reconstruction using the certified adjoint as backprojector."""
    gq = g.astype(np.float64).copy()
    if geometry == "fan":
        mag = (sod + odd) / sod
        s_iso = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n_bins)
        w = sod / np.sqrt(sod ** 2 + (s_iso * mag) ** 2)      # cosine weight
        gq = (gq.reshape(n_angles, n_bins) * w[None, :]).ravel()
    q = _filter_sino(gq, n_bins, n_angles, hann)
    rec = (A.T @ q).reshape(n, n)
    key = (geometry, n, n_bins, n_angles, hann)
    if key not in _SCALE_CACHE:
        if ref_for_scale is None:
            from .phantoms import shepp_logan_mod
            ref_for_scale = shepp_logan_mod(n)
        g0 = A @ ref_for_scale.ravel()
        g0q = g0.copy()
        if geometry == "fan":
            g0q = (g0q.reshape(n_angles, n_bins) * w[None, :]).ravel()
        r0 = (A.T @ _filter_sino(g0q, n_bins, n_angles, hann)).reshape(n, n)
        num = float(np.sum(r0 * ref_for_scale))
        _SCALE_CACHE[key] = float(np.sum(ref_for_scale ** 2)) / max(num, 1e-12)
    return np.maximum(rec * _SCALE_CACHE[key], 0.0)
