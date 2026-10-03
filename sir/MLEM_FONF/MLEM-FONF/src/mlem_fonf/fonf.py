"""Fractional-order multi-notch filter (FONF).

Implements the transfer function of Eq. (4),

    H_FONF(w) = G_alpha(w) * prod_m [1 - exp(-||w - w_m||^2 / (2 sigma_m^2))]^(alpha/2),
    G_alpha(w) = exp(-alpha ||w||^2 / (4 w_N^2)),

with w_N the Nyquist radius of the discrete spectrum (fixed, no free
parameter), and the data-driven notch selection rule of Sec. II-C.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter


def _freq_grid(n: int):
    f = np.fft.fftfreq(n)                       # cycles / pixel in [-0.5, 0.5)
    wx, wy = np.meshgrid(2 * np.pi * f, 2 * np.pi * f, indexing="xy")
    return wx, wy, np.sqrt(wx ** 2 + wy ** 2)


def h_fonf(n: int, alpha: float = 0.5, notches=()) -> np.ndarray:
    """Return H_FONF on the (unshifted) FFT grid.

    `notches` is a sequence of (wx_m, wy_m, sigma_m) in radians/pixel; the
    conjugate of each notch is applied automatically so the filter stays real.
    """
    wx, wy, wr = _freq_grid(n)
    w_nyq = np.pi                                # Nyquist radius (fixed)
    H = np.exp(-alpha * wr ** 2 / (4.0 * w_nyq ** 2))
    for (cx, cy, sg) in notches:
        for sx, sy in ((cx, cy), (-cx, -cy)):    # conjugate pair
            d2 = (wx - sx) ** 2 + (wy - sy) ** 2
            H *= (1.0 - np.exp(-d2 / (2.0 * sg ** 2))) ** (alpha / 2.0)
    return H


def select_notches(image: np.ndarray, sigma_cap_frac: float = 0.05,
                   k_max: int = 8, dc_guard: int = 3):
    """Sec. II-C rule: peaks of R = |F(f)| / S_bar(||w||) above tau = mu_R + 3 sigma_R.

    Returns a list of (wx_m, wy_m, sigma_m); empty when no structured peak is
    detected (K = 0, the broadband regime).
    """
    n = image.shape[0]
    F = np.fft.fftshift(np.abs(np.fft.fft2(image)))
    cy = cx = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - cy, xx - cx)
    rbin = r.astype(int)
    # radially averaged magnitude spectrum S_bar(rho)
    s_sum = np.bincount(rbin.ravel(), F.ravel())
    s_cnt = np.bincount(rbin.ravel())
    s_bar = s_sum / np.maximum(s_cnt, 1)
    R = F / np.maximum(s_bar[rbin], 1e-12)
    R[r <= dc_guard] = 0.0                       # exclude the DC neighbourhood
    tau = R[r > dc_guard].mean() + 3.0 * R[r > dc_guard].std()
    peaks = (R == maximum_filter(R, size=7)) & (R > tau)
    pys, pxs = np.nonzero(peaks)
    if len(pys) == 0:
        return []
    order = np.argsort(-R[pys, pxs])
    notches, used = [], np.zeros(n * n, bool)
    for idx in order:
        py, px = pys[idx], pxs[idx]
        if px < cx or (px == cx and py < cy):    # keep one of each conjugate pair
            continue
        # half-width at half-maximum along x, capped
        half = R[py, px] / 2.0
        w = 1
        while px + w < n and px - w >= 0 and min(R[py, px + w], R[py, px - w]) > half:
            w += 1
        sigma = min(w * 2 * np.pi / n, sigma_cap_frac * np.pi)
        wxm = (px - cx) * 2 * np.pi / n
        wym = (py - cy) * 2 * np.pi / n
        notches.append((wxm, wym, sigma))
        if len(notches) >= k_max:
            break
    return notches


def apply_filter(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    """f_hat = F^{-1}( H_FONF . F(f) ), real part."""
    return np.real(np.fft.ifft2(H * np.fft.fft2(image)))


# ------------------------------------------------------------------ v2 rule
def select_notches_v2(image: np.ndarray, sectors: int = 16, r0: int = 8,
                      kappa: float = 10.0, sigma_cap_frac: float = 0.05,
                      k_max: int = 8, coverage_min: float = 0.35,
                      r_max: float | None = None, return_z: bool = False):
    """Robust notch selection (journal version).

    Replaces the isotropic mean background of Sec. II-C with a sectoral
    MEDIAN background S(rho, theta-sector) and a MAD-standardized threshold
    z > kappa, with an annulus guard rho > r0. Designed so that anisotropic
    anatomical spectra (spine/ribs) do NOT trigger notches while genuine
    narrowband peaks (rings/stripes) do."""
    n = image.shape[0]
    F = np.fft.fftshift(np.abs(np.fft.fft2(image)))
    cy = cx = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - cy, xx - cx)
    th = np.arctan2(yy - cy, xx - cx)
    rbin = np.minimum(r.astype(int), n)                    # radial bins
    sbin = ((th + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
    cell = rbin * sectors + sbin
    ncell = (n + 1) * sectors
    flat = F.ravel(); cf = cell.ravel()
    order = np.argsort(cf)
    med = np.zeros(ncell)
    sorted_vals = flat[order]; sorted_cells = cf[order]
    bounds = np.searchsorted(sorted_cells, np.arange(ncell + 1))
    for c in range(ncell):
        a, b = bounds[c], bounds[c + 1]
        if b > a:
            med[c] = np.median(sorted_vals[a:b])
    # fallback for sparse cells: per-radius global median
    rmed = np.zeros(n + 1)
    rb = rbin.ravel(); ro = np.argsort(rb)
    rb_s, fv_s = rb[ro], flat[ro]
    rbounds = np.searchsorted(rb_s, np.arange(n + 2))
    for c in range(n + 1):
        a, b = rbounds[c], rbounds[c + 1]
        if b > a:
            rmed[c] = np.median(fv_s[a:b])
    counts = np.bincount(cf, minlength=ncell)
    sparse = counts < 8
    med[sparse] = rmed[(np.arange(ncell) // sectors)][sparse]
    B = med[cell]
    R = F / np.maximum(B, 1e-12)
    r_hi = 0.98 * (n // 2) if r_max is None else min(r_max, 0.98 * (n // 2))
    mask = (r > r0) & (r < r_hi)
    vals = R[mask]
    m0 = np.median(vals)
    mad = np.median(np.abs(vals - m0)) * 1.4826
    z = (R - m0) / max(mad, 1e-12)
    peaks = (R == maximum_filter(R, size=7)) & (z > kappa) & mask
    pys, pxs = np.nonzero(peaks)
    Fc = np.fft.fft2(image)
    fx = np.fft.fftfreq(n); FXg, FYg = np.meshgrid(fx, fx)

    def _coverage(px_off, py_off, bw=3.0):
        """Spatial-coherence of the candidate: one-sided band-pass envelope
        coverage. Global stripes/rings cover the field; anatomical texture
        forms localized packets."""
        d = np.hypot(FXg - px_off / n, FYg - py_off / n)
        env = np.abs(np.fft.ifft2(Fc * (d < bw / n)))
        return float(np.mean(env > 0.3 * env.max()))

    notches = []
    for idx in np.argsort(-z[pys, pxs]):
        py, px = pys[idx], pxs[idx]
        if px < cx or (px == cx and py < cy):
            continue
        if _coverage(px - cx, py - cy) < coverage_min:
            continue
        half = R[py, px] / 2.0
        w = 1
        while px + w < n and px - w >= 0 and min(R[py, px + w], R[py, px - w]) > half:
            w += 1
        sigma = min(w * 2 * np.pi / n, sigma_cap_frac * np.pi)
        notches.append(((px - cx) * 2 * np.pi / n, (py - cy) * 2 * np.pi / n, sigma))
        if len(notches) >= k_max:
            break
    return (notches, z) if return_z else notches


def select_notches_persistent(img_a: np.ndarray, img_b: np.ndarray,
                              tol_px: float = 3.0, **kw):
    """Persistence filter: keep only peaks detected in BOTH images within
    tol_px pixels (rings/stripes are stationary across iterates; anatomical
    flukes are not)."""
    n = img_a.shape[0]
    na = select_notches_v2(img_a, **kw)
    nb = select_notches_v2(img_b, **kw)
    keep = []
    for (ax, ay, s) in na:
        for (bx, by, _) in nb:
            if np.hypot(ax - bx, ay - by) * n / (2 * np.pi) <= tol_px:
                keep.append((ax, ay, s))
                break
    return keep
