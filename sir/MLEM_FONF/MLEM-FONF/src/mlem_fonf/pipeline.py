"""Algorithm 1' — the frozen, fully automatic MLEM+FONF pipeline (JBHI).

Locked components (see THEORY_LOG for the gates that froze each):
  * damped multiplicative MLEM step, gamma = 0.9        (gate G1)
  * fractional order alpha via Poisson-discrepancy       (A-FONF milestone)
  * notch bank via robust rule v2 + persistence          (gate G2a)
  * relaxation weight lam_dt = 0.95, non-negativity, tolerance stop

Zero hand-tuned parameters at run time: alpha and the notch set are selected
from the data; everything else is a frozen constant of the algorithm.
"""
from __future__ import annotations

import time

import numpy as np

from .adaptive import alpha_morozov
from .algorithms import mlem_fonf_damped
from .fonf import select_notches_persistent

GAMMA = 0.9
LAM_DT = 0.95
ALPHA_GRID = None          # resolution-aware: adaptive.alpha_grid_for(n)


def reconstruct(counts: np.ndarray, count_scale: float, A, n: int,
                n_bins: int, n_iter: int = 800, tol: float = 0.0,
                select_iter: int = 800, verbose: bool = False):
    """Run the frozen pipeline; returns (image, report).

    counts       raw Poisson sinogram counts
    count_scale  line-integral scale: g = counts * count_scale
    """
    t0 = time.time()
    g = counts * count_scale

    # Stage 1 — data-driven fractional order (Morozov)
    a_sel, diag = alpha_morozov(counts, count_scale, A, n, grid=ALPHA_GRID,
                                n_iter=select_iter, return_diag=True)
    f_sel = diag["reconstruction"]
    if verbose:
        print(f"[stage 1] alpha* = {a_sel}  deviances="
              f"{ {k: round(v, 2) for k, v in diag['deviances'].items()} }")

    # Stage 2 — notch detection with persistence on two iterates
    f_pair = mlem_fonf_damped(g, A, n, 20, alpha=a_sel, lam_dt=LAM_DT,
                              gamma=GAMMA, f0=f_sel)
    band = n_bins // 2 - 2          # acquisition-band guard: frequencies
    # beyond the detector Nyquist radius are unmeasurable inversion artifacts
    notches = select_notches_persistent(f_sel, f_pair, r_max=band)
    if verbose:
        print(f"[stage 2] K = {len(notches)} notch(es)")

    # Stage 3 — final damped run with the selected order and notch bank
    f = mlem_fonf_damped(g, A, n, n_iter, alpha=a_sel, lam_dt=LAM_DT,
                         gamma=GAMMA, notches=notches)
    report = dict(alpha=a_sel, K=len(notches), notches=notches,
                  deviances=diag["deviances"], gamma=GAMMA, lam_dt=LAM_DT,
                  runtime_s=time.time() - t0)
    return f, report
