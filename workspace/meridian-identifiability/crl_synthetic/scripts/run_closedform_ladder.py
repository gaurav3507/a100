#!/usr/bin/env python
"""
run_closedform_ladder.py — closed-form mixing recovery under a PAIRING-DEGRADATION ladder.

WHY THIS FILE EXISTS
    Option B (run_multienv_gate.py: one intervention per coordinate) has a VERIFIED-correct
    optimum, but gradient training plateaus in a local minimum (pair_sum stuck ~11.3 vs the
    true 10.0; per-coordinate MCC bimodal — some coords 1.000, others 0.001 — and NOT
    explained by intervention strength). The problem turns out to have a CLOSED FORM:

        env_c differs from the reference only on latent coordinate c, so the LATENT paired
        difference is 1-D along axis c. Since x = A z, the OBSERVED paired difference
        (x_ref - x_env_c) lies exactly along the c-th COLUMN of the mixing A. Verified:
        |cos| with true A[:,c] = 1.000000 for all 10 coords, and the observed difference is
        exactly rank-1 (2nd/1st singular value ~1e-16).

    So the mixing is recoverable directly, with no optimizer and no local minima — by taking
    the top singular vector of each observed paired-difference matrix.

    THE CATCH, AND THE POINT OF THIS FILE. This works ONLY because the pairing is PERFECT:
    env_c is the reference re-simulated with the SAME innovations, so everything except
    coordinate c's mechanism cancels. Real task-fMRI never gives same-innovation pairs — the
    same subject in a different session/run has independent noise. So a win at perfect
    pairing proves NOTHING about transfer to real data. RHO=1 IS SYNTHETIC-ONLY. The
    scientifically meaningful question is not "does the closed form work at RHO=1" (it must,
    by construction — that is the oracle sanity gate) but WHERE ON THE RHO LADDER RECOVERY
    COLLAPSES, because real task-fMRI sits at RHO ~ 0.

THE LADDER
    A pairing-quality knob RHO in [0, 1] enters ONLY through the innovations:
        eps_env = RHO * eps_ref + sqrt(1 - RHO^2) * eps_fresh
    so the env latent is  z_env_c = z_ref + counterfactual_diff_c + (eps_env - eps_ref),
    where the first added term is the exact 1-D mechanism difference on coordinate c and the
    second is (RHO-1)*eps_ref + sqrt(1-RHO^2)*eps_fresh — zero at RHO=1, full-rank below it.
    RHO=1 -> the exact counterfactual (current setup). RHO=0 -> the innovation mismatch is
    injected once and then decays. The full-rank mismatch is what erodes the rank-1 structure
    the estimator depends on, so the rank-1 ratio is the direct trust diagnostic.

    THE ADDITIVE LADDER IS OPTIMISTIC AT THE INDEPENDENT END, and there is a separate rung for
    the honest case. The additive knob adds the mismatch once; it does NOT let the mismatch
    COMPOUND through the VAR recursion. The genuine two-independent-sessions case re-simulates
    env_c from scratch through its own shifted VAR with fresh innovations (z_env[t] = B_env @
    z_env[t-1..] + eps_fresh[t]) — the realistic task-fMRI analogue (same subject, different
    run). That is the "independent_recursive" rung. It is strictly harder than additive RHO=0:
    verified, at the independent end the fraction of paired-difference energy on the true
    target coordinate is 0.335 additive vs 0.218 recursive, and the rank-1 ratio is 0.59
    additive vs 0.75 recursive. So the additive ladder is the controlled interpolation between
    the verified RHO=1 ceiling and the independent end; the independent_recursive rung is the
    one to QUOTE for transfer claims, and results.md reports it separately and prominently with
    a dual verdict (the additive collapse RHO AND the recursive result).

    Two further realism toggles, each independent so their effects are separable:
      (a) observation noise at amplitude SNR in {inf, 20, 10, 5, 2} (independent per env)
      (b) HRF: canonical double-gamma convolution of the latents before mixing (TR=2.0).
          NOTE the HRF is linear and per-coordinate, so it does NOT rotate the spatial
          mixing: at RHO=1 it leaves |cos| ~ 1 and rank-1 intact and only blurs the latent
          (lowering MCC). That separation is itself a result the ladder makes visible.

WHAT IS REPORTED PER RUNG
    per-column |cos| with the true (standardized) mixing column A_eff[:,c] (mean and min);
    the rank-1 ratio (2nd/1st singular value, mean and max); closed-form MCC, subspace R2,
    per-coordinate MCC; and the MSE-baseline MCC on identical data as the rotationally-blind
    floor. The RHO ladder additionally runs the Option B gradient encoder in three modes to
    test whether the local minimum was an initialization problem:
      (i)   random init (the current failing setup)
      (ii)  init from the closed-form pseudo-inverse, then trained
      (iii) closed-form only, no training
    Headline figure: MCC vs RHO, three curves + the MSE floor line. Second figure: rank-1
    ratio vs RHO (the estimator-trustworthiness curve).

RUN
    python run_closedform_ladder.py --oracle   # scaffold checks 1-5b + closed-form gate @RHO=1
    python run_closedform_ladder.py --smoke    # RHO=1.0 and 0.5, 1 seed, all three modes
    python run_closedform_ladder.py            # full ladder, 12 seeds, resumable

RUNTIME (ESTIMATE — NOT MEASURED)
    The RHO ladder dominates: 9 RHO x 12 seeds x (2 flow trainings + 1 MSE) is a few hundred
    trainings, plausibly 4-6 h on an A100. The noise/HRF rungs are closed-form + MSE only and
    are cheap. Resumable per (rung, seed) — kill and rerun to continue. For a fast preview use
    --seeds 3.

OUTPUT
    ./results_closedform_ladder/
        results.md                       RHO ladder headline table + noise/HRF tables + verdict
        seeds/<rung>_seed<k>.json        per-(rung, seed) metrics
        fig_rho_ladder_mcc.png           MCC vs RHO: closed-form / cf-init-trained / random-trained
        fig_rho_ladder_rank1.png         rank-1 ratio vs RHO (estimator trustworthiness)
"""

import argparse
import json
import math
import os
import platform
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.stats import kurtosis, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ----------------------------------------------------------------------------- config

OUT_DIR = "./results_closedform_ladder"
SEED_DIR = os.path.join(OUT_DIR, "seeds")

D_LATENT = 10
N_OBS = 50
MAX_LAG = 2
EDGES_PER_NODE = 1.5
BURN_IN = 200

WINDOW_L = 64
N_WINDOWS = 256
TEST_FRAC = 0.2

N_SEEDS = 12
MCC_GATE = 0.4

FLOW_COUPLINGS = 6
FLOW_HIDDEN = 256
TRAIN_STEPS = 5000
LR = 1e-3
BATCH_WINDOWS = 16
RIDGE = 1e-3

LAMBDA_PAIR = 1.0
LAMBDA_PRIOR = 0.0

AE_STEPS = 3000
AE_LR = 1e-3

LAGGED_COV_WARN = 0.1
ORACLE_SEEDS = 6
SQRT_EPS = 1e-8

PAIR_CORR_CONTENT_MIN = 0.98
PAIR_CORR_SHUFFLED_MAX = 0.25
ONED_L1L2_MAX = 1.15

REF_TARGET_RHO = 0.5
ENV_TARGET_RHO = 0.9
MIN_ALPHA = 0.02

# --- the ladder ---
HRF_TR = 2.0
RHO_LIST = [1.0, 0.99, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
SNR_LIST = [20.0, 10.0, 5.0, 2.0]        # amplitude SNR; noise std = signal std / SNR
DIST = "laplace"                          # innovations for the ladder (closed form is dist-agnostic)

# closed-form sanity gate thresholds (oracle, RHO=1, no noise, no HRF)
CF_COS_MIN = 0.999                         # every column must align with the truth
CF_RANK1_MAX = 1e-4                        # every paired difference must be essentially rank-1
CF_MCC_MIN = 0.99                          # unmixing must recover the latents


# ----------------------------------- simulator primitives (reused verbatim)

def sample_innovations(rng, shape, dist):
    if dist == "laplace":
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=shape)  # unit variance
    if dist == "gaussian":
        return rng.normal(0.0, 1.0, size=shape)
    raise ValueError(dist)


def make_var(rng, d, edges_per_node, max_lag):
    """Sparse lagged linear VAR. Returns list of (d, d) coefficient matrices, one per lag."""
    n_edges = int(round(edges_per_node * d))
    coefs = [np.zeros((d, d)) for _ in range(max_lag)]
    slots = [(l, i, j) for l in range(max_lag) for i in range(d) for j in range(d) if i != j]
    for k in rng.choice(len(slots), size=n_edges, replace=False):
        l, i, j = slots[k]
        coefs[l][i, j] = rng.uniform(0.6, 1.0) * rng.choice([-1.0, 1.0])
    for i in range(d):
        coefs[0][i, i] = rng.uniform(0.3, 0.5)   # self-lag ON, moderate: raise VAR predictability
    return stabilize(coefs)


def companion_rho(coefs):
    """Spectral radius of the VAR's companion matrix. rho < 1 <=> stationary."""
    d = coefs[0].shape[0]
    max_lag = len(coefs)
    comp = np.zeros((d * max_lag, d * max_lag))
    comp[:d, :] = np.concatenate(coefs, axis=1)
    if max_lag > 1:
        comp[d:, : d * (max_lag - 1)] = np.eye(d * (max_lag - 1))
    return float(np.max(np.abs(np.linalg.eigvals(comp))))


def stabilize(coefs, target_rho=0.9):
    """Scale coefs down until the companion spectral radius is <= target_rho.

    Bisection, not a single pass: `c * (target_rho / rho)` only scales the companion
    eigenvalues linearly when max_lag == 1. For max_lag >= 2 rho scales like sqrt(s) in the
    lag-2-dominant case, so one pass undershoots and the latents diverge to inf.
    rho(0) = 0 is always feasible, so the bracket is valid regardless of monotonicity, and
    the postcondition is asserted rather than assumed.
    """
    coefs = [c.copy() for c in coefs]
    if companion_rho(coefs) <= target_rho:
        return coefs
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if companion_rho([c * mid for c in coefs]) > target_rho:
            hi = mid
        else:
            lo = mid
    out = [c * lo for c in coefs]
    rho = companion_rho(out)
    if rho > target_rho + 1e-8:
        raise RuntimeError(f"stabilize failed to reach target_rho: rho={rho:.6f}")
    return out


def make_mixing(rng, d, n_obs):
    """Well-conditioned linear mixing d -> n_obs."""
    while True:
        A = rng.normal(0.0, 1.0, size=(n_obs, d)) / np.sqrt(d)
        s = np.linalg.svd(A, compute_uv=False)
        if s[-1] > 1e-3 and s[0] / s[-1] < 50.0:
            return A


# ----------------------------------- env construction (reused verbatim)

def _max_stable_alpha(build, target_rho):
    """Largest alpha in [0, 1] keeping companion rho <= target_rho, by bisection.

    rho(0) = the reference (stable), so the bracket is always valid regardless of whether
    rho(alpha) is monotone.
    """
    if companion_rho(build(1.0)) <= target_rho:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if companion_rho(build(mid)) > target_rho:
            hi = mid
        else:
            lo = mid
    return lo


def make_env_shift(rng, coefs_0, c, max_lag, target_rho=ENV_TARGET_RHO):
    """Build env_c: shift exactly ONE lagged off-diagonal coefficient whose TARGET row is c.

    Among ALL existing lagged off-diagonal edges into row c, evaluate the achievable
    stationary shift (highest alpha) of each and keep the best. This stops a coordinate from
    being starved by an unlucky single-edge pick: one edge may feed the dominant eigenmode
    and collapse to alpha ~ 0, while another edge into the same coordinate absorbs a full
    shift. A shared magnitude is used across candidates so they are compared on equal footing.
    If row c has no incoming edge at all, one is created. Either way env_c - reference is
    nonzero at exactly one (lag, c, j) entry, so the counterfactual difference is 1-D on
    coord c. Returns (coefs_c, edge=(l, c, j), alpha).

    A low best-alpha means every edge into c is stability-constrained; make_dataset aborts
    below MIN_ALPHA and the sanity block flags anything below 0.5.
    """
    d = coefs_0[0].shape[0]
    mag = rng.uniform(1.5, 2.5)   # one magnitude, so edges are compared apples-to-apples

    def make_build(l, j, base, target_val):
        def build(alpha):
            new = [cc.copy() for cc in coefs_0]
            new[l][c, j] = (1.0 - alpha) * base + alpha * target_val
            return new
        return build

    candidates = [(l, j) for l in range(max_lag) for j in range(d)
                  if j != c and abs(coefs_0[l][c, j]) > 1e-8]
    if candidates:
        best = None   # (alpha, l, j, base, target_val)
        for (l, j) in candidates:
            base = coefs_0[l][c, j]
            target_val = -np.sign(base) * mag          # sign flip + magnitude
            a = _max_stable_alpha(make_build(l, j, base, target_val), target_rho)
            if best is None or a > best[0]:
                best = (a, l, j, base, target_val)
        alpha, l, j, base, target_val = best
    else:
        l = int(rng.integers(max_lag))
        j = int(rng.choice([x for x in range(d) if x != c]))
        base = 0.0
        target_val = rng.choice([-1.0, 1.0]) * mag
        alpha = _max_stable_alpha(make_build(l, j, base, target_val), target_rho)

    return make_build(l, j, base, target_val)(alpha), (int(l), int(c), int(j)), float(alpha)


# ----------------------------------- counterfactual / simulate (reused verbatim)

def counterfactual_diff(z_ref_full, coefs_env, coefs_ref, max_lag):
    """dd[t] = sum_l (B_env[l] - B_ref[l]) @ z_ref[t-l-1]. Nonzero only in the shifted row.

    Vectorized: env_c is driven by the reference's OWN history, so this is a plain linear
    map of the reference series, not a recursion. z_env = z_ref + dd.
    """
    total, d = z_ref_full.shape
    dd = np.zeros((total, d))
    for l in range(max_lag):
        delta = coefs_env[l] - coefs_ref[l]              # nonzero only in one row
        dd[l + 1:] += z_ref_full[: total - l - 1] @ delta.T
    return dd


def simulate_reference(rng, coefs, total, dist):
    """Contiguous reference trajectory. Returns (z_full (total, d), eps (total, d))."""
    d = coefs[0].shape[0]
    max_lag = len(coefs)
    eps = sample_innovations(rng, (total, d), dist)
    z = np.zeros((total, d))
    for t in range(total):
        acc = np.zeros(d)
        for l in range(max_lag):
            if t - l - 1 >= 0:
                acc += coefs[l] @ z[t - l - 1]
        z[t] = acc + eps[t]
    return z, eps



# --------------------------------------------------- HRF and the pairing-degradation dataset


def hrf_kernel(tr=HRF_TR, length=32.0):
    """Canonical double-gamma HRF (SPM-style), sampled at TR. Area-normalized.

    peak = Gamma(shape 6, scale 1) pdf; undershoot = Gamma(16, 1) pdf; h = peak - undershoot/6.
    Linear and causal; convolving the latents with it before mixing does not rotate the
    spatial mixing, so at RHO=1 it leaves |cos| and rank-1 intact and only blurs the latent.
    """
    t = np.arange(0.0, length, tr)
    peak = t ** (6 - 1) * np.exp(-t) / math.gamma(6)
    undershoot = t ** (16 - 1) * np.exp(-t) / math.gamma(16)
    h = peak - undershoot / 6.0
    return h / h.sum()


def hrf_convolve(z_full, h):
    """Causal per-coordinate convolution of a contiguous latent series. z_full: (total, d)."""
    total, d = z_full.shape
    out = np.zeros_like(z_full)
    for k in range(d):
        out[:, k] = np.convolve(z_full[:, k], h)[:total]
    return out


def make_dataset(seed, dist, rho=1.0, snr=np.inf, hrf=False, recursive=False):
    """One-intervention-per-coordinate paired data with a pairing-quality knob RHO.

    Two constructions for env_c:

    ADDITIVE (default). RHO enters ONLY through the innovations: env_c gets eps_env =
    RHO*eps_ref + sqrt(1-RHO^2)*eps_fresh, applied on top of the reference's OWN history, so
        z_env_c = z_ref + counterfactual_diff_c + (eps_env - eps_ref)
    The counterfactual term is the exact 1-D mechanism difference on coordinate c; the
    innovation-mismatch term is zero at RHO=1 and full-rank below it.

    RECURSIVE (recursive=True). env_c is re-simulated from scratch through its OWN shifted
    VAR with fresh independent innovations: z_env[t] = B_env @ z_env[t-1..] + eps_fresh[t].
    This is the genuine two-independent-sessions case (same subject, different run) and the
    realistic task-fMRI analogue. It is NOT the additive RHO=0 case: additive injects the
    innovation mismatch once and lets it decay, but recursive lets the mismatch COMPOUND
    through the VAR recursion, so it is strictly harder. Verified: at the independent end,
    fraction of paired-difference energy on the true target coordinate is 0.335 additive vs
    0.218 recursive, and the rank-1 ratio is 0.59 additive vs 0.75 recursive. The additive
    ladder is therefore OPTIMISTIC at low RHO — it is the controlled interpolation between
    the verified RHO=1 ceiling and the independent end, while the recursive rung is the one
    to quote for transfer claims. In recursive mode RHO is ignored (there is no reference
    innovation to correlate with).

    Ground-truth z_all is the NEURAL latent (pre-HRF); x_all is mix(+HRF)(+noise) then
    shared-standardized. A_eff is the standardized mixing whose columns the closed form
    recovers.
    """
    rng_struct = np.random.default_rng(seed)
    rng_innov = np.random.default_rng(1_000_000 + seed)
    rng_mix = np.random.default_rng(2_000_000 + seed)
    rng_pair = np.random.default_rng(3_000_000 + seed)   # fresh innovations for the RHO mix
    rng_noise = np.random.default_rng(4_000_000 + seed)  # observation noise
    rng_rec = np.random.default_rng(5_000_000 + seed)    # fresh innovations for recursive envs

    coefs_0 = make_var(rng_struct, D_LATENT, EDGES_PER_NODE, MAX_LAG)
    coefs_0 = stabilize(coefs_0, target_rho=REF_TARGET_RHO)   # headroom for interventions
    coefs_envs, edges, alphas = [], [], []
    for c in range(D_LATENT):
        cc, edge, alpha = make_env_shift(rng_struct, coefs_0, c, MAX_LAG)
        coefs_envs.append(cc)
        edges.append(edge)
        alphas.append(alpha)

    # --- INVARIANTS on the mechanism (RHO-independent) ---
    if companion_rho(coefs_0) >= 1.0:
        raise RuntimeError(f"seed {seed} {dist}: reference VAR non-stationary.")
    weak = [(c, round(a, 4)) for c, a in enumerate(alphas) if a < MIN_ALPHA]
    if weak:
        raise RuntimeError(
            f"seed {seed} {dist}: interventions collapsed for envs {weak} (alpha < {MIN_ALPHA}). "
            f"Lower REF_TARGET_RHO or the intervention magnitude.")
    for c, cc in enumerate(coefs_envs):
        if companion_rho(cc) >= 1.0:
            raise RuntimeError(f"seed {seed} {dist}: env {c} non-stationary.")
        for l in range(MAX_LAG):
            other = np.delete(coefs_0[l] - cc[l], c, axis=0)
            if np.abs(other).max() > 1e-12:
                raise RuntimeError(f"seed {seed} {dist}: env {c} differs off its target row.")

    total = N_WINDOWS * WINDOW_L + BURN_IN
    z_ref_full, eps_ref = simulate_reference(rng_innov, coefs_0, total, dist)
    if not np.isfinite(z_ref_full).all():
        raise RuntimeError(f"seed {seed} {dist}: reference latents contain inf/NaN.")

    neural_full = [z_ref_full]
    if recursive:
        # Genuine independent re-simulation through env_c's own shifted VAR. Each env draws
        # its own fresh innovation sequence (rng_rec advances per call, so envs are mutually
        # independent). The mismatch compounds through the recursion.
        for c in range(D_LATENT):
            z_env_full, _ = simulate_reference(rng_rec, coefs_envs[c], total, dist)
            neural_full.append(z_env_full)
    else:
        for c in range(D_LATENT):
            dd = counterfactual_diff(z_ref_full, coefs_envs[c], coefs_0, MAX_LAG)  # 1-D on coord c
            eps_fresh = sample_innovations(rng_pair, (total, D_LATENT), dist)
            mismatch = (rho - 1.0) * eps_ref + np.sqrt(max(1.0 - rho ** 2, 0.0)) * eps_fresh
            neural_full.append(z_ref_full + dd + mismatch)

    bold_full = [hrf_convolve(z, hrf_kernel()) for z in neural_full] if hrf else neural_full

    def win(zf):
        return zf[BURN_IN:].reshape(N_WINDOWS, WINDOW_L, D_LATENT)

    z_all = np.stack([win(zf) for zf in neural_full], 0)     # NEURAL ground truth (E, nw, L, d)
    bold_all = np.stack([win(zf) for zf in bold_full], 0)
    for arr, nm in ((z_all, "neural"), (bold_all, "bold")):
        if not np.isfinite(arr).all() or np.abs(arr).max() > 1e6:
            raise RuntimeError(f"seed {seed} {dist}: {nm} latents blew up.")

    # At RHO=1 (additive) the mismatch is exactly zero, so the neural pair is bit-identical
    # off coord c. (HRF acts on bold only, so this holds regardless of the HRF toggle.) Below
    # RHO=1, and in recursive mode, the pair differs everywhere — that is the degradation.
    if rho == 1.0 and not recursive:
        for c in range(D_LATENT):
            noncol = [k for k in range(D_LATENT) if k != c]
            resid = float(np.abs(z_all[0][:, :, noncol] - z_all[1 + c][:, :, noncol]).max())
            if resid > 1e-6:
                raise RuntimeError(f"seed {seed} {dist}: PAIRING BROKEN for env {c} at RHO=1 "
                                   f"(off-target residual {resid:.3e}).")

    A = make_mixing(rng_mix, D_LATENT, N_OBS)
    x_raw = bold_all @ A.T                                    # (E, nw, L, N)

    if np.isfinite(snr):
        sig_std = x_raw.reshape(-1, N_OBS).std(0)             # per-channel signal amplitude
        noise = rng_noise.normal(size=x_raw.shape) * (sig_std / snr)[None, None, None, :]
        x_raw = x_raw + noise                                # INDEPENDENT noise per environment

    flat = x_raw.reshape(-1, N_OBS)
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    x_all = (x_raw - mu) / sd
    A_eff = A / sd[:, None]     # standardized mixing: column c is the direction the SVD recovers

    return dict(
        x_all=x_all.astype(np.float32), z_all=z_all.astype(np.float32),
        A=A, A_eff=A_eff, coefs_0=coefs_0, coefs_envs=coefs_envs,
        edges=edges, alphas=alphas, targets=list(range(D_LATENT)),
        rho_0=companion_rho(coefs_0), rho_envs=[companion_rho(cc) for cc in coefs_envs],
        rho=float(rho), snr=float(snr), hrf=bool(hrf), recursive=bool(recursive),
        pairing_mode="recursive" if recursive else "counterfactual",
    )


# ----------------------------------- data sanity (reused verbatim)

def lagged_cross_cov(z, lag):
    """E[z_t z_{t-lag}^T] over contiguous windows. z: (n_windows, L, d)."""
    a = z[:, lag:, :].reshape(-1, z.shape[2])
    b = z[:, : z.shape[1] - lag, :].reshape(-1, z.shape[2])
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a.T @ b) / a.shape[0]


def paired_corr_per_coord(z_a, z_b):
    """Pearson r between paired z_a[k,t] and z_b[k,t], per latent coordinate."""
    fa = z_a.reshape(-1, z_a.shape[-1])
    fb = z_b.reshape(-1, z_b.shape[-1])
    out = np.zeros(fa.shape[1])
    for k in range(fa.shape[1]):
        a, b = fa[:, k], fb[:, k]
        if a.std() < 1e-12 or b.std() < 1e-12:
            out[k] = 0.0
        else:
            out[k] = float(np.corrcoef(a, b)[0, 1])
    return out


def data_sanity(data, dist, seed, log):
    z_all = data["z_all"]
    z_ref = z_all[0]
    fa = z_ref.reshape(-1, D_LATENT)
    kurt = kurtosis(fa, axis=0, fisher=True, bias=False)

    # per-env: content correlation (should be 1.0 off the target) and the 1-D check
    content_r_min = 1.0
    target_r = []
    oned_l1l2 = []
    lag_diff_min = np.inf
    for c in range(D_LATENT):
        z_c = z_all[1 + c]
        corr = paired_corr_per_coord(z_ref, z_c)
        noncol = [k for k in range(D_LATENT) if k != c]
        content_r_min = min(content_r_min, float(np.min(np.abs(corr[noncol]))))
        target_r.append(float(abs(corr[c])))
        dd = (z_ref - z_c).reshape(-1, D_LATENT)
        prof = np.sqrt((dd ** 2).mean(0) + SQRT_EPS)
        oned_l1l2.append(float(prof.sum() / (np.linalg.norm(prof) + 1e-12)))
        for lag in range(1, MAX_LAG + 1):
            lag_diff_min = min(lag_diff_min,
                               float(np.linalg.norm(lagged_cross_cov(z_ref, lag)
                                                    - lagged_cross_cov(z_c, lag), "fro")))

    log("  --- DATA SANITY ---")
    log(f"  seed {seed}  dist {dist}  pairing counterfactual  ({D_LATENT} intervention envs)")
    log(f"  excess kurtosis of z_ref per coord = [{', '.join(f'{k:.2f}' for k in kurt)}]")
    log(f"    mean {kurt.mean():.3f}  (Laplace => clearly > 0; Gaussian => ~0)")
    log(f"  reference rho = {data['rho_0']:.4f}   env rho range = "
        f"[{min(data['rho_envs']):.4f}, {max(data['rho_envs']):.4f}]")
    log(f"  intervention alpha per env = [{', '.join(f'{a:.2f}' for a in data['alphas'])}]")
    log(f"  content correlation r_min over all envs = {content_r_min:.4f}  (expect ~1.0)")
    log(f"  target correlation r per env = [{', '.join(f'{r:.2f}' for r in target_r)}]")
    log(f"  per-env 1-D check L1/L2 = [{', '.join(f'{v:.3f}' for v in oned_l1l2)}]  "
        f"(expect ~1.0; > {ONED_L1L2_MAX} = not 1-D)")
    log("  edges shifted (lag, target c, source j):")
    for c, (l, cc, j) in enumerate(data["edges"]):
        log(f"    env {c}: lag {l+1}  z{cc} <- z{j}   alpha {data['alphas'][c]:.3f}")

    void = lag_diff_min < LAGGED_COV_WARN
    if void:
        log("  " + "!" * 68)
        log("  WARNING: some env is indistinguishable from the reference. TEST VOID.")
        log("  " + "!" * 68)
    if dist == "laplace" and kurt.mean() < 0.5:
        log(f"  WARNING: Laplace marginal kurtosis low (mean {kurt.mean():.3f}).")
    weak = [(c, round(data["alphas"][c], 3)) for c in range(D_LATENT)
            if data["alphas"][c] < 0.5]
    if weak:
        log("  " + "!" * 68)
        log(f"  WEAK INTERVENTIONS (alpha < 0.5), (coord, alpha): {weak}")
        log("  These coordinates carry a faint paired-difference signal and may not be")
        log("  pinned. Cross-reference the per-coordinate MCC printed after training: if a")
        log("  low aggregate MCC is driven by exactly these coordinates, that is why.")
        log("  " + "!" * 68)
    log("  --- END SANITY ---")

    return dict(kurtosis_mean=float(kurt.mean()), content_r_min=content_r_min,
                target_r=target_r, oned_l1l2=oned_l1l2, alphas=data["alphas"],
                rho_0=float(data["rho_0"]), lag_diff_min=float(lag_diff_min),
                pairing_mode="counterfactual", void=bool(void))


# ----------------------------------- flow model (reused verbatim)

class CouplingLayer(nn.Module):
    """RealNVP affine coupling on a fixed binary mask. Zero-init => starts at identity."""

    def __init__(self, dim, hidden, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        xm = x * self.mask
        s, t = self.net(xm).chunk(2, dim=-1)
        s = torch.tanh(s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        y = xm + (1.0 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(-1)


class FlowEncoder(nn.Module):
    """Pointwise encoder: linear projection N->d, then a RealNVP flow.

    Applied per timepoint (the true mixing is instantaneous). BOTH members of a pair go
    through the SAME encoder — that is what makes the paired difference meaningful.
    """

    def __init__(self, n_obs, d, n_couplings, hidden):
        super().__init__()
        self.proj = nn.Linear(n_obs, d, bias=False)
        self.decoder = nn.Linear(d, n_obs, bias=False)
        masks = []
        for k in range(n_couplings):
            m = torch.zeros(d)
            m[k % 2 :: 2] = 1.0
            masks.append(m)
        self.layers = nn.ModuleList([CouplingLayer(d, hidden, m) for m in masks])

    def encode(self, x):
        """x: (..., n_obs) -> z: (..., d), logdet: (...)"""
        shape = x.shape[:-1]
        h = self.proj(x.reshape(-1, x.shape[-1]))
        logdet = torch.zeros(h.shape[0], device=x.device)
        for layer in self.layers:
            h, ld = layer(h)
            logdet = logdet + ld
        return h.reshape(*shape, -1), logdet.reshape(*shape)

    def reconstruct(self, x):
        return self.decoder(self.proj(x))


# ----------------------------------- latent VAR + objective utils (reused verbatim)

def fit_latent_var(z, max_lag, ridge):
    """Differentiable ridge VAR fit inside a window batch.

    z: (B, L, d) contiguous. Returns B_hat: (d*max_lag, d), predicting z[t] from
    [z[t-1], ..., z[t-max_lag]]. Lagged pairs never cross a window boundary.
    """
    B, L, d = z.shape
    targets = z[:, max_lag:, :].reshape(-1, d)
    preds = [z[:, max_lag - l - 1 : L - l - 1, :] for l in range(max_lag)]
    P = torch.cat(preds, dim=-1).reshape(-1, d * max_lag)
    G = P.T @ P + ridge * P.shape[0] * torch.eye(P.shape[1], device=z.device, dtype=z.dtype)
    return torch.linalg.solve(G, P.T @ targets)


def standardize_coords(z_a, z_b):
    """Unit-variance per latent coordinate, pooled over both environments, IN-GRAPH.

    Anti-gaming guard: scaling coordinate k by c leaves this output EXACTLY unchanged (the
    pooled sd scales by the same c), so no sparsity penalty downstream can be reduced by
    shrinking a coordinate.
    """
    pooled = torch.cat([z_a.reshape(-1, z_a.shape[-1]), z_b.reshape(-1, z_b.shape[-1])], 0)
    sd = pooled.std(0, unbiased=False) + 1e-6
    return z_a / sd, z_b / sd


def sparsity_ratio(M):
    """L1/L2 of a tensor: minimized (=1) when one entry carries everything, max sqrt(numel)."""
    return M.abs().sum() / (M.pow(2).sum().sqrt() + 1e-8)


def standardize_multi(z):
    """Unit-variance per coordinate, pooled over ALL environments, IN-GRAPH.

    z: (E, ...) stacked over environments. Anti-gaming guard: scaling coordinate k by a
    constant scales the pooled sd by the same constant, so the standardized value (and every
    sparsity penalty downstream) is exactly invariant to per-coordinate rescaling.
    """
    pooled = z.reshape(-1, z.shape[-1])
    sd = pooled.std(0, unbiased=False) + 1e-6
    return z / sd


def diff_profile(dd):
    """Per-coordinate RMS of a paired difference. dd: (..., d).

    The +SQRT_EPS inside the sqrt is load-bearing: d/dx sqrt(x) -> inf as x -> 0, and the
    penalty drives content-coordinate differences to zero, so without it the objective NaNs
    exactly when it starts working (seen for real in candidate 1). Returns (profile, mag).
    """
    dd = dd.reshape(-1, dd.shape[-1])
    prof = (dd.pow(2).mean(0) + SQRT_EPS).sqrt()
    mag = (dd.pow(2).sum(-1) + SQRT_EPS).sqrt().mean()
    return prof, mag



# ---------------------------------------- Option B objective (reused), now with optional init


def train_multienv_model(data, device, log, init_A=None):
    """Encoder trained under sum_c L1/L2 of the per-env paired difference. EXACT gradient.

    Objective identical to run_multienv_gate.py. The ONLY addition is init_A: when given the
    closed-form mixing estimate (N x d), the linear projection is set to its pseudo-inverse
    and the decoder to init_A, so training STARTS at the closed-form solution (the flow
    couplings are identity at init). Mode (i) = init_A None (random); mode (ii) = init_A given.
    """
    x_all = torch.from_numpy(data["x_all"]).to(device)   # (E, nw, L, N)
    E, n = x_all.shape[0], x_all.shape[1]

    model = FlowEncoder(N_OBS, D_LATENT, FLOW_COUPLINGS, FLOW_HIDDEN).to(device)
    if init_A is not None:
        W = np.linalg.pinv(init_A)                       # (d, N) unmixing
        with torch.no_grad():
            model.proj.weight.copy_(torch.as_tensor(W, dtype=model.proj.weight.dtype,
                                                     device=device))
            model.decoder.weight.copy_(torch.as_tensor(init_A, dtype=model.decoder.weight.dtype,
                                                        device=device))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_STEPS)

    nonfinite_grad_steps = 0
    for step in range(TRAIN_STEPS):
        idx = torch.randint(0, n, (BATCH_WINDOWS,), device=device)   # SAME index for all envs
        xb = x_all[:, idx]                                           # (E, B, L, N)
        z, logdet = model.encode(xb)

        rec = ((model.reconstruct(xb) - xb) ** 2).mean(dim=(1, 2, 3)).sum()
        nll = (z.abs().sum(-1) - logdet).mean() if LAMBDA_PRIOR > 0 else z.new_zeros(())

        zs = standardize_multi(z)
        l_pair = z.new_zeros(())
        for c in range(E - 1):
            prof, _ = diff_profile(zs[0] - zs[1 + c])
            l_pair = l_pair + sparsity_ratio(prof)

        loss = rec + LAMBDA_PRIOR * nll + LAMBDA_PAIR * l_pair

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"step {step}: non-finite loss. rec={rec.item()}, l_pair={l_pair.item()}, "
                f"z finite={bool(torch.isfinite(z).all())}, "
                f"latent sd min={float(z.reshape(-1, D_LATENT).std(0).min())}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(gnorm):
            nonfinite_grad_steps += 1
            if nonfinite_grad_steps == 1:
                log(f"      !! step {step}: non-finite gradient; skipping.")
            opt.zero_grad(set_to_none=True)
            sched.step()
            continue
        opt.step()
        sched.step()

        if step % 1000 == 0:
            log(f"      step {step:5d}  loss {loss.item():9.3f}  rec {rec.item():7.3f}  "
                f"pair_sum {l_pair.item():6.3f}")

    if nonfinite_grad_steps:
        frac = nonfinite_grad_steps / TRAIN_STEPS
        log(f"    WARNING: {nonfinite_grad_steps}/{TRAIN_STEPS} ({frac:.1%}) steps skipped.")
        if frac > 0.05:
            raise RuntimeError(f"{frac:.1%} steps had non-finite gradients; not trustworthy.")

    return model, dict(nonfinite_grad_steps=int(nonfinite_grad_steps))


# ----------------------------------- MSE baseline (reused verbatim)

def train_mse_baseline(data, device, log):
    """Plain pointwise linear autoencoder over ALL environments pooled. Rotationally blind."""
    x = torch.from_numpy(data["x_all"]).to(device).reshape(-1, N_OBS)

    enc = nn.Linear(N_OBS, D_LATENT, bias=False).to(device)
    dec = nn.Linear(D_LATENT, N_OBS, bias=False).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=AE_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=AE_STEPS)

    n = x.shape[0]
    for step in range(AE_STEPS):
        idx = torch.randint(0, n, (BATCH_WINDOWS * WINDOW_L,), device=device)
        xb_ = x[idx]
        rec = ((dec(enc(xb_)) - xb_) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        rec.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            log(f"    step {step:5d}  rec {rec.item():9.6f}")

    class Wrap(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.enc, self.dec = enc, dec

        def encode(self, x):
            shape = x.shape[:-1]
            z = self.enc(x.reshape(-1, x.shape[-1]))
            return z.reshape(*shape, -1), torch.zeros(*shape, device=x.device)

        def reconstruct(self, x):
            return self.dec(self.enc(x))

    return Wrap(enc, dec), dict(nonfinite_grad_steps=0)


# ----------------------------------- metrics (reused verbatim)

def corr_matrix(z_true, z_hat):
    d = z_true.shape[1]
    C = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            r = spearmanr(z_true[:, i], z_hat[:, j]).statistic
            C[i, j] = 0.0 if not np.isfinite(r) else abs(r)
    return C


def mcc_and_matching(z_true, z_hat):
    """Hungarian-matched mean |Spearman|. Returns (mcc, true_of_hat, per_true_coord)."""
    C = corr_matrix(z_true, z_hat)
    row, col = linear_sum_assignment(-C)
    mcc = float(C[row, col].mean())
    true_of_hat = {int(c): int(r) for r, c in zip(row, col)}
    per_true = np.zeros(z_true.shape[1])
    for r, c in zip(row, col):
        per_true[r] = C[r, c]
    return mcc, true_of_hat, per_true


def subspace_r2(z_true, z_hat):
    lr = LinearRegression().fit(z_hat, z_true)
    return float(r2_score(z_true, lr.predict(z_hat), multioutput="uniform_average"))


def recon_r(x, x_hat):
    return float(np.corrcoef(x.ravel(), x_hat.ravel())[0, 1])


def gini(v):
    """0 = uniform, ->1 = concentrated on one entry."""
    v = np.sort(np.abs(np.asarray(v, dtype=np.float64)))
    n = len(v)
    if v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def permute_vector(v, true_of_hat, d):
    """Remap a per-coordinate profile from recovered into ground-truth coordinate order."""
    perm = np.array([true_of_hat[i] for i in range(d)])
    out = np.zeros_like(v)
    out[perm] = v
    return out


def var_fit_quality(za, zb):
    """One-step-ahead prediction R2 of the fitted latent VAR, per environment, on a
    held-out tail of the windows. Are these latents a FAITHFUL dynamical model, or did the
    encoder degrade the fit to buy sparsity? za, zb: (n_win, L, d) torch.

    A VAR is closed under rotation, so a RANDOM rotation of the true latents should keep
    R2 ~ TRUE's R2. If TRAINED's R2 is clearly below TRUE's while its paired difference is
    sparser, the encoder degraded the dynamics to buy sparsity (specification failure), not
    identified the rotation.

    NOTE for PAIRING_MODE="counterfactual": z_b is a one-step counterfactual sequence, not
    an env-B trajectory, so its VAR R2 is only loosely interpretable. z_a's is the one to
    read. Both are reported.
    """
    def r2_env(z):
        n = z.shape[0]
        cut = max(1, int(0.8 * n))
        z_tr, z_te = z[:cut], z[cut:]
        za_s, _ = standardize_coords(z_tr, z_tr)
        B = fit_latent_var(za_s, MAX_LAG, RIDGE)
        L = z_te.shape[1]
        z_te_s, _ = standardize_coords(z_te, z_te)
        tgt = z_te_s[:, MAX_LAG:, :].reshape(-1, D_LATENT)
        preds = [z_te_s[:, MAX_LAG - l - 1 : L - l - 1, :] for l in range(MAX_LAG)]
        P = torch.cat(preds, dim=-1).reshape(-1, D_LATENT * MAX_LAG)
        pred = P @ B
        ss_res = ((tgt - pred) ** 2).sum().item()
        ss_tot = ((tgt - tgt.mean(0)) ** 2).sum().item()
        return 1.0 - ss_res / (ss_tot + 1e-12)
    return r2_env(za), r2_env(zb)


# ----------------------------------- io (reused verbatim)

def atomic_write(path, text_or_obj, is_json):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        if is_json:
            json.dump(text_or_obj, f, indent=2)
        else:
            f.write(text_or_obj)
    os.replace(tmp, path)


def seed_path(condition, seed):
    return os.path.join(SEED_DIR, f"{condition}_seed{seed:02d}.json")



# ------------------------------------------------------------------ closed-form estimator


def closedform_estimate(x_all):
    """Estimate the mixing columns from observed paired differences. x_all: (E, nw, L, N).

    For env c, D_c = x_ref - x_env_c over all paired timepoints, mean-centred over time.
    D_c[t] = A_eff[:,c] * g[t] + (degradation), so its top RIGHT singular vector (the
    N-dimensional direction) estimates A_eff[:,c] up to sign. rank1[c] = s2/s1 is 0 at
    perfect pairing and rises as the pairing degrades. Column c is known by construction
    (env 1+c -> column c); no Hungarian matching. SVD in float64 for the rank-1 read.
    """
    E, nw, L, N = x_all.shape
    d = E - 1
    A_hat = np.zeros((N, d))
    rank1 = np.zeros(d)
    for c in range(d):
        D = (x_all[0] - x_all[1 + c]).reshape(-1, N).astype(np.float64)
        D = D - D.mean(0, keepdims=True)
        _, S, Vt = np.linalg.svd(D, full_matrices=False)
        A_hat[:, c] = Vt[0]
        rank1[c] = S[1] / (S[0] + 1e-30)
    return A_hat, rank1


def column_cos(A_hat, A_eff):
    """Per-column |cos| between estimated and true (standardized) mixing columns."""
    cos = np.zeros(A_hat.shape[1])
    for c in range(A_hat.shape[1]):
        a, b = A_hat[:, c], A_eff[:, c]
        cos[c] = abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return cos


def eval_closedform(A_hat, x_te, z_te):
    """Unmix held-out observations with pinv(A_hat) and score against the true latents."""
    W = np.linalg.pinv(A_hat)                                # (d, N)
    z_hat = x_te.reshape(-1, N_OBS) @ W.T
    z_true = z_te.reshape(-1, D_LATENT)
    m, _, per_true = mcc_and_matching(z_true, z_hat)
    return m, per_true.tolist(), subspace_r2(z_true, z_hat)


def eval_model(model, x_te, z_te, device):
    """Encode held-out observations and score against the true latents."""
    with torch.no_grad():
        z_hat = model.encode(torch.from_numpy(x_te).to(device))[0].reshape(-1, D_LATENT).cpu().numpy()
    z_true = z_te.reshape(-1, D_LATENT)
    m, _, per_true = mcc_and_matching(z_true, z_hat)
    return m, per_true.tolist(), subspace_r2(z_true, z_hat)


# ------------------------------------------------------------------------------- one rung


def run_rung(rung, seed, device, log):
    t0 = time.time()
    data = make_dataset(seed, DIST, rho=rung["rho"], snr=rung["snr"], hrf=rung["hrf"],
                        recursive=rung.get("recursive", False))

    n_test = int(TEST_FRAC * N_WINDOWS)
    tr = slice(0, N_WINDOWS - n_test)
    te = slice(N_WINDOWS - n_test, N_WINDOWS)
    x_tr, x_te = data["x_all"][:, tr], data["x_all"][:, te]
    z_tr_true, z_te_true = data["z_all"][:, tr], data["z_all"][:, te]

    # --- closed form (mode iii): estimate A from train, evaluate on held-out ---
    A_hat, rank1 = closedform_estimate(x_tr)
    cos = column_cos(A_hat, data["A_eff"])
    cf_mcc, cf_percoord, cf_r2 = eval_closedform(A_hat, x_te, z_te_true)

    # --- MSE floor on identical data ---
    mse_model, _ = train_mse_baseline({"x_all": x_tr}, device, log)
    mse_mcc, _, _ = eval_model(mse_model, x_te, z_te_true, device)

    res = dict(
        rung=rung["name"], rho=rung["rho"], snr=rung["snr"], hrf=rung["hrf"],
        recursive=bool(rung.get("recursive", False)), seed=seed,
        cos_mean=float(cos.mean()), cos_min=float(cos.min()), cos=cos.tolist(),
        rank1_mean=float(rank1.mean()), rank1_max=float(rank1.max()), rank1=rank1.tolist(),
        cf_mcc=cf_mcc, cf_subspace_r2=cf_r2, cf_per_coord_mcc=cf_percoord,
        mse_mcc=mse_mcc, alphas=data["alphas"], trained=bool(rung["train"]),
    )

    if rung["train"]:
        log(f"       [mode i] random-init training")
        m_rand, _ = train_multienv_model({"x_all": x_tr}, device, log, init_A=None)
        rand_mcc, rand_pc, rand_r2 = eval_model(m_rand, x_te, z_te_true, device)
        log(f"       [mode ii] closed-form-init training")
        m_cf, _ = train_multienv_model({"x_all": x_tr}, device, log, init_A=A_hat)
        cfinit_mcc, cfinit_pc, cfinit_r2 = eval_model(m_cf, x_te, z_te_true, device)
        res.update(rand_mcc=rand_mcc, rand_per_coord_mcc=rand_pc, rand_subspace_r2=rand_r2,
                   cfinit_mcc=cfinit_mcc, cfinit_per_coord_mcc=cfinit_pc,
                   cfinit_subspace_r2=cfinit_r2)

    res["seconds"] = time.time() - t0
    return res


# -------------------------------------------------------------------------- figures


def _agg(results, rungs, key):
    """mean, std of `key` across seeds, per rung (in rung order). Missing -> nan."""
    out = []
    for r in rungs:
        vals = [x[key] for x in results if x["rung"] == r["name"] and key in x]
        if vals:
            out.append((float(np.mean(vals)), float(np.std(vals))))
        else:
            out.append((float("nan"), float("nan")))
    return out


def fig_rho_ladder_mcc(results, rho_rungs, recursive_rungs=None):
    if not any(x["rung"] == r["name"] for r in rho_rungs for x in results):
        return None
    rhos = [r["rho"] for r in rho_rungs]
    cf = _agg(results, rho_rungs, "cf_mcc")
    cfi = _agg(results, rho_rungs, "cfinit_mcc")
    rnd = _agg(results, rho_rungs, "rand_mcc")
    mse = _agg(results, rho_rungs, "mse_mcc")
    mse_floor = np.nanmean([m[0] for m in mse])

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for series, lab, col in ((cf, "closed-form (no training)", "#55A868"),
                             (cfi, "closed-form-init, trained", "#4C72B0"),
                             (rnd, "random-init, trained", "#C44E52")):
        mean = [s[0] for s in series]
        sd = [s[1] for s in series]
        ax.errorbar(rhos, mean, yerr=sd, marker="o", capsize=3, label=lab, color=col)
    ax.axhline(mse_floor, color="k", ls="--", lw=1, label=f"MSE floor ({mse_floor:.2f})")

    # the transfer-relevant rung: star markers at x=0, distinct from the additive RHO=0 circle
    if recursive_rungs:
        rec = _agg(results, recursive_rungs, "cf_mcc")
        reci = _agg(results, recursive_rungs, "cfinit_mcc")
        recr = _agg(results, recursive_rungs, "rand_mcc")
        for (series, col) in ((rec, "#55A868"), (reci, "#4C72B0"), (recr, "#C44E52")):
            if series and np.isfinite(series[0][0]):
                ax.errorbar([0.0], [series[0][0]], yerr=[series[0][1]], marker="*",
                            markersize=16, capsize=3, color=col, markeredgecolor="k",
                            linestyle="none")
        if rec and np.isfinite(rec[0][0]):
            ax.annotate("independent\n(recursive)", xy=(0.0, rec[0][0]),
                        xytext=(0.12, min(0.95, rec[0][0] + 0.18)), fontsize=8,
                        ha="left", arrowprops=dict(arrowstyle="->", lw=0.8))
        ax.plot([], [], marker="*", markersize=12, color="0.4", markeredgecolor="k",
                linestyle="none", label="independent_recursive (transfer)")

    ax.set_xlabel("RHO  (pairing quality; 1 = perfect counterfactual, 0 = independent runs)")
    ax.set_ylabel("MCC vs true latents")
    ax.set_title("Recovery vs pairing quality\nRHO=1 is synthetic-only; stars = genuine "
                 "independent pairing (real task-fMRI)")
    ax.invert_xaxis()   # perfect pairing on the left, realistic on the right
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_rho_ladder_mcc.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_rho_ladder_rank1(results, rho_rungs, recursive_rungs=None):
    if not any(x["rung"] == r["name"] for r in rho_rungs for x in results):
        return None
    rhos = [r["rho"] for r in rho_rungs]
    r1 = _agg(results, rho_rungs, "rank1_mean")
    cos = _agg(results, rho_rungs, "cos_mean")

    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    ax1.errorbar(rhos, [s[0] for s in r1], yerr=[s[1] for s in r1], marker="o", capsize=3,
                 color="#C44E52", label="rank-1 ratio (2nd/1st singular value)")
    ax1.set_xlabel("RHO (pairing quality)")
    ax1.set_ylabel("rank-1 ratio  (0 = pure rank-1, estimator trustworthy)", color="#C44E52")
    ax1.tick_params(axis="y", labelcolor="#C44E52")
    ax1.invert_xaxis()
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.errorbar(rhos, [s[0] for s in cos], yerr=[s[1] for s in cos], marker="s", capsize=3,
                 color="#55A868", label="mean |cos| with true column")
    ax2.set_ylabel("mean |cos| with true mixing column", color="#55A868")
    ax2.tick_params(axis="y", labelcolor="#55A868")

    # recursive markers at x=0: rank1 (star, left axis) and cos (star, right axis)
    if recursive_rungs:
        rec_r1 = _agg(results, recursive_rungs, "rank1_mean")
        rec_cos = _agg(results, recursive_rungs, "cos_mean")
        if rec_r1 and np.isfinite(rec_r1[0][0]):
            ax1.plot([0.0], [rec_r1[0][0]], marker="*", markersize=16, color="#C44E52",
                     markeredgecolor="k", linestyle="none")
        if rec_cos and np.isfinite(rec_cos[0][0]):
            ax2.plot([0.0], [rec_cos[0][0]], marker="*", markersize=16, color="#55A868",
                     markeredgecolor="k", linestyle="none")
        ax1.plot([], [], marker="*", markersize=12, color="0.4", markeredgecolor="k",
                 linestyle="none", label="independent_recursive")

    ax1.legend(fontsize=8, loc="center left")
    ax1.set_title("Estimator trustworthiness vs pairing quality\n"
                  "stars = genuine independent (recursive) pairing")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_rho_ladder_rank1.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ------------------------------------------------------------------------- reporting


def write_report(results, rho_rungs, snr_rungs, recursive_rungs, other_rungs, figs, meta):
    L = []
    L.append("# Closed-form mixing recovery under a pairing-degradation ladder\n")
    L.append(f"- torch {meta['torch']}, numpy {meta['numpy']}, scipy {meta['scipy']}, "
             f"sklearn {meta['sklearn']}")
    L.append(f"- device: {meta['device']} ({meta['gpu']})")
    L.append(f"- d={D_LATENT}, N={N_OBS}, {D_LATENT} intervention envs + 1 reference, "
             f"windows={N_WINDOWS}x{WINDOW_L}\n")
    L.append("> **RHO=1 is synthetic-only** — it requires the same-innovation counterfactual "
             "pairing that real task-fMRI never provides. The scientific question is WHERE ON "
             "THE RHO LADDER recovery collapses, since real data sits near RHO=0.\n")

    def row(rung, extra_train):
        rs = [x for x in results if x["rung"] == rung["name"]]
        if not rs:
            return None
        def mv(k):
            v = [x[k] for x in rs if k in x and np.isfinite(x[k])]
            return (np.mean(v), np.std(v)) if v else (float("nan"), 0.0)
        cells = [f"{mv('cos_mean')[0]:.4f}", f"{mv('cos_min')[0]:.4f}",
                 f"{mv('rank1_mean')[0]:.2e}", f"{mv('rank1_max')[0]:.2e}",
                 f"{mv('cf_mcc')[0]:.3f}", f"{mv('mse_mcc')[0]:.3f}"]
        if extra_train:
            cells += [f"{mv('cfinit_mcc')[0]:.3f}", f"{mv('rand_mcc')[0]:.3f}"]
        return cells

    L.append("## RHO ladder — the controlled interpolation\n")
    L.append("The additive RHO knob injects the innovation mismatch once and lets it decay; it "
             "is the clean interpolation between the verified RHO=1 ceiling and the independent "
             "end, but it does NOT let the mismatch compound through the VAR recursion, so it is "
             "**optimistic at low RHO**. Read it together with the independent_recursive rung "
             "below.\n")
    L.append("| RHO | cos mean | cos min | rank1 mean | rank1 max | CF MCC | MSE floor | "
             "cf-init MCC | rand-init MCC |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rho_rungs:
        c = row(r, True)
        if c:
            L.append(f"| {r['rho']:.2f} | " + " | ".join(c) + " |")
    L.append("")

    L.append("## INDEPENDENT_RECURSIVE — the transfer-relevant rung\n")
    L.append("Genuine two-independent-sessions construction: env_c re-simulated from scratch "
             "through its own shifted VAR with fresh innovations, so the mismatch COMPOUNDS "
             "through the recursion. This is the realistic task-fMRI analogue (same subject, "
             "different run) and the number to quote for transfer claims. No noise, no HRF, so "
             "it is directly comparable to the additive RHO=0 row above — the gap between them "
             "is exactly how optimistic the additive ladder is at the independent end.\n")
    L.append("| construction | cos mean | cos min | rank1 mean | rank1 max | CF MCC | "
             "MSE floor | cf-init MCC | rand-init MCC |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in recursive_rungs:
        c = row(r, True)
        if c:
            L.append(f"| {r['name']} | " + " | ".join(c) + " |")
    # the additive RHO=0 row repeated here for a side-by-side read
    add0 = next((r for r in rho_rungs if r["rho"] == 0.0), None)
    if add0:
        c = row(add0, True)
        if c:
            L.append("| additive RHO=0 (optimistic) | " + " | ".join(c) + " |")
    L.append("")

    L.append("## Observation-noise ladder (RHO=1, HRF off)\n")
    L.append("| SNR | cos mean | cos min | rank1 mean | rank1 max | CF MCC | MSE floor |")
    L.append("|---|---|---|---|---|---|---|")
    for r in snr_rungs:
        c = row(r, False)
        if c:
            L.append(f"| {r['snr']:.0f} | " + " | ".join(c) + " |")
    L.append("")

    if other_rungs:
        L.append("## Other rungs (HRF, combined-realistic)\n")
        L.append("| rung | RHO | SNR | HRF | cos mean | rank1 mean | CF MCC | MSE floor |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in other_rungs:
            rs = [x for x in results if x["rung"] == r["name"]]
            if not rs:
                continue
            def mv(k):
                v = [x[k] for x in rs if k in x and np.isfinite(x[k])]
                return np.mean(v) if v else float("nan")
            L.append(f"| {r['name']} | {r['rho']:.2f} | {r['snr']:.0f} | {r['hrf']} "
                     f"| {mv('cos_mean'):.4f} | {mv('rank1_mean'):.2e} | {mv('cf_mcc'):.3f} "
                     f"| {mv('mse_mcc'):.3f} |")
        L.append("")

    # --- verdict ---
    L.append("## Verdict\n")
    def rung_mv(name, k):
        v = [x[k] for x in results if x["rung"] == name and k in x and np.isfinite(x[k])]
        return np.mean(v) if v else float("nan")

    mse_floor = np.nanmean([rung_mv(r["name"], "mse_mcc") for r in rho_rungs])
    collapse_rho = None
    for r in rho_rungs:                          # RHO_LIST is descending, so first below floor
        cf = rung_mv(r["name"], "cf_mcc")
        if np.isfinite(cf) and np.isfinite(mse_floor) and cf <= mse_floor + 0.05:
            collapse_rho = r["rho"]
            break
    if collapse_rho is not None:
        L.append(f"- **Additive ladder: closed-form recovery collapses to the MSE floor at "
                 f"RHO ~ {collapse_rho:.2f}** (CF MCC within 0.05 of the {mse_floor:.3f} floor). "
                 "This is the controlled interpolation and is OPTIMISTIC at the independent end.")
    else:
        L.append(f"- **Additive ladder: closed-form MCC stays above the MSE floor "
                 f"({mse_floor:.3f}) across the entire RHO sweep, including RHO=0.** Optimistic; "
                 "read the recursive result before concluding anything about transfer.")

    # --- the transfer-relevant verdict: the independent_recursive rung ---
    rec_name = recursive_rungs[0]["name"] if recursive_rungs else None
    rec_cf = rung_mv(rec_name, "cf_mcc") if rec_name else float("nan")
    rec_cos = rung_mv(rec_name, "cos_mean") if rec_name else float("nan")
    rec_rank1 = rung_mv(rec_name, "rank1_mean") if rec_name else float("nan")
    rec_cfinit = rung_mv(rec_name, "cfinit_mcc") if rec_name else float("nan")
    rec_rand = rung_mv(rec_name, "rand_mcc") if rec_name else float("nan")
    add0_cf = rung_mv(add0["name"], "cf_mcc") if add0 else float("nan")
    if np.isfinite(rec_cf):
        clears = rec_cf > mse_floor + 0.05
        L.append(
            f"- **INDEPENDENT_RECURSIVE (the transfer number): closed-form MCC {rec_cf:.3f} "
            f"vs MSE floor {mse_floor:.3f}** ({'clears' if clears else 'AT/BELOW'} the floor); "
            f"mean |cos| {rec_cos:.3f}, rank-1 ratio {rec_rank1:.3f}. "
            f"cf-init trained {rec_cfinit:.3f}, random-init trained {rec_rand:.3f}. "
            f"For comparison the additive RHO=0 closed-form MCC is {add0_cf:.3f} — the gap "
            "is how much the additive ladder overstates recovery at the independent end.")
        if not clears:
            L.append("  - **At genuine independent pairing the closed form does not beat the "
                     "rotationally-blind floor. This is the honest transfer verdict: the "
                     "method does not carry to real task-fMRI.**")
        else:
            L.append("  - The closed form clears the floor even under genuine independent "
                     "pairing. Surprising for transfer — scrutinize the rank-1 ratio (a high "
                     "value means the recovered column is not a trustworthy mixing estimate).")

    cf_top = rung_mv(rho_rungs[0]["name"], "cf_mcc")
    rand_top = rung_mv(rho_rungs[0]["name"], "rand_mcc")
    cfinit_top = rung_mv(rho_rungs[0]["name"], "cfinit_mcc")
    L.append(f"- At RHO=1: closed-form MCC {cf_top:.3f}, random-init trained {rand_top:.3f}, "
             f"closed-form-init trained {cfinit_top:.3f}.")
    if np.isfinite(rand_top) and np.isfinite(cfinit_top):
        if cfinit_top > rand_top + 0.05 and cfinit_top > 0.9:
            L.append("  - Closed-form init clears the local minimum that random init fell into: "
                     "**the Option B plateau was an initialization problem.**")
        elif cfinit_top <= rand_top + 0.05:
            L.append("  - Closed-form init does NOT beat random init: the objective actively "
                     "pulls away from the closed-form optimum, so the plateau is a landscape "
                     "problem, not just initialization.")
    L.append("")

    L.append("## Caveats\n")
    L.append("1. **RHO=1 is not a real-data result.** It needs same-innovation counterfactual "
             "pairs. The transfer-relevant rows are the low-RHO end.")
    L.append("2. **The HRF is linear and per-coordinate**, so it does not rotate the mixing: "
             "it leaves |cos| and rank-1 intact at RHO=1 and only blurs the latent (lowering "
             "MCC via imperfect deconvolution). Do not read the HRF MCC drop as a mixing-"
             "recovery failure — the cos column shows mixing recovery is untouched.")
    L.append("3. **Ground truth is the NEURAL latent** (pre-HRF). Under the HRF toggle the "
             "recoverable quantity is the convolved latent, so MCC understates mixing recovery; "
             "|cos| is the clean mixing metric.")
    L.append("4. **SNR is amplitude** (noise std = signal std / SNR), independent per env.\n")

    if figs:
        L.append("## Figures\n")
        for f in figs:
            L.append(f"- `{os.path.basename(f)}`")
        L.append("")

    atomic_write(os.path.join(OUT_DIR, "results.md"), "\n".join(L), is_json=False)


# ---------------------------------------------------------------------- oracle test


def run_oracle(log, device):
    """Scaffold checks 1-5b at RHO=1 (default make_dataset), plus the closed-form sanity gate.

    Checks 0-5b are carried over from run_multienv_gate.py: simulator integrity, VAR-fit
    convention, permute round-trips, end-to-end unmixing, pairing reality, and 1-D-ness. The
    NEW gate asserts the closed-form estimator recovers every mixing column at perfect pairing
    (|cos| ~ 1, rank-1 ~ 0, MCC ~ 1); if it does not, a sign/column-order convention is wrong.
    """
    log("=" * 72)
    log("ORACLE HARNESS TEST (no encoder training)")
    log("=" * 72)
    fails = []

    for seed in range(max(N_SEEDS, ORACLE_SEEDS)):
        try:
            make_dataset(seed, DIST)          # RHO=1 default; raises on any integrity break
        except RuntimeError as e:
            log(f"[FAIL] seed {seed}: simulator integrity — {e}")
            fails.append(f"simulator integrity (seed {seed}): {e}")
    if not fails:
        log(f"[ok  ] simulator integrity over {max(N_SEEDS, ORACLE_SEEDS)} seeds")

    # 1. VAR-fit convention on the reference's true latents.
    for seed in range(ORACLE_SEEDS):
        data = make_dataset(seed, DIST)
        B = fit_latent_var(torch.from_numpy(data["z_all"][0]).double(), MAX_LAG, 1e-8).numpy()
        true_B = np.zeros_like(B)
        for l in range(MAX_LAG):
            for i in range(D_LATENT):
                for j in range(D_LATENT):
                    true_B[l * D_LATENT + j, i] = data["coefs_0"][l][i, j]
        err = float(np.abs(B - true_B).max())
        ok = err < 0.05
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: VAR fit, max|B_hat - B_true| = {err:.4f}")
        if not ok:
            fails.append(f"VAR fit convention (seed {seed}, err {err:.4f})")

    log("[skip] check 2 (mechanism-difference localization) — n/a under the counterfactual.")

    # 3. permute_vector round-trip
    rng = np.random.default_rng(0)
    v_true = rng.random(D_LATENT)
    perm = rng.permutation(D_LATENT)
    v_rec = np.array([v_true[perm[i]] for i in range(D_LATENT)])
    err = float(np.abs(permute_vector(v_rec, {i: int(perm[i]) for i in range(D_LATENT)},
                                      D_LATENT) - v_true).max())
    ok = err < 1e-12
    log(f"[{'ok  ' if ok else 'FAIL'}] permute_vector round-trip, max err {err:.2e}")
    if not ok:
        fails.append(f"permute_vector ({err:.2e})")

    # 4. end to end: oracle unmixing (true A) through the MCC readout, over all envs.
    for seed in range(ORACLE_SEEDS):
        data = make_dataset(seed, DIST)
        E = D_LATENT + 1
        z_all = data["z_all"]
        flat = np.concatenate([(z_all[e].reshape(-1, D_LATENT) @ data["A"].T) for e in range(E)], 0)
        A_eff = data["A"] / (flat.std(0) + 1e-8)[:, None]
        perm = np.random.default_rng(seed).permutation(D_LATENT)
        W = np.linalg.pinv(A_eff)[perm, :]
        x_all = data["x_all"]
        z_hat = x_all.reshape(-1, N_OBS) @ W.T
        m, _, _ = mcc_and_matching(z_all.reshape(-1, D_LATENT), z_hat)
        ok = m > 0.95
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: ORACLE end to end, MCC {m:.4f}")
        if not ok:
            fails.append(f"oracle end to end (seed {seed}, MCC {m:.3f})")

    # 5. pairing reality (RHO=1): content shared, shuffling destroys it.
    log("-" * 72)
    log("CHECK 5 — IS THE PAIRING REAL AT RHO=1?")
    for seed in range(ORACLE_SEEDS):
        data = make_dataset(seed, DIST)
        z_all = data["z_all"]
        c_un_all, c_sh_all = [], []
        for c in range(D_LATENT):
            corr = paired_corr_per_coord(z_all[0], z_all[1 + c])
            noncol = [k for k in range(D_LATENT) if k != c]
            c_un_all.append(float(np.min(np.abs(corr[noncol]))))
            srng = np.random.default_rng(777 + seed * 31 + c)
            fb = z_all[1 + c].reshape(-1, D_LATENT)
            shuf = srng.permutation(fb.shape[0])
            c_sh_all.append(float(np.max(np.abs(
                paired_corr_per_coord(z_all[0], fb[shuf].reshape(z_all[1 + c].shape))))))
        c_un, c_sh = float(np.min(c_un_all)), float(np.max(c_sh_all))
        ok = c_un >= PAIR_CORR_CONTENT_MIN and c_sh <= PAIR_CORR_SHUFFLED_MAX
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: content r_min={c_un:.4f}, "
            f"shuffled r_max={c_sh:.3f}")
        if c_un < PAIR_CORR_CONTENT_MIN:
            fails.append(f"pairing broken (seed {seed}): content r={c_un:.3f}")
        if c_sh > PAIR_CORR_SHUFFLED_MAX:
            fails.append(f"pairing fake (seed {seed}): shuffled r={c_sh:.3f}")

    # 5b. each paired difference 1-D on its target coordinate.
    log("-" * 72)
    log("CHECK 5b — IS EACH PAIRED DIFFERENCE 1-D AT RHO=1?")
    for seed in range(ORACLE_SEEDS):
        data = make_dataset(seed, DIST)
        z_all = data["z_all"]
        bad = []
        for c in range(D_LATENT):
            dd = (z_all[0] - z_all[1 + c]).reshape(-1, D_LATENT)
            prof = np.sqrt((dd ** 2).mean(0) + SQRT_EPS)
            l1l2 = float(prof.sum() / (np.linalg.norm(prof) + 1e-12))
            if l1l2 > ONED_L1L2_MAX or int(np.argmax(prof)) != c:
                bad.append((c, round(l1l2, 3), int(np.argmax(prof))))
        ok = not bad
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: 1-D check, offenders {bad}")
        if bad:
            fails.append(f"not 1-D (seed {seed}): {bad}")

    # NEW — CLOSED-FORM SANITY GATE at RHO=1 (no noise, no HRF).
    log("-" * 72)
    log("CHECK CF — CLOSED-FORM ESTIMATOR SANITY GATE (RHO=1, no noise, no HRF)")
    for seed in range(ORACLE_SEEDS):
        data = make_dataset(seed, DIST)
        A_hat, rank1 = closedform_estimate(data["x_all"])
        cos = column_cos(A_hat, data["A_eff"])
        m, _, _ = eval_closedform(A_hat, data["x_all"], data["z_all"])
        bad_cols = [(c, round(float(cos[c]), 4), float(f"{rank1[c]:.1e}"))
                    for c in range(D_LATENT)
                    if cos[c] < CF_COS_MIN or rank1[c] > CF_RANK1_MAX]
        ok = (not bad_cols) and m > CF_MCC_MIN
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: min|cos| {cos.min():.6f} "
            f"(>= {CF_COS_MIN}), max rank1 {rank1.max():.2e} (<= {CF_RANK1_MAX}), MCC {m:.4f}")
        if bad_cols:
            fails.append(f"closed-form columns failed (seed {seed}): (col, |cos|, rank1) "
                         f"{bad_cols}. A sign/column-order convention is wrong.")
        if m <= CF_MCC_MIN:
            fails.append(f"closed-form MCC {m:.3f} <= {CF_MCC_MIN} at RHO=1 (seed {seed}).")

    log("=" * 72)
    if fails:
        log("ORACLE TEST FAILED. Do not trust any ladder numbers until this is fixed:")
        for f in fails:
            log("  - " + f)
        return False
    log("ORACLE TEST PASSED — simulator, pairing, 1-D structure, and the closed-form gate hold.")
    return True


# ------------------------------------------------------------------------------ main


def build_rungs():
    rho_rungs = [dict(name=f"rho{r:.2f}", rho=r, snr=np.inf, hrf=False, train=True)
                 for r in RHO_LIST]
    snr_rungs = [dict(name=f"snr{s:.0f}", rho=1.0, snr=s, hrf=False, train=False)
                 for s in SNR_LIST]
    # The transfer-relevant rung: genuine two-independent-sessions re-simulation. No noise/HRF
    # so it is directly comparable to the additive RHO=0 point and isolates the recursion.
    recursive_rungs = [dict(name="independent_recursive", rho=0.0, snr=np.inf, hrf=False,
                            recursive=True, train=True)]
    other_rungs = [dict(name="hrf", rho=1.0, snr=np.inf, hrf=True, train=False),
                   dict(name="realistic", rho=0.0, snr=5.0, hrf=True, train=True)]
    return rho_rungs, snr_rungs, recursive_rungs, other_rungs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true", help="self-test + closed-form gate; run first")
    ap.add_argument("--smoke", action="store_true", help="oracle, then RHO 1.0 and 0.5, 1 seed")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    os.makedirs(SEED_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "n/a"

    def log(msg):
        print(msg, flush=True)

    log("=" * 72)
    log("closed-form mixing recovery — pairing-degradation ladder")
    log(f"python  {platform.python_version()}  ({sys.platform})")
    log(f"torch   {torch.__version__}   cuda_available={torch.cuda.is_available()}")
    log(f"numpy   {np.__version__}   scipy {scipy.__version__}   sklearn {sklearn.__version__}")
    log(f"device  {device}   gpu: {gpu}")
    log("=" * 72)

    meta = dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, device=device, gpu=gpu)

    if args.oracle or args.smoke:
        if not run_oracle(log, device):
            sys.exit(1)
        log("")
        if args.oracle:
            return

    rho_rungs, snr_rungs, recursive_rungs, other_rungs = build_rungs()
    n_seeds = args.seeds
    if args.smoke:
        rho_rungs = [r for r in rho_rungs if r["rho"] in (1.0, 0.5)]
        if not any(r["rho"] == 0.5 for r in rho_rungs):
            rho_rungs.append(dict(name="rho0.50", rho=0.5, snr=np.inf, hrf=False, train=True))
        snr_rungs, other_rungs = [], []      # keep recursive_rungs: it exercises the new path
        n_seeds = 1
        log("SMOKE MODE: RHO 1.0 and 0.5 + independent_recursive, 1 seed\n")

    all_rungs = rho_rungs + snr_rungs + recursive_rungs + other_rungs
    results = []
    for rung in all_rungs:
        for seed in range(n_seeds):
            p = os.path.join(SEED_DIR, f"{rung['name']}_seed{seed:02d}.json")
            if os.path.exists(p):
                with open(p) as f:
                    results.append(json.load(f))
                log(f"[skip] {rung['name']} seed {seed:02d}")
                continue
            log(f"[run ] {rung['name']} (rho={rung['rho']}, snr={rung['snr']}, hrf={rung['hrf']}) "
                f"seed {seed:02d}")
            res = run_rung(rung, seed, device, log)
            atomic_write(p, res, is_json=True)
            results.append(res)
            extra = (f"  cf-init {res['cfinit_mcc']:.3f}  rand {res['rand_mcc']:.3f}"
                     if res.get("trained") else "")
            log(f"       cos {res['cos_mean']:.4f}  rank1 {res['rank1_mean']:.2e}  "
                f"CF-MCC {res['cf_mcc']:.3f}  MSE {res['mse_mcc']:.3f}{extra}  "
                f"({res['seconds']:.0f}s)")

    figs = [f for f in (fig_rho_ladder_mcc(results, rho_rungs, recursive_rungs),
                        fig_rho_ladder_rank1(results, rho_rungs, recursive_rungs)) if f]
    write_report(results, rho_rungs, snr_rungs, recursive_rungs, other_rungs, figs, meta)

    log("")
    log("=" * 72)
    if args.smoke:
        log("SMOKE MODE — 2 RHO points, 1 seed. Not the ladder.")
    log(f"wrote {os.path.join(OUT_DIR, 'results.md')}")


if __name__ == "__main__":
    main()
