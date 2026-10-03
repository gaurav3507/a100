#!/usr/bin/env python
"""
run_nonlinear_mixing.py — does ANY of this survive when the mixing is NONLINEAR?

WHY THIS FILE EXISTS
    Every experiment so far used LINEAR mixing (x = z @ A.T). But causal representation
    learning is DEFINED by recovering latents through an unknown NONLINEAR mixing; with
    linear mixing the problem is linear ICA — solved and crowded — and the closed-form
    estimator's entire basis ("the observed paired difference lies along column c of A")
    exists ONLY because A is a matrix. This file removes that crutch and asks what, if
    anything, survives.

    It reuses run_closedform_ladder.py WHOLESALE. The one substantive change is the mixing.
    Everything else — simulator, environment construction, RHO/pairing ladder, the
    independent_recursive rung, the closed-form estimator, the FlowEncoder, the cf-init and
    random-init training modes, the metrics, the oracle checks, resumability, reporting —
    is byte-identical to the linear file, so the numbers are directly comparable.

THE MIXING  (config switch MIXING_MODE in {"linear", "nonlinear"})
    linear    : x = A @ z. The control / reproduction rung; make_mixing unchanged.
    nonlinear : x = A @ f(z), where f: R^d -> R^d is a fixed-per-seed invertible smooth
                nonlinear map (the UNKNOWN ground-truth mixing, not learned) and A is the
                same linear d->N projection. f is a residual MLP with LeakyReLU(0.2):
                    f(z) = z + s * h(z),   h(z) = W3 @ leaky(W2 @ leaky(W1 @ z))
                Each W is square and well-conditioned (condition number <= NONLIN_COND_MAX,
                resampled otherwise) and scaled so the spectral norm of the h-Jacobian stays
                below NONLIN_LIP < 1, which guarantees f = I + s*h is invertible for every
                s in [0, 1] (I + M is nonsingular when ||M|| < 1). NONLIN_STRENGTH = s scales
                the nonlinearity:
                    s = 0.0  -> f = identity EXACTLY -> x = A @ z -> the linear file, bit for
                               bit (the mandatory control: every number must reproduce it).
                    s = 0.25, 0.5, 1.0 -> a near-linear-to-strongly-nonlinear ladder.
    Invertibility is asserted numerically: over the sampled latent points, det(J_f) must not
    change sign and must stay bounded away from 0. A non-invertible mixing makes the ground
    truth unrecoverable in principle and voids the run.

WHAT IS REPORTED, at each NONLIN_STRENGTH and at RHO=1 and independent_recursive
    1. CLOSED-FORM estimator: per-env rank-1 ratio, per-column |cos| with the true (linear,
       s=0) mixing direction A_eff[:,c] — the object the estimator ASSUMES exists — and
       element-wise MCC. EXPECTED to collapse as s rises, because under a nonlinear f the
       paired difference no longer lies along a fixed column. We quantify where it breaks.
    2. FLOW-ENCODER training, random-init AND closed-form-init, under the Option B paired
       objective. This is the first setting where the flow has a real job: under linear
       mixing the linear projection alone sufficed and the flow was dead weight. MCC is
       reported against the MSE-baseline floor MEASURED ON THE SAME nonlinear data per rung
       (the floor is re-measured every rung, never assumed — it differs from the linear case).
    3. The rotation-search question, ADAPTED and WEAKER. Under nonlinear mixing the ambiguity
       is no longer a rotation, so the linear rotation search does not apply. Instead we
       report whether the paired objective at the TRUE latents is lower than at latents from
       random invertible perturbations of the true unmixing. This is a WEAKER check: it
       samples a handful of alternatives rather than optimizing over the whole ambiguity
       group, so it can fail to find a better solution that exists. A pass is suggestive, not
       a proof; a fail (truth beaten) is still a genuine disproof.

CONTROLS AND EXPECTED FAILURES
    At s = 0.0 every number MUST reproduce the linear results (closed-form MCC ~ 1.0 at
    RHO=1, |cos| ~ 1.0, rank-1 ~ 1e-8). If it does not, the refactor broke something and the
    oracle aborts naming the check. At s = 1.0 the closed-form sanity gate is EXPECTED to
    fail — that is the result, not a bug — so the oracle reports it as an expected failure
    and does NOT abort. The code marks the distinction explicitly so a genuine bug is still
    caught.

RUN
    python run_nonlinear_mixing.py --oracle   # checks at s=0 (must pass) and s=1 (gate exp-fail)
    python run_nonlinear_mixing.py --smoke    # s in {0,1}, RHO=1, 1 seed, closed-form + cf-init
    python run_nonlinear_mixing.py            # full: s x {RHO=1, independent_recursive}, 6 seeds

HEADLINE VERDICT
    (a) at what NONLIN_STRENGTH the closed-form estimator collapses, and
    (b) whether the flow encoder under the paired objective clears the re-measured MSE floor
        at s = 1.0 — i.e. whether ANY of this works in the actual (nonlinear) CRL setting.

IS THE PAIRING DOING ANY WORK? (two baselines added on top of the reused file)
    Recovering latent directions from BETWEEN-CONDITION covariance differences is long
    established in neuroimaging: Common Spatial Subspace Decomposition (Fu et al., Neuroimage
    1999) decomposes covariance matrices across task conditions into condition-specific and
    common spatial factors, and the joint-diagonalization / DSS family does the same. The only
    baseline previously in this file was an MSE autoencoder, which does not test that at all.
    Two facts make the concern concrete: (a) at independent pairing, time-shuffling the env
    data (destroying the pairing, preserving marginals) moved |cos| 0.8385 -> 0.8399, i.e. not
    at all, and a plain covariance difference scored 0.9132, BETTER than the paired estimator;
    (b) CF-MCC at independent_recursive was FLAT across nonlinearity (0.840/0.842/0.843/0.845
    for s = 0/0.25/0.5/1.0), the signature of a second-order method. So two arms are added,
    scored exactly like the closed form:
      CSP_baseline    : covariance difference C_ref - C_env per environment, NO pairing, top
                        eigenvector -> A_hat -> standard readout (csp_cos / csp_mcc).
      shuffled_control: the paired closed form on TIME-SHUFFLED env data — pairing destroyed,
                        marginals/covariances preserved (shuf_cos / shuf_mcc). shuf ~ cf means
                        the pairing contributes nothing at that rung.
    Seeds raised 6 -> 12 (the independent_recursive CF-MCC spread was 0.845 +/- 0.194, too wide
    to trust the mean); per-seed CF-MCC values are printed so bimodality is visible. Figure
    calls are wrapped so a container matplotlib/numpy mismatch can never lose a completed sweep.
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

OUT_DIR = "./results_nonlinear_mixing"
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

REF_TARGET_RHO = 0.5
ENV_TARGET_RHO = 0.9
MIN_ALPHA = 0.02

HRF_TR = 2.0
DIST = "laplace"

# --- THE ONE SUBSTANTIVE CHANGE: nonlinear mixing ---
MIXING_MODE = "nonlinear"          # "linear" reproduces the closed-form-ladder file exactly
NONLIN_STRENGTH_LIST = [0.0, 0.25, 0.5, 1.0]
NONLIN_LAYERS = 3                  # square weight matrices in h (W1, W2, W3)
NONLIN_SLOPE = 0.2                 # LeakyReLU negative slope
NONLIN_COND_MAX = 50.0             # reject/resample a weight if condition number exceeds this
NONLIN_LIP = 0.8                   # spectral norm bound on J_h => f = I + s*h invertible for s<=1
NONLIN_SIGMA_MIN = 0.05            # invertibility gate: min singular value of J_f over samples
                                   # (>= 1 - NONLIN_LIP = 0.2 by construction; det shrinks with
                                   # dimension so sigma_min, not |det|, is the right measure)
NONLIN_N_JAC_CHECK = 256           # sampled points for the Jacobian invertibility check

# closed-form sanity gate thresholds (oracle). At s=0 these MUST hold; at s=1 the gate is
# EXPECTED to fail (reported, not aborted).
CF_COS_MIN = 0.999
CF_RANK1_MAX = 1e-4
CF_MCC_MIN = 0.99

# the two pairing conditions swept against each NONLIN_STRENGTH
CONDITIONS = [("rho1", dict(rho=1.0, recursive=False)),
              ("independent_recursive", dict(rho=0.0, recursive=True))]

# adapted (weaker) identifiability probe: how many random invertible perturbations to try
N_PERTURB = 8


# --------------------------------- simulator (reused verbatim from linear file)

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


def make_mixing(rng, d, n_obs):
    """Well-conditioned linear mixing d -> n_obs."""
    while True:
        A = rng.normal(0.0, 1.0, size=(n_obs, d)) / np.sqrt(d)
        s = np.linalg.svd(A, compute_uv=False)
        if s[-1] > 1e-3 and s[0] / s[-1] < 50.0:
            return A


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



# ----------------------------------------------------- nonlinear mixing (the one change)


def _rand_wellcond(rng, d, target_spec, cond_max=NONLIN_COND_MAX):
    """Random square weight with condition number <= cond_max, top singular value = target_spec."""
    for _ in range(1000):
        W = rng.normal(size=(d, d))
        s = np.linalg.svd(W, compute_uv=False)
        if s[0] / s[-1] <= cond_max:
            return W * (target_spec / s[0])
    raise RuntimeError("could not sample a well-conditioned weight")


def make_nonlin_flow(seed, d):
    """Fixed-per-seed nonlinear map f(z) = z + s*h(z), h = W3 leaky(W2 leaky(W1 z)).

    Returns a dict of weights. Each W has condition number <= NONLIN_COND_MAX and top singular
    value NONLIN_LIP**(1/NONLIN_LAYERS), so the spectral norm of the h-Jacobian (a product of
    the three weights times LeakyReLU derivatives, all <= 1) is bounded by NONLIN_LIP < 1.
    Hence f = I + s*h is invertible for every s in [0,1]. h(0)=0, so f fixes the origin and,
    at s=0, f = identity EXACTLY (the linear reproduction).
    """
    rng = np.random.default_rng(6_000_000 + seed)
    spec = NONLIN_LIP ** (1.0 / NONLIN_LAYERS)
    Ws = [_rand_wellcond(rng, d, spec) for _ in range(NONLIN_LAYERS)]
    return dict(Ws=Ws, slope=NONLIN_SLOPE)


def _leaky(u, slope):
    return np.where(u >= 0.0, u, slope * u)


def _leaky_deriv(u, slope):
    return np.where(u >= 0.0, 1.0, slope)


def apply_flow(flow, z, strength):
    """f(z) = z + strength * h(z). z: (..., d) -> (..., d). strength=0 => identity exactly."""
    if strength == 0.0:
        return z
    W1, W2, W3 = flow["Ws"]
    slope = flow["slope"]
    a1 = _leaky(z @ W1.T, slope)
    a2 = _leaky(a1 @ W2.T, slope)
    h = a2 @ W3.T
    return z + strength * h


def flow_jacobian_stats(flow, z, strength):
    """Per-point invertibility diagnostics of J_f = I + strength*J_h over rows of z (M, d).

    Returns (all_det_positive, min_sigma): whether det(J_f) keeps a constant (positive) sign,
    and the smallest singular value of J_f across the points. min_sigma is the RIGHT "bounded
    away from 0" measure — det shrinks with dimension (~sigma^d) and can be tiny for a perfectly
    invertible map at d=10, whereas sigma_min(I + M) >= 1 - ||M|| is the true conditioning.
    """
    if strength == 0.0:
        return True, 1.0
    W1, W2, W3 = flow["Ws"]
    slope = flow["slope"]
    d = z.shape[1]
    u1 = z @ W1.T
    a1 = _leaky(u1, slope)
    u2 = a1 @ W2.T
    I = np.eye(d)
    det_pos = True
    min_sigma = np.inf
    for i in range(z.shape[0]):
        D1 = np.diag(_leaky_deriv(u1[i], slope))
        D2 = np.diag(_leaky_deriv(u2[i], slope))
        Jf = I + strength * (W3 @ D2 @ W2 @ D1 @ W1)
        if np.linalg.det(Jf) <= 0:
            det_pos = False
        min_sigma = min(min_sigma, float(np.linalg.svd(Jf, compute_uv=False)[-1]))
    return det_pos, min_sigma


def assert_invertible(flow, z_sample, strength, seed):
    """The mixing must be invertible on the sampled range, or the ground truth is unrecoverable.

    Gate on the min singular value of J_f (bounded from 0), and confirm det keeps a constant
    sign. By construction ||strength*J_h|| <= NONLIN_LIP < 1, so sigma_min >= 1 - NONLIN_LIP and
    this passes with margin; it fires only if NONLIN_LIP is set too close to 1 or a bug makes f
    fold.
    """
    if strength == 0.0:
        return
    idx = np.random.default_rng(7_000_000 + seed).choice(
        z_sample.shape[0], size=min(NONLIN_N_JAC_CHECK, z_sample.shape[0]), replace=False)
    det_pos, min_sigma = flow_jacobian_stats(flow, z_sample[idx], strength)
    if (not det_pos) or min_sigma < NONLIN_SIGMA_MIN:
        raise RuntimeError(
            f"seed {seed} strength {strength}: mixing NOT invertible on the sampled range "
            f"(min sigma(J_f) = {min_sigma:.3e}, det sign constant = {det_pos}). The ground "
            "truth is unrecoverable; run void. If min sigma is only slightly low, lower "
            "NONLIN_LIP.")


def make_dataset(seed, dist, rho=1.0, snr=np.inf, hrf=False, recursive=False,
                 nonlin_strength=0.0):
    """Identical to run_closedform_ladder.make_dataset EXCEPT the mixing is x = A @ f(z), with
    f the fixed-per-seed nonlinear flow (strength 0 => f = identity => the linear file exactly).

    Ground-truth z_all is the NEURAL latent; x_all is A @ f(bold) (+noise) then standardized.
    A_eff is the standardized LINEAR mixing (the s=0 column direction) — the target the
    closed-form estimator assumes exists, kept for the |cos| comparison across strengths.
    """
    rng_struct = np.random.default_rng(seed)
    rng_innov = np.random.default_rng(1_000_000 + seed)
    rng_mix = np.random.default_rng(2_000_000 + seed)
    rng_pair = np.random.default_rng(3_000_000 + seed)
    rng_noise = np.random.default_rng(4_000_000 + seed)
    rng_rec = np.random.default_rng(5_000_000 + seed)

    coefs_0 = make_var(rng_struct, D_LATENT, EDGES_PER_NODE, MAX_LAG)
    coefs_0 = stabilize(coefs_0, target_rho=REF_TARGET_RHO)
    coefs_envs, edges, alphas = [], [], []
    for c in range(D_LATENT):
        cc, edge, alpha = make_env_shift(rng_struct, coefs_0, c, MAX_LAG)
        coefs_envs.append(cc)
        edges.append(edge)
        alphas.append(alpha)

    if companion_rho(coefs_0) >= 1.0:
        raise RuntimeError(f"seed {seed} {dist}: reference VAR non-stationary.")
    weak = [(c, round(a, 4)) for c, a in enumerate(alphas) if a < MIN_ALPHA]
    if weak:
        raise RuntimeError(f"seed {seed} {dist}: interventions collapsed for envs {weak}.")
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
        raise RuntimeError(f"seed {seed} {dist}: reference latents inf/NaN.")

    neural_full = [z_ref_full]
    if recursive:
        for c in range(D_LATENT):
            z_env_full, _ = simulate_reference(rng_rec, coefs_envs[c], total, dist)
            neural_full.append(z_env_full)
    else:
        for c in range(D_LATENT):
            dd = counterfactual_diff(z_ref_full, coefs_envs[c], coefs_0, MAX_LAG)
            eps_fresh = sample_innovations(rng_pair, (total, D_LATENT), dist)
            mismatch = (rho - 1.0) * eps_ref + np.sqrt(max(1.0 - rho ** 2, 0.0)) * eps_fresh
            neural_full.append(z_ref_full + dd + mismatch)

    bold_full = [hrf_convolve(z, hrf_kernel()) for z in neural_full] if hrf else neural_full

    def win(zf):
        return zf[BURN_IN:].reshape(N_WINDOWS, WINDOW_L, D_LATENT)

    z_all = np.stack([win(zf) for zf in neural_full], 0)
    bold_all = np.stack([win(zf) for zf in bold_full], 0)
    for arr, nm in ((z_all, "neural"), (bold_all, "bold")):
        if not np.isfinite(arr).all() or np.abs(arr).max() > 1e6:
            raise RuntimeError(f"seed {seed} {dist}: {nm} latents blew up.")

    if rho == 1.0 and not recursive:
        for c in range(D_LATENT):
            noncol = [k for k in range(D_LATENT) if k != c]
            resid = float(np.abs(z_all[0][:, :, noncol] - z_all[1 + c][:, :, noncol]).max())
            if resid > 1e-6:
                raise RuntimeError(f"seed {seed} {dist}: PAIRING BROKEN for env {c} at RHO=1.")

    # --- THE MIXING: x = A @ f(bold), f nonlinear (strength 0 => identity => linear file) ---
    flow = make_nonlin_flow(seed, D_LATENT)
    strength = 0.0 if MIXING_MODE == "linear" else float(nonlin_strength)
    assert_invertible(flow, bold_all.reshape(-1, D_LATENT), strength, seed)
    mixed_latent = apply_flow(flow, bold_all, strength)          # f(bold), (E, nw, L, d)

    A = make_mixing(rng_mix, D_LATENT, N_OBS)
    x_raw = mixed_latent @ A.T                                   # (E, nw, L, N)

    if np.isfinite(snr):
        sig_std = x_raw.reshape(-1, N_OBS).std(0)
        noise = rng_noise.normal(size=x_raw.shape) * (sig_std / snr)[None, None, None, :]
        x_raw = x_raw + noise

    flat = x_raw.reshape(-1, N_OBS)
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    x_all = (x_raw - mu) / sd
    A_eff = A / sd[:, None]     # the LINEAR (s=0) column direction the closed form assumes

    return dict(
        x_all=x_all.astype(np.float32), z_all=z_all.astype(np.float32),
        A=A, A_eff=A_eff, flow=flow, nonlin_strength=strength,
        coefs_0=coefs_0, coefs_envs=coefs_envs, edges=edges, alphas=alphas,
        targets=list(range(D_LATENT)),
        rho_0=companion_rho(coefs_0), rho_envs=[companion_rho(cc) for cc in coefs_envs],
        rho=float(rho), snr=float(snr), hrf=bool(hrf), recursive=bool(recursive),
        pairing_mode="recursive" if recursive else "counterfactual",
    )


# --------------------------------- data sanity (reused verbatim)

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


# --------------------------------- flow model (reused verbatim)

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


# --------------------------------- objective utils (reused verbatim)

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


# --------------------------------- training (reused verbatim)

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


# --------------------------------- metrics (reused verbatim)

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


# --------------------------------- closed-form estimator (reused verbatim)

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


# --------------------------------- io (reused verbatim)

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



# ---------------------------------- adapted (weaker) identifiability probe: no rotation


def paired_objective(z_all_np):
    """Option B summed L1/L2 of the per-env paired differences. z_all_np: (E, nw, L, d)."""
    z = torch.from_numpy(np.asarray(z_all_np, dtype=np.float32))
    zs = standardize_multi(z)
    total = 0.0
    for c in range(z.shape[0] - 1):
        prof, _ = diff_profile(zs[0] - zs[1 + c])
        total += float(sparsity_ratio(prof))
    return total


def perturbation_check(z_all_np, seed, n_perturb=N_PERTURB):
    """Is the paired objective at the TRUE latents lower than at random INVERTIBLE perturbations
    of them? Under nonlinear mixing the ambiguity is not a rotation, so we cannot run the linear
    rotation search; this samples a handful of well-conditioned invertible linear maps instead.

    WEAKER than the rotation search: it tries N_PERTURB alternatives rather than OPTIMIZING over
    the ambiguity group, so a lower-objective solution can exist and go unfound. A pass (truth
    favored) is suggestive; a fail (an alternative beats the truth) is a genuine disproof. Note
    this depends only on the latents and the objective, NOT on the mixing, so it is the same
    across NONLIN_STRENGTH — it probes the objective landscape, while the flow-encoder MCC
    probes whether that landscape is REACHABLE through the nonlinear mixing.
    """
    obj_true = paired_objective(z_all_np)
    E, nw, L, d = z_all_np.shape
    rng = np.random.default_rng(8_000_000 + seed)
    objs = []
    for _ in range(n_perturb):
        G = _rand_wellcond(rng, d, 1.0)
        z_pert = (z_all_np.reshape(E, -1, d) @ G.T).reshape(E, nw, L, d)
        objs.append(paired_objective(z_pert))
    objs = np.array(objs)
    return dict(obj_true=float(obj_true), obj_pert_mean=float(objs.mean()),
                obj_pert_min=float(objs.min()),
                true_favored_frac=float(np.mean(obj_true < objs)))


# ------------------------------------------------------------------------------- one rung


def csp_estimate(x_all_tr):
    """CSP-style covariance-difference baseline — uses NO pairing at all.

    Common Spatial Subspace Decomposition (Fu et al., Neuroimage 1999) and the joint-
    diagonalization / DSS family recover latent directions from BETWEEN-CONDITION covariance
    differences, no timepoint correspondence used. For each env e: mean-centre the observed
    data of the reference and of condition e over all timepoints, form C_ref - C_env, take its
    top eigenvector (by |eigenvalue|; the K=1 direction) as that environment's estimate.
    Coordinates are then recovered exactly as the paired closed form does (assemble A_hat,
    pseudo-invert, standard readout), so the comparison is apples to apples.
    """
    E, nw, L, N = x_all_tr.shape
    d = E - 1
    C_ref = np.cov(x_all_tr[0].reshape(-1, N).astype(np.float64), rowvar=False)
    A_hat = np.zeros((N, d))
    for e in range(d):
        C_e = np.cov(x_all_tr[1 + e].reshape(-1, N).astype(np.float64), rowvar=False)
        w, V = np.linalg.eigh(C_ref - C_e)                 # symmetric -> real eigenpairs
        A_hat[:, e] = V[:, int(np.argmax(np.abs(w)))]      # top by |eigenvalue|
    return A_hat


def shuffle_env_time(x_all_tr, seed):
    """Permute the TIME index of each env condition (reference kept fixed). Destroys the pairing
    while preserving every marginal and covariance property — the direct pairing-necessity test:
    run the paired closed form on this and see if its estimate survives."""
    E, nw, L, N = x_all_tr.shape
    out = x_all_tr.copy()
    rng = np.random.default_rng(9_000_000 + seed)
    for e in range(1, E):
        flat = out[e].reshape(-1, N)
        out[e] = flat[rng.permutation(flat.shape[0])].reshape(nw, L, N)
    return out


def run_rung(strength, cond_name, cond, seed, device, log):
    t0 = time.time()
    data = make_dataset(seed, DIST, rho=cond["rho"], recursive=cond["recursive"],
                        nonlin_strength=strength)
    n_test = int(TEST_FRAC * N_WINDOWS)
    tr = slice(0, N_WINDOWS - n_test)
    te = slice(N_WINDOWS - n_test, N_WINDOWS)
    x_tr, x_te = data["x_all"][:, tr], data["x_all"][:, te]
    z_te_true = data["z_all"][:, te]

    # 1. closed form (assumes a linear column exists)
    A_hat, rank1 = closedform_estimate(x_tr)
    cos = column_cos(A_hat, data["A_eff"])
    cf_mcc, cf_pc, cf_r2 = eval_closedform(A_hat, x_te, z_te_true)

    # 1b. CSP-style covariance-difference baseline (NO pairing) — the 1999-technique control
    A_csp = csp_estimate(x_tr)
    csp_cos = column_cos(A_csp, data["A_eff"])
    csp_mcc, _, csp_r2 = eval_closedform(A_csp, x_te, z_te_true)

    # 1c. pairing-necessity control: the SAME paired estimator on time-shuffled env data
    A_shuf, _ = closedform_estimate(shuffle_env_time(x_tr, seed))
    shuf_cos = column_cos(A_shuf, data["A_eff"])
    shuf_mcc, _, _ = eval_closedform(A_shuf, x_te, z_te_true)

    # 2. MSE floor on THIS nonlinear data (re-measured, never assumed) + both flow modes
    mse_model, _ = train_mse_baseline({"x_all": x_tr}, device, log)
    mse_mcc, _, _ = eval_model(mse_model, x_te, z_te_true, device)
    log(f"       [flow random-init] strength={strength} {cond_name}")
    m_rand, _ = train_multienv_model({"x_all": x_tr}, device, log, init_A=None)
    rand_mcc, _, _ = eval_model(m_rand, x_te, z_te_true, device)
    log(f"       [flow cf-init]")
    m_cf, _ = train_multienv_model({"x_all": x_tr}, device, log, init_A=A_hat)
    cfinit_mcc, _, _ = eval_model(m_cf, x_te, z_te_true, device)

    # 3. adapted objective-landscape probe (mixing-independent)
    pert = perturbation_check(data["z_all"][:, tr], seed)

    log(f"       strength={strength} {cond_name} seed {seed}: CF-MCC {cf_mcc:.3f}  "
        f"CSP {csp_mcc:.3f}  SHUF {shuf_mcc:.3f}  MSE {mse_mcc:.3f}  "
        f"rand {rand_mcc:.3f}  cf-init {cfinit_mcc:.3f}")
    return dict(strength=float(strength), cond=cond_name, seed=seed,
                cos_mean=float(cos.mean()), cos_min=float(cos.min()),
                rank1_mean=float(rank1.mean()), rank1_max=float(rank1.max()),
                cf_mcc=cf_mcc, cf_subspace_r2=cf_r2, mse_mcc=mse_mcc,
                rand_mcc=rand_mcc, cfinit_mcc=cfinit_mcc,
                obj_true=pert["obj_true"], obj_pert_mean=pert["obj_pert_mean"],
                obj_pert_min=pert["obj_pert_min"], true_favored_frac=pert["true_favored_frac"],
                # NEW: covariance-difference baseline + pairing-necessity control
                csp_cos_mean=float(csp_cos.mean()), csp_cos_min=float(csp_cos.min()),
                csp_mcc=csp_mcc, csp_subspace_r2=csp_r2,
                shuf_cos_mean=float(shuf_cos.mean()), shuf_mcc=shuf_mcc,
                seconds=time.time() - t0)


# -------------------------------------------------------------------------- figures


def _agg(rs, key):
    v = [r[key] for r in rs if key in r and np.isfinite(r[key])]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), 0.0)


def fig_nl_mcc(results):
    if not results:
        return None
    conds = [c for c, _ in CONDITIONS]
    fig, axes = plt.subplots(1, len(conds), figsize=(6 * len(conds), 4.6), squeeze=False)
    for ax, cond in zip(axes[0], conds):
        ss = sorted({r["strength"] for r in results if r["cond"] == cond})
        def curve(key):
            return [_agg([r for r in results if r["cond"] == cond and r["strength"] == s], key)
                    for s in ss]
        for key, lab, col, ls in (("cf_mcc", "closed-form (paired)", "#55A868", "-"),
                                   ("csp_mcc", "CSP covariance-diff", "#8172B3", "-"),
                                   ("shuf_mcc", "paired, time-shuffled", "#CCB974", ":"),
                                   ("cfinit_mcc", "flow cf-init", "#4C72B0", "-"),
                                   ("rand_mcc", "flow random-init", "#C44E52", "-"),
                                   ("mse_mcc", "MSE floor", "#888888", "--")):
            c = curve(key)
            ax.errorbar(ss, [x[0] for x in c], yerr=[x[1] for x in c], marker="o", capsize=3,
                        label=lab, color=col, ls=ls)
        ax.set_xlabel("NONLIN_STRENGTH  (0 = linear control)")
        ax.set_ylabel("MCC vs true latents")
        ax.set_title(f"{cond}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Does anything recover the latents as the mixing goes nonlinear?")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_nl_mcc.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_nl_estimator(results):
    if not results:
        return None
    conds = [c for c, _ in CONDITIONS]
    fig, axes = plt.subplots(1, len(conds), figsize=(6 * len(conds), 4.6), squeeze=False)
    for ax, cond in zip(axes[0], conds):
        ss = sorted({r["strength"] for r in results if r["cond"] == cond})
        cosm = [_agg([r for r in results if r["cond"] == cond and r["strength"] == s], "cos_mean")[0]
                for s in ss]
        r1 = [_agg([r for r in results if r["cond"] == cond and r["strength"] == s], "rank1_mean")[0]
              for s in ss]
        ax.plot(ss, cosm, marker="s", color="#55A868", label="mean |cos| with true column")
        ax.set_ylabel("mean |cos|", color="#55A868")
        ax.set_ylim(0, 1.05)
        ax2 = ax.twinx()
        ax2.plot(ss, r1, marker="o", color="#C44E52", label="rank-1 ratio")
        ax2.set_ylabel("rank-1 ratio", color="#C44E52")
        ax.set_xlabel("NONLIN_STRENGTH")
        ax.set_title(f"{cond}: closed-form estimator trustworthiness")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_nl_estimator.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ------------------------------------------------------------------------- reporting


def write_report(results, meta):
    L = []
    L.append("# Nonlinear mixing: does anything survive when x = A f(z)?\n")
    L.append(f"- torch {meta['torch']}, numpy {meta['numpy']}, device {meta['device']}")
    L.append(f"- MIXING_MODE = {MIXING_MODE}, strengths {NONLIN_STRENGTH_LIST}, "
             f"d={D_LATENT}, N={N_OBS}\n")
    L.append("Reuses run_closedform_ladder.py wholesale; the ONLY change is x = A @ f(z) with f "
             "a fixed invertible nonlinear map. Strength 0 = f identity = the linear file "
             "exactly (the control). The MSE floor is re-measured on the nonlinear data at every "
             "rung.\n")

    for cond, _ in CONDITIONS:
        rs = [r for r in results if r["cond"] == cond]
        if not rs:
            continue
        L.append(f"## {cond}\n")
        L.append("| strength | cos mean | rank1 mean | CF-MCC | CSP-MCC | SHUF-MCC | MSE floor | "
                 "flow rand | flow cf-init | obj true<pert |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in sorted({r["strength"] for r in rs}):
            g = [r for r in rs if r["strength"] == s]
            def m(k):
                return _agg(g, k)[0]
            L.append(f"| {s:.2f} | {m('cos_mean'):.3f} | {m('rank1_mean'):.2e} "
                     f"| {m('cf_mcc'):.3f} | {m('csp_mcc'):.3f} | {m('shuf_mcc'):.3f} "
                     f"| {m('mse_mcc'):.3f} | {m('rand_mcc'):.3f} | {m('cfinit_mcc'):.3f} "
                     f"| {m('true_favored_frac'):.2f} |")
        L.append("")
        # per-seed cf_mcc so bimodality is visible (the mean alone hid a wide spread)
        L.append("Per-seed CF-MCC (bimodality check):\n")
        for s in sorted({r["strength"] for r in rs}):
            g = sorted([r for r in rs if r["strength"] == s], key=lambda r: r["seed"])
            vals = ", ".join(f"{r['cf_mcc']:.3f}" for r in g)
            a = _agg(g, "cf_mcc")
            L.append(f"- strength {s:.2f} (mean {a[0]:.3f} ± {a[1]:.3f}, n={len(g)}): [{vals}]")
        L.append("")

    # --- IS THE PAIRING DOING ANY WORK? ---
    L.append("## IS THE PAIRING DOING ANY WORK?\n")
    L.append("CSP-MCC uses a between-condition covariance difference with NO pairing "
             "(the Common Spatial Subspace Decomposition idea, Neuroimage 1999). SHUF-MCC is the "
             "paired closed form run on time-shuffled env data — pairing destroyed, marginals and "
             "covariances preserved. If CSP ~ CF, a 1999 second-order technique already does the "
             "job; if SHUF ~ CF, the pairing itself contributes nothing at that rung.\n")
    L.append("| condition | strength | CF-MCC | CSP-MCC | SHUF-MCC | CF-CSP | CF-SHUF |")
    L.append("|---|---|---|---|---|---|---|")
    for cond, _ in CONDITIONS:
        for s in sorted({r["strength"] for r in results if r["cond"] == cond}):
            g = [r for r in results if r["cond"] == cond and r["strength"] == s]
            if not g:
                continue
            cf, csp, shuf = _agg(g, "cf_mcc")[0], _agg(g, "csp_mcc")[0], _agg(g, "shuf_mcc")[0]
            L.append(f"| {cond} | {s:.2f} | {cf:.3f} | {csp:.3f} | {shuf:.3f} "
                     f"| {cf - csp:+.3f} | {cf - shuf:+.3f} |")
    L.append("")
    for cond, _ in CONDITIONS:
        g1 = [r for r in results if r["cond"] == cond and r["strength"] == 1.0]
        if not g1:
            continue
        cf1, csp1, shuf1 = _agg(g1, "cf_mcc")[0], _agg(g1, "csp_mcc")[0], _agg(g1, "shuf_mcc")[0]
        # (i) does CSP match CF at strength 1?
        if csp1 >= cf1 - 0.05:
            L.append(f"- **{cond}, strength 1.0: the paired machinery adds NOTHING over a "
                     f"covariance-difference method (CSP {csp1:.3f} vs CF {cf1:.3f}).** Under "
                     "nonlinear mixing the estimator is behaving as a second-order method; the "
                     "contribution of this line of work is therefore the identifiability "
                     "ANALYSIS, not the estimator.")
        else:
            # (ii) does CSP collapse as strength rises while CF holds?
            g0 = [r for r in results if r["cond"] == cond and r["strength"] == 0.0]
            csp0 = _agg(g0, "csp_mcc")[0] if g0 else float("nan")
            if np.isfinite(csp0) and (csp0 - csp1) > 0.15 and (cf1 >= cf1 - 0.05):
                L.append(f"- **{cond}: CSP collapses with nonlinearity (CSP {csp0:.3f} -> {csp1:.3f}) "
                         f"while CF holds ({cf1:.3f}).** Evidence the paired estimator exploits "
                         "structure BEYOND second order.")
            else:
                L.append(f"- **{cond}, strength 1.0: CF {cf1:.3f} exceeds CSP {csp1:.3f} by "
                         f"{cf1 - csp1:.3f}** (CSP did not clearly collapse). Suggestive that "
                         "pairing helps here, but not the clean beyond-second-order signature.")
        # (iii) direct pairing-necessity test
        L.append(f"  - pairing-necessity (direct): CF {cf1:.3f} vs SHUF {shuf1:.3f} "
                 f"(delta {cf1 - shuf1:+.3f}). "
                 + ("Shuffling barely changes the estimate -> the pairing is NOT doing the work "
                    "here." if abs(cf1 - shuf1) < 0.05 else
                    "Shuffling degrades the estimate -> the pairing carries real signal here."))
    L.append("")

    # --- s=0 reproduction control ---
    L.append("## Controls and verdict\n")
    for cond, _ in CONDITIONS:
        r0 = [r for r in results if r["cond"] == cond and r["strength"] == 0.0]
        if not r0:
            continue
        cf0, cos0, rk0 = _agg(r0, "cf_mcc")[0], _agg(r0, "cos_mean")[0], _agg(r0, "rank1_mean")[0]
        ok = cf0 > 0.9 and cos0 > 0.99 and rk0 < 1e-4
        L.append(f"- **s=0 reproduction ({cond}): {'OK' if ok else 'BROKEN'}** — closed-form MCC "
                 f"{cf0:.3f}, |cos| {cos0:.3f}, rank1 {rk0:.1e}. "
                 + ("Matches the linear file." if ok else
                    "Does NOT match the linear results — the refactor changed something."))

    # (a) where does the closed form collapse?
    for cond, _ in CONDITIONS:
        rs = [r for r in results if r["cond"] == cond]
        if not rs:
            continue
        ss = sorted({r["strength"] for r in rs})
        floor = _agg([r for r in rs if r["strength"] == 0.0], "mse_mcc")[0]
        coll = next((s for s in ss if s > 0 and _agg([r for r in rs if r["strength"] == s], "cf_mcc")[0]
                     <= _agg([r for r in rs if r["strength"] == s], "mse_mcc")[0] + 0.05), None)
        if coll is not None:
            L.append(f"- **(a) {cond}: the closed-form estimator collapses to the MSE floor by "
                     f"NONLIN_STRENGTH = {coll:.2f}** — as expected, there is no column to recover "
                     "once the mixing is nonlinear.")
        else:
            L.append(f"- **(a) {cond}: the closed form never fully collapses in the tested range** "
                     "— unexpected; check whether the nonlinearity is strong enough (cos should "
                     "fall well below 1 at strength 1).")

    # (b) does the flow clear the floor at strength 1?
    for cond, _ in CONDITIONS:
        r1 = [r for r in results if r["cond"] == cond and r["strength"] == 1.0]
        if not r1:
            continue
        floor = _agg(r1, "mse_mcc")
        best_key = "cfinit_mcc" if _agg(r1, "cfinit_mcc")[0] >= _agg(r1, "rand_mcc")[0] else "rand_mcc"
        best = _agg(r1, best_key)
        pooled = float(np.sqrt(best[1] ** 2 + floor[1] ** 2))
        clears = best[0] - floor[0] > pooled
        L.append(f"- **(b) {cond} at strength 1.0: flow {'CLEARS' if clears else 'does NOT clear'} "
                 f"the MSE floor** — best flow ({best_key.replace('_mcc','')}) MCC {best[0]:.3f} "
                 f"± {best[1]:.3f} vs floor {floor[0]:.3f} ± {floor[1]:.3f} (pooled sd {pooled:.3f}). "
                 + ("Something survives the nonlinear CRL setting."
                    if clears else
                    "In the actual (nonlinear) CRL setting the paired objective does not beat a "
                    "rotationally-blind baseline — this line of attack does not transfer."))
    L.append("")

    L.append("## Caveats\n")
    L.append("1. **`obj true<pert` is a WEAK probe.** It checks whether the paired objective at "
             "the true latents beats a few random invertible perturbations; it does NOT optimize "
             "over the ambiguity group like the linear rotation search, and it is mixing-"
             "independent (a property of the objective and the latents). A pass is suggestive, a "
             "fail is a real disproof.")
    L.append("2. **|cos| is measured against the LINEAR (s=0) column A_eff[:,c]** — the object the "
             "closed form assumes exists. Under a nonlinear f there is no such column, so a "
             "falling |cos| is exactly the collapse being quantified, not a metric artifact.")
    L.append("3. **The MSE floor is re-measured per rung on the nonlinear data** and differs from "
             "the linear case; do not compare flow MCC to the linear floor.\n")

    # Figures must NEVER be able to lose a completed sweep: a matplotlib/numpy mismatch in the
    # container once crashed the whole run at fig.savefig AFTER all science was done. Wrap each
    # call, log the failure, and write results.md regardless.
    figs = []
    for figfn in (fig_nl_mcc, fig_nl_estimator):
        try:
            f = figfn(results)
            if f:
                figs.append(f)
        except Exception as e:  # noqa: BLE001 - a figure error must not lose the results
            L.append(f"> figure {figfn.__name__} FAILED to render ({type(e).__name__}: {e}); "
                     "the numbers above are complete and unaffected.")
            print(f"[warn] {figfn.__name__} failed: {type(e).__name__}: {e}", flush=True)
    if figs:
        L.append("## Figures\n")
        for f in figs:
            L.append(f"- `{os.path.basename(f)}`")
        L.append("")

    atomic_write(os.path.join(OUT_DIR, "results.md"), "\n".join(L), is_json=False)


# ---------------------------------------------------------------------- oracle test


def _cf_gate_numbers(seed, strength):
    data = make_dataset(seed, DIST, nonlin_strength=strength)
    A_hat, rank1 = closedform_estimate(data["x_all"])
    cos = column_cos(A_hat, data["A_eff"])
    m, _, _ = eval_closedform(A_hat, data["x_all"], data["z_all"])
    return float(cos.min()), float(rank1.max()), float(m)


def run_oracle(log, device):
    """Reused checks at s=0 (must pass — the linear reproduction) AND the closed-form gate at
    s=1 (EXPECTED to fail — reported, not aborted). The distinction is explicit so a genuine bug
    (e.g. s=0 not reproducing linear, or the mixing turning out non-invertible) still aborts.
    """
    log("=" * 72)
    log("ORACLE HARNESS TEST")
    log(f"MIXING_MODE = {MIXING_MODE}")
    log("=" * 72)
    fails = []

    # 0. integrity + invertibility at s=0 and s=1
    for seed in range(ORACLE_SEEDS):
        for s in (0.0, 1.0):
            try:
                make_dataset(seed, DIST, nonlin_strength=s)     # runs assert_invertible
            except RuntimeError as e:
                log(f"[FAIL] seed {seed} s={s}: build/invertibility — {e}")
                fails.append(f"build/invertibility (seed {seed}, s={s}): {e}")
    if not fails:
        log(f"[ok  ] build + invertibility over {ORACLE_SEEDS} seeds at s in {{0,1}}")

    # 1. VAR-fit convention on true latents (s=0; strength-independent since z is unmixed)
    for seed in range(min(ORACLE_SEEDS, 3)):
        data = make_dataset(seed, DIST, nonlin_strength=0.0)
        B = fit_latent_var(torch.from_numpy(data["z_all"][0]).double(), MAX_LAG, 1e-8).numpy()
        true_B = np.zeros_like(B)
        for l in range(MAX_LAG):
            for i in range(D_LATENT):
                for j in range(D_LATENT):
                    true_B[l * D_LATENT + j, i] = data["coefs_0"][l][i, j]
        err = float(np.abs(B - true_B).max())
        ok = err < 0.05
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: VAR fit, max|B_hat-B_true| = {err:.4f}")
        if not ok:
            fails.append(f"VAR fit convention (seed {seed}, err {err:.4f})")

    # 2. THE CRITICAL CONTROL — s=0 must reproduce the linear closed-form (MCC~1, cos~1, rank1~0)
    log("-" * 72)
    log("CONTROL — s=0 must reproduce the linear results")
    for seed in range(min(ORACLE_SEEDS, 3)):
        cmin, r1max, m = _cf_gate_numbers(seed, 0.0)
        ok = cmin >= CF_COS_MIN and r1max <= CF_RANK1_MAX and m >= CF_MCC_MIN
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: s=0 closed-form min|cos| {cmin:.6f}, "
            f"max rank1 {r1max:.2e}, MCC {m:.4f}")
        if not ok:
            fails.append(f"s=0 does NOT reproduce linear (seed {seed}): min|cos| {cmin:.4f}, "
                         f"rank1 {r1max:.1e}, MCC {m:.3f}. The refactor broke the linear case.")

    # 3. EXPECTED FAILURE — s=1 closed-form gate should FAIL (there is no column). Not aborted.
    log("-" * 72)
    log("EXPECTED FAILURE — s=1 closed-form gate should fail (nonlinear => no column to recover)")
    passed_at_1 = 0
    for seed in range(min(ORACLE_SEEDS, 3)):
        cmin, r1max, m = _cf_gate_numbers(seed, 1.0)
        gate_pass = cmin >= CF_COS_MIN and r1max <= CF_RANK1_MAX and m >= CF_MCC_MIN
        if gate_pass:
            passed_at_1 += 1
            log(f"[warn] seed {seed}: s=1 closed-form gate PASSED (min|cos| {cmin:.3f}, "
                f"rank1 {r1max:.1e}, MCC {m:.3f}) — nonlinearity may be too weak to matter.")
        else:
            log(f"[exp-fail] seed {seed}: s=1 closed-form gate failed AS EXPECTED "
                f"(min|cos| {cmin:.3f}, rank1 {r1max:.2e}, MCC {m:.3f}).")
    if passed_at_1 == min(ORACLE_SEEDS, 3):
        log("[warn] the closed-form gate passed at s=1 on EVERY seed. Not a bug, but the "
            "nonlinearity is too weak to test the question — raise NONLIN_LIP.")

    log("=" * 72)
    if fails:
        log("ORACLE TEST FAILED:")
        for f in fails:
            log("  - " + f)
        return False
    log("ORACLE TEST PASSED — s=0 reproduces the linear closed form; s=1 gate fails as expected "
        "(reported, not a bug); mixing invertible throughout.")
    return True


# ------------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true", help="self-test (s=0 control, s=1 exp-fail)")
    ap.add_argument("--smoke", action="store_true",
                    help="strength 0 and 1, RHO=1, 1 seed, closed-form + cf-init training")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    os.makedirs(SEED_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "n/a"

    def log(msg):
        print(msg, flush=True)

    log("=" * 72)
    log("nonlinear mixing — does anything survive when x = A f(z)?")
    log(f"python  {platform.python_version()}  ({sys.platform})")
    log(f"torch   {torch.__version__}   cuda_available={torch.cuda.is_available()}")
    log(f"numpy   {np.__version__}   scipy {scipy.__version__}   sklearn {sklearn.__version__}")
    log(f"device  {device}   gpu: {gpu}   MIXING_MODE {MIXING_MODE}")
    log("=" * 72)

    meta = dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, device=device, gpu=gpu)

    if args.oracle:
        sys.exit(0 if run_oracle(log, device) else 1)

    strengths = list(NONLIN_STRENGTH_LIST)
    conds = list(CONDITIONS)
    n_seeds = args.seeds
    if args.smoke:
        strengths = [0.0, 1.0]
        conds = [c for c in CONDITIONS if c[0] == "rho1"]
        n_seeds = 1
        log("SMOKE MODE: strength in {0,1}, RHO=1, 1 seed\n")

    # A cached JSON is only reusable if it has the NEW keys (csp/shuf). Older files from before
    # this baseline was added are incomplete -> re-run that seed rather than report partial rows.
    required_keys = {"csp_mcc", "csp_cos_mean", "csp_cos_min", "csp_subspace_r2",
                     "shuf_mcc", "shuf_cos_mean"}
    results = []
    for cond_name, cond in conds:
        for strength in strengths:
            for seed in range(n_seeds):
                p = os.path.join(SEED_DIR, f"nl_{cond_name}_s{strength:.2f}_seed{seed:02d}.json")
                if os.path.exists(p):
                    with open(p) as f:
                        cached = json.load(f)
                    if required_keys.issubset(cached.keys()):
                        results.append(cached)
                        log(f"[skip] {cond_name} strength={strength} seed {seed:02d}")
                        continue
                    log(f"[rerun] {cond_name} strength={strength} seed {seed:02d} "
                        "(cached file predates the CSP/shuffle baseline; missing new keys)")
                log(f"[run ] {cond_name} strength={strength} seed {seed:02d}")
                res = run_rung(strength, cond_name, cond, seed, device, log)
                atomic_write(p, res, is_json=True)
                results.append(res)

    write_report(results, meta)
    log("")
    log("=" * 72)
    if args.smoke:
        log("SMOKE MODE — strengths {0,1}, RHO=1, 1 seed. Not the full ladder.")
    log(f"wrote {os.path.join(OUT_DIR, 'results.md')}")


if __name__ == "__main__":
    main()