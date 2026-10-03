#!/usr/bin/env python
"""
run_nshift_boundary.py — WHERE is the boundary between identifiable and falsified, as the
number of latent coordinates each environment perturbs grows from 1?

THE QUESTION
    Option B works because each environment shifts EXACTLY ONE coordinate: the paired
    difference is 1-D, L1/L2 of a 1-hot vector is uniquely 1, and no rotation can out-sparse
    it, so the optimum is provably at the truth. Candidate 2 (6 coordinates shifted at once)
    was FALSIFIED: the direct rotation search found BEST_ROTATION beating TRUE (2.165 vs
    2.394), because a multi-coordinate difference admits WITHIN-BLOCK rotations that look
    sparser than reality. Real cognitive conditions shift MANY mechanisms at once, so real
    data sits in the falsified regime. This file finds the boundary.

THE PARAMETER — K, distinct TARGET coordinates per environment (NOT edges shifted)
    The load-bearing structure is how many latent COORDINATES the paired difference spans,
    not how many edges are shifted. Shifting several edges all INTO the same target c still
    lands entirely on coordinate c -> the difference is 1-D. What breaks 1-D-ness is shifting
    edges into DIFFERENT targets. So we sweep K = number of distinct target coordinates each
    environment perturbs:
        K=1 -> current Option B (1-D difference; verified working)
        K=2,3,4,6 -> the difference spans a K-dim subspace, with within-subspace rotational
                     freedom that a sparser-than-truth rotation can exploit.
    Design: d=10 environments, environment c targets the balanced circulant set
    {c, c+1, ..., c+K-1 mod d}, so every coordinate is a target in exactly K environments.
    K=1 reduces exactly to one-intervention-per-coordinate.

THE DECISIVE TEST (cheap; numpy/torch, NO encoder training; run first with --rotation-only)
    For each K and seed, on TRUE latents, run the direct rotation search: minimize the summed
    L1/L2 of the per-environment paired-difference coordinate profile over rotations M, from
    >= 200 random restarts (identity is seeded as restart 0, so BEST <= TRUE always). Report:
        TRUE (objective at the truth), RANDOM (at a random rotation), BEST_ROTATION (best found)
        FALSIFIED = BEST_ROTATION beats TRUE by more than tolerance.
      BEST_ROTATION ~ TRUE  -> optimum still at the truth; identifiable in principle.
      BEST_ROTATION < TRUE  -> FALSIFIED at this K; a sparser-than-truth solution exists.
    When it beats TRUE, the block-preservation diagnostic reports whether the winning rotation
    keeps each difference on its true target coordinates (a within-block rotation) or moves
    the block. >= 200 restarts, >= 6 seeds.

THE RECOVERY TEST (only for K values that PASS the rotation search)
    Closed-form estimator + closed-form-init trained encoder, at RHO=1 and at
    independent_recursive. For K>1 the per-environment paired difference spans K dimensions,
    so the estimator is generalized: the top-K RIGHT singular vectors of each observed paired
    difference are a K-dim SUBSPACE estimate, scored by PRINCIPAL ANGLES against the true K-dim
    subspace (mean principal-angle cosine), NOT a single |cos|. This recovers a SUBSPACE per
    environment; individual mixing columns are pinned ONLY where the K-dim subspaces across
    environments intersect appropriately (a coordinate targeted by several environments is the
    shared direction of their subspaces). We recover columns by that intersection (top
    eigenvector of the summed subspace projectors) and report the resulting MCC vs the MSE
    floor, alongside the subspace principal-angle cosine and the rank-K gap.

OUTPUT
    results.md: a table over K (TRUE / RANDOM / BEST_ROTATION, falsified y/n, block-preserved
    y/n, and where applicable the closed-form recovery metrics), the headline verdict (the
    LARGEST K at which the optimum is still at the truth), and a figure with K on the x-axis
    and TRUE vs BEST_ROTATION as two curves — the point where they separate is the boundary.

RUN
    python run_nshift_boundary.py --rotation-only   # the cheap decisive test across all K
    python run_nshift_boundary.py --smoke           # K=1 and K=6, 1 seed, rotation + recovery
    python run_nshift_boundary.py                    # full: rotation test + recovery for passers
"""

import argparse
import json
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
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ----------------------------------------------------------------------------- config

OUT_DIR = "./results_nshift_boundary"
SEED_DIR = os.path.join(OUT_DIR, "seeds")

D_LATENT = 10
N_OBS = 50
MAX_LAG = 2
EDGES_PER_NODE = 1.5
BURN_IN = 200

WINDOW_L = 64
N_WINDOWS = 256
TEST_FRAC = 0.2

N_SEEDS = 6
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

SQRT_EPS = 1e-8            # gradient-safe floor for the TRAINING objective (recovery path)
ROT_EPS = 1e-12            # much smaller floor for the ROTATION objective: no gradient is
                           # driven to zero there, and a tiny floor keeps L1/L2 magnitude-
                           # invariant so a WEAK intervention still reads as a clean K-hot
                           # profile (the per-coordinate floor would otherwise inflate it,
                           # worse at larger d with more off-target coordinates).
REF_TARGET_RHO = 0.5
ENV_TARGET_RHO = 0.9
MIN_ALPHA = 0.02           # strict floor for make_dataset (recovery): weak interventions
                           # leave coordinates unpinned, which corrupts recovery.
MIN_ALPHA_LATENTS = 1e-5   # loose floor for make_latents (rotation test): the test depends on
                           # the paired-difference SUPPORT, not magnitude, so any nonzero
                           # stationary intervention is valid. Aborts only on the pathological.
ORACLE_SEEDS = 6
DIST = "laplace"

# --- the K sweep (d=10 recovery path) ---
K_LIST = [1, 2, 3, 4, 6]

# --- the d x K boundary sweep (--rotation-only): same K/d ratios {.1,.2,.3,.4,.6} at each d ---
DK_SWEEP = [(10, [1, 2, 3, 4, 6]),
            (20, [2, 4, 6, 8, 12]),
            (30, [3, 6, 9, 12, 18])]

# --- the n_envs sweep (--nenv-sweep): vary the NUMBER OF ENVIRONMENTS independently of d ---
# All prior runs used n_envs = d. Real task fMRI gives ~7 conditions (HCP: 7 tasks + rest)
# regardless of latent dimension, so n_envs << d is the realistic, previously-untested regime.
NENV_D = 30
NENV_KS = [6, 12]                          # the clean rungs at d=30
NENV_LIST = [30, 20, 15, 10, 8, 7, 5]      # 30 = one-per-coordinate; 7 = HCP-realistic
HCP_NENV = 7

# --- the rotation search (the decisive test) ---
ROT_RESTARTS = 200         # >= 200 random restarts per (d, K, seed)
ROT_RESTART_CHUNK = 50     # restarts optimized per batch; chunking bounds memory at large d
ROT_STEPS = 80             # gradient steps per restart
ROT_LR = 0.05
ROT_SUBSAMPLE = 1500       # timepoints subsampled for the search (speed/memory; structure kept)
ROT_BEAT_FRAC = 0.02       # BEST below TRUE by more than max(this*TRUE, ROT_ABS_TOL) = falsified
ROT_ABS_TOL = 0.10
BLOCK_PRESERVE_HI = 0.90   # mean per-env energy-on-true-target above this = within-block
BLOCK_MOVED_LO = 0.70      # below this = the rotation moved the block
DEGEN_ENERGY = 0.95        # non-falsified but block energy below this = DEGENERATE PLATEAU
                           # (a flat/near-tied optimum landing on the wrong solution; not trainable)

# --- recovery (only for passing K) ---
RECOVERY_CONDITIONS = [("rho1", dict(rho=1.0, recursive=False)),
                       ("independent_recursive", dict(rho=0.0, recursive=True))]


# --------------------------------- simulator primitives (reused verbatim)

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



# ------------------------------------------ K-coordinate environment construction (new)


def target_sets(d, K):
    """Balanced circulant target assignment: env c targets {c, c+1, ..., c+K-1} mod d.

    Every coordinate is a target in EXACTLY K environments, so coverage is balanced. K=1
    reduces to one-intervention-per-coordinate (env c targets {c}).
    """
    return [sorted([(c + j) % d for j in range(K)]) for c in range(d)]


def target_sets_nenv(d, K, n_envs):
    """n_envs environments (not d), each spanning K coordinates, starts spread evenly over d.

    env e targets {start_e, ..., start_e+K-1} mod d with start_e = round(e*d/n_envs), so the
    K-blocks tile the circle as evenly as n_envs allows. Coverage (the union) and the per-
    coordinate coverage count are reported by the caller; coordinates covered by NO environment
    are rotationally free and cannot be identified. n_envs = d reproduces target_sets(d, K).
    """
    return [sorted([(int(round(e * d / n_envs)) + j) % d for j in range(K)])
            for e in range(n_envs)]


def make_env_multi(rng, coefs_0, targets, max_lag, target_rho=ENV_TARGET_RHO):
    """Build an environment perturbing K DISTINCT target coordinates (one edge into each row).

    For each target t, pick the incoming off-diagonal edge into row t that individually admits
    the largest stationary shift (via _max_stable_alpha), then move ALL K chosen edges together
    by a single shared alpha, bisected so the combined environment is stationary. alpha=0 is the
    reference (stable), so the bracket is valid. env - reference is nonzero on exactly the K
    target rows, so the counterfactual paired difference spans exactly those K coordinates.
    Returns (coefs, edges=[(l,t,j)...], alpha).
    """
    d = coefs_0[0].shape[0]
    mag = rng.uniform(1.5, 2.5)                      # shared magnitude across this env's edges
    chosen = []                                      # (l, t, j, base, target_val)
    for t in targets:
        cands = [(l, j) for l in range(max_lag) for j in range(d)
                 if j != t and abs(coefs_0[l][t, j]) > 1e-8]
        if cands:
            best = None
            for (l, j) in cands:
                base = coefs_0[l][t, j]
                tv = -np.sign(base) * mag

                def build_one(alpha, l=l, j=j, base=base, tv=tv):
                    new = [cc.copy() for cc in coefs_0]
                    new[l][t, j] = (1.0 - alpha) * base + alpha * tv
                    return new
                a = _max_stable_alpha(build_one, target_rho)
                if best is None or a > best[0]:
                    best = (a, l, j, base, tv)
            _, l, j, base, tv = best
        else:
            l = int(rng.integers(max_lag))
            j = int(rng.choice([x for x in range(d) if x != t]))
            base, tv = 0.0, rng.choice([-1.0, 1.0]) * mag
        chosen.append((l, t, j, base, tv))

    def build(alpha):
        new = [cc.copy() for cc in coefs_0]
        for (l, t, j, base, tv) in chosen:
            new[l][t, j] = (1.0 - alpha) * base + alpha * tv
        return new

    if companion_rho(build(1.0)) <= target_rho:
        alpha = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if companion_rho(build(mid)) > target_rho:
                hi = mid
            else:
                lo = mid
        alpha = lo
    edges = [(int(l), int(t), int(j)) for (l, t, j, _b, _tv) in chosen]
    return build(alpha), edges, float(alpha)


def make_dataset(seed, dist, K, rho=1.0, recursive=False, shared_targets=None):
    """d environments, each perturbing K distinct target coordinates (balanced circulant).

    ADDITIVE (default): env is the reference's history + the exact K-dim mechanism difference
    + innovation mismatch ((rho-1)*eps_ref + sqrt(1-rho^2)*eps_fresh). At rho=1 the difference
    is exactly the K-dim mechanism term. RECURSIVE: env is re-simulated from scratch through
    its own shifted VAR with fresh innovations (the genuine independent case). z_all is the
    neural ground truth; x_all is mixed + shared-standardized; A_eff are the standardized
    mixing columns; target_sets record which coordinates each env spans.

    shared_targets: if given, EVERY environment targets that same K-dim set instead of the
    circulant assignment. This is the candidate-2 shared-block case — KNOWN to be falsifiable
    (one within-block rotation reduces every environment's objective at once) — used only by
    the oracle to verify the rotation search has the power to find a sparser-than-truth
    solution when one provably exists.
    """
    rng_struct = np.random.default_rng(seed)
    rng_innov = np.random.default_rng(1_000_000 + seed)
    rng_mix = np.random.default_rng(2_000_000 + seed)
    rng_pair = np.random.default_rng(3_000_000 + seed)
    rng_rec = np.random.default_rng(5_000_000 + seed)

    coefs_0 = make_var(rng_struct, D_LATENT, EDGES_PER_NODE, MAX_LAG)
    coefs_0 = stabilize(coefs_0, target_rho=REF_TARGET_RHO)

    tsets = ([sorted(shared_targets)] * D_LATENT if shared_targets is not None
             else target_sets(D_LATENT, K))
    coefs_envs, edges_all, alphas = [], [], []
    for c in range(D_LATENT):
        cc, edges, alpha = make_env_multi(rng_struct, coefs_0, tsets[c], MAX_LAG)
        coefs_envs.append(cc)
        edges_all.append(edges)
        alphas.append(alpha)

    if companion_rho(coefs_0) >= 1.0:
        raise RuntimeError(f"seed {seed} K={K}: reference VAR non-stationary.")
    weak = [(c, round(a, 4)) for c, a in enumerate(alphas) if a < MIN_ALPHA]
    if weak:
        raise RuntimeError(f"seed {seed} K={K}: interventions collapsed for envs {weak} "
                           f"(alpha < {MIN_ALPHA}). Lower REF_TARGET_RHO or the magnitude.")
    for c, cc in enumerate(coefs_envs):
        if companion_rho(cc) >= 1.0:
            raise RuntimeError(f"seed {seed} K={K}: env {c} non-stationary.")
        # env may differ from the reference ONLY on its K target rows
        allowed = set(tsets[c])
        for l in range(MAX_LAG):
            for r in range(D_LATENT):
                if r not in allowed and np.abs(coefs_0[l][r] - cc[l][r]).max() > 1e-12:
                    raise RuntimeError(f"seed {seed} K={K}: env {c} differs off its target rows.")

    total = N_WINDOWS * WINDOW_L + BURN_IN
    z_ref_full, eps_ref = simulate_reference(rng_innov, coefs_0, total, dist)
    if not np.isfinite(z_ref_full).all():
        raise RuntimeError(f"seed {seed} K={K}: reference latents inf/NaN.")

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

    def win(zf):
        return zf[BURN_IN:].reshape(N_WINDOWS, WINDOW_L, D_LATENT)

    z_all = np.stack([win(zf) for zf in neural_full], 0)
    if not np.isfinite(z_all).all() or np.abs(z_all).max() > 1e6:
        raise RuntimeError(f"seed {seed} K={K}: latents blew up.")

    # At rho=1 additive, the neural difference is supported EXACTLY on the K target coords.
    if rho == 1.0 and not recursive:
        for c in range(D_LATENT):
            off = [k for k in range(D_LATENT) if k not in set(tsets[c])]
            if off:
                resid = float(np.abs(z_all[0][:, :, off] - z_all[1 + c][:, :, off]).max())
                if resid > 1e-6:
                    raise RuntimeError(f"seed {seed} K={K}: env {c} difference leaks off its "
                                       f"K target coords (residual {resid:.3e}).")

    A = make_mixing(rng_mix, D_LATENT, N_OBS)
    x_raw = z_all @ A.T
    flat = x_raw.reshape(-1, N_OBS)
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    x_all = (x_raw - mu) / sd
    A_eff = A / sd[:, None]

    return dict(
        x_all=x_all.astype(np.float32), z_all=z_all.astype(np.float32),
        A=A, A_eff=A_eff, coefs_0=coefs_0, coefs_envs=coefs_envs,
        edges=edges_all, alphas=alphas, target_sets=tsets, K=int(K),
        rho=float(rho), recursive=bool(recursive),
    )


def make_latents(seed, dist, d, K, shared_targets=None, n_envs=None):
    """d-PARAMETERIZED latent-only builder for the rotation sweep (no mixing, no N_OBS).

    The rotation search operates purely in latent space, so the mixing is irrelevant to it;
    skipping it lets d vary freely (10, 20, 30) without touching N_OBS or the encoder. Builds
    the rho=1 additive latents (env = reference history + exact K-dim mechanism difference),
    with the same invariants as make_dataset.

    n_envs controls HOW MANY environments (default d, one per coordinate). When n_envs < d the
    targets are spread over d via target_sets_nenv, and 'covered' / 'coverage_count' record
    which coordinates are perturbed by at least one environment (the rest are rotationally
    free). Returns z_all (E=n_envs+1, nw, L, d), target_sets, alphas, covered, coverage_count.
    """
    rng_struct = np.random.default_rng(seed)
    rng_innov = np.random.default_rng(1_000_000 + seed)
    n_env = d if n_envs is None else int(n_envs)

    coefs_0 = make_var(rng_struct, d, EDGES_PER_NODE, MAX_LAG)
    coefs_0 = stabilize(coefs_0, target_rho=REF_TARGET_RHO)
    if shared_targets is not None:
        tsets = [sorted(shared_targets)] * n_env
    elif n_envs is None:
        tsets = target_sets(d, K)
    else:
        tsets = target_sets_nenv(d, K, n_env)
    coefs_envs, alphas = [], []
    for c in range(n_env):
        cc, _edges, alpha = make_env_multi(rng_struct, coefs_0, tsets[c], MAX_LAG)
        coefs_envs.append(cc)
        alphas.append(alpha)

    if companion_rho(coefs_0) >= 1.0:
        raise RuntimeError(f"seed {seed} d={d} K={K}: reference VAR non-stationary.")
    # Loose floor: the rotation test needs only the paired-difference SUPPORT, not magnitude,
    # so a weak stationary intervention is valid. Abort only on the pathological (near-zero),
    # where the difference would sink into the ROT_EPS floor and stop being cleanly K-hot.
    dead = [(c, round(a, 6)) for c, a in enumerate(alphas) if a < MIN_ALPHA_LATENTS]
    if dead:
        raise RuntimeError(f"seed {seed} d={d} K={K}: interventions vanished for envs {dead} "
                           f"(alpha < {MIN_ALPHA_LATENTS}); that coordinate cannot take any "
                           "stationary intervention. Lower REF_TARGET_RHO for more headroom.")
    for c, cc in enumerate(coefs_envs):
        if companion_rho(cc) >= 1.0:
            raise RuntimeError(f"seed {seed} d={d} K={K}: env {c} non-stationary.")
        allowed = set(tsets[c])
        for l in range(MAX_LAG):
            for r in range(d):
                if r not in allowed and np.abs(coefs_0[l][r] - cc[l][r]).max() > 1e-12:
                    raise RuntimeError(f"seed {seed} d={d} K={K}: env {c} differs off targets.")

    total = N_WINDOWS * WINDOW_L + BURN_IN
    z_ref_full, _ = simulate_reference(rng_innov, coefs_0, total, dist)
    if not np.isfinite(z_ref_full).all():
        raise RuntimeError(f"seed {seed} d={d} K={K}: reference latents inf/NaN.")
    neural_full = [z_ref_full]
    for c in range(n_env):
        neural_full.append(z_ref_full + counterfactual_diff(z_ref_full, coefs_envs[c],
                                                            coefs_0, MAX_LAG))

    def win(zf):
        return zf[BURN_IN:].reshape(N_WINDOWS, WINDOW_L, d)

    z_all = np.stack([win(zf) for zf in neural_full], 0)
    if not np.isfinite(z_all).all() or np.abs(z_all).max() > 1e6:
        raise RuntimeError(f"seed {seed} d={d} K={K}: latents blew up.")
    for c in range(n_env):
        off = [k for k in range(d) if k not in set(tsets[c])]
        if off and float(np.abs(z_all[0][:, :, off] - z_all[1 + c][:, :, off]).max()) > 1e-6:
            raise RuntimeError(f"seed {seed} d={d} K={K}: env {c} difference leaks off targets.")

    coverage_count = [sum(1 for ts in tsets if k in set(ts)) for k in range(d)]
    covered = sorted([k for k in range(d) if coverage_count[k] > 0])
    return dict(z_all=z_all.astype(np.float32), target_sets=tsets, alphas=alphas,
                K=int(K), d=int(d), n_envs=int(n_env),
                covered=covered, coverage_count=coverage_count)


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


# --------------------------------- latent VAR + objective utils (reused verbatim)

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



# --------------------------------------------- the decisive test: batched rotation search


def _rot_objective(z, M_batch):
    """Summed L1/L2 of the per-env paired-difference coordinate profile, for a BATCH of
    rotations. z: (E, T, d). M_batch: (R, d, d). Returns (R,) losses.

    Numerically identical to standardize_multi + diff_profile + sparsity_ratio summed over
    envs (the training objective), just vectorized over R rotations. z' = M z per restart;
    standardize pooled over all envs; per env c, L1/L2 of the per-coordinate RMS of
    (z'_ref - z'_env_c).
    """
    E, T, d = z.shape
    R = M_batch.shape[0]
    rot = torch.einsum("rkm,etm->retk", M_batch, z)          # (R, E, T, d)
    flat = rot.reshape(R, E * T, d)
    sd = flat.std(dim=1, unbiased=False) + 1e-6              # (R, d)
    rot_s = (flat / sd[:, None, :]).reshape(R, E, T, d)
    loss = rot.new_zeros(R)
    for c in range(E - 1):
        dd = rot_s[:, 0] - rot_s[:, 1 + c]                  # (R, T, d)
        prof = (dd.pow(2).mean(1) + ROT_EPS).sqrt()         # (R, d); tiny floor => scale-invariant
        loss = loss + prof.sum(-1) / (prof.pow(2).sum(-1).sqrt() + 1e-12)
    return loss


def rotation_search(z_np, seed, device, restarts=ROT_RESTARTS, steps=ROT_STEPS):
    """TRUE, RANDOM, and BEST_ROTATION of the summed L1/L2 objective on TRUE latents.

    z_np: (E, nw, L, d) — d is inferred, so this is d-agnostic. Subsampled to ROT_SUBSAMPLE
    timepoints. Identity is evaluated as the seed of BEST, so BEST <= TRUE by construction; any
    improvement is a genuine sparser-than-truth rotation. Restarts are run in chunks of
    ROT_RESTART_CHUNK (bounds memory at large d, since the working tensor is R*(d+1)*T*d).
    """
    E = z_np.shape[0]
    d = z_np.shape[-1]
    z_flat = z_np.reshape(E, -1, d)
    T = z_flat.shape[1]
    rng = np.random.default_rng(1234 + seed)
    idx = rng.choice(T, size=min(ROT_SUBSAMPLE, T), replace=False)
    z = torch.from_numpy(z_flat[:, idx, :]).to(device).float()          # (E, T', d)

    eye = torch.eye(d, device=device).unsqueeze(0)
    true_val = float(_rot_objective(z, eye)[0].item())

    Qs = []
    for r in range(4):                                                  # a few random rotations
        g = np.random.default_rng(9000 + seed * 10 + r)
        Q, _ = np.linalg.qr(g.normal(size=(d, d)))
        Qs.append(Q)
    Qb = torch.from_numpy(np.stack(Qs)).to(device).float()
    random_val = float(_rot_objective(z, Qb).mean().item())

    best_val, best_M = true_val, np.eye(d)                              # identity seed => BEST <= TRUE
    done = 0
    chunk_i = 0
    while done < restarts:
        R = min(ROT_RESTART_CHUNK, restarts - done)
        ggen = torch.Generator(device="cpu").manual_seed(seed * 100003 + chunk_i)
        Ap = (0.3 * torch.randn(R, d, d, generator=ggen)).to(device).requires_grad_(True)
        opt = torch.optim.Adam([Ap], lr=ROT_LR)
        for _ in range(steps):
            M = torch.matrix_exp(Ap - Ap.transpose(-1, -2))            # (R, d, d), orthogonal
            loss = _rot_objective(z, M)
            with torch.no_grad():
                bi = int(loss.argmin().item())
                if float(loss[bi].item()) < best_val:
                    best_val = float(loss[bi].item())
                    best_M = M[bi].detach().cpu().numpy().copy()
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()
            opt.step()
        done += R
        chunk_i += 1
    return true_val, random_val, best_val, best_M


def is_signed_permutation(M, tol=0.15):
    A = np.abs(M)
    return bool(np.all(np.sort(A, 1)[:, -1] > 1 - tol) and np.all(np.sort(A, 1)[:, -2] < tol)
               and np.all(np.sort(A, 0)[-1, :] > 1 - tol))


def block_preservation(best_M, tsets):
    """Does the winning rotation keep each env's difference on its true target coordinates?

    z' = M z, so dd_c' = M dd_c and dd_c lives in span{e_t : t in T_c}. Coordinate k of dd_c'
    is carried by row k of M[:, T_c]. energy_on_true_c = fraction of ||M[:, T_c]||_F^2 that
    lands on the true target rows T_c (= 1 if M maps span{T_c} to itself, i.e. within-block).
    Averaged over environments. Also reports whether M is a signed permutation.
    """
    d = best_M.shape[0]
    per_env = []
    for tset in tsets:
        sub = best_M[:, tset]                          # (d, K)
        row_e = (sub ** 2).sum(1)                      # (d,)
        per_env.append(float(row_e[tset].sum() / (row_e.sum() + 1e-30)))
    return dict(energy_on_true=float(np.mean(per_env)),
                energy_on_true_min=float(np.min(per_env)),
                signed_permutation=is_signed_permutation(best_M))


# ------------------------------------------------- subspace recovery for K > 1 (new)


def principal_angle_cos(U_est, U_true):
    """Mean cosine of principal angles between two orthonormal N x K subspaces (1 = identical)."""
    s = np.linalg.svd(U_est.T @ U_true, compute_uv=False)
    return float(np.clip(s, 0.0, 1.0).mean())


def subspace_estimate(x_all_tr, K):
    """Per-env K-dim subspace estimate + rank-K gap, from observed paired differences.

    For env c, D_c = x_ref - x_env_c (mean-centred over time); its top-K right singular
    vectors are an orthonormal N x K estimate of span{A_eff[:,t] : t in T_c}. rankK_gap =
    s[K]/s[K-1] (0 = the difference is cleanly K-dimensional; K=1 recovers the old rank1 ratio).
    """
    E, nw, L, N = x_all_tr.shape
    d = E - 1
    U_list, gaps = [], []
    for c in range(d):
        D = (x_all_tr[0] - x_all_tr[1 + c]).reshape(-1, N).astype(np.float64)
        D = D - D.mean(0, keepdims=True)
        _, S, Vt = np.linalg.svd(D, full_matrices=False)
        U_list.append(Vt[:K].T)                        # (N, K), orthonormal
        gaps.append(float(S[K] / (S[K - 1] + 1e-30)) if K < len(S) else 0.0)
    return U_list, np.array(gaps)


def true_subspaces(A_eff, tsets, K):
    """Orthonormal N x K bases of the true per-env subspaces span{A_eff[:,t] : t in T_c}."""
    U = []
    for tset in tsets:
        Q, _ = np.linalg.qr(A_eff[:, tset])
        U.append(Q[:, :K])
    return U


def recover_columns_by_intersection(U_list, tsets, d, N):
    """Recover individual mixing columns as the direction shared by the subspaces that span
    each coordinate. Coordinate t is a target of envs {c : t in T_c}; A_eff[:,t] lies in each
    of their subspaces, so it is the top eigenvector of the summed projectors sum_c U_c U_c^T
    (eigenvalue ~ #envs at perfect pairing). Only pinned when those subspaces intersect in that
    one direction — the K>1 caveat made concrete.
    """
    proj25 = [U @ U.T for U in U_list]                 # per-env N x N projectors
    A_hat = np.zeros((N, d))
    for t in range(d):
        envs = [c for c in range(d) if t in set(tsets[c])]
        P = sum(proj25[c] for c in envs)
        w, V = np.linalg.eigh(P)
        A_hat[:, t] = V[:, -1]                          # top eigenvector = shared direction
    return A_hat


def eval_columns(A_hat, x_te, z_te):
    """Unmix held-out observations with pinv(A_hat) and score against the true latents."""
    W = np.linalg.pinv(A_hat)
    z_hat = x_te.reshape(-1, N_OBS) @ W.T
    z_true = z_te.reshape(-1, D_LATENT)
    m, _, per_true = mcc_and_matching(z_true, z_hat)
    return m, per_true.tolist(), subspace_r2(z_true, z_hat)


# ------------------------------------------------------------------------ run drivers


def run_rotation_test(K, seed, device, log, d=D_LATENT):
    t0 = time.time()
    data = make_latents(seed, DIST, d, K)
    n_test = int(TEST_FRAC * N_WINDOWS)
    tr = slice(0, N_WINDOWS - n_test)
    z_tr = data["z_all"][:, tr]

    true_val, random_val, best_val, best_M = rotation_search(z_tr, seed, device)
    beat = true_val - best_val
    falsified = beat > max(ROT_ABS_TOL, ROT_BEAT_FRAC * true_val)
    bp = block_preservation(best_M, data["target_sets"])
    # a non-falsified rung whose winning rotation still landed off the true targets is a
    # DEGENERATE PLATEAU: the optimum went flat (near-tied value at the wrong solution), which
    # is not trainable even though it clears the falsification tolerance.
    degenerate = (not falsified) and bp["energy_on_true"] < DEGEN_ENERGY

    frac_gap = beat / (true_val + 1e-12)     # (TRUE - BEST)/TRUE, comparable across d and K
    flag = "FALSIFIED" if falsified else ("DEGENERATE" if degenerate else "ok")
    log(f"       d={d} K={K} (K/d={K/d:.2f}) seed {seed}: TRUE {true_val:.3f}  "
        f"RANDOM {random_val:.3f}  BEST {best_val:.3f}  {flag}  "
        f"block-e-o-t {bp['energy_on_true']:.2f}")
    return dict(phase="rotation", d=int(d), K=int(K), K_over_d=float(K / d), seed=seed,
                true_val=true_val, random_val=random_val, best_val=best_val,
                beat=float(beat), frac_gap=float(frac_gap), falsified=bool(falsified),
                degenerate=bool(degenerate),
                block_energy_on_true=bp["energy_on_true"],
                block_energy_on_true_min=bp["energy_on_true_min"],
                signed_permutation=bp["signed_permutation"],
                min_alpha=float(min(data["alphas"])), alphas=data["alphas"],
                seconds=time.time() - t0)


def run_recovery(K, seed, cond_name, cond, device, log):
    t0 = time.time()
    data = make_dataset(seed, DIST, K, rho=cond["rho"], recursive=cond["recursive"])
    n_test = int(TEST_FRAC * N_WINDOWS)
    tr = slice(0, N_WINDOWS - n_test)
    te = slice(N_WINDOWS - n_test, N_WINDOWS)
    x_tr, x_te = data["x_all"][:, tr], data["x_all"][:, te]
    z_te = data["z_all"][:, te]

    U_est, gaps = subspace_estimate(x_tr, K)
    U_true = true_subspaces(data["A_eff"], data["target_sets"], K)
    pa = np.array([principal_angle_cos(U_est[c], U_true[c]) for c in range(D_LATENT)])

    A_hat = recover_columns_by_intersection(U_est, data["target_sets"], D_LATENT, N_OBS)
    cf_mcc, cf_pc, cf_r2 = eval_columns(A_hat, x_te, z_te)

    mse_model, _ = train_mse_baseline({"x_all": x_tr}, device, log)
    mse_mcc, _, _ = eval_model(mse_model, x_te, z_te, device)

    log(f"       [cf-init] training")
    m_cf, _ = train_multienv_model({"x_all": x_tr}, device, log, init_A=A_hat)
    cfinit_mcc, _, _ = eval_model(m_cf, x_te, z_te, device)

    log(f"       K={K} {cond_name} seed {seed}: subspace-cos {pa.mean():.3f}  "
        f"rankK-gap {gaps.mean():.2e}  CF-MCC {cf_mcc:.3f}  cf-init {cfinit_mcc:.3f}  "
        f"MSE {mse_mcc:.3f}")
    return dict(phase="recovery", K=int(K), seed=seed, cond=cond_name,
                subspace_cos_mean=float(pa.mean()), subspace_cos_min=float(pa.min()),
                rankK_gap_mean=float(gaps.mean()), rankK_gap_max=float(gaps.max()),
                cf_mcc=cf_mcc, cf_subspace_r2=cf_r2, cf_per_coord_mcc=cf_pc,
                cfinit_mcc=cfinit_mcc, mse_mcc=mse_mcc, seconds=time.time() - t0)



# -------------------------------------------------------------------------- figures


def _rot_by_k(results, K):
    return [r for r in results if r.get("phase") == "rotation" and r["K"] == K]


def _rec(results, K, cond):
    return [r for r in results if r.get("phase") == "recovery" and r["K"] == K and r["cond"] == cond]


def fig_boundary(results):
    ks = [K for K in K_LIST if _rot_by_k(results, K)]
    if not ks:
        return None
    tru = [np.mean([r["true_val"] for r in _rot_by_k(results, K)]) for K in ks]
    tru_sd = [np.std([r["true_val"] for r in _rot_by_k(results, K)]) for K in ks]
    best = [np.mean([r["best_val"] for r in _rot_by_k(results, K)]) for K in ks]
    best_sd = [np.std([r["best_val"] for r in _rot_by_k(results, K)]) for K in ks]
    rand = [np.mean([r["random_val"] for r in _rot_by_k(results, K)]) for K in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ks, tru, yerr=tru_sd, marker="o", capsize=3, color="#55A868",
                label="TRUE (objective at the truth)")
    ax.errorbar(ks, best, yerr=best_sd, marker="s", capsize=3, color="#C44E52",
                label="BEST_ROTATION (best found)")
    ax.plot(ks, rand, marker="^", ls=":", color="0.5", label="RANDOM (reference ceiling)")
    # shade where they separate (falsified)
    for K in ks:
        rs = _rot_by_k(results, K)
        if np.mean([r["falsified"] for r in rs]) >= 0.5:
            ax.axvspan(K - 0.25, K + 0.25, color="#C44E52", alpha=0.08)
    ax.set_xlabel("K  (distinct target coordinates each environment perturbs)")
    ax.set_ylabel("summed L1/L2 of per-env paired-difference profile")
    ax.set_title("Where the optimum leaves the truth\n"
                 "TRUE and BEST_ROTATION coincide until the boundary; shaded K = falsified")
    ax.set_xticks(ks)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_boundary.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ------------------------------------------ d x K boundary sweep (--rotation-only)


_DCOL = {10: "#4C72B0", 20: "#DD8452", 30: "#55A868"}


def _dsweep_ds(results):
    return sorted({r["d"] for r in results if r.get("phase") == "rotation"})


def _by_dk(results, d, K):
    return [r for r in results if r.get("phase") == "rotation" and r["d"] == d and r["K"] == K]


def fig_gap_vs_k(results):
    ds = _dsweep_ds(results)
    if not ds:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in ds:
        Ks = sorted({r["K"] for r in results if r.get("phase") == "rotation" and r["d"] == d})
        y = [np.mean([r["frac_gap"] for r in _by_dk(results, d, K)]) for K in Ks]
        sd = [np.std([r["frac_gap"] for r in _by_dk(results, d, K)]) for K in Ks]
        ax.errorbar(Ks, y, yerr=sd, marker="o", capsize=3, color=_DCOL.get(d),
                    label=f"d={d}")
    ax.set_xlabel("K  (absolute number of coordinates perturbed per environment)")
    ax.set_ylabel("(TRUE - BEST) / TRUE   (0 = optimum at the truth)")
    ax.set_title("Falsification gap vs ABSOLUTE K\n"
                 "curves aligned here => the boundary is a constant K")
    ax.legend(title="latent dim")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_gap_vs_K.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_gap_vs_kd(results):
    ds = _dsweep_ds(results)
    if not ds:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in ds:
        rs = [r for r in results if r.get("phase") == "rotation" and r["d"] == d]
        kds = sorted({r["K_over_d"] for r in rs})
        y = [np.mean([r["frac_gap"] for r in rs if abs(r["K_over_d"] - kd) < 1e-9]) for kd in kds]
        sd = [np.std([r["frac_gap"] for r in rs if abs(r["K_over_d"] - kd) < 1e-9]) for kd in kds]
        ax.errorbar(kds, y, yerr=sd, marker="o", capsize=3, color=_DCOL.get(d), label=f"d={d}")
    ax.set_xlabel("K / d   (fraction of coordinates perturbed per environment)")
    ax.set_ylabel("(TRUE - BEST) / TRUE   (0 = optimum at the truth)")
    ax.set_title("Falsification gap vs FRACTIONAL K/d\n"
                 "curves COLLAPSE here => the constraint is fractional (larger d tolerates more)")
    ax.legend(title="latent dim")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_gap_vs_Kd.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_blockenergy_vs_kd(results):
    ds = _dsweep_ds(results)
    if not ds:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in ds:
        rs = [r for r in results if r.get("phase") == "rotation" and r["d"] == d]
        kds = sorted({r["K_over_d"] for r in rs})
        y = [np.mean([r["block_energy_on_true"] for r in rs if abs(r["K_over_d"] - kd) < 1e-9])
             for kd in kds]
        ax.plot(kds, y, marker="o", color=_DCOL.get(d), label=f"d={d}")
    ax.axhline(DEGEN_ENERGY, color="k", ls="--", lw=1, label=f"degenerate below {DEGEN_ENERGY}")
    ax.set_xlabel("K / d")
    ax.set_ylabel("block energy-on-true  (1 = within-block; low = optimum moved off the truth)")
    ax.set_title("Degenerate-plateau onset vs K/d\n"
                 "a rung can clear the falsification tolerance yet still be untrainable here")
    ax.legend(title="latent dim")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_blockenergy_vs_Kd.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def write_report_dsweep(results, meta):
    L = []
    L.append("# N-shift boundary vs latent dimension: absolute K or fractional K/d?\n")
    L.append(f"- torch {meta['torch']}, numpy {meta['numpy']}, scipy {meta['scipy']}, "
             f"sklearn {meta['sklearn']}")
    L.append(f"- device: {meta['device']} ({meta['gpu']})")
    L.append(f"- rotation search: {ROT_RESTARTS} restarts x {ROT_STEPS} steps, "
             f"subsample {ROT_SUBSAMPLE}\n")
    L.append("A rung BREAKS if it is FALSIFIED (a sparser-than-truth rotation exists) OR "
             "DEGENERATE (not falsified within tolerance, but the winning rotation's block "
             f"energy-on-true fell below {DEGEN_ENERGY} — a flat/near-tied optimum sitting on "
             "the wrong solution, which is not trainable). **Block energy is reported for every "
             "rung, not just the pass/fail verdict.**\n")

    ds = _dsweep_ds(results)
    L.append("## Full sweep\n")
    L.append("| d | K | K/d | TRUE | RANDOM | BEST | (TRUE-BEST)/TRUE | falsified | "
             "block e-o-t | degenerate | min α |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in ds:
        Ks = sorted({r["K"] for r in results if r.get("phase") == "rotation" and r["d"] == d})
        for K in Ks:
            rs = _by_dk(results, d, K)
            def m(k):
                return np.mean([r[k] for r in rs])
            L.append(f"| {d} | {K} | {K/d:.2f} | {m('true_val'):.3f} | {m('random_val'):.3f} "
                     f"| {m('best_val'):.3f} | {m('frac_gap'):.4f} "
                     f"| {np.mean([r['falsified'] for r in rs]):.2f} "
                     f"| {m('block_energy_on_true'):.2f} "
                     f"| {np.mean([r['degenerate'] for r in rs]):.2f} "
                     f"| {min(r.get('min_alpha', float('nan')) for r in rs):.3f} |")
    L.append("")
    L.append("*min α is the smallest intervention strength across environments; at larger d "
             "the reference is scaled down harder, so some coordinates take only a weak "
             "stationary intervention. The rotation test depends on the paired-difference "
             "SUPPORT, not its magnitude (L1/L2 is scale-invariant under the small ROT_EPS "
             "floor), so weak interventions remain valid — but a very small min α is worth "
             "noting.*\n")

    # per-d break point (smallest K where majority breaks), in absolute K and in K/d
    def breaks(rs):
        return (np.mean([r["falsified"] for r in rs]) >= 0.5
                or np.mean([r["degenerate"] for r in rs]) >= 0.5)
    L.append("## Boundary per latent dimension\n")
    L.append("| d | breaks first at K | = K/d |")
    L.append("|---|---|---|")
    bpoints = {}
    for d in ds:
        Ks = sorted({r["K"] for r in results if r.get("phase") == "rotation" and r["d"] == d})
        first = next((K for K in Ks if breaks(_by_dk(results, d, K))), None)
        bpoints[d] = first
        L.append(f"| {d} | {first if first is not None else 'no break in range'} "
                 f"| {f'{first/d:.2f}' if first is not None else '—'} |")
    L.append("")

    # headline: constant K vs constant K/d
    L.append("## Verdict — absolute K or fractional K/d?\n")
    broke = {d: k for d, k in bpoints.items() if k is not None}
    if len(broke) >= 2:
        ks = list(broke.values())
        kds = [k / d for d, k in broke.items()]
        k_spread = max(ks) - min(ks)
        kd_spread = max(kds) - min(kds)
        if kd_spread < k_spread / 2 and kd_spread <= 0.15:
            L.append(f"- **The boundary is FRACTIONAL (~constant K/d ≈ {np.mean(kds):.2f}).** "
                     f"Break points in K/d span only {kd_spread:.2f} across d while the absolute "
                     f"K span is {k_spread}. Larger latent spaces tolerate proportionally more "
                     "simultaneous shifts — the constraint is on the fraction of coordinates "
                     "perturbed per environment, not their count.")
        elif k_spread <= 1:
            L.append(f"- **The boundary is ABSOLUTE (~constant K ≈ {int(np.median(ks))}).** "
                     f"Break points in K span only {k_spread} across d while K/d spans "
                     f"{kd_spread:.2f}. The tolerance is a fixed NUMBER of simultaneous shifts "
                     "regardless of latent dimension.")
        else:
            L.append(f"- Mixed: K break points span {k_spread}, K/d span {kd_spread:.2f}. "
                     "Neither a clean constant-K nor a clean constant-K/d law; read the curves.")
    elif len(broke) == 0:
        L.append("- **No rung broke anywhere in the swept range at any d.** The balanced "
                 "multi-environment design keeps the optimum at the truth across all tested "
                 "K and d — consistent with the K=6 (d=10) oracle result. If nothing breaks "
                 "even at K/d=0.6, extend the sweep to higher K/d to find the ceiling.")
    else:
        d0 = next(iter(broke))
        L.append(f"- Only d={d0} broke (at K={broke[d0]}); the others held across the range. "
                 "Extend the sweep to resolve the scaling law.")
    L.append("- The figures make the test visual: if the gap-vs-K curves for the three d "
             "values are offset but the gap-vs-K/d curves lie on top of each other, the "
             "constraint is fractional.")
    L.append("")

    figs = [f for f in (fig_gap_vs_k(results), fig_gap_vs_kd(results),
                        fig_blockenergy_vs_kd(results)) if f]
    if figs:
        L.append("## Figures\n")
        for f in figs:
            L.append(f"- `{os.path.basename(f)}`")
        L.append("")

    atomic_write(os.path.join(OUT_DIR, "results.md"), "\n".join(L), is_json=False)


def run_dsweep(n_seeds, device, log, meta, smoke=False):
    sweep = DK_SWEEP
    if smoke:
        sweep = [(10, [1, 6]), (20, [6])]
        n_seeds = 1
        log("SMOKE (rotation-only): d=10 K in {1,6}, d=20 K=6, 1 seed\n")
    results = []
    for d, Ks in sweep:
        for K in Ks:
            for seed in range(n_seeds):
                p = os.path.join(SEED_DIR, f"rotation_d{d}_K{K}_seed{seed:02d}.json")
                if os.path.exists(p):
                    with open(p) as f:
                        results.append(json.load(f))
                    log(f"[skip] d={d} K={K} seed {seed:02d}")
                    continue
                log(f"[run ] d={d} K={K} (K/d={K/d:.2f}) seed {seed:02d}")
                res = run_rotation_test(K, seed, device, log, d=d)
                atomic_write(p, res, is_json=True)
                results.append(res)
    write_report_dsweep(results, meta)


# ---------------------------------- n_envs sweep: environments independent of d (--nenv-sweep)


def run_nenv_test(d, K, n_envs, seed, device, log):
    t0 = time.time()
    data = make_latents(seed, DIST, d, K, n_envs=n_envs)
    covered = data["covered"]
    cov_frac = len(covered) / d
    n_test = int(TEST_FRAC * N_WINDOWS)
    tr = slice(0, N_WINDOWS - n_test)
    z_tr = data["z_all"][:, tr]

    # The paired differences are identically zero on uncovered coordinates, so the objective
    # lives entirely in the covered subspace: this rotation search IS the covered-set test.
    # (Rotating within uncovered coords does nothing; spreading a difference into them only
    # raises L1/L2, so the optimizer never does it.) Uncovered coords are free regardless.
    true_val, random_val, best_val, best_M = rotation_search(z_tr, seed, device)
    beat = true_val - best_val
    falsified = beat > max(ROT_ABS_TOL, ROT_BEAT_FRAC * true_val)
    bp = block_preservation(best_M, data["target_sets"])
    degenerate = (not falsified) and bp["energy_on_true"] < DEGEN_ENERGY
    covered_pinned = (not falsified) and (not degenerate)
    # aggregate ELEMENT-WISE identifiable fraction of ALL d coords: covered coords count only if
    # pinned; uncovered coords never count (rotationally free).
    agg_ident_frac = cov_frac if covered_pinned else 0.0

    flag = "FALSIFIED" if falsified else ("DEGENERATE" if degenerate else "ok")
    log(f"       d={d} K={K} n_envs={n_envs} seed {seed}: TRUE {true_val:.3f}  BEST {best_val:.3f}"
        f"  {flag}  block-e-o-t {bp['energy_on_true']:.2f}  coverage {len(covered)}/{d}")
    return dict(phase="nenv", d=int(d), K=int(K), n_envs=int(n_envs), seed=seed,
                true_val=true_val, random_val=random_val, best_val=best_val,
                beat=float(beat), frac_gap=float(beat / (true_val + 1e-12)),
                falsified=bool(falsified), degenerate=bool(degenerate),
                block_energy_on_true=bp["energy_on_true"],
                covered_pinned=bool(covered_pinned),
                coverage=len(covered), coverage_frac=float(cov_frac),
                max_coverage_count=int(max(data["coverage_count"])),
                min_coverage_count=int(min(data["coverage_count"])),
                agg_ident_frac=float(agg_ident_frac),
                min_alpha=float(min(data["alphas"])), covered=covered,
                seconds=time.time() - t0)


def fig_nenv(results):
    rs = [r for r in results if r.get("phase") == "nenv"]
    if not rs:
        return None
    Ks = sorted({r["K"] for r in rs})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    col = {6: "#4C72B0", 12: "#DD8452"}
    for K in Ks:
        ns = sorted({r["n_envs"] for r in rs if r["K"] == K})
        gap = [np.mean([r["frac_gap"] for r in rs if r["K"] == K and r["n_envs"] == n]) for n in ns]
        gsd = [np.std([r["frac_gap"] for r in rs if r["K"] == K and r["n_envs"] == n]) for n in ns]
        be = [np.mean([r["block_energy_on_true"] for r in rs if r["K"] == K and r["n_envs"] == n])
              for n in ns]
        ax1.errorbar(ns, gap, yerr=gsd, marker="o", capsize=3, color=col.get(K), label=f"K={K}")
        ax2.plot(ns, be, marker="s", color=col.get(K), label=f"K={K}")
    for ax in (ax1, ax2):
        ax.axvline(HCP_NENV, color="k", ls=":", lw=1)
        ax.annotate("HCP ~7", xy=(HCP_NENV, ax.get_ylim()[1]), fontsize=8, ha="center", va="top")
        ax.set_xlabel("number of environments (n_envs)")
        ax.grid(alpha=0.3)
        ax.legend()
    ax1.set_ylabel("(TRUE - BEST) / TRUE   (0 = optimum at the truth)")
    ax1.set_title(f"Falsification gap vs n_envs (d={NENV_D})")
    ax2.axhline(DEGEN_ENERGY, color="0.5", ls="--", lw=1)
    ax2.set_ylabel("block energy-on-true  (1 = within-block; low = optimum moved off truth)")
    ax2.set_title("Degeneracy vs n_envs")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_nenv.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def write_report_nenv(results, meta):
    L = []
    L.append("# Number of environments vs identifiability (n_envs independent of d)\n")
    L.append(f"- torch {meta['torch']}, numpy {meta['numpy']}, device {meta['device']}")
    L.append(f"- d={NENV_D}, K in {NENV_KS}, n_envs swept over {NENV_LIST}, rotation search "
             f"{ROT_RESTARTS} restarts x {ROT_STEPS} steps\n")
    L.append("Every prior run used n_envs = d (one environment per coordinate). Real task fMRI "
             f"gives ~{HCP_NENV} conditions (HCP: 7 tasks + rest) regardless of latent "
             "dimension, so **n_envs << d is the realistic regime** and this is its first test. "
             "Each environment perturbs K distinct coordinates; the paired difference is zero on "
             "any coordinate no environment touches, so uncovered coordinates are rotationally "
             "free and cannot be identified — identifiability over the COVERED set is reported "
             "separately from the aggregate.\n")

    rs = [r for r in results if r.get("phase") == "nenv"]
    L.append("## Sweep\n")
    L.append("| K | n_envs | coverage | cov-count (min..max) | TRUE | BEST | (T-B)/T | "
             "falsified | block e-o-t | degenerate | covered pinned | agg. identifiable |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for K in sorted({r["K"] for r in rs}):
        for n in sorted({r["n_envs"] for r in rs if r["K"] == K}, reverse=True):
            g = [r for r in rs if r["K"] == K and r["n_envs"] == n]
            def m(k):
                return np.mean([r[k] for r in g])
            L.append(f"| {K} | {n} | {m('coverage'):.1f}/{NENV_D} "
                     f"| {min(r['min_coverage_count'] for r in g)}..{max(r['max_coverage_count'] for r in g)} "
                     f"| {m('true_val'):.2f} | {m('best_val'):.2f} | {m('frac_gap'):.4f} "
                     f"| {np.mean([r['falsified'] for r in g]):.2f} "
                     f"| {m('block_energy_on_true'):.2f} "
                     f"| {np.mean([r['degenerate'] for r in g]):.2f} "
                     f"| {np.mean([r['covered_pinned'] for r in g]):.2f} "
                     f"| {m('agg_ident_frac'):.2f} |")
    L.append("")
    L.append("*coverage = distinct coordinates perturbed by >=1 environment (rest are free); "
             "cov-count = per-coordinate coverage count range; covered pinned = fraction of "
             "seeds where the covered coordinates are element-wise identifiable (not falsified, "
             "not degenerate); agg. identifiable = element-wise identifiable fraction of ALL d "
             "coordinates (covered_frac if pinned, else 0).*\n")

    # headline: minimum n_envs at which covered coords stay pinned, per K
    L.append("## Verdict\n")
    for K in sorted({r["K"] for r in rs}):
        ns = sorted({r["n_envs"] for r in rs if r["K"] == K})
        pinned_ns = [n for n in ns
                     if np.mean([r["covered_pinned"] for r in rs if r["K"] == K and r["n_envs"] == n]) >= 0.5]
        min_pin = min(pinned_ns) if pinned_ns else None
        if min_pin is not None:
            hcp_ok = any(n <= HCP_NENV for n in pinned_ns)
            L.append(f"- **K={K}: covered coordinates stay uniquely identified down to n_envs = "
                     f"{min_pin}.** "
                     + (f"n_envs = {HCP_NENV} (HCP-realistic) IS sufficient here."
                        if hcp_ok else
                        f"n_envs = {HCP_NENV} (HCP-realistic) is NOT sufficient — the smallest "
                        f"working count ({min_pin}) is well above it."))
        else:
            L.append(f"- **K={K}: NO tested n_envs keeps the covered coordinates identified** "
                     "(all falsified or degenerate). The optimum is off the truth throughout.")
    L.append(f"- The realistic count n_envs ≈ {HCP_NENV} is marked on the figure. Whether it "
             "lands in the identified region is the transfer-relevant answer: with ~7 "
             "conditions and a latent space of tens of dimensions, element-wise identification "
             "requires each coordinate to be constrained by enough overlapping environments, "
             "which few conditions cannot supply.")
    L.append("- **Coverage is not the bottleneck here** (K*n_envs >= d for these rungs, so the "
             "covered set is essentially all of d); the limiting factor is OVERLAP — how many "
             "environments constrain each coordinate — which falls as n_envs falls.\n")

    figs = [f for f in (fig_nenv(results),) if f]
    if figs:
        L.append("## Figures\n")
        for f in figs:
            L.append(f"- `{os.path.basename(f)}`")
        L.append("")

    atomic_write(os.path.join(OUT_DIR, "results.md"), "\n".join(L), is_json=False)


def run_nenv_sweep(n_seeds, device, log, meta, smoke=False):
    ks, nlist = NENV_KS, NENV_LIST
    if smoke:
        ks, nlist, n_seeds = [6], [30, 7], 1
        log("SMOKE (nenv): d=30 K=6, n_envs in {30,7}, 1 seed\n")
    results = []
    for K in ks:
        for n in nlist:
            for seed in range(n_seeds):
                p = os.path.join(SEED_DIR, f"nenv_d{NENV_D}_K{K}_n{n}_seed{seed:02d}.json")
                if os.path.exists(p):
                    with open(p) as f:
                        results.append(json.load(f))
                    log(f"[skip] nenv K={K} n_envs={n} seed {seed:02d}")
                    continue
                log(f"[run ] nenv K={K} n_envs={n} seed {seed:02d}")
                res = run_nenv_test(NENV_D, K, n, seed, device, log)
                atomic_write(p, res, is_json=True)
                results.append(res)
    write_report_nenv(results, meta)


# ------------------------------------------------------------------------- reporting


def write_report(results, meta):
    L = []
    L.append("# N-shift boundary: where multi-coordinate shifts falsify the sparse optimum\n")
    L.append(f"- torch {meta['torch']}, numpy {meta['numpy']}, scipy {meta['scipy']}, "
             f"sklearn {meta['sklearn']}")
    L.append(f"- device: {meta['device']} ({meta['gpu']})")
    L.append(f"- d={D_LATENT}, N={N_OBS}, {D_LATENT} envs, balanced circulant targets, "
             f"rotation search {ROT_RESTARTS} restarts x {ROT_STEPS} steps\n")

    def rfrac(K):
        rs = _rot_by_k(results, K)
        return np.mean([r["falsified"] for r in rs]) if rs else float("nan")

    L.append("## The decisive test — rotation search over K\n")
    L.append("BEST_ROTATION is seeded with identity, so BEST <= TRUE always; BEST < TRUE means "
             "a genuine sparser-than-truth rotation exists (FALSIFIED). block e-o-t = mean "
             "per-env energy the winning rotation keeps on the true target coordinates "
             "(~1 = within-block rotation; low = it moves the block).\n")
    L.append("| K | TRUE | RANDOM | BEST_ROTATION | beat | falsified frac | block e-o-t | "
             "signed-perm |")
    L.append("|---|---|---|---|---|---|---|---|")
    for K in K_LIST:
        rs = _rot_by_k(results, K)
        if not rs:
            continue
        def m(k):
            return np.mean([r[k] for r in rs])
        L.append(f"| {K} | {m('true_val'):.3f} | {m('random_val'):.3f} | {m('best_val'):.3f} "
                 f"| {m('beat'):.3f} | {rfrac(K):.2f} | {m('block_energy_on_true'):.2f} "
                 f"| {np.mean([r['signed_permutation'] for r in rs]):.2f} |")
    L.append("")

    # headline verdict: largest K not falsified in a majority of seeds
    passing = [K for K in K_LIST if _rot_by_k(results, K) and rfrac(K) < 0.5]
    boundary = max(passing) if passing else None
    L.append("## Verdict\n")
    if boundary is not None:
        L.append(f"- **The optimum is still at the truth up to K = {boundary}.** At K > "
                 f"{boundary} a sparser-than-truth rotation exists, so the sparse-paired-"
                 "difference objective is falsified there — no training can recover the latents "
                 "because the objective's own minimum is not at the truth.")
    else:
        L.append("- Even K=1 shows a sparser-than-truth rotation — that would contradict the "
                 "1-D floor and points to a bug in the rotation search or the construction.")
    L.append(f"- Real cognitive conditions perturb many coordinates at once (large K), so real "
             f"task-fMRI sits at K > {boundary if boundary else 1}: **in the falsified regime.** "
             "The boundary here is the honest ceiling on how many simultaneous mechanism "
             "shifts this identifiability argument tolerates.")
    for K in K_LIST:
        rs = _rot_by_k(results, K)
        if rs and rfrac(K) >= 0.5:
            eot = np.mean([r["block_energy_on_true"] for r in rs])
            kind = ("within-block (block identifiable, individual coords free)"
                    if eot >= BLOCK_PRESERVE_HI else
                    "moves the block (not even block-identifiable)" if eot <= BLOCK_MOVED_LO
                    else "partial (mixes within/across blocks)")
            L.append(f"  - K={K}: the winning rotation is {kind} (block e-o-t {eot:.2f}).")
    L.append("")

    # recovery table (passing K only)
    rec_ks = sorted({r["K"] for r in results if r.get("phase") == "recovery"})
    if rec_ks:
        L.append("## Closed-form recovery (K values that PASSED the rotation search)\n")
        L.append("For K>1 the estimator recovers a K-dim SUBSPACE per environment (top-K right "
                 "singular vectors); subspace-cos is the mean principal-angle cosine vs the true "
                 "subspace. Individual columns are pinned ONLY where the subspaces intersect — "
                 "recovered here via the shared-direction (projector-sum) construction, scored "
                 "as CF-MCC.\n")
        L.append("| K | condition | subspace-cos | rankK-gap | CF-MCC | cf-init MCC | MSE floor |")
        L.append("|---|---|---|---|---|---|---|")
        for K in rec_ks:
            for cond_name, _ in RECOVERY_CONDITIONS:
                rs = _rec(results, K, cond_name)
                if not rs:
                    continue
                def m(k):
                    v = [r[k] for r in rs if k in r and np.isfinite(r[k])]
                    return np.mean(v) if v else float("nan")
                L.append(f"| {K} | {cond_name} | {m('subspace_cos_mean'):.3f} "
                         f"| {m('rankK_gap_mean'):.2e} | {m('cf_mcc'):.3f} "
                         f"| {m('cfinit_mcc'):.3f} | {m('mse_mcc'):.3f} |")
        L.append("")

    L.append("## Caveats\n")
    L.append("1. **K counts target COORDINATES, not edges.** Several edges into one target "
             "stay 1-D; only distinct targets add subspace dimensions.")
    L.append("2. **For K>1 the closed form recovers subspaces, not columns.** Columns are "
             "pinned only where per-env K-dim subspaces intersect in a single shared direction; "
             "the CF-MCC column reflects that intersection succeeding, which it does at RHO=1 "
             "and degrades under independent pairing.")
    L.append("3. **The rotation search is a necessary test, not a sufficient proof.** It "
             "searches orthogonal rotations from many restarts; not finding a sparser rotation "
             "is strong evidence but not a theorem. Finding one IS a disproof.")
    L.append("4. **RHO=1 / additive is synthetic-only** (same-innovation pairing); the "
             "independent_recursive recovery row is the transfer-relevant one.\n")

    figs = [f for f in (fig_boundary(results),) if f]
    if figs:
        L.append("## Figures\n")
        for f in figs:
            L.append(f"- `{os.path.basename(f)}`")
        L.append("")

    atomic_write(os.path.join(OUT_DIR, "results.md"), "\n".join(L), is_json=False)


# ---------------------------------------------------------------------- oracle test


def run_oracle(log, device):
    """Self-test: simulator integrity, the rotation search reproduces Option B (K=1 not
    falsified) AND can find a known sparser-than-truth solution (K=6 falsified), and the K=1
    closed form recovers the mixing.
    """
    log("=" * 72)
    log("ORACLE HARNESS TEST")
    log("=" * 72)
    fails = []

    for seed in range(ORACLE_SEEDS):
        for K in K_LIST:
            try:
                make_dataset(seed, DIST, K)
            except RuntimeError as e:
                log(f"[FAIL] seed {seed} K={K}: simulator integrity — {e}")
                fails.append(f"integrity (seed {seed}, K={K}): {e}")
    if not fails:
        log(f"[ok  ] simulator integrity over {ORACLE_SEEDS} seeds x {len(K_LIST)} K values")

    # K=1 must NOT be falsified (the 1-D floor: BEST ~ TRUE).
    for seed in range(min(ORACLE_SEEDS, 3)):
        data = make_dataset(seed, DIST, 1)
        tv, rv, bv, _ = rotation_search(data["z_all"], seed, device)
        falsified = (tv - bv) > max(ROT_ABS_TOL, ROT_BEAT_FRAC * tv)
        ok = not falsified
        log(f"[{'ok  ' if ok else 'FAIL'}] K=1 seed {seed}: TRUE {tv:.3f} BEST {bv:.3f} "
            f"RANDOM {rv:.3f} -> {'not falsified' if ok else 'FALSIFIED (should not be!)'}")
        if not ok:
            fails.append(f"K=1 falsified (seed {seed}): TRUE {tv:.3f} vs BEST {bv:.3f}. The "
                         "1-D floor was violated — bug in the search or the construction.")

    # POWER CHECK: a KNOWN-falsifiable case (all envs share one 4-D block, candidate-2 style)
    # MUST be flagged falsified. This validates the search can find a sparser-than-truth
    # rotation when one provably exists, WITHOUT presupposing whether the circulant K=6 design
    # (10 different overlapping blocks) falsifies — that is the experiment, not an invariant.
    for seed in range(min(ORACLE_SEEDS, 3)):
        data = make_dataset(seed, DIST, 4, shared_targets=[0, 1, 2, 3])
        tv, rv, bv, _ = rotation_search(data["z_all"], seed, device)
        falsified = (tv - bv) > max(ROT_ABS_TOL, ROT_BEAT_FRAC * tv)
        ok = falsified
        log(f"[{'ok  ' if ok else 'FAIL'}] POWER (shared 4-D block) seed {seed}: TRUE {tv:.3f} "
            f"BEST {bv:.3f} -> {'falsified (as it must)' if ok else 'NOT falsified'}")
        if not ok:
            fails.append(f"power check failed (seed {seed}): the search did not falsify a "
                         "provably-falsifiable shared-block case. Raise ROT_RESTARTS / ROT_STEPS "
                         "— the discriminator lacks power and every rotation result is suspect.")

    # INFORMATIONAL: whether the circulant K=6 design falsifies is the EXPERIMENT, not asserted.
    for seed in range(min(ORACLE_SEEDS, 2)):
        data = make_dataset(seed, DIST, 6)
        tv, rv, bv, _ = rotation_search(data["z_all"], seed, device)
        falsified = (tv - bv) > max(ROT_ABS_TOL, ROT_BEAT_FRAC * tv)
        log(f"[info] circulant K=6 seed {seed}: TRUE {tv:.3f} BEST {bv:.3f} "
            f"-> {'falsified' if falsified else 'NOT falsified'} (this is the experiment, "
            "not a pass/fail)")

    # K=1 closed form recovers the mixing (subspace-cos ~1, MCC ~1) at RHO=1.
    for seed in range(min(ORACLE_SEEDS, 3)):
        data = make_dataset(seed, DIST, 1)
        U_est, _ = subspace_estimate(data["x_all"], 1)
        U_true = true_subspaces(data["A_eff"], data["target_sets"], 1)
        cos = np.mean([principal_angle_cos(U_est[c], U_true[c]) for c in range(D_LATENT)])
        A_hat = recover_columns_by_intersection(U_est, data["target_sets"], D_LATENT, N_OBS)
        m, _, _ = eval_columns(A_hat, data["x_all"], data["z_all"])
        ok = cos > 0.99 and m > 0.99
        log(f"[{'ok  ' if ok else 'FAIL'}] K=1 seed {seed}: closed-form subspace-cos {cos:.4f}, "
            f"MCC {m:.4f}")
        if not ok:
            fails.append(f"K=1 closed form (seed {seed}): cos {cos:.3f}, MCC {m:.3f}.")

    log("=" * 72)
    if fails:
        log("ORACLE TEST FAILED:")
        for f in fails:
            log("  - " + f)
        return False
    log("ORACLE TEST PASSED — integrity, rotation-search power (K=1 clean; a KNOWN-falsifiable "
        "shared-block case is correctly falsified), and K=1 closed form all hold. Whether the "
        "circulant multi-env design falsifies at any K is the experiment (see the [info] lines).")
    return True


# ------------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true", help="self-test only")
    ap.add_argument("--rotation-only", dest="rotation_only", action="store_true",
                    help="the d x K boundary sweep (d=10,20,30), no training")
    ap.add_argument("--nenv-sweep", dest="nenv_sweep", action="store_true",
                    help="vary the NUMBER of environments at d=30 (HCP-realistic n_envs << d)")
    ap.add_argument("--smoke", action="store_true", help="tiny sweep, 1 seed")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    os.makedirs(SEED_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "n/a"

    def log(msg):
        print(msg, flush=True)

    log("=" * 72)
    log("n-shift boundary — where multi-coordinate shifts falsify the sparse optimum")
    log(f"python  {platform.python_version()}  ({sys.platform})")
    log(f"torch   {torch.__version__}   cuda_available={torch.cuda.is_available()}")
    log(f"numpy   {np.__version__}   scipy {scipy.__version__}   sklearn {sklearn.__version__}")
    log(f"device  {device}   gpu: {gpu}")
    log("=" * 72)

    meta = dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, device=device, gpu=gpu)

    if args.oracle:
        sys.exit(0 if run_oracle(log, device) else 1)

    # --- the d x K boundary sweep is the --rotation-only deliverable ---
    if args.rotation_only:
        run_dsweep(args.seeds, device, log, meta, smoke=args.smoke)
        log("")
        log("=" * 72)
        log("ROTATION-ONLY d x K sweep — no recovery run.")
        log(f"wrote {os.path.join(OUT_DIR, 'results.md')}")
        return

    # --- the n_envs sweep (environments independent of d) ---
    if args.nenv_sweep:
        run_nenv_sweep(args.seeds, device, log, meta, smoke=args.smoke)
        log("")
        log("=" * 72)
        log(f"NENV sweep (d={NENV_D}) — no recovery run.")
        log(f"wrote {os.path.join(OUT_DIR, 'results.md')}")
        return

    n_seeds = args.seeds
    k_list = list(K_LIST)
    if args.smoke:
        k_list = [1, 6]
        n_seeds = 1
        log("SMOKE MODE: K in {1, 6}, 1 seed\n")

    results = []

    # --- phase 1: the rotation test (d=10) ---
    log("PHASE 1 — rotation search (the decisive test)")
    for K in k_list:
        for seed in range(n_seeds):
            p = os.path.join(SEED_DIR, f"rotation_d{D_LATENT}_K{K}_seed{seed:02d}.json")
            if os.path.exists(p):
                with open(p) as f:
                    results.append(json.load(f))
                log(f"[skip] rotation K={K} seed {seed:02d}")
                continue
            log(f"[run ] rotation K={K} seed {seed:02d}")
            res = run_rotation_test(K, seed, device, log)
            atomic_write(p, res, is_json=True)
            results.append(res)

    # --- phase 2: recovery for passing K (skipped in --rotation-only) ---
    if not args.rotation_only:
        def rfrac(K):
            rs = [r for r in results if r.get("phase") == "rotation" and r["K"] == K]
            return np.mean([r["falsified"] for r in rs]) if rs else 1.0
        passing = [K for K in k_list if rfrac(K) < 0.5]
        log(f"\nPHASE 2 — recovery for passing K = {passing}")
        for K in passing:
            for cond_name, cond in RECOVERY_CONDITIONS:
                for seed in range(n_seeds):
                    p = os.path.join(SEED_DIR, f"recovery_K{K}_{cond_name}_seed{seed:02d}.json")
                    if os.path.exists(p):
                        with open(p) as f:
                            results.append(json.load(f))
                        log(f"[skip] recovery K={K} {cond_name} seed {seed:02d}")
                        continue
                    log(f"[run ] recovery K={K} {cond_name} seed {seed:02d}")
                    res = run_recovery(K, seed, cond_name, cond, device, log)
                    atomic_write(p, res, is_json=True)
                    results.append(res)

    write_report(results, meta)
    log("")
    log("=" * 72)
    if args.smoke:
        log("SMOKE MODE — K in {1,6}, 1 seed. Not the full sweep.")
    log(f"wrote {os.path.join(OUT_DIR, 'results.md')}")


if __name__ == "__main__":
    main()
