#!/usr/bin/env python
"""
run_multienv_gate.py — ceiling-probe gate for ELEMENT-WISE causal representation learning
via ONE-INTERVENTION-PER-COORDINATE paired data.

QUESTION
    With d environments, each a counterfactual that shifts exactly ONE distinct latent
    coordinate's incoming mechanism, does the summed sparse-paired-difference objective
    recover the latents ELEMENT-WISE (MCC clearing the MSE floor), as the one-intervention-
    per-node theory (Squires et al.; Varici et al.) promises?

WHY THIS ONE IS DIFFERENT — the three prior candidates all died, each for its own reason:
    Candidate 0 (coefficient-difference sparsity): optimum in the WRONG place; the trained
        encoder out-sparsed the true latents.
    Candidate 1 (score-function-difference sparsity): optimum verifiably AT the truth, but
        the bilevel gradient through the inner-optimized score net could not REACH it.
    Candidate 2 (paired difference, 6-D style block): optimum in the WRONG place again — a
        rotation INSIDE the 6-dimensional style block beat the truth (oracle check 6 caught
        it). Block-identifiable, not element-wise.

    Candidate 3 (this file) removes the escape structurally. Each environment shifts ONE
    coordinate, so its paired difference is 1-DIMENSIONAL. Two facts, both verified
    numerically by the user before this was written:
      (1) A 1-D paired difference has L1/L2 >= 1, uniquely = 1 at axis-alignment. L1/L2 of
          ANY nonnegative vector is >= 1 (Cauchy-Schwarz), = 1 only when it is 1-hot. A
          single-coordinate direction rotated off-axis only grows its L1/L2. So no rotation
          can make it sparser than the truth. There is NO sparser-than-truth escape — the
          exact failure mode of candidates 0 and 2 is impossible by construction.
      (2) Summed over d environments each shifting a DISTINCT coordinate, the objective
          sum_c L1/L2(profile_c) >= d, with equality exactly at the truth (and its signed
          permutations). Verified: 200 optimization starts, the truth (= d) was never
          beaten, and the minimizers were permutation-like.

    So the summed objective's global minimum is d, achieved at the truth, and check 6 tests
    exactly that. Because the L1/L2 >= 1 floor is distribution-free, a PASS here is the
    first genuine element-wise identification in the queue. A FAIL — given the oracle
    confirms the optimum sits at the truth — would be a pure optimization/reachability
    result (candidate 1's disease), cleanly separable from an identifiability failure.

THE DATA — d environments, one intervention each (all counterfactual, exact pairing)
    Reference VAR coefs_0 (fixed simulator: bisection stabilize, self-loops on, loud coefs),
    then re-stabilized to REF_TARGET_RHO (< the env ceiling) so every coordinate has room to
    take a stationary intervention edge — without that headroom the dominant-mode coordinate
    bisects to a null intervention (caught by the first A100 oracle run).
    For each coordinate c, env_c shifts exactly ONE lagged off-diagonal coefficient whose
    TARGET row is c (an existing incoming edge if one exists, else a freshly created one),
    magnitude 1.5-2.5, with the alpha-bisection keeping env_c stationary. env_c therefore
    differs from the reference on exactly coordinate c's incoming dynamics and nowhere else.

    Pairing is the counterfactual construction from candidate 2, made exact and vectorized:
    env_c is driven by the SAME innovations AND the SAME history as the reference, so
        z_env_c[t] = z_ref[t] + (B_env_c - B_ref) @ past_ref[t]
    and (B_env_c - B_ref) is nonzero only in row c. Hence z_env_c - z_ref is supported on
    coordinate c ALONE, and every other coordinate is BIT-IDENTICAL to the reference. That
    is the 1-D difference the whole guarantee rests on; make_dataset asserts it.

THE OBJECTIVE — element-wise, EXACT gradient, no score net, no bilevel
    Encode the reference and every env_c with the SAME FlowEncoder. Standardize all latents
    to unit variance IN-GRAPH, pooled over ALL environments (the anti-gaming guard: scaling
    a coordinate scales the pooled sd identically and cancels). For each env_c form the
    paired difference dd_c = standardize(z_ref) - standardize(z_env_c) and penalize the
    L1/L2 of its per-coordinate norm profile.
        loss = sum_e recon(x_e) + LAMBDA_PRIOR*prior + LAMBDA_PAIR * sum_c L1/L2(profile_c)
    Every term is a closed-form function of the encoder's outputs, so d(loss)/d(encoder) is
    the exact total derivative — no inner optimization, nothing detached. LAMBDA_PRIOR = 0
    by default so any result is attributable to the pairing alone (candidate 1's confound).

READOUT — TRUE / TRAINED / RANDOM on the SUMMED paired-difference sparsity
    Summed L1/L2 in each model's OWN coordinates. TRUE = d (the floor, the truth). RANDOM =
    a random rotation of the true latents (well above d). TRAINED is the thing under test.
    TRAINED cannot beat TRUE — the floor forbids it — so the only outcomes are IDENTIFIED
    (TRAINED ~ d) or NOT-ACTED (TRAINED ~ RANDOM). This time element-wise MCC IS the right
    metric: the theory promises element-wise recovery, so MCC clearing the MSE floor is a
    genuine pass, not an arithmetic artifact. Every coordinate is a target of exactly one
    environment, so there are no "untouched" coordinates to leave rotationally free.

RUN — in this order
    python run_multienv_gate.py --oracle     # self-test; check 6 MUST pass by the math
    python run_multienv_gate.py --smoke      # oracle + 1 seed multienv + mse on Laplace
    python run_multienv_gate.py --sparsity   # detailed TRUE/TRAINED/RANDOM printout, seed 0
    python run_multienv_gate.py              # full 2x2 sweep, 12 seeds

RUNTIME (ESTIMATE — NOT MEASURED. Watch the first seed's printed timing.)
    d+1 = 11 environments encoded per step, but no score net anywhere. Rough guess 30-60 min
    for the full 2x2 at 12 seeds on an A100-80GB. Resumable: per-condition per-seed JSON,
    atomic write, finished seeds skipped.

OUTPUT
    ./results_multienv_gate/
        results.md                         summary table + verdict + caveats
        seeds/<condition>_seed<k>.json     per-run metrics
        fig_true_trained_random_<c>.png    the key readout, across seeds
        fig_profile_matrix_<c>.png         per-env paired-difference profile (should be ~I)
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
from scipy.stats import kurtosis, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ----------------------------------------------------------------------------- config

OUT_DIR = "./results_multienv_gate"
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
MCC_GATE = 0.4             # legacy gate. The MSE-floor delta is the real MCC verdict.

FLOW_COUPLINGS = 6
FLOW_HIDDEN = 256
TRAIN_STEPS = 5000
LR = 1e-3
BATCH_WINDOWS = 16         # 11 envs per step, so smaller than candidate 2's 32
RIDGE = 1e-3

# --- element-wise paired objective ---
LAMBDA_PAIR = 1.0          # weight on sum_c L1/L2(profile_c); recon is also summed over envs
LAMBDA_PRIOR = 0.0         # OFF by default: keep the result attributable to the pairing

AE_STEPS = 3000
AE_LR = 1e-3

CONDITIONS = [
    ("multienv_laplace", "multienv", "laplace"),
    ("multienv_gaussian", "multienv", "gaussian"),
    ("mse_laplace", "mse", "laplace"),
    ("mse_gaussian", "mse", "gaussian"),
]

LAGGED_COV_WARN = 0.1
ORACLE_SEEDS = 6
SQRT_EPS = 1e-8            # sqrt'(0) is infinite and the penalty aims at 0; see diff_profile

# --- intervention headroom (this is what the first A100 oracle run forced) ---
# make_var stabilizes the reference to rho = 0.9, the ENV ceiling. When the raw VAR is above
# 0.9 (loud coefficients usually are) that leaves the reference sitting AT 0.9 with zero room
# to add an intervention edge: the bisection then drives alpha -> 0 for whichever coordinate
# feeds the dominant eigenmode, env_c collapses back onto the reference, and its paired
# difference is 0 (not 1-D). Re-stabilizing the reference well below the env ceiling gives
# every coordinate room to take a stationary intervention. MIN_ALPHA is the fail-fast floor.
REF_TARGET_RHO = 0.5      # reference re-stabilized here; envs still allowed up to 0.9
ENV_TARGET_RHO = 0.9
MIN_ALPHA = 0.02          # an env whose intervention bisects below this is degenerate -> abort

# check 5 / 5b thresholds — the pairing must be real and each difference must be 1-D
PAIR_CORR_CONTENT_MIN = 0.98    # content coords are EXACTLY identical here -> r = 1
PAIR_CORR_SHUFFLED_MAX = 0.25
ONED_L1L2_MAX = 1.15           # per-env true paired difference must be ~1-hot (L1/L2 ~ 1)

# check 6 thresholds — the optimum must be AT the truth (= d)
ROT_SEARCH_STEPS = 600
ROT_SEARCH_LR = 0.02
ROT_SEARCH_TOL = 0.5           # summed over d envs; a rotation beating TRUE by more = bug
TRUE_SUM_TOL = 1.0             # TRUE summed L1/L2 must be within this of d

# ------------------------------------------------- simulator primitives (reused verbatim)

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


def make_dataset(seed, dist):
    # Separate streams: for a given seed the VAR, the interventions and the mixing are
    # IDENTICAL across dist; only the innovations differ, so Laplace vs Gaussian is a
    # controlled comparison.
    rng_struct = np.random.default_rng(seed)
    rng_innov = np.random.default_rng(1_000_000 + seed)
    rng_mix = np.random.default_rng(2_000_000 + seed)

    coefs_0 = make_var(rng_struct, D_LATENT, EDGES_PER_NODE, MAX_LAG)
    # Re-stabilize the reference BELOW the env ceiling so every coordinate has room to take
    # a stationary intervention. Without this, coordinates feeding the dominant eigenmode
    # bisect to alpha = 0 and their paired difference vanishes (the first A100 oracle failure).
    coefs_0 = stabilize(coefs_0, target_rho=REF_TARGET_RHO)
    coefs_envs, edges, alphas = [], [], []
    for c in range(D_LATENT):
        cc, edge, alpha = make_env_shift(rng_struct, coefs_0, c, MAX_LAG)
        coefs_envs.append(cc)
        edges.append(edge)
        alphas.append(alpha)

    # --- INVARIANTS (assert, do not assume) ---
    if companion_rho(coefs_0) >= 1.0:
        raise RuntimeError(f"seed {seed} {dist}: reference VAR non-stationary.")
    weak = [(c, round(a, 4)) for c, a in enumerate(alphas) if a < MIN_ALPHA]
    if weak:
        raise RuntimeError(
            f"seed {seed} {dist}: interventions collapsed for envs {weak} "
            f"(alpha < {MIN_ALPHA}). Those coordinates have no stability headroom even after "
            f"re-stabilizing the reference to rho={REF_TARGET_RHO}. Lower REF_TARGET_RHO "
            "further, or lower the intervention magnitude in make_env_shift.")
    for c, cc in enumerate(coefs_envs):
        if companion_rho(cc) >= 1.0:
            raise RuntimeError(f"seed {seed} {dist}: env {c} non-stationary "
                               f"(rho={companion_rho(cc):.4f}).")
        # env_c may differ from the reference ONLY in row c
        for l in range(MAX_LAG):
            diff = coefs_0[l] - cc[l]
            other = np.delete(diff, c, axis=0)
            if np.abs(other).max() > 1e-12:
                raise RuntimeError(
                    f"seed {seed} {dist}: env {c} differs from the reference OFF its target "
                    f"row (max {np.abs(other).max():.3e}); the difference would not be 1-D.")

    total = N_WINDOWS * WINDOW_L + BURN_IN
    z_ref_full, _ = simulate_reference(rng_innov, coefs_0, total, dist)
    if not np.isfinite(z_ref_full).all():
        raise RuntimeError(f"seed {seed} {dist}: reference latents contain inf/NaN.")

    z_full = [z_ref_full]
    for c in range(D_LATENT):
        z_full.append(z_ref_full + counterfactual_diff(z_ref_full, coefs_envs[c],
                                                        coefs_0, MAX_LAG))

    def win(zf):
        return zf[BURN_IN:].reshape(N_WINDOWS, WINDOW_L, D_LATENT)

    z_all = np.stack([win(zf) for zf in z_full], 0)      # (E, nw, L, d), index 0 = reference
    if not np.isfinite(z_all).all() or np.abs(z_all).max() > 1e6:
        raise RuntimeError(f"seed {seed} {dist}: latents blew up "
                           f"(max |z| = {np.abs(z_all).max():.3e}).")

    # THE PAIRING INVARIANT: env (1+c) is bit-identical to the reference off coordinate c.
    for c in range(D_LATENT):
        noncol = [k for k in range(D_LATENT) if k != c]
        resid = float(np.abs(z_all[0][:, :, noncol] - z_all[1 + c][:, :, noncol]).max())
        if resid > 1e-6:
            raise RuntimeError(
                f"seed {seed} {dist}: PAIRING BROKEN for env {c}. Off-target residual "
                f"{resid:.3e}; the difference is not 1-D on coordinate {c}.")

    A = make_mixing(rng_mix, D_LATENT, N_OBS)
    x_raw = z_all @ A.T                                  # (E, nw, L, N)
    flat = x_raw.reshape(-1, N_OBS)
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    x_all = (x_raw - mu) / sd

    return dict(
        x_all=x_all.astype(np.float32), z_all=z_all.astype(np.float32),
        coefs_0=coefs_0, coefs_envs=coefs_envs, A=A,
        edges=edges, alphas=alphas, targets=list(range(D_LATENT)),
        rho_0=companion_rho(coefs_0), rho_envs=[companion_rho(cc) for cc in coefs_envs],
        pairing_mode="counterfactual",
    )


# ------------------------------------------------------------------- data sanity block

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


# ----------------------------------------------------------- flow model (reused verbatim)

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

# --------------------------------------------------- latent VAR + guards (reused verbatim)

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


# ---------------------------------------------- THE OBJECTIVE (element-wise, exact gradient)


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


def measure_multienv_diff(z):
    """The key readout, in the latents' OWN coordinates. z: (E, ...) stacked. No MCC, no fit.

    summed_l1l2 = sum_c L1/L2(profile_c). At the truth every profile is 1-hot so each term
    is 1 and the sum is d (= n_env). Closed form — this candidate needs no score net.
    """
    with torch.no_grad():
        zs = standardize_multi(z.float())
        n_env = z.shape[0] - 1
        per_env, profiles, total = [], [], 0.0
        for c in range(n_env):
            prof, _ = diff_profile(zs[0] - zs[1 + c])
            l1l2 = float(prof.abs().sum() / (prof.pow(2).sum().sqrt() + 1e-12))
            per_env.append(l1l2)
            profiles.append((prof / (prof.max() + 1e-12)).cpu().numpy().tolist())
            total += l1l2
    return dict(summed_l1l2=float(total), mean_l1l2=float(total / max(n_env, 1)),
                per_env_l1l2=per_env, profiles=profiles)


def train_multienv_model(data, device, log):
    """Encoder trained under sum_c L1/L2 of the per-env paired difference. EXACT gradient.

    loss = sum_e recon(x_e) + LAMBDA_PRIOR*prior + LAMBDA_PAIR * sum_c L1/L2(profile_c)

    No score net, no inner optimization, nothing detached: d(loss)/d(encoder) is the true
    total derivative, which is the structural break from candidate 1.
    """
    x_all = torch.from_numpy(data["x_all"]).to(device)   # (E, nw, L, N)
    E, n = x_all.shape[0], x_all.shape[1]

    model = FlowEncoder(N_OBS, D_LATENT, FLOW_COUPLINGS, FLOW_HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_STEPS)

    nonfinite_grad_steps = 0
    for step in range(TRAIN_STEPS):
        idx = torch.randint(0, n, (BATCH_WINDOWS,), device=device)   # SAME index for all envs
        xb = x_all[:, idx]                                           # (E, B, L, N)
        z, logdet = model.encode(xb)                                # (E, B, L, d), (E, B, L)

        rec = ((model.reconstruct(xb) - xb) ** 2).mean(dim=(1, 2, 3)).sum()   # sum over envs
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
                log(f"      !! step {step}: non-finite gradient; skipping. "
                    f"rec={rec.item():.4f} l_pair={l_pair.item():.4f}")
            opt.zero_grad(set_to_none=True)
            sched.step()
            continue
        opt.step()
        sched.step()

        if step % 500 == 0:
            with torch.no_grad():
                sd_min = float(z.reshape(-1, D_LATENT).std(0).min())
            log(f"      step {step:5d}  loss {loss.item():9.3f}  rec {rec.item():7.3f}  "
                f"pair_sum {l_pair.item():6.3f}  sd_min {sd_min:7.4f}")

    if nonfinite_grad_steps:
        frac = nonfinite_grad_steps / TRAIN_STEPS
        log(f"    WARNING: {nonfinite_grad_steps}/{TRAIN_STEPS} ({frac:.1%}) steps skipped "
            "for non-finite gradients.")
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


# ----------------------------------------------------------------- metrics (reused verbatim)

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

def permute_mechanism(D, true_of_hat, d, max_lag):
    """Remap a (d*max_lag, d) mechanism difference from recovered into ground-truth order."""
    perm = np.array([true_of_hat[i] for i in range(d)])
    out = np.zeros_like(D)
    for l in range(max_lag):
        blk = D[l * d : (l + 1) * d, :]
        out[l * d : (l + 1) * d, :] = blk[np.ix_(np.argsort(perm), np.argsort(perm))]
    return out

def true_support_mask(shift_support, d, max_lag):
    """Mask over the (d*max_lag, d) mechanism matrix. Row = lagged source, col = target."""
    m = np.zeros((d * max_lag, d), dtype=bool)
    for (l, i, j) in shift_support:
        m[l * d + j, i] = True
    return m

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

def random_rotation(seed, d):
    rng = np.random.default_rng(10_000 + seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return Q


# ------------------------------------------- TRUE / TRAINED / RANDOM (the key readout)


def reference_multienv_diffs(seed, dist, log=None):
    """TRUE and RANDOM summed-paired-difference references for a (seed, dist). Closed form."""
    data = make_dataset(seed, dist)
    n_test = int(TEST_FRAC * N_WINDOWS)
    tr = slice(0, N_WINDOWS - n_test)
    z_tr = torch.from_numpy(data["z_all"][:, tr])            # (E, nw_tr, L, d)

    true = measure_multienv_diff(z_tr)
    true["var_r2"] = var_fit_quality(z_tr[0].double(), z_tr[0].double())[0]

    Q = torch.from_numpy(random_rotation(seed, D_LATENT)).float()
    z_rot = (z_tr.reshape(z_tr.shape[0], -1, D_LATENT) @ Q).reshape(z_tr.shape)
    rand = measure_multienv_diff(z_rot)
    rand["var_r2"] = var_fit_quality(z_rot[0].double(), z_rot[0].double())[0]

    return dict(true=true, random=rand, targets=data["targets"])


def rotation_search_multi(z, device, steps=ROT_SEARCH_STEPS, seed=0):
    """Directly optimize a rotation of the TRUE latents to minimize the summed objective.

    z: (E, ...) stacked. M = expm(A - A^T) is orthogonal, so the search cannot cheat by
    collapsing coordinates. Ap starts near 0 (M ~ I ~ the truth), so the search begins AT
    the truth and can only try to go lower. By the L1/L2 >= 1 floor the sum cannot drop
    below d, so BEST >= TRUE = d whenever every env difference is genuinely 1-D. If BEST
    lands clearly below TRUE, some env's TRUE difference was NOT 1-hot (TRUE > d) and a
    rotation clawed it back toward d -> a setup bug, which check 6 reports.
    Returns (best_sum, best_M).
    """
    E = z.shape[0]
    zs = z.reshape(E, -1, D_LATENT).to(device).float()
    g = torch.Generator(device="cpu").manual_seed(seed)
    Ap = (0.01 * torch.randn(D_LATENT, D_LATENT, generator=g)).to(device).requires_grad_(True)
    opt = torch.optim.Adam([Ap], lr=ROT_SEARCH_LR)

    best, best_M = float("inf"), None
    for _ in range(steps):
        M = torch.matrix_exp(Ap - Ap.T)
        rot = zs @ M.T
        rot_s = standardize_multi(rot)
        loss = rot.new_zeros(())
        for c in range(E - 1):
            prof, _ = diff_profile(rot_s[0] - rot_s[1 + c])
            loss = loss + sparsity_ratio(prof)
        if float(loss.item()) < best:
            best = float(loss.item())
            best_M = M.detach().cpu().numpy().copy()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return best, best_M


def is_signed_permutation(M, tol=0.1):
    """Is M (approximately) a signed permutation? Each row and column one dominant |entry|."""
    A = np.abs(M)
    row_ok = np.all(np.sort(A, axis=1)[:, -1] > 1.0 - tol)
    col_ok = np.all(np.sort(A, axis=0)[-1, :] > 1.0 - tol)
    row_sec = np.all(np.sort(A, axis=1)[:, -2] < tol)
    return bool(row_ok and col_ok and row_sec)


def sparsity_diagnostic(seed, device, log):
    """Detailed standalone printout of the key readout, on Laplace data. --sparsity mode."""
    log("=" * 72)
    log(f"MULTI-ENV PAIRED-DIFFERENCE DIAGNOSTIC (seed {seed}) — permutation-free")
    log("=" * 72)

    data = make_dataset(seed, "laplace")
    n_test = int(TEST_FRAC * N_WINDOWS)
    tr = slice(0, N_WINDOWS - n_test)
    train = dict(x_all=data["x_all"][:, tr], z_all=data["z_all"][:, tr])

    refs = reference_multienv_diffs(seed, "laplace", log)
    t, r = refs["true"], refs["random"]

    log("    [trained] training the multi-env paired encoder")
    model, _ = train_multienv_model(train, device, log)
    model.eval()
    with torch.no_grad():
        z = model.encode(torch.from_numpy(train["x_all"]).to(device))[0].cpu()
    m_pd = measure_multienv_diff(z)
    var_m = var_fit_quality(z[0].double(), z[0].double())[0]

    best, best_M = rotation_search_multi(torch.from_numpy(train["z_all"]), device, seed=seed)

    log("-" * 72)
    log(f"  TRUE unmixing   : summed L1/L2 = {t['summed_l1l2']:.3f}  (d = {D_LATENT})  "
        f"per-env mean = {t['mean_l1l2']:.3f}  VAR R2 = {t['var_r2']:.3f}")
    log(f"  TRAINED model   : summed L1/L2 = {m_pd['summed_l1l2']:.3f}  "
        f"per-env mean = {m_pd['mean_l1l2']:.3f}  VAR R2 = {var_m:.3f}")
    log(f"  RANDOM rotation : summed L1/L2 = {r['summed_l1l2']:.3f}  "
        f"per-env mean = {r['mean_l1l2']:.3f}  VAR R2 = {r['var_r2']:.3f}")
    log(f"  BEST ROTATION   : summed L1/L2 = {best:.3f}   <- direct search "
        f"(is signed permutation: {is_signed_permutation(best_M)})")
    log("-" * 72)
    log(f"  Each per-env L1/L2 >= 1 ALWAYS (Cauchy-Schwarz), = 1 only when 1-hot. So the")
    log(f"  summed objective >= d = {D_LATENT}, achieved at the truth. TRAINED cannot beat")
    log("  TRUE; the only failure is TRAINED ~ RANDOM (objective did not act).")
    log("  VERDICT:")
    log(f"    TRAINED ~ TRUE (~{D_LATENT}) with VAR R2 ~ TRUE -> element-wise IDENTIFIED.")
    log("    TRAINED ~ RANDOM -> the objective did not act (candidate 1's disease; here the")
    log("      gradient is exact, so suspect LAMBDA_PAIR / optimizer / signal strength).")
    log("=" * 72)


# ------------------------------------------------------------------------- single run


def _encode_all(model, x_np, device):
    """Encode a stacked (E, nw, L, N) array. Returns (E, nw, L, d) torch on cpu."""
    with torch.no_grad():
        return model.encode(torch.from_numpy(x_np).to(device))[0].cpu()


def run_one(condition, objective, dist, seed, device, log, sanity_cache, ref_cache):
    t0 = time.time()
    data = make_dataset(seed, dist)
    E = D_LATENT + 1

    key = (seed, dist)
    if key not in sanity_cache:
        sanity_cache[key] = data_sanity(data, dist, seed, log)
    sanity = sanity_cache[key]

    n_test = int(TEST_FRAC * N_WINDOWS)
    tr = slice(0, N_WINDOWS - n_test)
    te = slice(N_WINDOWS - n_test, N_WINDOWS)
    train = dict(x_all=data["x_all"][:, tr], z_all=data["z_all"][:, tr])
    test = dict(x_all=data["x_all"][:, te], z_all=data["z_all"][:, te])

    if key not in ref_cache:
        ref_cache[key] = reference_multienv_diffs(seed, dist, log)
    refs = ref_cache[key]

    torch.manual_seed(seed)
    if objective == "multienv":
        model, train_info = train_multienv_model(train, device, log)
    else:
        model, train_info = train_mse_baseline(train, device, log)
    model.eval()

    # HELD-OUT windows: recovery metrics.
    z_te = _encode_all(model, test["x_all"], device)                 # (E, nw, L, d)
    z_hat = z_te.reshape(-1, D_LATENT).numpy()
    z_true = test["z_all"].reshape(-1, D_LATENT)
    with torch.no_grad():
        x_te = torch.from_numpy(test["x_all"]).to(device)
        x_rec = model.reconstruct(x_te).cpu().numpy()

    if not np.isfinite(z_hat).all():
        raise RuntimeError(f"{condition} seed {seed}: non-finite latents; training diverged.")

    m, true_of_hat, per_true = mcc_and_matching(z_true, z_hat)
    r2 = subspace_r2(z_true, z_hat)
    rr = recon_r(test["x_all"], x_rec)

    # TRAIN windows: the paired-difference readout (matches the TRUE/RANDOM reference windows).
    z_tr = _encode_all(model, train["x_all"], device)
    trained_pd = measure_multienv_diff(z_tr)
    var_m = var_fit_quality(z_tr[0].double(), z_tr[0].double())[0]

    # localization: does each env's profile peak at its true target coordinate?
    targets = refs["targets"]
    hits = 0
    prof_matrix = []
    for c in range(D_LATENT):
        prof = np.array(trained_pd["profiles"][c])
        prof_true = permute_vector(prof, true_of_hat, D_LATENT)      # into true-coord order
        prof_matrix.append((prof_true / (prof_true.max() + 1e-12)).tolist())
        if int(np.argmax(prof_true)) == targets[c]:
            hits += 1
    hit_rate = hits / D_LATENT

    # PER-COORDINATE DIAGNOSTIC — the key readout for attributing a low aggregate MCC.
    # per_true[c] is the Hungarian-matched MCC of true coordinate c; alphas[c] is the
    # intervention strength on coordinate c. Both are indexed by true coordinate, so a
    # starved coordinate (low alpha) failing to be recovered (low MCC) shows up on one line.
    per_coord_mcc = per_true.tolist()
    if objective == "multienv":
        log(f"       per-coordinate (coord: alpha -> MCC), aggregate MCC {m:.3f}:")
        for c in range(D_LATENT):
            mark = "   <-- weak intervention" if data["alphas"][c] < 0.5 else ""
            log(f"         z{c}: alpha {data['alphas'][c]:.3f} -> MCC {per_true[c]:.3f}{mark}")

    # every coordinate is a target of exactly one env => all touched, none untouched
    res = dict(
        condition=condition, objective=objective, dist=dist, seed=seed,
        mcc=m, mcc_shift_touched=m, mcc_shift_untouched=float("nan"),
        subspace_r2=r2, recon_r=rr,
        summed_l1l2_trained=trained_pd["summed_l1l2"],
        summed_l1l2_true=refs["true"]["summed_l1l2"],
        summed_l1l2_random=refs["random"]["summed_l1l2"],
        mean_l1l2_trained=trained_pd["mean_l1l2"],
        mean_l1l2_true=refs["true"]["mean_l1l2"],
        mean_l1l2_random=refs["random"]["mean_l1l2"],
        var_r2_trained=[var_m], var_r2_true=[refs["true"]["var_r2"]],
        var_r2_random=[refs["random"]["var_r2"]],
        pair_topk_hit_rate=hit_rate,
        profile_matrix=prof_matrix, targets=targets,
        alphas=data["alphas"], per_coord_mcc=per_coord_mcc,
        nonfinite_grad_steps=train_info["nonfinite_grad_steps"],
        sanity=sanity, seconds=time.time() - t0,
    )
    return res


# ------------------------------------------------------------------------------- io


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


# -------------------------------------------------------------------------- figures


def fig_true_trained_random(condition, results):
    """THE key figure: summed paired-difference sparsity, TRUE vs TRAINED vs RANDOM."""
    if not results:
        return None
    rs = sorted(results, key=lambda r: r["seed"])
    seeds = [r["seed"] for r in rs]
    tru = [r["summed_l1l2_true"] for r in rs]
    trn = [r["summed_l1l2_trained"] for r in rs]
    rnd = [r["summed_l1l2_random"] for r in rs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    xs = np.arange(len(seeds))
    w = 0.27
    ax1.bar(xs - w, tru, w, label="TRUE (floor = d, identified)", color="#55A868")
    ax1.bar(xs, trn, w, label="TRAINED", color="#4C72B0")
    ax1.bar(xs + w, rnd, w, label="RANDOM (not identified)", color="#C44E52")
    ax1.axhline(D_LATENT, color="k", ls=":", lw=1, label=f"d = {D_LATENT} (global floor)")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(seeds)
    ax1.set_xlabel("seed")
    ax1.set_ylabel("summed paired-difference L1/L2")
    ax1.set_title(f"{condition}: key readout\nTRAINED cannot beat TRUE; goal is TRAINED = TRUE")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(xs - w, [np.mean(r["var_r2_true"]) for r in rs], w, label="TRUE", color="#55A868")
    ax2.bar(xs, [np.mean(r["var_r2_trained"]) for r in rs], w, label="TRAINED", color="#4C72B0")
    ax2.bar(xs + w, [np.mean(r["var_r2_random"]) for r in rs], w, label="RANDOM", color="#C44E52")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(seeds)
    ax2.set_xlabel("seed")
    ax2.set_ylabel("reference VAR one-step R2")
    ax2.set_title("dynamics faithfulness control\n(VAR rotation-closed: RANDOM ~ TRUE)")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"fig_true_trained_random_{condition}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_profile_matrix(condition, results):
    """Per-env paired-difference profile in true-coord order (seed 0). Should be ~identity."""
    r0 = next((r for r in results if r["seed"] == 0), None)
    if r0 is None or "profile_matrix" not in r0:
        return None
    P = np.array(r0["profile_matrix"])            # (n_env, d), row c should peak at target c
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    im = ax.imshow(P, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    for c, tgt in enumerate(r0["targets"]):
        ax.add_patch(plt.Rectangle((tgt - 0.5, c - 0.5), 1, 1, fill=False,
                                   edgecolor="red", lw=2.0))
    ax.set_xlabel("recovered coordinate (true-coord order)")
    ax.set_ylabel("environment (intervention target)")
    ax.set_title(f"{condition}, seed 0: per-env paired-difference profile\n"
                 f"red = true target; diagonal = element-wise recovery "
                 f"(MCC {r0['mcc']:.3f}, hit {r0['pair_topk_hit_rate']:.2f})")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"fig_profile_matrix_{condition}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ------------------------------------------------------------------------- reporting


def summarize(all_results):
    rows = []
    for cond, obj, dist in CONDITIONS:
        rs = [r for r in all_results if r["condition"] == cond]
        if not rs:
            continue

        def ms(key, fn=None):
            v = np.array([fn(r) if fn else r[key] for r in rs], dtype=np.float64)
            return float(np.nanmean(v)), float(np.nanstd(v))

        rows.append(dict(
            condition=cond, objective=obj, dist=dist, n=len(rs),
            mcc=ms("mcc"), r2=ms("subspace_r2"), rr=ms("recon_r"),
            hit=ms("pair_topk_hit_rate"),
            s_true=ms("summed_l1l2_true"), s_trained=ms("summed_l1l2_trained"),
            s_random=ms("summed_l1l2_random"),
            var_true=ms(None, lambda r: np.mean(r["var_r2_true"])),
            var_trained=ms(None, lambda r: np.mean(r["var_r2_trained"])),
            var_random=ms(None, lambda r: np.mean(r["var_r2_random"])),
        ))
    return rows


def write_report(rows, all_results, figs, meta):
    L, verdict = [], []
    L.append("# Multi-env one-intervention-per-coordinate gate\n")
    L.append(f"- torch {meta['torch']}, numpy {meta['numpy']}, scipy {meta['scipy']}, "
             f"sklearn {meta['sklearn']}")
    L.append(f"- device: {meta['device']} ({meta['gpu']})")
    L.append(f"- d={D_LATENT}, N={N_OBS}, max_lag={MAX_LAG}, {D_LATENT} intervention envs + "
             f"1 reference, windows={N_WINDOWS}x{WINDOW_L}, seeds up to {N_SEEDS}")
    L.append(f"- LAMBDA_PAIR={LAMBDA_PAIR}, LAMBDA_PRIOR={LAMBDA_PRIOR} (prior off => result "
             "attributable to the pairing alone)")
    L.append("- ceiling probe: no HRF, no observation noise, exact counterfactual pairing")
    L.append("- signal: sum_c L1/L2 of the per-env paired difference; each term >= 1 by "
             f"Cauchy-Schwarz, so the objective >= d = {D_LATENT}, uniquely met at the "
             "truth. No sparser-than-truth escape exists.\n")

    voids = [r for r in all_results if r["sanity"]["void"]]
    if voids:
        L.append(f"> **{len(voids)} runs had a VOID sanity check.** Numbers meaningless.\n")

    L.append("## THE KEY READOUT — permutation-free summed paired-difference sparsity\n")
    L.append(f"Summed L1/L2 in each model's OWN coordinates. TRUE = d = {D_LATENT} (the "
             "global floor); TRAINED cannot go below it. **TRAINED ~ TRUE -> element-wise "
             "identified. TRAINED ~ RANDOM -> the objective did not act.**\n")
    L.append("| condition | summed L1/L2 TRUE | TRAINED | RANDOM | VAR R2 TRUE | "
             "VAR R2 TRAINED | VAR R2 RANDOM |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(
            f"| {r['condition']} "
            f"| {r['s_true'][0]:.3f} +/- {r['s_true'][1]:.3f} "
            f"| {r['s_trained'][0]:.3f} +/- {r['s_trained'][1]:.3f} "
            f"| {r['s_random'][0]:.3f} +/- {r['s_random'][1]:.3f} "
            f"| {r['var_true'][0]:.3f} | {r['var_trained'][0]:.3f} | {r['var_random'][0]:.3f} |")
    L.append("")

    L.append("## Recovery metrics (mean +/- std over seeds)\n")
    L.append("| condition | objective | data | n | MCC | subspace R2 | recon r | "
             "per-env localization hit |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(
            f"| {r['condition']} | {r['objective']} | {r['dist']} | {r['n']} "
            f"| {r['mcc'][0]:.3f} +/- {r['mcc'][1]:.3f} "
            f"| {r['r2'][0]:.3f} +/- {r['r2'][1]:.3f} "
            f"| {r['rr'][0]:.3f} +/- {r['rr'][1]:.3f} "
            f"| {r['hit'][0]:.2f} +/- {r['hit'][1]:.2f} |")
    L.append("")

    by = {r["condition"]: r for r in rows}
    L.append("## Verdict\n")

    for dist in ("laplace", "gaussian"):
        cond = f"multienv_{dist}"
        if cond not in by:
            continue
        r = by[cond]
        t, m_, rd = r["s_true"][0], r["s_trained"][0], r["s_random"][0]
        span = abs(rd - t)
        tol = max(0.1 * max(span, 1e-9), 0.3)
        if abs(m_ - t) <= tol:
            call = "IDENTIFIED - TRAINED matches TRUE (element-wise recovery)."
        elif abs(m_ - rd) <= tol:
            call = ("NOT IDENTIFIED - TRAINED sits at RANDOM. The gradient is exact, so this "
                    "is not a bilevel failure: suspect LAMBDA_PAIR, the optimizer, or the "
                    "signal strength (check per-env alpha).")
        else:
            frac = (rd - m_) / (span + 1e-12)
            call = f"PARTIAL - TRAINED is {frac:.0%} of the way from RANDOM to TRUE."
        line = f"**multienv + {dist} - {call}** (TRUE {t:.3f}, TRAINED {m_:.3f}, RANDOM {rd:.3f})"
        L.append(line + "\n")
        verdict.append(line)

        vt, vm = r["var_true"][0], r["var_trained"][0]
        if vm < vt - 0.1:
            line = (f"  - reference VAR R2 dropped from {vt:.3f} (TRUE) to {vm:.3f} (TRAINED): "
                    "the encoder degraded the dynamics.")
            L.append(line + "\n")
            verdict.append(line)

    for dist in ("laplace", "gaussian"):
        mc, ms_ = f"multienv_{dist}", f"mse_{dist}"
        if mc not in by or ms_ not in by:
            continue
        a, sa = by[mc]["mcc"]
        b, sb = by[ms_]["mcc"]
        delta = a - b
        pooled = float(np.sqrt(sa ** 2 + sb ** 2))
        line = (f"**MCC vs the MSE floor ({dist}): multienv {a:.3f} +/- {sa:.3f}, "
                f"MSE {b:.3f} +/- {sb:.3f}, DELTA {delta:+.3f}** (pooled sd {pooled:.3f})")
        L.append(line + "\n")
        verdict.append(line)
        if delta > pooled:
            line = ("  - Clears the rotationally-blind floor by more than a pooled sd. Because "
                    "the theory promises ELEMENT-WISE recovery here, this is a genuine pass, "
                    "not a matching artifact.")
        else:
            line = ("  - Does NOT clear the MSE floor by a pooled sd. Not element-wise "
                    "identification, whatever the absolute MCC.")
        L.append(line + "\n")
        verdict.append(line)

    L.append("## Caveats — read before quoting these numbers\n")
    L.append("1. **TRAINED cannot beat TRUE, by construction.** Each per-env term is L1/L2 of "
             "a nonnegative vector, hence >= 1, so the sum >= d and the truth is a global "
             "minimum. The degenerate out-sparsing that killed candidates 0 and 2 cannot "
             "happen. The only failure is TRAINED failing to reach TRUE, which the exact "
             "gradient is meant to prevent.")
    L.append("2. **The pairing is a free counterfactual.** env_c shares the reference's "
             "innovations AND history, so the difference is exactly 1-D on coordinate c and "
             "content is bit-identical elsewhere. Real data does not hand you paired "
             "counterfactuals; a pass here says nothing about obtaining them.")
    L.append("3. **The Gaussian arm is not a negative control.** The paired difference "
             "(B_env - B_ref) @ past is structural, independent of the innovation "
             "distribution, so both arms should behave alike. A difference implicates the "
             "flow, not identifiability.")
    L.append("4. **Weak interventions (small alpha) weaken a coordinate's signal.** If some "
             "env's alpha is small its paired difference is faint and that coordinate may not "
             "be pinned. The sanity block prints per-env alpha; oracle check 5b enforces "
             "1-D-ness.")
    L.append("5. **Ceiling probe only.** No HRF, no noise, linear instantaneous mixing, "
             "correctly specified lag order. Necessary, not sufficient.\n")

    if figs:
        L.append("## Figures\n")
        for f in figs:
            L.append(f"- `{os.path.basename(f)}`")
        L.append("")

    atomic_write(os.path.join(OUT_DIR, "results.md"), "\n".join(L), is_json=False)
    return verdict


# ---------------------------------------------------------------------- oracle test


def run_oracle(log, device):
    """Harness self-test. Must pass or the run aborts. Check 6 must pass BY THE MATH:
    the summed objective's floor is d, achieved at the truth, so a rotation cannot beat it.
    """
    log("=" * 72)
    log("ORACLE HARNESS TEST (no encoder training)")
    log(f"one intervention per coordinate: {D_LATENT} envs + 1 reference, counterfactual")
    log("=" * 72)
    fails = []

    # 0. SIMULATOR INTEGRITY across every seed the sweep will use.
    weak_alpha = []
    for seed in range(max(N_SEEDS, ORACLE_SEEDS)):
        for dist in ("laplace", "gaussian"):
            try:
                data = make_dataset(seed, dist)
            except RuntimeError as e:
                log(f"[FAIL] seed {seed} {dist}: simulator integrity — {e}")
                fails.append(f"simulator integrity (seed {seed}, {dist}): {e}")
                continue
            weak_alpha += [(seed, dist, c, a) for c, a in enumerate(data["alphas"]) if a < 0.3]
    if not fails:
        log(f"[ok  ] simulator integrity over {max(N_SEEDS, ORACLE_SEEDS)} seeds x 2 dists: "
            "stationary, finite, each env differs on ONE target row, pairing 1-D")
    if weak_alpha:
        log(f"[warn] {len(weak_alpha)} (seed,dist,env) had intervention alpha < 0.3 "
            "(weak signal on that coordinate).")

    # 1. VAR-fit convention on the reference's true latents.
    for seed in range(ORACLE_SEEDS):
        data = make_dataset(seed, "laplace")
        B = fit_latent_var(torch.from_numpy(data["z_all"][0]).double(), MAX_LAG, 1e-8).numpy()
        true_B = np.zeros_like(B)
        for l in range(MAX_LAG):
            for i in range(D_LATENT):
                for j in range(D_LATENT):
                    true_B[l * D_LATENT + j, i] = data["coefs_0"][l][i, j]
        err = float(np.abs(B - true_B).max())
        ok = err < 0.05
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: VAR fit on reference latents, "
            f"max|B_hat - B_true| = {err:.4f}")
        if not ok:
            fails.append(f"VAR fit convention (seed {seed}, err {err:.4f})")

    # 2. mechanism-difference localization is not applicable (counterfactual envs).
    log("[skip] check 2 (mechanism-difference localization) — not applicable under the "
        "counterfactual construction; env latents are not env VAR trajectories.")

    # 3. permute round-trips
    rng = np.random.default_rng(0)
    D_true = rng.random((D_LATENT * MAX_LAG, D_LATENT))
    perm = rng.permutation(D_LATENT)
    D_rec = np.zeros_like(D_true)
    for l in range(MAX_LAG):
        for r in range(D_LATENT):
            for c in range(D_LATENT):
                D_rec[l * D_LATENT + r, c] = D_true[l * D_LATENT + perm[r], perm[c]]
    err = float(np.abs(permute_mechanism(D_rec, {i: int(perm[i]) for i in range(D_LATENT)},
                                         D_LATENT, MAX_LAG) - D_true).max())
    ok = err < 1e-10
    log(f"[{'ok  ' if ok else 'FAIL'}] permute_mechanism round-trip, max err {err:.2e}")
    if not ok:
        fails.append(f"permute_mechanism ({err:.2e})")
    v_true = rng.random(D_LATENT)
    perm2 = rng.permutation(D_LATENT)
    v_rec = np.array([v_true[perm2[i]] for i in range(D_LATENT)])
    err = float(np.abs(permute_vector(v_rec, {i: int(perm2[i]) for i in range(D_LATENT)},
                                      D_LATENT) - v_true).max())
    ok = err < 1e-12
    log(f"[{'ok  ' if ok else 'FAIL'}] permute_vector round-trip, max err {err:.2e}")
    if not ok:
        fails.append(f"permute_vector ({err:.2e})")

    # 4. end to end: oracle unmixing through the real readout, over ALL envs.
    for seed in range(ORACLE_SEEDS):
        data = make_dataset(seed, "laplace")
        A = data["A"]
        E = D_LATENT + 1
        z_all = data["z_all"]
        flat = np.concatenate([(z_all[e].reshape(-1, D_LATENT) @ A.T) for e in range(E)], 0)
        A_eff = A / (flat.std(0) + 1e-8)[:, None]
        perm = np.random.default_rng(seed).permutation(D_LATENT)
        W = torch.from_numpy(np.linalg.pinv(A_eff)[perm, :]).float()
        x_all = torch.from_numpy(data["x_all"])
        z_hat = (x_all.reshape(-1, N_OBS) @ W.T).numpy()
        z_true = z_all.reshape(-1, D_LATENT)
        m, true_of_hat, _ = mcc_and_matching(z_true, z_hat)
        found = true_of_hat == {i: int(perm[i]) for i in range(D_LATENT)}
        ok = m > 0.95
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: ORACLE end to end, MCC {m:.4f}, "
            f"permutation recovered: {found}")
        if not ok:
            fails.append(f"oracle end to end (seed {seed}, MCC {m:.3f})")

    # 5. THE PAIRING IS REAL — content shared, shuffling destroys it.
    log("-" * 72)
    log("CHECK 5 — IS THE PAIRING REAL?")
    for seed in range(ORACLE_SEEDS):
        data = make_dataset(seed, "laplace")
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
        c_un = float(np.min(c_un_all))
        c_sh = float(np.max(c_sh_all))
        ok_un = c_un >= PAIR_CORR_CONTENT_MIN
        ok_sh = c_sh <= PAIR_CORR_SHUFFLED_MAX
        ok = ok_un and ok_sh
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: content r_min={c_un:.4f} "
            f"(need >= {PAIR_CORR_CONTENT_MIN}), shuffled r_max={c_sh:.3f} "
            f"(need <= {PAIR_CORR_SHUFFLED_MAX})")
        if not ok_un:
            fails.append(f"PAIRING BROKEN (seed {seed}): content only paired at r={c_un:.3f}.")
        if not ok_sh:
            fails.append(f"PAIRING FAKE (seed {seed}): shuffled r={c_sh:.3f} too high.")

    # 5b. EACH env difference is 1-D on its target coordinate.
    log("-" * 72)
    log("CHECK 5b — IS EACH PAIRED DIFFERENCE 1-D ON ITS TARGET COORDINATE?")
    for seed in range(ORACLE_SEEDS):
        data = make_dataset(seed, "laplace")
        z_all = data["z_all"]
        bad = []
        l1l2s = []
        for c in range(D_LATENT):
            dd = (z_all[0] - z_all[1 + c]).reshape(-1, D_LATENT)
            prof = np.sqrt((dd ** 2).mean(0) + SQRT_EPS)
            l1l2 = float(prof.sum() / (np.linalg.norm(prof) + 1e-12))
            l1l2s.append(l1l2)
            if l1l2 > ONED_L1L2_MAX or int(np.argmax(prof)) != c:
                bad.append((c, round(l1l2, 3), int(np.argmax(prof)), round(data["alphas"][c], 3)))
        ok = not bad
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: per-env L1/L2 "
            f"max={max(l1l2s):.3f} (need <= {ONED_L1L2_MAX}), all argmax on target = {ok}")
        if bad:
            fails.append(f"NOT 1-D (seed {seed}): envs (c, L1/L2, argmax, alpha) {bad}. "
                         "The one-intervention-per-node premise is broken; the guarantee "
                         "does not hold. Likely an env whose alpha was bisected too small.")

    # 6. IS THE OPTIMUM AT THE TRUTH? Direct rotation search. MUST pass by the math.
    log("-" * 72)
    log("CHECK 6 — IS THE OBJECTIVE'S OPTIMUM AT THE TRUTH? (direct rotation search)")
    for seed in range(min(ORACLE_SEEDS, 3)):
        data = make_dataset(seed, "laplace")
        n_test = int(TEST_FRAC * N_WINDOWS)
        tr = slice(0, N_WINDOWS - n_test)
        z_tr = torch.from_numpy(data["z_all"][:, tr])
        true = measure_multienv_diff(z_tr)
        Q = torch.from_numpy(random_rotation(seed, D_LATENT)).float()
        z_rot = (z_tr.reshape(z_tr.shape[0], -1, D_LATENT) @ Q).reshape(z_tr.shape)
        rand = measure_multienv_diff(z_rot)
        best, best_M = rotation_search_multi(z_tr, device, seed=seed)

        t_sum = true["summed_l1l2"]
        true_is_d = abs(t_sum - D_LATENT) <= TRUE_SUM_TOL
        sparser = t_sum < rand["summed_l1l2"] - 0.5
        optimum_at_truth = best >= t_sum - ROT_SEARCH_TOL
        perm_like = is_signed_permutation(best_M)
        ok = true_is_d and sparser and optimum_at_truth
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: TRUE={t_sum:.3f} (d={D_LATENT})  "
            f"RANDOM={rand['summed_l1l2']:.3f}  BEST_ROTATION={best:.3f}  "
            f"(best is signed permutation: {perm_like})")
        if not true_is_d:
            fails.append(f"TRUE != d (seed {seed}): summed L1/L2 = {t_sum:.3f} vs d={D_LATENT}. "
                         "Some env difference is not 1-D; see check 5b.")
        if not sparser:
            fails.append(f"NO SIGNAL (seed {seed}): TRUE ({t_sum:.3f}) not below RANDOM "
                         f"({rand['summed_l1l2']:.3f}).")
        if not optimum_at_truth:
            fails.append(
                f"OPTIMUM NOT AT THE TRUTH (seed {seed}): rotation search reached {best:.3f}, "
                f"below TRUE {t_sum:.3f} by {t_sum - best:.3f} (tol {ROT_SEARCH_TOL}). By the "
                "L1/L2 >= 1 floor this is impossible if every env is 1-D, so an env's TRUE "
                "difference was not 1-hot (TRUE inflated above d) — a SETUP BUG, see check 5b, "
                "not an objective failure.")

    log("=" * 72)
    if fails:
        log("ORACLE TEST FAILED. Do not trust any sweep numbers until this is fixed:")
        for f in fails:
            log("  - " + f)
        return False
    log("ORACLE TEST PASSED — simulator, pairing, 1-D structure, and the optimum-at-truth "
        "guarantee all hold.")
    return True


# ------------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true",
                    help="harness self-test only (no encoder training); run this first")
    ap.add_argument("--smoke", action="store_true",
                    help="oracle, then one seed of multienv+laplace and mse+laplace")
    ap.add_argument("--sparsity", action="store_true",
                    help="detailed TRUE/TRAINED/RANDOM printout, seed 0")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    os.makedirs(SEED_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "n/a"

    def log(msg):
        print(msg, flush=True)

    log("=" * 72)
    log("multi-env one-intervention-per-coordinate gate — ceiling probe")
    log(f"python  {platform.python_version()}  ({sys.platform})")
    log(f"torch   {torch.__version__}   cuda_available={torch.cuda.is_available()}")
    log(f"numpy   {np.__version__}   scipy {scipy.__version__}   sklearn {sklearn.__version__}")
    log(f"device  {device}   gpu: {gpu}")
    log(f"d {D_LATENT}   envs {D_LATENT}+1   LAMBDA_PAIR {LAMBDA_PAIR}   "
        f"LAMBDA_PRIOR {LAMBDA_PRIOR}")
    log("=" * 72)

    if args.sparsity:
        sparsity_diagnostic(0, device, log)
        return

    meta = dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, device=device, gpu=gpu)

    if args.oracle or args.smoke:
        if not run_oracle(log, device):
            sys.exit(1)
        log("")
        if args.oracle:
            return

    conditions = CONDITIONS
    n_seeds = args.seeds
    if args.smoke:
        conditions = [c for c in CONDITIONS if c[0] in ("multienv_laplace", "mse_laplace")]
        n_seeds = 1
        log("SMOKE MODE: 1 seed, multienv_laplace + mse_laplace only\n")

    sanity_cache, ref_cache = {}, {}
    all_results = []
    for cond, obj, dist in conditions:
        for seed in range(n_seeds):
            p = seed_path(cond, seed)
            if os.path.exists(p):
                with open(p) as f:
                    all_results.append(json.load(f))
                log(f"[skip] {cond} seed {seed:02d} (already done)")
                continue
            log(f"[run ] {cond} seed {seed:02d}")
            res = run_one(cond, obj, dist, seed, device, log, sanity_cache, ref_cache)
            atomic_write(p, res, is_json=True)
            all_results.append(res)
            log(f"       MCC {res['mcc']:.3f}  R2 {res['subspace_r2']:.3f}  "
                f"recon_r {res['recon_r']:.3f}  summed L1/L2 true/trained/random "
                f"{res['summed_l1l2_true']:.2f}/{res['summed_l1l2_trained']:.2f}/"
                f"{res['summed_l1l2_random']:.2f}  ({res['seconds']:.0f}s)")

    figs = []
    for cond, _, _ in conditions:
        rs = [r for r in all_results if r["condition"] == cond]
        for fn in (fig_true_trained_random, fig_profile_matrix):
            f = fn(cond, rs)
            if f:
                figs.append(f)

    rows = summarize(all_results)
    verdict = write_report(rows, all_results, figs, meta)

    log("")
    log("=" * 72)
    for v in verdict:
        log(v.replace("**", ""))
    log("=" * 72)
    if args.smoke:
        log("SMOKE MODE — 1 seed only. Numbers are not the sweep.")
    log(f"wrote {os.path.join(OUT_DIR, 'results.md')}")


if __name__ == "__main__":
    main()