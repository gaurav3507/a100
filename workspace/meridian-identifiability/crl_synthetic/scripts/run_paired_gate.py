#!/usr/bin/env python
"""
run_paired_gate.py — synthetic ceiling-probe gate for a PAIRED / CONTRASTIVE causal
representation learning objective.

QUESTION
    Does a paired-counterfactual contrastive objective recover latents up to rotation, on
    clean two-environment sparse-lagged-VAR data? And — the question that killed the two
    previous candidates — is the objective's optimum actually AT the truth, and can an
    exact gradient reach it?

WHERE THIS SITS (why candidate 2 exists)
    Candidate 0, coefficient-difference sparsity: the optimum was in the WRONG PLACE. The
        trained encoder out-sparsed the true latents. Falsified.
    Candidate 1, score-function-difference sparsity: the optimum was verifiably AT the
        truth (oracle: TRUE score-diff sparser than RANDOM), but the bilevel gradient
        through the inner-optimized score net could not REACH it. TRAINED sat at RANDOM.
        More inner DSM steps did not help. The failure was the bilevel approximation.
    Candidate 2, this file: NO score net, NO inner optimization, NO bilevel problem. The
        identifying signal is in the DATA (paired), and the encoder gradient flows straight
        through the loss. It is the exact total derivative. This structurally cannot hit
        candidate 1's failure mode. It can still hit candidate 0's, which is why oracle
        check 6 exists — see below.

THE PAIRING — the crux. Read this before trusting anything downstream.
    Theory: content-style / paired counterfactual (von Kügelgen et al. 2021; Locatello et
    al. 2020). The SAME underlying state observed under two conditions, where a few latents
    ("style") shift and the rest ("content") is SHARED. Under the correct unmixing the pair
    differs on few coordinates; under a wrong rotation the difference smears across all of
    them. That breaks rotation.

    Everything rests on content being genuinely shared. Two constructions were considered,
    and they are NOT equivalent:

    PAIRING_MODE = "counterfactual"   (DEFAULT — the one that satisfies the theory)
        Hold the innovations AND the history fixed; change only the mechanism:
            z_b[t] = B_b @ past_a[t] + eps[t]      (past_a is env A's own history)
            z_a[t] = B_a @ past_a[t] + eps[t]
        =>  z_b[t] - z_a[t] = (B_b - B_a) @ past_a[t]
        which is EXACTLY supported on the shifted TARGET rows. Every other coordinate has
        z_b[k] == z_a[k] bit-for-bit. Content exactly shared, style exactly the shifted
        mechanism. This is the von Kügelgen setting. make_dataset ASSERTS the invariant.

    PAIRING_MODE = "resimulate"       (the literal "same innovations, re-simulate" recipe)
        Share only the innovations and let env B run its own trajectory. Then, writing
        dd[t] = z_a[t] - z_b[t]:
            dd[t] = sum_l (B_a[l] - B_b[l]) z_a[t-l-1]  +  sum_l B_b[l] dd[t-l-1]
                    ^^^^^^^^^^^^^^ sparse ^^^^^^^^^^^^     ^^^^ recursion: smears ^^^^
        The first term is sparse. The second is a VAR recursion ON THE DIFFERENCE, so once
        dd is nonzero at a shifted target it propagates to every downstream coordinate and,
        within a few steps of a 64-step window, to essentially all of them. Content is NOT
        shared; it decays apart at rate rho. The paired difference is only approximately
        sparse and the TRUE-vs-RANDOM separation is correspondingly weak.

    Both are implemented. The oracle measures BOTH and prints them side by side, so the
    choice is made on measured numbers rather than on this comment. The default is
    "counterfactual" because "resimulate" cannot satisfy the check-5 requirement that
    unshifted coordinates be near-perfectly paired.

WHY THE GRADIENT IS EXACT
    loss = recon(x_a) + recon(x_b) + LAMBDA_PRIOR*prior + LAMBDA_PAIR*sparse_paired_diff
    Every term is a closed-form function of the encoder's outputs. There is no inner
    optimization, no estimated quantity standing between the encoder and the loss, and
    nothing detached. d(loss)/d(encoder) is the true total derivative. Candidate 1 failed
    because it was not; this cannot fail that way.

    LAMBDA_PRIOR DEFAULTS TO 0.0 ON PURPOSE. Candidate 1's headline caveat was that an
    ICA-style heavy-tail prior sat in the loss alongside the term under test, so a positive
    result could not be attributed. With the prior off, any result here is attributable to
    the pairing alone. Turn it on only to ask a different question.

ORACLE — must pass or the run aborts
    Checks 1-4  carried over: VAR-fit convention, mechanism-difference localization,
                permute_vector round-trip, and end-to-end MCC ~ 1.0 through the true
                unmixing. Plus check 0, simulator integrity across every sweep seed.
    Check 5     THE PAIRING IS REAL. Per-coordinate correlation between paired z_a[t] and
                z_b[t] on TRUE latents: unshifted coords must be near 1 (shared content),
                shifted targets clearly lower (style), and a SHUFFLED pairing must destroy
                it (otherwise "pairing" means nothing). A fake pairing gives a fake result,
                so this aborts.
    Check 6     THE OPTIMUM IS AT THE TRUTH. This is the check that would have killed
                candidate 0 in thirty seconds. It ignores the encoder entirely and directly
                optimizes a ROTATION of the ground-truth latents to minimize this exact
                penalty. If any rotation beats the truth, the objective's optimum is not at
                the truth and the sweep is pointless no matter how well it trains.

READOUT — permutation-free TRUE / TRAINED / RANDOM, same logic that caught candidate 0
    Sparsity (L1/L2 over coordinates) of the paired difference |z_a[t] - z_b[t]|, measured
    in each model's OWN coordinates, no MCC and no permutation:
        TRUE latents      -> the floor. What "identified" looks like.
        TRAINED model     -> the thing under test.
        RANDOM rotation   -> the scrambled ceiling. What "not identified" looks like.
    TRAINED ~ TRUE -> identified. TRAINED ~ RANDOM -> the objective did not act.
    TRAINED < TRUE -> DEGENERATE, the optimum is not at the truth (candidate 0's death).
    VAR R2 rides alongside as the dynamics-faithfulness control (a VAR is rotation-closed,
    so RANDOM's R2 should match TRUE's).

    No score net here, so TRUE/RANDOM references are closed-form and essentially free.

RUN — in this order
    python run_paired_gate.py --oracle     # self-test + pairing + rotation search. Must pass.
    python run_paired_gate.py --smoke      # oracle + 1 seed paired + mse on Laplace
    python run_paired_gate.py --sparsity   # detailed TRUE/TRAINED/RANDOM printout, seed 0
    python run_paired_gate.py              # full 2x2 sweep, 12 seeds

RUNTIME (ESTIMATE — NOT MEASURED. Watch the first seed's printed timing.)
    Much cheaper than candidate 1: no score nets anywhere. Roughly 15-25 min for the full
    2x2 at 12 seeds on an A100-80GB. Resumable: per-condition per-seed JSON, atomic write,
    finished seeds skipped.

OUTPUT
    ./results_paired_gate/
        results.md                        summary table + verdict + caveats
        seeds/<condition>_seed<k>.json    per-run metrics
        fig_paired_diff_<condition>.png   per-coordinate paired difference vs true support
        fig_true_trained_random_<c>.png   the key readout, across seeds
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

OUT_DIR = "./results_paired_gate"
SEED_DIR = os.path.join(OUT_DIR, "seeds")

D_LATENT = 10
N_OBS = 50
MAX_LAG = 2
EDGES_PER_NODE = 1.5
N_SHIFT = 6                # off-diagonal lagged coefficients that differ across envs

WINDOW_L = 64              # contiguous timepoints per window
N_WINDOWS = 256            # per environment
TEST_FRAC = 0.2

N_SEEDS = 12
MCC_GATE = 0.4             # legacy gate. NOT the verdict — the MSE floor at d=10 is ~0.5.

# --- THE PAIRING. See the header. "counterfactual" is the only mode that satisfies the
# --- check-5 requirement that unshifted coordinates be near-perfectly paired.
PAIRING_MODE = "counterfactual"    # "counterfactual" | "resimulate"

FLOW_COUPLINGS = 6
FLOW_HIDDEN = 256
TRAIN_STEPS = 5000
LR = 1e-3
BATCH_WINDOWS = 32
RIDGE = 1e-3               # ridge on the latent VAR solve (diagnostic only)

# --- paired objective ---
LAMBDA_PAIR = 10.0         # weight on sparsity of the paired difference
# Default 0.0 DELIBERATELY. Candidate 1's headline caveat was that a heavy-tail prior sat
# in the loss beside the term under test, so a pass could not be attributed to the term.
# With this at 0 any result here is attributable to the pairing alone.
LAMBDA_PRIOR = 0.0

AE_STEPS = 3000
AE_LR = 1e-3

CONDITIONS = [
    ("paired_laplace", "paired", "laplace"),
    ("paired_gaussian", "paired", "gaussian"),
    ("mse_laplace", "mse", "laplace"),
    ("mse_gaussian", "mse", "gaussian"),
]

LAGGED_COV_WARN = 0.1      # ||C_a(lag) - C_b(lag)||_F below this => mechanisms indistinguishable

# Oracle breadth. Deliberately > 1: in the previous file seed 0 alone hid two simulator
# faults (it needed no stabilize rescale, so it looked clean while seeds 1-2 diverged).
ORACLE_SEEDS = 6

# check 5 thresholds — the pairing must be REAL
PAIR_CORR_UNSHIFTED_MIN = 0.90   # counterfactual mode gives exactly 1.0
PAIR_CORR_SHUFFLED_MAX = 0.25    # shuffling the pair index must destroy the correlation
PAIR_TARGET_GAP_MIN = 0.10       # targets must be clearly less paired than unshifted

# check 6 threshold — the optimum must be AT the truth
ROT_SEARCH_STEPS = 600
ROT_SEARCH_LR = 0.02
ROT_SEARCH_TOL = 0.05            # a rotation beating TRUE by more than this = falsified

SQRT_EPS = 1e-8   # see paired_diff_profile: sqrt'(0) is infinite and the penalty aims at 0

# ------------------------------------------------- simulator (reused from the FIXED file)
#
# stabilize / companion_rho / make_var / shift_var / make_mixing / sample_innovations are
# carried over verbatim from run_score_fn_gate.py, INCLUDING the two bug fixes that file
# introduced (single-pass rescale did not stabilize for max_lag >= 2; shift_var globally
# rescaled env B and made the mechanism difference dense). simulate_windows is replaced by
# simulate_paired_windows, which is the whole point of this file.


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


def shift_var(rng, coefs, n_shift, target_rho=0.9):
    """Change n_shift off-diagonal LAGGED coefficients. Returns (coefs_b, support, alpha).

    Never touches the untouched coefficients: interpolates ONLY the shifted entries from
    their env-A value toward the desired value, bisecting the weight alpha so env B stays
    stationary. alpha = 0 reproduces env A exactly (always stable), so the bracket is always
    valid. Env A and env B therefore differ EXACTLY on the shift support, which is the
    premise the whole experiment rests on; make_dataset asserts it.
    """
    d = coefs[0].shape[0]
    offdiag = [
        (l, i, j)
        for l in range(len(coefs))
        for i in range(d)
        for j in range(d)
        if i != j and abs(coefs[l][i, j]) > 1e-8
    ]
    if not offdiag:
        raise RuntimeError("VAR has no off-diagonal edges to shift")
    n_shift = min(n_shift, len(offdiag))
    picks = [offdiag[k] for k in rng.choice(len(offdiag), size=n_shift, replace=False)]
    desired = {}
    for (l, i, j) in picks:
        old = coefs[l][i, j]
        desired[(l, i, j)] = -np.sign(old) * rng.uniform(1.5, 2.5)  # sign flip + magnitude

    def build(alpha):
        new = [c.copy() for c in coefs]
        for (l, i, j), v in desired.items():
            new[l][i, j] = (1.0 - alpha) * coefs[l][i, j] + alpha * v
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
    support = [(int(l), int(i), int(j)) for (l, i, j) in desired]
    return build(alpha), support, float(alpha)


def simulate_paired_windows(rng, coefs_a, coefs_b, n_windows, L, dist, mode, burn_in=200):
    """PAIRED trajectories. Returns (z_a, z_b), each (n_windows, L, d), INDEX-ALIGNED:
    z_a[k, t] and z_b[k, t] are the two members of one pair. The loss depends on this.

    Both modes share ONE innovation sequence eps. They differ in what else is shared:

    "counterfactual": z_b[t] = B_b @ past_a[t] + eps[t]. Same noise AND same history; only
        the mechanism differs. Then z_b[t] - z_a[t] = (B_b - B_a) @ past_a[t], which is
        EXACTLY supported on the shifted TARGET rows — every other coordinate is
        bit-identical. Content exactly shared. This is the von Kügelgen setting.

    "resimulate": z_b runs its own trajectory off the shared eps. Writing dd = z_a - z_b,
        dd[t] = sum_l (B_a[l]-B_b[l]) z_a[t-l-1] + sum_l B_b[l] dd[t-l-1]. The second term
        is a VAR recursion on the difference, so the difference SMEARS across all
        coordinates within a few steps. Content is not shared, only slowly-decaying. The
        paired difference is at best approximately sparse. Kept so the oracle can measure
        the gap rather than argue about it.
    """
    d = coefs_a[0].shape[0]
    max_lag = len(coefs_a)
    total = n_windows * L + burn_in
    eps = sample_innovations(rng, (total, d), dist)      # SHARED across the pair

    z_a = np.zeros((total, d))
    for t in range(total):
        acc = np.zeros(d)
        for l in range(max_lag):
            if t - l - 1 >= 0:
                acc += coefs_a[l] @ z_a[t - l - 1]
        z_a[t] = acc + eps[t]

    z_b = np.zeros((total, d))
    if mode == "counterfactual":
        for t in range(total):
            acc = np.zeros(d)
            for l in range(max_lag):
                if t - l - 1 >= 0:
                    acc += coefs_b[l] @ z_a[t - l - 1]      # env A's history
            z_b[t] = acc + eps[t]
    elif mode == "resimulate":
        for t in range(total):
            acc = np.zeros(d)
            for l in range(max_lag):
                if t - l - 1 >= 0:
                    acc += coefs_b[l] @ z_b[t - l - 1]      # env B's own history
            z_b[t] = acc + eps[t]
    else:
        raise ValueError(f"unknown PAIRING_MODE: {mode}")

    z_a, z_b = z_a[burn_in:], z_b[burn_in:]
    return z_a.reshape(n_windows, L, d), z_b.reshape(n_windows, L, d)


def make_mixing(rng, d, n_obs):
    """Well-conditioned linear mixing d -> n_obs."""
    while True:
        A = rng.normal(0.0, 1.0, size=(n_obs, d)) / np.sqrt(d)
        s = np.linalg.svd(A, compute_uv=False)
        if s[-1] > 1e-3 and s[0] / s[-1] < 50.0:
            return A


def true_target_coords(shift_support):
    """The shifted TARGET coordinates — where the paired difference lives.

    z_b[t] - z_a[t] = (B_b - B_a) @ past, and (B_b - B_a) has nonzero rows exactly at the
    shift targets i. Source coords j do NOT appear: they enter through `past`, which is
    shared. Under "counterfactual" this is exact and make_dataset asserts it.
    """
    return sorted({int(i) for (_l, i, _j) in shift_support})


def make_dataset(seed, dist):
    # Separate streams so that for a given seed the VAR, the shift support and the mixing
    # are IDENTICAL across dist. Laplace vs Gaussian then differ only in the innovations,
    # which makes the two arms a controlled comparison rather than two unrelated problems.
    rng_struct = np.random.default_rng(seed)
    rng_innov = np.random.default_rng(1_000_000 + seed)
    rng_mix = np.random.default_rng(2_000_000 + seed)

    coefs_a = make_var(rng_struct, D_LATENT, EDGES_PER_NODE, MAX_LAG)
    coefs_b, shift_support, shift_alpha = shift_var(rng_struct, coefs_a, N_SHIFT)
    targets = true_target_coords(shift_support)

    # --- INVARIANTS. Assert, do not assume. Every one of these was silently violated at
    # --- some point by an earlier version of this simulator.

    # (1) both environments stationary
    for name, cf in (("a", coefs_a), ("b", coefs_b)):
        rho = companion_rho(cf)
        if rho >= 1.0:
            raise RuntimeError(
                f"seed {seed} {dist}: env {name} is non-stationary (companion rho={rho:.4f}). "
                "Latents would diverge to inf.")

    # (2) the environments differ EXACTLY on the shift support and nowhere else
    off_support_max = 0.0
    for l in range(MAX_LAG):
        diff = np.abs(coefs_a[l] - coefs_b[l])
        for (ll, i, j) in shift_support:
            if ll == l:
                diff[i, j] = 0.0
        off_support_max = max(off_support_max, float(diff.max()))
    if off_support_max > 1e-10:
        raise RuntimeError(
            f"seed {seed} {dist}: env A and env B differ OFF the shift support by "
            f"{off_support_max:.3e}. The mechanism difference is not sparse; premise void.")

    z_a, z_b = simulate_paired_windows(rng_innov, coefs_a, coefs_b, N_WINDOWS, WINDOW_L,
                                       dist, PAIRING_MODE)

    # (3) finite
    for name, z in (("a", z_a), ("b", z_b)):
        if not np.isfinite(z).all():
            raise RuntimeError(f"seed {seed} {dist}: env {name} latents contain inf/NaN.")
        if np.abs(z).max() > 1e6:
            raise RuntimeError(f"seed {seed} {dist}: env {name} latents blew up "
                               f"(max |z| = {np.abs(z).max():.3e}).")

    # (4) THE PAIRING INVARIANT. Under "counterfactual" the algebra says the pair is
    #     bit-identical off the shifted target rows. If that is not true, the derivation is
    #     wrong or the construction is miswired, and the entire contrastive signal is
    #     fiction. This is the cheapest possible place to catch it.
    if PAIRING_MODE == "counterfactual":
        nontarget = [k for k in range(D_LATENT) if k not in targets]
        if nontarget:
            resid = float(np.abs(z_a[:, :, nontarget] - z_b[:, :, nontarget]).max())
            if resid > 1e-6:
                raise RuntimeError(
                    f"seed {seed} {dist}: PAIRING BROKEN. Under 'counterfactual' the pair "
                    f"must be identical off the shifted targets {targets}, but the max "
                    f"residual on non-target coords is {resid:.3e}. The contrastive signal "
                    "is not what the header claims.")

    A = make_mixing(rng_mix, D_LATENT, N_OBS)
    x_a = z_a @ A.T
    x_b = z_b @ A.T
    # ONE shared invertible rescale over BOTH members. A per-environment standardization
    # would leak the environment identity into the observations and hand the encoder a
    # shortcut that has nothing to do with the mechanism.
    flat = np.concatenate([x_a.reshape(-1, N_OBS), x_b.reshape(-1, N_OBS)], axis=0)
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    x_a = (x_a - mu) / sd
    x_b = (x_b - mu) / sd
    return dict(
        x_a=x_a.astype(np.float32), x_b=x_b.astype(np.float32),
        z_a=z_a.astype(np.float32), z_b=z_b.astype(np.float32),
        coefs_a=coefs_a, coefs_b=coefs_b, A=A,
        shift_support=shift_support, shift_alpha=shift_alpha, targets=targets,
        rho_a=companion_rho(coefs_a), rho_b=companion_rho(coefs_b),
        pairing_mode=PAIRING_MODE,
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
    z_a, z_b = data["z_a"], data["z_b"]
    fa = z_a.reshape(-1, D_LATENT)

    cov_diff = float(np.linalg.norm(np.cov(fa.T) - np.cov(z_b.reshape(-1, D_LATENT).T), "fro"))
    lag_diffs = {}
    for lag in range(1, MAX_LAG + 1):
        d_ = float(np.linalg.norm(lagged_cross_cov(z_a, lag) - lagged_cross_cov(z_b, lag), "fro"))
        lag_diffs[lag] = d_
    kurt = kurtosis(fa, axis=0, fisher=True, bias=False)

    log("  --- DATA SANITY ---")
    log(f"  seed {seed}  dist {dist}  pairing_mode {data['pairing_mode']}")
    log(f"  ||cov(z_a) - cov(z_b)||_F                = {cov_diff:.4f}")
    for lag in range(1, MAX_LAG + 1):
        log(f"  ||laggedcov_{lag}(z_a) - laggedcov_{lag}(z_b)||_F = {lag_diffs[lag]:.4f}")
    log(f"  excess kurtosis of z_a per coord         = [{', '.join(f'{k:.2f}' for k in kurt)}]")
    log(f"    mean {kurt.mean():.3f}  (Laplace innovations => expect clearly > 0; Gaussian => ~0)")
    log(f"  companion rho: env a = {data['rho_a']:.4f}, env b = {data['rho_b']:.4f}")
    log(f"  shift attenuation alpha = {data['shift_alpha']:.4f}  "
        f"(1.0 = full requested shift was stationary as asked)")
    log("  ground-truth shifted coefficients (lag, target i, source j): a -> b")
    for (l, i, j) in data["shift_support"]:
        va = data["coefs_a"][l][i, j]
        vb = data["coefs_b"][l][i, j]
        log(f"    lag {l+1}  z{i} <- z{j}   {va:+.3f} -> {vb:+.3f}   |delta| = {abs(vb - va):.3f}")

    targets = data["targets"]
    log(f"  => paired-difference support (TARGET coords) = {targets}  "
        f"({len(targets)}/{D_LATENT} coords)")

    corr = paired_corr_per_coord(z_a, z_b)
    log("  PAIRED correlation r(z_a[k,t], z_b[k,t]) per coordinate — the contrastive signal:")
    log("    " + "  ".join(f"z{k}{'*' if k in targets else ' '}={corr[k]:+.3f}"
                           for k in range(D_LATENT)))
    log("    (* = shifted target. Unshifted should be ~1 = shared content; targets lower.)")

    max_lag_diff = max(lag_diffs.values())
    void = max_lag_diff < LAGGED_COV_WARN
    if void:
        log("  " + "!" * 68)
        log("  WARNING: lagged cross-covariance difference between environments is ~0.")
        log("  The environments are INDISTINGUISHABLE in the mechanism. THIS TEST IS VOID.")
        log("  " + "!" * 68)
    if dist == "laplace" and kurt.mean() < 0.5:
        log("  WARNING: Laplace data has near-Gaussian marginal kurtosis "
            f"(mean {kurt.mean():.3f}). Mixing/VAR filtering may have killed the tails.")
    if data["shift_alpha"] < 0.1:
        log(f"  WARNING: shift attenuation alpha = {data['shift_alpha']:.4f} < 0.1. The "
            "requested shift was almost entirely scaled away to keep env B stationary.")
    log("  --- END SANITY ---")

    nontarget = [k for k in range(D_LATENT) if k not in targets]
    return dict(cov_diff=cov_diff, lagged_cov_diff={str(k): v for k, v in lag_diffs.items()},
                kurtosis=kurt.tolist(), kurtosis_mean=float(kurt.mean()),
                n_target_coords=len(targets), shift_alpha=float(data["shift_alpha"]),
                rho_a=float(data["rho_a"]), rho_b=float(data["rho_b"]),
                paired_corr=corr.tolist(),
                paired_corr_unshifted_min=float(corr[nontarget].min()) if nontarget else float("nan"),
                paired_corr_target_mean=float(corr[targets].mean()) if targets else float("nan"),
                pairing_mode=data["pairing_mode"], void=bool(void))


# --------------------------------------------------------- flow model (UNCHANGED — reused)


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


# ---------------------------------------------- latent VAR + guards (UNCHANGED — reused)


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


# ------------------------------------------------ THE OBJECTIVE (paired, exact gradient)


def paired_diff_profile(za, zb):
    """Per-coordinate RMS of the PAIRED difference, on unit-variance latents.

    za, zb: (..., d), INDEX-ALIGNED — za[k,t] and zb[k,t] must be the two members of one
    pair. Returns (profile: (d,), mean magnitude).

    The +SQRT_EPS inside both sqrts is NOT cosmetic. d/dx sqrt(x) = 1/(2 sqrt(x)) -> inf as
    x -> 0, and driving per-coordinate paired differences to zero on the content
    coordinates is precisely what this penalty is FOR. Without the eps the objective walks
    its own gradient into a singularity and NaNs exactly when it starts working. That
    happened for real in candidate 1.
    """
    za_s, zb_s = standardize_coords(za, zb)
    dd = (za_s - zb_s).reshape(-1, za.shape[-1])
    prof = (dd.pow(2).mean(0) + SQRT_EPS).sqrt()
    mag = (dd.pow(2).sum(-1) + SQRT_EPS).sqrt().mean()
    return prof, mag


def measure_paired_diff(za, zb):
    """The key readout, in the latents' OWN coordinates. No permutation, no MCC, no fitting.

    Closed form — this is why candidate 2 is cheap. Candidate 1 needed a whole DSM score
    net fit here (and again for each reference), which is also exactly where its bilevel
    gradient problem lived.
    """
    with torch.no_grad():
        prof, mag = paired_diff_profile(za.float(), zb.float())
        prof = prof.cpu().numpy()
    return dict(
        profile=prof.tolist(),
        l1l2=float(np.abs(prof).sum() / (np.linalg.norm(prof) + 1e-12)),
        gini=gini(prof),
        magnitude=float(mag.item()),
    )


def train_paired_model(data, device, log):
    """Encoder trained under the sparse PAIRED-DIFFERENCE penalty.

    loss = recon(x_a) + recon(x_b) + LAMBDA_PRIOR * prior + LAMBDA_PAIR * paired_sparsity

    THE GRADIENT IS EXACT. Every term is a closed-form function of the encoder's outputs.
    No inner optimization, no estimated quantity between encoder and loss, nothing
    detached. This is the structural difference from candidate 1, whose gradient was a
    bilevel approximation that could not reach a known-correct optimum.
    """
    x_a = torch.from_numpy(data["x_a"]).to(device)   # (n_win, L, N)
    x_b = torch.from_numpy(data["x_b"]).to(device)
    n = x_a.shape[0]

    model = FlowEncoder(N_OBS, D_LATENT, FLOW_COUPLINGS, FLOW_HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_STEPS)

    nonfinite_grad_steps = 0
    for step in range(TRAIN_STEPS):
        # ONE index for BOTH members. Drawing two independent index sets would silently
        # destroy the pairing and turn the penalty into a comparison of unrelated windows —
        # the loss would still go down and the run would still finish, meaninglessly.
        idx = torch.randint(0, n, (BATCH_WINDOWS,), device=device)
        xa, xb = x_a[idx], x_b[idx]

        za, lda = model.encode(xa)
        zb, ldb = model.encode(xb)

        rec = (((model.reconstruct(xa) - xa) ** 2).mean()
               + ((model.reconstruct(xb) - xb) ** 2).mean())
        nll = (za.abs().sum(-1) - lda).mean() + (zb.abs().sum(-1) - ldb).mean()

        prof, _mag = paired_diff_profile(za, zb)
        l_pair = sparsity_ratio(prof)      # L1/L2 over the d per-coordinate norms

        loss = rec + LAMBDA_PRIOR * nll + LAMBDA_PAIR * l_pair

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"step {step}: non-finite loss. Component diagnostic — rec={rec.item()}, "
                f"nll={nll.item()}, l_pair={l_pair.item()}, "
                f"prof finite={bool(torch.isfinite(prof).all())}, "
                f"prof min={float(prof.min())}, prof max={float(prof.max())}, "
                f"za finite={bool(torch.isfinite(za).all())}, "
                f"latent sd min={float(torch.cat([za, zb], 0).reshape(-1, D_LATENT).std(0).min())}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        # clip_grad_norm_ CANNOT rescue a non-finite gradient: it computes the norm (inf or
        # NaN) and scales by it, so NaN stays NaN and poisons every later step silently.
        if not torch.isfinite(gnorm):
            nonfinite_grad_steps += 1
            if nonfinite_grad_steps == 1:
                log(f"      !! step {step}: non-finite gradient (norm={gnorm}); skipping. "
                    f"rec={rec.item():.4f} l_pair={l_pair.item():.4f} "
                    f"prof_min={float(prof.min()):.3e}")
            opt.zero_grad(set_to_none=True)
            sched.step()
            continue
        opt.step()
        sched.step()

        if step % 500 == 0:
            with torch.no_grad():
                sd_min = float(torch.cat([za, zb], 0).reshape(-1, D_LATENT).std(0).min())
            log(f"      step {step:5d}  loss {loss.item():9.3f}  rec {rec.item():7.4f}  "
                f"pair_l1l2 {l_pair.item():6.3f}  sd_min {sd_min:7.4f}")

    if nonfinite_grad_steps:
        frac = nonfinite_grad_steps / TRAIN_STEPS
        log(f"    WARNING: {nonfinite_grad_steps}/{TRAIN_STEPS} ({frac:.1%}) steps had a "
            "non-finite gradient and were skipped.")
        if frac > 0.05:
            raise RuntimeError(
                f"{frac:.1%} of steps had non-finite gradients; results not trustworthy.")

    return model, dict(nonfinite_grad_steps=int(nonfinite_grad_steps))


def train_mse_baseline(data, device, log):
    """Plain pointwise linear autoencoder. Rotationally blind by construction. UNCHANGED.

    Sees both members as unpaired samples — it has no pairing term, which is the point.
    """
    x = torch.from_numpy(np.concatenate([data["x_a"], data["x_b"]], 0)).to(device)
    x = x.reshape(-1, N_OBS)

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


# ----------------------------------------------------------- metrics (UNCHANGED — reused)


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


# --------------------------------------------- TRUE / TRAINED / RANDOM (the key readout)


def random_rotation(seed, d):
    rng = np.random.default_rng(10_000 + seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return Q


def reference_paired_diffs(seed, dist, log=None):
    """TRUE and RANDOM paired-difference references for a (seed, dist).

    Depend only on the data, not on the objective, so both arms share them. Closed form.
    """
    data = make_dataset(seed, dist)
    n_test = int(TEST_FRAC * N_WINDOWS)
    tr = slice(0, N_WINDOWS - n_test)
    za_true = torch.from_numpy(data["z_a"][tr])
    zb_true = torch.from_numpy(data["z_b"][tr])

    true_pd = measure_paired_diff(za_true, zb_true)
    ra_t, rb_t = var_fit_quality(za_true.double(), zb_true.double())
    true_pd["var_r2"] = [ra_t, rb_t]

    Q = random_rotation(seed, D_LATENT)
    za_rot = torch.from_numpy((za_true.numpy().reshape(-1, D_LATENT) @ Q).reshape(za_true.shape))
    zb_rot = torch.from_numpy((zb_true.numpy().reshape(-1, D_LATENT) @ Q).reshape(zb_true.shape))
    rand_pd = measure_paired_diff(za_rot, zb_rot)
    ra_r, rb_r = var_fit_quality(za_rot.double(), zb_rot.double())
    rand_pd["var_r2"] = [ra_r, rb_r]

    return dict(true=true_pd, random=rand_pd, target_coords=data["targets"])


def rotation_search(za, zb, device, steps=ROT_SEARCH_STEPS, seed=0, log=None):
    """Directly optimize a ROTATION of the TRUE latents to minimize this exact penalty.

    No encoder, no flow, no training of anything else. This asks the one question that
    killed candidate 0: IS THE OBJECTIVE'S OPTIMUM AT THE TRUTH? Identity = the truth. If
    gradient descent over rotations finds something meaningfully sparser than identity,
    then the objective prefers a non-truth solution and no amount of good training will
    save the sweep.

    M = expm(A - A^T) is orthogonal by construction, so the search stays well-conditioned
    and cannot cheat by collapsing coordinates.

    LIMITATION, stated plainly: this searches ROTATIONS. The encoder's linear projection
    can realize any invertible map, and scaling is neutralized by standardize_coords but
    shears are not in this search class. So "no rotation beats the truth" is a NECESSARY
    condition, not a sufficient one. A pass here does not prove the optimum is at the
    truth; a failure DOES prove it is not.
    """
    d = za.shape[-1]
    za = za.reshape(-1, d).to(device).float()
    zb = zb.reshape(-1, d).to(device).float()
    g = torch.Generator(device="cpu").manual_seed(seed)
    Ap = (0.01 * torch.randn(d, d, generator=g)).to(device).requires_grad_(True)
    opt = torch.optim.Adam([Ap], lr=ROT_SEARCH_LR)

    best = float("inf")
    for _ in range(steps):
        M = torch.matrix_exp(Ap - Ap.T)          # orthogonal
        prof, _ = paired_diff_profile(za @ M.T, zb @ M.T)
        loss = sparsity_ratio(prof)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        best = min(best, float(loss.item()))
    return best


def sparsity_diagnostic(seed, device, log):
    """Detailed standalone printout of the key readout, on Laplace data. --sparsity mode."""
    log("=" * 72)
    log(f"PAIRED-DIFFERENCE DIAGNOSTIC (seed {seed}) — permutation-free, oracle-referenced")
    log("=" * 72)

    data = make_dataset(seed, "laplace")
    n_test = int(TEST_FRAC * N_WINDOWS)
    tr = slice(0, N_WINDOWS - n_test)
    train = {k: data[k][tr] for k in ("x_a", "x_b", "z_a", "z_b")}

    refs = reference_paired_diffs(seed, "laplace", log)
    t, r = refs["true"], refs["random"]

    log("    [trained] training the paired encoder")
    model, _ = train_paired_model(train, device, log)
    model.eval()
    with torch.no_grad():
        za = model.encode(torch.from_numpy(train["x_a"]).to(device))[0].cpu()
        zb = model.encode(torch.from_numpy(train["x_b"]).to(device))[0].cpu()
    m_pd = measure_paired_diff(za, zb)
    ra_m, rb_m = var_fit_quality(za.double(), zb.double())

    rot_best = rotation_search(torch.from_numpy(train["z_a"]),
                               torch.from_numpy(train["z_b"]), device, seed=seed, log=log)

    log("-" * 72)
    log(f"  TRUE unmixing   : pair L1/L2 = {t['l1l2']:.3f}  Gini = {t['gini']:.3f}  "
        f"|d| = {t['magnitude']:.3f}  VAR R2 = [{t['var_r2'][0]:.3f}, {t['var_r2'][1]:.3f}]")
    log(f"  TRAINED model   : pair L1/L2 = {m_pd['l1l2']:.3f}  Gini = {m_pd['gini']:.3f}  "
        f"|d| = {m_pd['magnitude']:.3f}  VAR R2 = [{ra_m:.3f}, {rb_m:.3f}]")
    log(f"  RANDOM rotation : pair L1/L2 = {r['l1l2']:.3f}  Gini = {r['gini']:.3f}  "
        f"|d| = {r['magnitude']:.3f}  VAR R2 = [{r['var_r2'][0]:.3f}, {r['var_r2'][1]:.3f}]")
    log(f"  BEST ROTATION   : pair L1/L2 = {rot_best:.3f}   <- direct search over rotations")
    log(f"  true paired-difference support (target coords) = {refs['target_coords']}")
    log("-" * 72)
    log(f"  L1/L2 is over the d per-coordinate norms: 1.0 = one coordinate carries the whole")
    log(f"  difference, {np.sqrt(D_LATENT):.2f} = perfectly uniform across all {D_LATENT}.")
    log(f"  With |targets| = {len(refs['target_coords'])}, a difference localized on exactly")
    log(f"  those targets with EQUAL magnitudes sits at sqrt({len(refs['target_coords'])}) = "
        f"{np.sqrt(max(len(refs['target_coords']), 1)):.3f} — that is the k-sparse CEILING,")
    log("  not a floor. Unequal magnitudes push L1/L2 down toward 1.")
    log("  VERDICT:")
    log("    BEST ROTATION clearly below TRUE -> the objective's optimum is NOT at the")
    log("      truth. Falsified before training even matters (candidate 0's death).")
    log("    TRAINED ~ TRUE on BOTH sparsity and VAR R2 -> genuinely identified.")
    log("    TRAINED ~ RANDOM -> the objective did not act (candidate 1's death).")
    log("    TRAINED < TRUE with VAR R2 << TRUE -> degenerate: bought sparsity by")
    log("      destroying the dynamics.")
    log("=" * 72)


# ------------------------------------------------------------------------- single run


def run_one(condition, objective, dist, seed, device, log, sanity_cache, ref_cache):
    t0 = time.time()
    data = make_dataset(seed, dist)

    key = (seed, dist)
    if key not in sanity_cache:
        sanity_cache[key] = data_sanity(data, dist, seed, log)
    sanity = sanity_cache[key]

    n_test = int(TEST_FRAC * N_WINDOWS)
    tr = slice(0, N_WINDOWS - n_test)
    te = slice(N_WINDOWS - n_test, N_WINDOWS)
    train = {k: data[k][tr] for k in ("x_a", "x_b", "z_a", "z_b")}
    test = {k: data[k][te] for k in ("x_a", "x_b", "z_a", "z_b")}

    if key not in ref_cache:
        ref_cache[key] = reference_paired_diffs(seed, dist, log)
    refs = ref_cache[key]

    torch.manual_seed(seed)
    if objective == "paired":
        model, train_info = train_paired_model(train, device, log)
    else:
        model, train_info = train_mse_baseline(train, device, log)
    model.eval()

    with torch.no_grad():
        # HELD-OUT windows: recovery metrics.
        xa = torch.from_numpy(test["x_a"]).to(device)
        xb = torch.from_numpy(test["x_b"]).to(device)
        za, _ = model.encode(xa)
        zb, _ = model.encode(xb)
        x_rec = model.reconstruct(torch.cat([xa, xb], 0)).cpu().numpy()

        z_hat = torch.cat([za, zb], 0).reshape(-1, D_LATENT).cpu().numpy()
        z_true = np.concatenate([test["z_a"], test["z_b"]], 0).reshape(-1, D_LATENT)

        pooled = torch.cat([za.reshape(-1, D_LATENT), zb.reshape(-1, D_LATENT)], 0)
        sds = pooled.std(0).cpu().numpy()

        # TRAIN windows: the paired-difference readout. MUST match the windows used by the
        # TRUE/RANDOM references, or the three-way comparison is measured on different
        # sample counts and the L1/L2 numbers are not comparable. The comparison asks where
        # the OBJECTIVE'S OPTIMUM sits — a question about the training data. MCC stays
        # held-out.
        za_tr, _ = model.encode(torch.from_numpy(train["x_a"]).to(device))
        zb_tr, _ = model.encode(torch.from_numpy(train["x_b"]).to(device))

    if not np.isfinite(z_hat).all():
        raise RuntimeError(
            f"{condition} seed {seed}: encoder produced non-finite latents "
            f"({np.isnan(z_hat).sum()} NaN of {z_hat.size}). Training diverged.")

    m, true_of_hat, per_true = mcc_and_matching(z_true, z_hat)
    r2 = subspace_r2(z_true, z_hat)
    rr = recon_r(np.concatenate([test["x_a"], test["x_b"]], 0), x_rec)

    trained_pd = measure_paired_diff(za_tr.cpu(), zb_tr.cpu())
    ra_m, rb_m = var_fit_quality(za_tr.double().cpu(), zb_tr.double().cpu())

    # localization: is the paired difference on the coordinates the shift actually targeted?
    prof = np.array(trained_pd["profile"])
    prof_true_order = permute_vector(prof, true_of_hat, D_LATENT)
    prof_norm = prof_true_order / (prof_true_order.max() + 1e-12)
    targets = refs["target_coords"]
    mask = np.zeros(D_LATENT, dtype=bool)
    mask[targets] = True
    k = int(mask.sum())
    hit = float(mask[np.argsort(prof_norm)[::-1][:k]].sum()) / max(k, 1)

    touched = sorted({c for (_, i, j) in data["shift_support"] for c in (i, j)})
    untouched = [c for c in range(D_LATENT) if c not in touched]

    res = dict(
        condition=condition, objective=objective, dist=dist, seed=seed,
        pairing_mode=PAIRING_MODE,
        mcc=m,
        mcc_shift_touched=float(per_true[touched].mean()) if touched else float("nan"),
        mcc_shift_untouched=float(per_true[untouched].mean()) if untouched else float("nan"),
        n_touched=len(touched), subspace_r2=r2, recon_r=rr,
        # the key readout, permutation-free
        pair_l1l2_trained=trained_pd["l1l2"],
        pair_l1l2_true=refs["true"]["l1l2"],
        pair_l1l2_random=refs["random"]["l1l2"],
        pair_gini_trained=trained_pd["gini"],
        pair_gini_true=refs["true"]["gini"],
        pair_gini_random=refs["random"]["gini"],
        pair_mag_trained=trained_pd["magnitude"],
        pair_mag_true=refs["true"]["magnitude"],
        pair_mag_random=refs["random"]["magnitude"],
        var_r2_trained=[ra_m, rb_m],
        var_r2_true=refs["true"]["var_r2"],
        var_r2_random=refs["random"]["var_r2"],
        # localization
        pair_profile=prof_norm.tolist(),
        pair_topk_hit_rate=hit,
        target_coords=targets,
        true_support=[list(s) for s in data["shift_support"]],
        scale_gaming_check=dict(
            latent_sd_min=float(sds.min()), latent_sd_max=float(sds.max()),
            latent_sd_ratio=float(sds.max() / (sds.min() + 1e-12)),
        ),
        nonfinite_grad_steps=train_info["nonfinite_grad_steps"],
        sanity=sanity,
        seconds=time.time() - t0,
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


def fig_paired_diff(condition, results):
    """Per-coordinate paired difference at true target coords vs elsewhere, across seeds."""
    if not results:
        return None
    on, off = [], []
    for r in results:
        p = np.array(r["pair_profile"])
        mask = np.zeros(D_LATENT, dtype=bool)
        mask[r["target_coords"]] = True
        on.extend(p[mask].tolist())
        off.extend(p[~mask].tolist())
    if not on or not off:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.boxplot([on, off], showfliers=False)   # `labels=` kwarg moved in mpl 3.9
    ax1.set_xticklabels(["true target coords", "content coords"])
    jrng = np.random.default_rng(0)
    ax1.scatter(jrng.normal(1, 0.05, len(on)), on, s=14, alpha=0.6, color="#C44E52", zorder=3)
    ax1.set_ylabel("normalized per-coordinate |z_a - z_b|")
    ax1.set_title(f"{condition}: paired difference\nvs ground-truth shift targets "
                  f"(n={len(results)} seeds)")
    ax1.grid(alpha=0.3, axis="y")

    hits = [r["pair_topk_hit_rate"] for r in results]
    ax2.hist(hits, bins=np.linspace(0, 1, 11), color="#4C72B0", alpha=0.85)
    ax2.set_xlabel("top-k hit rate (k = #true target coords)")
    ax2.set_ylabel("seeds")
    ax2.set_title(f"support recovery\nmean hit {np.mean(hits):.2f}")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"fig_paired_diff_{condition}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_true_trained_random(condition, results):
    """THE key figure: paired-difference sparsity for TRUE vs TRAINED vs RANDOM, per seed."""
    if not results:
        return None
    rs = sorted(results, key=lambda r: r["seed"])
    seeds = [r["seed"] for r in rs]
    tru = [r["pair_l1l2_true"] for r in rs]
    trn = [r["pair_l1l2_trained"] for r in rs]
    rnd = [r["pair_l1l2_random"] for r in rs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    xs = np.arange(len(seeds))
    w = 0.27
    ax1.bar(xs - w, tru, w, label="TRUE (floor / identified)", color="#55A868")
    ax1.bar(xs, trn, w, label="TRAINED", color="#4C72B0")
    ax1.bar(xs + w, rnd, w, label="RANDOM (ceiling / not identified)", color="#C44E52")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(seeds)
    ax1.set_xlabel("seed")
    ax1.set_ylabel("paired-difference L1/L2  (lower = sparser)")
    ax1.set_title(f"{condition}: permutation-free key readout\n"
                  "TRAINED below TRUE = degenerate, not identified")
    ax1.axhline(1.0, color="k", ls=":", lw=1, label="1.0 = single coordinate")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(xs - w, [np.mean(r["var_r2_true"]) for r in rs], w, label="TRUE", color="#55A868")
    ax2.bar(xs, [np.mean(r["var_r2_trained"]) for r in rs], w, label="TRAINED", color="#4C72B0")
    ax2.bar(xs + w, [np.mean(r["var_r2_random"]) for r in rs], w, label="RANDOM", color="#C44E52")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(seeds)
    ax2.set_xlabel("seed")
    ax2.set_ylabel("latent VAR one-step R²")
    ax2.set_title("dynamics faithfulness control\n(VAR is rotation-closed: RANDOM ≈ TRUE)")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"fig_true_trained_random_{condition}.png")
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
            mcc_on=ms("mcc_shift_touched"), mcc_off=ms("mcc_shift_untouched"),
            p_true=ms("pair_l1l2_true"), p_trained=ms("pair_l1l2_trained"),
            p_random=ms("pair_l1l2_random"), hit=ms("pair_topk_hit_rate"),
            var_true=ms(None, lambda r: np.mean(r["var_r2_true"])),
            var_trained=ms(None, lambda r: np.mean(r["var_r2_trained"])),
            var_random=ms(None, lambda r: np.mean(r["var_r2_random"])),
            mag_trained=ms("pair_mag_trained"), mag_true=ms("pair_mag_true"),
        ))
    return rows


def write_report(rows, all_results, figs, meta):
    L, verdict = [], []
    L.append("# Paired / contrastive gate — ceiling probe\n")
    L.append(f"- torch {meta['torch']}, numpy {meta['numpy']}, scipy {meta['scipy']}, "
             f"sklearn {meta['sklearn']}")
    L.append(f"- device: {meta['device']} ({meta['gpu']})")
    L.append(f"- d={D_LATENT}, N={N_OBS}, max_lag={MAX_LAG}, N_SHIFT={N_SHIFT}, "
             f"windows={N_WINDOWS}x{WINDOW_L}, seeds up to {N_SEEDS}")
    L.append(f"- **PAIRING_MODE = {PAIRING_MODE}**, LAMBDA_PAIR = {LAMBDA_PAIR}, "
             f"LAMBDA_PRIOR = {LAMBDA_PRIOR}")
    L.append("- ceiling probe: no HRF, no observation noise")
    L.append("- identifying signal: group sparsity, over latent coordinates, of the PAIRED "
             "difference |z_a[t] - z_b[t]| on unit-variance latents")
    L.append("- the encoder gradient is EXACT (no score net, no inner optimization, nothing "
             "detached), which is the structural difference from the score-function "
             "candidate that failed to reach a known-correct optimum\n")

    if LAMBDA_PRIOR == 0.0:
        L.append("> The heavy-tail prior is OFF (LAMBDA_PRIOR = 0), so any result here is "
                 "attributable to the pairing alone rather than to an ICA-style prior.\n")

    voids = [r for r in all_results if r["sanity"]["void"]]
    if voids:
        L.append(f"> **{len(voids)} runs had a VOID data sanity check.** Numbers meaningless.\n")

    L.append("## THE KEY READOUT — permutation-free paired-difference sparsity\n")
    L.append("L1/L2 over the d per-coordinate norms of the paired difference, each measured "
             "in that model's OWN coordinates. No MCC, no permutation. "
             f"1.0 = one coordinate carries everything; {np.sqrt(D_LATENT):.2f} = uniform "
             f"across all {D_LATENT}.\n")
    L.append("**TRAINED ≈ TRUE → identified. TRAINED ≈ RANDOM → the objective did not act "
             "(how the score-function candidate died). TRAINED < TRUE → DEGENERATE, the "
             "optimum is not at the truth (how the coefficient candidate died).**\n")
    L.append("| condition | pair L1/L2 TRUE | pair L1/L2 TRAINED | pair L1/L2 RANDOM | "
             "VAR R² TRUE | VAR R² TRAINED | VAR R² RANDOM |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(
            f"| {r['condition']} "
            f"| {r['p_true'][0]:.3f} ± {r['p_true'][1]:.3f} "
            f"| {r['p_trained'][0]:.3f} ± {r['p_trained'][1]:.3f} "
            f"| {r['p_random'][0]:.3f} ± {r['p_random'][1]:.3f} "
            f"| {r['var_true'][0]:.3f} | {r['var_trained'][0]:.3f} | {r['var_random'][0]:.3f} |"
        )
    L.append("")

    L.append("## Recovery metrics (mean ± std over seeds)\n")
    L.append("| condition | objective | data | n | MCC | subspace R² | recon r | "
             "pair top-k hit | MCC touched | MCC untouched |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(
            f"| {r['condition']} | {r['objective']} | {r['dist']} | {r['n']} "
            f"| {r['mcc'][0]:.3f} ± {r['mcc'][1]:.3f} "
            f"| {r['r2'][0]:.3f} ± {r['r2'][1]:.3f} "
            f"| {r['rr'][0]:.3f} ± {r['rr'][1]:.3f} "
            f"| {r['hit'][0]:.2f} ± {r['hit'][1]:.2f} "
            f"| {r['mcc_on'][0]:.3f} ± {r['mcc_on'][1]:.3f} "
            f"| {r['mcc_off'][0]:.3f} ± {r['mcc_off'][1]:.3f} |"
        )
    L.append("")

    by = {r["condition"]: r for r in rows}
    L.append("## Verdict\n")

    for dist in ("laplace", "gaussian"):
        cond = f"paired_{dist}"
        if cond not in by:
            continue
        r = by[cond]
        t, m_, rd = r["p_true"][0], r["p_trained"][0], r["p_random"][0]
        span = abs(rd - t)
        tol = max(0.05 * max(span, 1e-9), 0.02)
        if m_ < t - tol:
            call = ("DEGENERATE — TRAINED out-sparsed TRUE. The optimum is NOT at the truth. "
                    "Same failure class as the coefficient objective.")
        elif abs(m_ - t) <= tol:
            call = "IDENTIFIED — TRAINED matches TRUE sparsity."
        elif abs(m_ - rd) <= tol:
            call = ("NOT IDENTIFIED — TRAINED sits at RANDOM. The objective did not act. "
                    "Note this CANNOT be the bilevel-gradient failure here; the gradient is "
                    "exact. Look at LAMBDA_PAIR, the optimizer, or the signal strength.")
        else:
            frac = (rd - m_) / (span + 1e-12)
            call = (f"PARTIAL — TRAINED is {frac:.0%} of the way from RANDOM to TRUE.")
        line = (f"**paired + {dist} — {call}** "
                f"(TRUE {t:.3f}, TRAINED {m_:.3f}, RANDOM {rd:.3f})")
        L.append(line + "\n")
        verdict.append(line)

        vt, vm = r["var_true"][0], r["var_trained"][0]
        if vm < vt - 0.1:
            line = (f"  - VAR R² dropped from {vt:.3f} (TRUE) to {vm:.3f} (TRAINED): the "
                    "encoder degraded the dynamics. Any sparsity gain is bought, not earned.")
            L.append(line + "\n")
            verdict.append(line)

    for dist in ("laplace", "gaussian"):
        pc, ms_ = f"paired_{dist}", f"mse_{dist}"
        if pc not in by or ms_ not in by:
            continue
        a, sa = by[pc]["mcc"]
        b, sb = by[ms_]["mcc"]
        delta = a - b
        pooled = float(np.sqrt(sa ** 2 + sb ** 2))
        line = (f"**MCC vs the MSE floor ({dist}): paired {a:.3f} ± {sa:.3f}, "
                f"MSE {b:.3f} ± {sb:.3f}, DELTA {delta:+.3f}** (pooled sd {pooled:.3f})")
        L.append(line + "\n")
        verdict.append(line)
        if delta <= pooled:
            line = ("  - The paired objective does not clear the rotationally-blind baseline "
                    "by more than one pooled sd. Whatever the absolute MCC is, that is not "
                    "identification.")
            L.append(line + "\n")
            verdict.append(line)

    if "paired_laplace" in by:
        m, s = by["paired_laplace"]["mcc"]
        L.append(f"Legacy absolute gate, reported but NOT the verdict: paired+laplace MCC "
                 f"{m:.3f} ± {s:.3f} vs {MCC_GATE}. See caveat 1.\n")

    L.append("## Caveats — read before quoting these numbers\n")
    L.append(f"1. **The absolute MCC>{MCC_GATE} gate is not informative at d={D_LATENT}.** A "
             "rotationally blind linear autoencoder scores around 0.5 here purely from "
             "Hungarian matching on a 10x10 correlation matrix. The verdict uses the "
             "paired-minus-MSE delta on identical data and the TRUE/TRAINED/RANDOM "
             "comparison instead.")
    L.append(f"2. **PAIRING_MODE = {PAIRING_MODE}.** Under 'counterfactual' the pair shares "
             "both the innovations AND the history, so z_b - z_a = (B_b - B_a) @ past is "
             "EXACTLY supported on the shifted targets and content is bit-identical "
             "elsewhere — the von Kügelgen setting, asserted in make_dataset. Under "
             "'resimulate' only the innovations are shared, the difference obeys its own VAR "
             "recursion and smears across all coordinates, and content is NOT shared. The "
             "two are not interchangeable and the oracle prints both.")
    L.append("3. **The Gaussian arm is not a negative control here at all.** The paired "
             "signal is structural, not distributional: z_b - z_a = (B_b - B_a) @ past holds "
             "regardless of the innovation distribution. Both arms should behave the same. "
             "If they do not, that is informative about the flow, not about identifiability.")
    L.append("4. **The rotation search (oracle check 6) is a necessary condition, not a "
             "sufficient one.** It searches ROTATIONS. The encoder can realize any invertible "
             "map; scaling is neutralized by standardize_coords but shears are not in the "
             "search class. No rotation beating the truth does NOT prove the optimum is at "
             "the truth. A rotation beating the truth DOES prove it is not.")
    L.append(f"5. **Coordinates the shift never touches are not pinned by the paired term.** "
             "The paired difference lives on the shifted TARGET coordinates only. Any "
             "rotation acting purely on the content coordinates leaves the penalty unchanged, "
             "so those coordinates are free under this term alone. With LAMBDA_PRIOR = 0 "
             "there is nothing else pinning them. Read the MCC touched/untouched split "
             "before the aggregate MCC — this caps achievable MCC well below 1 by "
             "construction.")
    L.append("6. **Ceiling probe only.** No HRF, no noise, linear instantaneous mixing, and "
             "a perfectly aligned pairing handed to the model for free. Real data does not "
             "come with counterfactual pairs. A pass here is necessary, not sufficient, and "
             "says nothing about whether the pairing is obtainable outside a simulator.\n")

    if figs:
        L.append("## Figures\n")
        for f in figs:
            L.append(f"- `{os.path.basename(f)}`")
        L.append("")

    atomic_write(os.path.join(OUT_DIR, "results.md"), "\n".join(L), is_json=False)
    return verdict


# ---------------------------------------------------------------------- oracle test


def run_oracle(log, device):
    """Harness self-test. Must pass or the run aborts.

    Checks 0-4 carried over from run_score_fn_gate.py. Checks 5 and 6 are new and are the
    two that matter for this candidate: is the PAIRING real, and is the OPTIMUM at the truth.
    """
    log("=" * 72)
    log("ORACLE HARNESS TEST (no encoder training)")
    log(f"PAIRING_MODE = {PAIRING_MODE}")
    log("=" * 72)
    fails = []

    # 0. SIMULATOR INTEGRITY across every seed the sweep will use. Seed 0 alone has hidden
    #    real faults before; never let it speak for the sweep.
    bad_alpha = []
    for seed in range(max(N_SEEDS, ORACLE_SEEDS)):
        for dist in ("laplace", "gaussian"):
            try:
                data = make_dataset(seed, dist)
            except RuntimeError as e:
                log(f"[FAIL] seed {seed} {dist}: simulator integrity — {e}")
                fails.append(f"simulator integrity (seed {seed}, {dist}): {e}")
                continue
            if data["shift_alpha"] < 0.1:
                bad_alpha.append((seed, dist, data["shift_alpha"]))
    if not fails:
        log(f"[ok  ] simulator integrity over {max(N_SEEDS, ORACLE_SEEDS)} seeds x 2 dists: "
            "stationary, finite, env A vs env B differ ONLY on the shift support, and the "
            "pairing invariant holds")
    if bad_alpha:
        log(f"[warn] {len(bad_alpha)} (seed,dist) had shift attenuation alpha < 0.1:")
        for (s, d_, a) in bad_alpha[:6]:
            log(f"         seed {s} {d_}: alpha = {a:.4f}")

    # 1. does the latent VAR fit recover the ground-truth coefficients on true latents?
    for seed in range(ORACLE_SEEDS):
        data = make_dataset(seed, "laplace")
        B = fit_latent_var(torch.from_numpy(data["z_a"]).double(), MAX_LAG, 1e-8).numpy()
        true_B = np.zeros_like(B)
        for l in range(MAX_LAG):
            for i in range(D_LATENT):
                for j in range(D_LATENT):
                    true_B[l * D_LATENT + j, i] = data["coefs_a"][l][i, j]
        err = float(np.abs(B - true_B).max())
        ok = err < 0.05
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: VAR fit on true latents, "
            f"max|B_hat - B_true| = {err:.4f}")
        if not ok:
            fails.append(f"VAR fit convention (seed {seed}, err {err:.4f})")

    # 2. does |B_a - B_b| land on the ground-truth shift support, in true coordinates?
    #    NOTE under "counterfactual" z_b is a one-step counterfactual sequence, not an env-B
    #    trajectory, so a VAR fit to z_b does NOT recover B_b and this check does not apply.
    #    It is kept for "resimulate", where z_b IS an env-B trajectory.
    if PAIRING_MODE == "resimulate":
        for seed in range(ORACLE_SEEDS):
            data = make_dataset(seed, "laplace")
            za_s, zb_s = standardize_coords(torch.from_numpy(data["z_a"]).double(),
                                            torch.from_numpy(data["z_b"]).double())
            D = np.abs(fit_latent_var(za_s, MAX_LAG, RIDGE).numpy()
                       - fit_latent_var(zb_s, MAX_LAG, RIDGE).numpy())
            D /= D.max()
            mask = true_support_mask(data["shift_support"], D_LATENT, MAX_LAG)
            k = int(mask.sum())
            hit = float(mask.ravel()[np.argsort(D.ravel())[::-1][:k]].sum()) / k
            ok = hit == 1.0
            log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: mech diff on true latents, "
                f"top-{k} hit {hit:.2f}")
            if not ok:
                fails.append(f"support localization (seed {seed}, hit {hit})")
    else:
        log("[skip] check 2 (mechanism-difference localization) — not applicable under "
            "'counterfactual', where z_b is a one-step counterfactual and not an env-B "
            "trajectory, so a VAR fit to z_b does not recover B_b by construction.")

    # 3. permute_mechanism / permute_vector round-trips
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

    # 4. end to end: oracle encoder through the real readout
    for seed in range(ORACLE_SEEDS):
        data = make_dataset(seed, "laplace")
        A = data["A"]
        flat = np.concatenate([(data["z_a"] @ A.T).reshape(-1, N_OBS),
                               (data["z_b"] @ A.T).reshape(-1, N_OBS)], 0)
        A_eff = A / (flat.std(0) + 1e-8)[:, None]
        perm = np.random.default_rng(seed).permutation(D_LATENT)
        W = torch.from_numpy(np.linalg.pinv(A_eff)[perm, :]).float()

        def enc(x):
            s = x.shape[:-1]
            return (x.reshape(-1, x.shape[-1]) @ W.T).reshape(*s, -1)

        za = enc(torch.from_numpy(data["x_a"]))
        zb = enc(torch.from_numpy(data["x_b"]))
        z_hat = torch.cat([za, zb], 0).reshape(-1, D_LATENT).numpy()
        z_true = np.concatenate([data["z_a"], data["z_b"]], 0).reshape(-1, D_LATENT)
        m, true_of_hat, _ = mcc_and_matching(z_true, z_hat)
        found_perm = true_of_hat == {i: int(perm[i]) for i in range(D_LATENT)}
        ok = m > 0.95
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: ORACLE end to end, MCC {m:.4f}, "
            f"permutation recovered: {found_perm}")
        if not ok:
            fails.append(f"oracle end to end (seed {seed}, MCC {m:.3f})")

    # 5. THE PAIRING IS REAL. Everything contrastive depends on this. A fake pairing gives a
    #    fake result, so this aborts. Three things must hold on TRUE latents:
    #      (a) content coordinates are strongly paired  -> shared content exists
    #      (b) shifted targets are clearly less paired  -> style differs there
    #      (c) SHUFFLING the pair index destroys it     -> the pairing is the index, not an
    #          artifact of the two environments merely having similar marginals
    log("-" * 72)
    log("CHECK 5 — IS THE PAIRING REAL? (load-bearing; a fake pairing gives a fake result)")
    for seed in range(ORACLE_SEEDS):
        data = make_dataset(seed, "laplace")
        z_a, z_b = data["z_a"], data["z_b"]
        targets = data["targets"]
        nontarget = [k for k in range(D_LATENT) if k not in targets]

        corr = paired_corr_per_coord(z_a, z_b)
        # shuffled control: break the pair index, keep everything else
        srng = np.random.default_rng(777 + seed)
        fb = z_b.reshape(-1, D_LATENT)
        shuf = srng.permutation(fb.shape[0])
        corr_shuf = paired_corr_per_coord(z_a, fb[shuf].reshape(z_b.shape))

        c_un = float(np.min(np.abs(corr[nontarget]))) if nontarget else float("nan")
        c_tg = float(np.mean(np.abs(corr[targets]))) if targets else float("nan")
        c_sh = float(np.max(np.abs(corr_shuf)))

        ok_un = (not nontarget) or (c_un >= PAIR_CORR_UNSHIFTED_MIN)
        ok_gap = (not targets) or (not nontarget) or (c_tg <= c_un - PAIR_TARGET_GAP_MIN)
        ok_sh = c_sh <= PAIR_CORR_SHUFFLED_MAX
        ok = ok_un and ok_gap and ok_sh
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: content r_min={c_un:.3f} "
            f"(need >= {PAIR_CORR_UNSHIFTED_MIN}), target r_mean={c_tg:.3f} "
            f"(need <= r_min - {PAIR_TARGET_GAP_MIN}), shuffled r_max={c_sh:.3f} "
            f"(need <= {PAIR_CORR_SHUFFLED_MAX})")
        if not ok_un:
            fails.append(
                f"PAIRING BROKEN (seed {seed}): content coords are only paired at "
                f"r={c_un:.3f} < {PAIR_CORR_UNSHIFTED_MIN}. There is no shared content, so "
                f"the contrastive signal does not exist. With PAIRING_MODE='{PAIRING_MODE}' "
                f"this is expected: re-simulation lets the difference smear across all "
                f"coordinates via its own VAR recursion. Set PAIRING_MODE='counterfactual'.")
        if not ok_gap:
            fails.append(f"PAIRING USELESS (seed {seed}): targets are as paired as content "
                         f"({c_tg:.3f} vs {c_un:.3f}); there is no style to separate.")
        if not ok_sh:
            fails.append(f"PAIRING FAKE (seed {seed}): shuffling the pair index leaves "
                         f"r={c_sh:.3f}. The 'pairing' is not carrying the signal — the two "
                         f"environments just have similar marginals.")

    # 6. IS THE OPTIMUM AT THE TRUTH? The check that would have killed candidate 0 in 30
    #    seconds. No encoder: directly optimize a rotation of the TRUE latents against this
    #    exact penalty. Identity = the truth.
    log("-" * 72)
    log("CHECK 6 — IS THE OBJECTIVE'S OPTIMUM AT THE TRUTH? (direct rotation search)")
    for seed in range(min(ORACLE_SEEDS, 3)):
        data = make_dataset(seed, "laplace")
        n_test = int(TEST_FRAC * N_WINDOWS)
        tr = slice(0, N_WINDOWS - n_test)
        za = torch.from_numpy(data["z_a"][tr])
        zb = torch.from_numpy(data["z_b"][tr])

        true_pd = measure_paired_diff(za, zb)
        Q = random_rotation(seed, D_LATENT)
        za_r = torch.from_numpy((za.numpy().reshape(-1, D_LATENT) @ Q).reshape(za.shape))
        zb_r = torch.from_numpy((zb.numpy().reshape(-1, D_LATENT) @ Q).reshape(zb.shape))
        rand_pd = measure_paired_diff(za_r, zb_r)
        best = rotation_search(za, zb, device, seed=seed)

        n_t = len(data["targets"])
        # sqrt(k) is the k-sparse CEILING, not a floor: for a k-nonzero vector L1/L2 runs
        # from 1 (one coordinate dominant) to sqrt(k) (all k equal). TRUE sitting below
        # sqrt(n_targets) just means the true targets carry unequal magnitudes.
        k_ceil = float(np.sqrt(max(n_t, 1)))
        sparser_than_random = true_pd["l1l2"] < rand_pd["l1l2"] - 0.02
        optimum_at_truth = best >= true_pd["l1l2"] - ROT_SEARCH_TOL
        ok = sparser_than_random and optimum_at_truth
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: TRUE={true_pd['l1l2']:.3f}  "
            f"RANDOM={rand_pd['l1l2']:.3f}  BEST_ROTATION={best:.3f}  "
            f"(k-sparse ceiling sqrt({n_t}) = {k_ceil:.3f}; TRUE below it just means the "
            f"targets carry unequal magnitudes)")
        if not sparser_than_random:
            fails.append(
                f"NO SIGNAL (seed {seed}): the paired difference on TRUE latents "
                f"({true_pd['l1l2']:.3f}) is not sparser than under a random rotation "
                f"({rand_pd['l1l2']:.3f}). The objective has nothing to give the encoder; "
                f"the sweep cannot succeed for any honest reason.")
        if not optimum_at_truth:
            fails.append(
                f"OPTIMUM NOT AT THE TRUTH (seed {seed}): a direct rotation search reached "
                f"L1/L2 = {best:.3f}, beating the true latents' {true_pd['l1l2']:.3f} by "
                f"{true_pd['l1l2'] - best:.3f} (tol {ROT_SEARCH_TOL}). This objective PREFERS "
                f"a non-truth solution, so training it well would make identification WORSE. "
                f"This is the coefficient objective's failure mode, detected before the "
                f"sweep instead of after it. FALSIFIED — do not run the sweep.")

    log("=" * 72)
    if fails:
        log("ORACLE TEST FAILED. Do not trust any sweep numbers until this is fixed:")
        for f in fails:
            log("  - " + f)
        return False
    log("ORACLE TEST PASSED — simulator, readout, pairing, and objective optimum are sound.")
    return True


# ------------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true",
                    help="harness self-test only (no encoder training); run this first")
    ap.add_argument("--smoke", action="store_true",
                    help="oracle, then one seed of paired+laplace and mse+laplace")
    ap.add_argument("--sparsity", action="store_true",
                    help="detailed TRUE/TRAINED/RANDOM paired-difference printout, seed 0")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    os.makedirs(SEED_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "n/a"

    def log(msg):
        print(msg, flush=True)

    log("=" * 72)
    log("paired / contrastive gate — ceiling probe")
    log(f"python  {platform.python_version()}  ({sys.platform})")
    log(f"torch   {torch.__version__}   cuda_available={torch.cuda.is_available()}")
    log(f"numpy   {np.__version__}   scipy {scipy.__version__}   sklearn {sklearn.__version__}")
    log(f"device  {device}   gpu: {gpu}")
    log(f"PAIRING_MODE {PAIRING_MODE}   LAMBDA_PAIR {LAMBDA_PAIR}   "
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
        conditions = [c for c in CONDITIONS if c[0] in ("paired_laplace", "mse_laplace")]
        n_seeds = 1
        log("SMOKE MODE: 1 seed, paired_laplace + mse_laplace only\n")

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
                f"recon_r {res['recon_r']:.3f}  pair L1/L2 true/trained/random "
                f"{res['pair_l1l2_true']:.3f}/{res['pair_l1l2_trained']:.3f}/"
                f"{res['pair_l1l2_random']:.3f}  ({res['seconds']:.0f}s)")

    figs = []
    for cond, _, _ in conditions:
        rs = [r for r in all_results if r["condition"] == cond]
        for fn in (fig_paired_diff, fig_true_trained_random):
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