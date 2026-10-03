#!/usr/bin/env python
"""
run_score_fn_gate.py — synthetic ceiling-probe gate for a SCORE-FUNCTION-difference
causal representation learning objective.

QUESTION
    Does penalizing the CROSS-ENVIRONMENT SCORE-FUNCTION DIFFERENCE recover latents up to
    rotation, on clean two-environment sparse-lagged-VAR data? And is the objective's
    optimum actually AT the truth — the exact thing that failed for the previous
    coefficient-sparsity objective, which was falsified by out-sparsing ground truth?

WHY THE SCORE, AND WHICH SCORE (read this — it is the one real design decision here)
    Varici et al. (score-based CRL): if two environments differ only in the mechanism of
    node i, then log p_a and log p_b differ only in the factor p(z_i | pa_i). Every other
    factor is IDENTICAL and CANCELS in the difference. So
        d(z) = grad_z log p_a(z) - grad_z log p_b(z)
    is supported only on {i} U pa_i. Sparse. Under a rotation z' = M z the score maps as
    grad_z' log p' = M^-T grad_z log p, so d'(z') = M^-T d(z). A sparse d becomes DENSE
    unless M^-T is a permutation times a scaling. That is the identifying signal, and it
    bites harder than coefficient sparsity because it is a pointwise function constraint,
    not a single matrix.

    The cancellation argument is what makes it sparse, and it dictates WHICH density.
    Here the shift is in the LAGGED VAR coefficients, so the factor that changes is the
    CONDITIONAL p_e(z[t] | z[t-1..t-max_lag]). This file therefore estimates the
    CONDITIONAL score
        s_e(z_t ; past) = grad_{z_t} log p_e(z_t | past)
    and NOT the marginal score of z[t].

    That distinction is load-bearing, so spelling it out:
      * MARGINAL score of z[t] is grad log p_e(z[t]) = -Sigma_e^-1 z[t] in the Gaussian
        case, where Sigma_e is the STATIONARY covariance. Changing one lagged coefficient
        perturbs the whole stationary covariance, so Sigma_a^-1 - Sigma_b^-1 is DENSE.
        The marginal score difference is dense and carries NO sparse signal. Penalizing it
        would be the same class of mistake as the original pointwise-encoder bug.
      * CONDITIONAL score w.r.t. z[t] has, per coordinate k, s_k = psi(eps_k) where
        eps = z[t] - sum_l B_l z[t-l-1] and psi = (log p_innovation)'. Between environments
        eps differs ONLY at the shifted TARGET coordinates i, so the difference is supported
        exactly on {shifted targets}. Sparse, and exactly the cancellation Varici relies on.
      * The JOINT over a whole window is only APPROXIMATELY sparse: it factors as
        p(z[0..max_lag-1]) * prod_t p(z_t|past_t), and the leading stationary term differs
        densely. Conditioning drops that term entirely. This is why the conditional is the
        right object and not a shortcut.

    Consequence for the readout: the score-difference support here is the set of shifted
    TARGET coordinates, {i for (lag,i,j) in shift_support}. Source coordinates j are NOT in
    the support of the z_t-score difference (they would appear in the score w.r.t. past,
    which DSM on z_t does not give us). The oracle test asserts exactly this.

ESTIMATOR
    Denoising score matching, multi-noise-level (NCSN-style) for stability. The net predicts
    the noise: eps_hat(z_tilde, past, sigma, e), and the score is s = -eps_hat/sigma. The
    weighted DSM loss is then plain ||eps_hat - eps||^2, which is O(1) across noise levels.
    One shared trunk conditioned on a one-hot environment index, so both environments get
    identical capacity (a per-env net could underfit one side and manufacture a fake sparse
    difference).

    ENCODER/SCORE COUPLING — explicit, since it is a bilevel problem:
      Stage A  RECON_WARMUP steps: encoder on reconstruction + prior only.
      Stage B  SCORE_WARMUP steps: encoder FROZEN, score net fit by DSM on its latents.
      Stage C  TRAIN_STEPS: alternate SCORE_INNER_STEPS DSM steps on DETACHED latents,
               then one encoder step through the sparsity penalty.
    The encoder gradient flows only through the EVALUATION POINTS of the score net, not
    through the score function's own dependence on the encoder. That is the standard
    approximation to the total derivative and it is an approximation, not the exact
    gradient. Stage B plus the inner loop keep the score estimate close enough to its
    optimum for the approximation to be meaningful. The achieved DSM loss is printed and
    stored so the known failure mode (noisy score estimate -> meaningless penalty) is
    visible rather than silent.

THE KEY READOUT — permutation-free, oracle-referenced
    Reused verbatim in spirit from the falsified run, because it is what caught the
    falsification. In the model's OWN coordinates, with no MCC and no permutation, compare
    score-difference sparsity for:
        TRUE   latents   -> the floor. What "identified" looks like.
        TRAINED model    -> the thing under test.
        RANDOM rotation  -> the scrambled ceiling. What "not identified" looks like.
    TRAINED ~ TRUE            -> identified.
    TRAINED ~ RANDOM          -> the objective did not act.
    TRAINED sparser than TRUE -> DEGENERATE. The optimum is not at the truth; the encoder
                                 found a solution that out-sparses reality. This is exactly
                                 how the coefficient objective died. VAR R2 is carried along
                                 to tell degeneracy (dynamics destroyed) from identification.

RUN — in this order
    python run_score_fn_gate.py --oracle    # harness self-test. Must pass or the run aborts.
    python run_score_fn_gate.py --smoke     # oracle + 1 seed score_fn + mse on Laplace
    python run_score_fn_gate.py --sparsity  # detailed TRUE/TRAINED/RANDOM printout, seed 0
    python run_score_fn_gate.py             # full 2x2 sweep, 12 seeds

    --oracle trains no ENCODER, but it does fit four small score nets (check 5), so budget
    a few minutes rather than seconds.

RUNTIME (ESTIMATE — NOT MEASURED. Watch the first seed's printed timing and adjust.)
    Roughly 60-90 min for the full 2x2 at 12 seeds on an A100-80GB. The score-net fits
    dominate: each run fits one, and each (seed,dist) fits two cached references.
    Resumable: per-condition per-seed JSON, atomic write, finished seeds skipped.

OUTPUT
    ./results_score_fn_gate/
        results.md                        summary table + verdict + caveats
        seeds/<condition>_seed<k>.json    per-run metrics
        fig_score_diff_<condition>.png    per-coordinate score difference vs true support
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

OUT_DIR = "./results_score_fn_gate"
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
MCC_GATE = 0.4             # legacy gate. NOT the verdict — see MSE floor discussion below.

FLOW_COUPLINGS = 6
FLOW_HIDDEN = 256
TRAIN_STEPS = 5000         # stage C: encoder steps under the score penalty
RECON_WARMUP = 500         # stage A: encoder on reconstruction + prior only
SCORE_WARMUP = 2000        # stage B: score net fit against the frozen warmed-up encoder
SCORE_INNER_STEPS = 10      # stage C: DSM steps per encoder step
LR = 1e-3
BATCH_WINDOWS = 32
RIDGE = 1e-3               # ridge on the in-graph latent VAR solve (diagnostic only now)

# --- score-function objective ---
LAMBDA_SCORE = 10.0        # weight on cross-environment score-difference sparsity
LAMBDA_PRIOR = 0.1         # weight on the heavy-tail prior (rotation-breaking, higher-order)
SCORE_HIDDEN = 256
SCORE_LAYERS = 3
SCORE_LR = 2e-3
# Multi-level DSM. sigma_min doubles as the evaluation noise for the "clean" score.
DSM_SIGMAS = (0.05, 0.1, 0.2, 0.4)
DSM_SIGMA_EVAL = 0.05
SCORE_FIT_STEPS = 2500     # steps for a standalone score-net fit (readout / TRUE / RANDOM)

# Oracle breadth. The cheap numpy-only checks run on this many seeds. Deliberately > 1:
# seed 0 alone hid BOTH simulator faults (it needed no stabilize rescale, so it looked clean
# while seeds 1-2 diverged to inf and carried a dense mechanism difference).
ORACLE_SEEDS = 6

AE_STEPS = 3000
AE_LR = 1e-3

CONDITIONS = [
    ("score_fn_laplace", "score_fn", "laplace"),
    ("score_fn_gaussian", "score_fn", "gaussian"),
    ("mse_laplace", "mse", "laplace"),
    ("mse_gaussian", "mse", "gaussian"),
]

# ||C_a(lag) - C_b(lag)||_F below this => environments indistinguishable in the mechanism.
# Measured range over 12 seeds at the previous settings: min 2.13, mean 3.78. 0.1 is ~20x
# below that floor, so it fires only on a real break (e.g. a shift stabilize() scaled away).
LAGGED_COV_WARN = 0.1

# ------------------------------------------- simulator (reused, WITH TWO BUG FIXES)
#
# Reused verbatim from run_score_gate.py EXCEPT stabilize() and shift_var(), which were
# both broken and are fixed below with the reasoning inline. The same two bugs are still
# present in run_score_gate.py. Both were caught by --oracle on a live A100 run:
#   * stabilize()  single-pass rescale does not stabilize for max_lag >= 2 -> env B
#                  diverged to inf on seeds 1 and 2 -> float32 overflow -> SVD crash.
#   * shift_var()  globally rescaled env B, so EVERY untouched coefficient also differed
#                  across environments -> B_a - B_b DENSE -> the sparse-shift premise void.
# Seed 0 masked both (it needed no rescale), which is why the oracle passed on seed 0.


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

    FIXED (was silently broken for max_lag >= 2, and the break was load-bearing).
    The previous version did ONE pass of `c * (target_rho / rho)`. That is only correct
    for max_lag == 1, where the companion IS B and its eigenvalues scale linearly with s.
    For max_lag >= 2 they do not. Counterexample, d=1, B1=[[0]], B2=[[4]]:
        companion [[0,4],[1,0]] -> rho = 2. One pass scales by 0.9/2 = 0.45, giving
        B2 = [[1.8]], lambda^2 = 1.8, rho = 1.34. STILL EXPLOSIVE.
    In the lag-2-dominant regime rho scales like sqrt(s), not s, so a single pass
    undershoots and the simulated latents diverge to inf/NaN. Observed live: seeds 1 and 2
    of the previous config produced inf latents, float32 overflow, and an SVD crash.

    Bisection on the scale factor is used instead of iterating the same wrong update:
    rho(0) = 0 is always feasible, so the bracket is valid regardless of whether rho(s) is
    monotone, and the postcondition is asserted rather than assumed.
    """
    coefs = [c.copy() for c in coefs]
    if companion_rho(coefs) <= target_rho:
        return coefs
    lo, hi = 0.0, 1.0                      # rho(0) = 0 feasible; rho(1) > target infeasible
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

    FIXED — the previous version broke the experiment's central premise.
    It replaced the chosen entries and then called stabilize() on the WHOLE matrix, which
    rescales EVERY coefficient of env B by some s < 1. Env B then became
    s * (env A with n_shift entries changed), so every UNTOUCHED coefficient also differed
    between environments by the factor s. That makes B_a - B_b DENSE, when the entire
    experiment rests on it being sparse. It was invisible on seed 0 only because s = 1
    there (no rescale fired), which is why the oracle passed on seed 0 and failed on 1-2.

    Fix: never touch the untouched coefficients. Interpolate ONLY the shifted entries from
    their env-A value toward the desired shifted value, and bisect the interpolation weight
    alpha so env B is stationary. alpha = 0 reproduces env A exactly (always stable), so the
    bracket is always valid. alpha = 1 means the full requested shift was stable as asked.

    Env A and env B now differ EXACTLY on the shift support, by construction. make_dataset
    asserts this invariant. A small alpha means the requested shift had to be attenuated to
    keep env B stationary; it is returned and reported, and the data-sanity void check
    catches the case where attenuation left no usable signal.
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
        lo, hi = 0.0, 1.0                  # alpha = 0 IS env A, which is already stable
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if companion_rho(build(mid)) > target_rho:
                hi = mid
            else:
                lo = mid
        alpha = lo
    support = [(int(l), int(i), int(j)) for (l, i, j) in desired]
    return build(alpha), support, float(alpha)


def simulate_windows(rng, coefs, n_windows, L, dist, burn_in=200):
    """Contiguous time series, chopped into windows. Time order preserved. -> (n_windows, L, d)."""
    d = coefs[0].shape[0]
    max_lag = len(coefs)
    total = n_windows * L + burn_in
    eps = sample_innovations(rng, (total, d), dist)
    z = np.zeros((total, d))
    for t in range(total):
        acc = np.zeros(d)
        for l in range(max_lag):
            if t - l - 1 >= 0:
                acc += coefs[l] @ z[t - l - 1]
        z[t] = acc + eps[t]
    z = z[burn_in:]
    return z.reshape(n_windows, L, d)


def make_mixing(rng, d, n_obs):
    """Well-conditioned linear mixing d -> n_obs."""
    while True:
        A = rng.normal(0.0, 1.0, size=(n_obs, d)) / np.sqrt(d)
        s = np.linalg.svd(A, compute_uv=False)
        if s[-1] > 1e-3 and s[0] / s[-1] < 50.0:
            return A


def make_dataset(seed, dist):
    # Separate streams so that for a given seed the VAR, the shift support and the mixing are
    # IDENTICAL across dist. Laplace vs Gaussian then differ only in the innovations, which
    # makes the two arms a controlled comparison rather than two unrelated problems.
    rng_struct = np.random.default_rng(seed)
    rng_innov = np.random.default_rng(1_000_000 + seed)
    rng_mix = np.random.default_rng(2_000_000 + seed)

    coefs_a = make_var(rng_struct, D_LATENT, EDGES_PER_NODE, MAX_LAG)
    coefs_b, shift_support, shift_alpha = shift_var(rng_struct, coefs_a, N_SHIFT)

    # --- INVARIANTS. All three were silently violated by the previous simulator, which is
    # --- how inf latents and a dense B_a - B_b reached the readout unnoticed. Assert, do
    # --- not assume: a violation here voids every number downstream.

    # (1) both environments stationary
    for name, cf in (("a", coefs_a), ("b", coefs_b)):
        rho = companion_rho(cf)
        if rho >= 1.0:
            raise RuntimeError(
                f"seed {seed} {dist}: env {name} is non-stationary (companion rho={rho:.4f}). "
                "Latents would diverge to inf. This is the stabilize() bug.")

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
            f"{off_support_max:.3e}. The mechanism difference is not sparse and the whole "
            "premise is void. This is the shift_var() global-rescale bug.")

    z_a = simulate_windows(rng_innov, coefs_a, N_WINDOWS, WINDOW_L, dist)
    z_b = simulate_windows(rng_innov, coefs_b, N_WINDOWS, WINDOW_L, dist)

    # (3) simulated latents are finite (belt and braces on top of (1))
    for name, z in (("a", z_a), ("b", z_b)):
        if not np.isfinite(z).all():
            raise RuntimeError(f"seed {seed} {dist}: env {name} latents contain inf/NaN.")
        if np.abs(z).max() > 1e6:
            raise RuntimeError(f"seed {seed} {dist}: env {name} latents blew up "
                               f"(max |z| = {np.abs(z).max():.3e}).")

    A = make_mixing(rng_mix, D_LATENT, N_OBS)
    x_a = z_a @ A.T
    x_b = z_b @ A.T
    # shared invertible rescale of observations (does not affect identifiability)
    flat = np.concatenate([x_a.reshape(-1, N_OBS), x_b.reshape(-1, N_OBS)], axis=0)
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    x_a = (x_a - mu) / sd
    x_b = (x_b - mu) / sd
    return dict(
        x_a=x_a.astype(np.float32), x_b=x_b.astype(np.float32),
        z_a=z_a.astype(np.float32), z_b=z_b.astype(np.float32),
        coefs_a=coefs_a, coefs_b=coefs_b, A=A,
        shift_support=shift_support, shift_alpha=shift_alpha,
        rho_a=companion_rho(coefs_a), rho_b=companion_rho(coefs_b),
    )


def true_target_coords(shift_support):
    """Support of the CONDITIONAL score difference w.r.t. z[t]: the shifted TARGET coords.

    Sources j do not appear (they live in the score w.r.t. past, which DSM on z_t does not
    estimate). See the header note on which score.
    """
    return sorted({int(i) for (_l, i, _j) in shift_support})


# ------------------------------------------------ data sanity block (UNCHANGED — reused)


def lagged_cross_cov(z, lag):
    """E[z_t z_{t-lag}^T] over contiguous windows. z: (n_windows, L, d)."""
    a = z[:, lag:, :].reshape(-1, z.shape[2])
    b = z[:, : z.shape[1] - lag, :].reshape(-1, z.shape[2])
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a.T @ b) / a.shape[0]


def data_sanity(data, dist, seed, log):
    z_a, z_b = data["z_a"], data["z_b"]
    fa = z_a.reshape(-1, D_LATENT)
    fb = z_b.reshape(-1, D_LATENT)

    cov_diff = float(np.linalg.norm(np.cov(fa.T) - np.cov(fb.T), "fro"))
    lag_diffs = {}
    for lag in range(1, MAX_LAG + 1):
        d_ = float(np.linalg.norm(lagged_cross_cov(z_a, lag) - lagged_cross_cov(z_b, lag), "fro"))
        lag_diffs[lag] = d_
    kurt = kurtosis(fa, axis=0, fisher=True, bias=False)

    log("  --- DATA SANITY ---")
    log(f"  seed {seed}  dist {dist}")
    log(f"  ||cov(z_a) - cov(z_b)||_F                = {cov_diff:.4f}")
    for lag in range(1, MAX_LAG + 1):
        log(f"  ||laggedcov_{lag}(z_a) - laggedcov_{lag}(z_b)||_F = {lag_diffs[lag]:.4f}")
    log("    note: the marginal covariance moves too (a lagged coefficient change propagates")
    log("    into the stationary covariance). This is exactly why this file scores the")
    log("    CONDITIONAL p(z_t|past) and not the marginal p(z_t): the marginal score")
    log("    difference is dense and carries no sparse signal. See the header.")
    log(f"  excess kurtosis of z_a per coord         = [{', '.join(f'{k:.2f}' for k in kurt)}]")
    log(f"    mean {kurt.mean():.3f}  (Laplace innovations => expect clearly > 0; Gaussian => ~0)")
    log(f"  companion rho: env a = {data['rho_a']:.4f}, env b = {data['rho_b']:.4f} "
        f"(both must be < 1 or the latents diverge)")
    log(f"  shift attenuation alpha = {data['shift_alpha']:.4f}  "
        f"(1.0 = full requested shift was stationary as asked; smaller = it had to be "
        f"scaled back toward env A to keep env B stationary)")
    log("  ground-truth shifted coefficients (lag, target i, source j): a -> b")
    for (l, i, j) in data["shift_support"]:
        va = data["coefs_a"][l][i, j]
        vb = data["coefs_b"][l][i, j]
        log(f"    lag {l+1}  z{i} <- z{j}   {va:+.3f} -> {vb:+.3f}   |delta| = {abs(vb - va):.3f}")
    tt = true_target_coords(data["shift_support"])
    log(f"  => conditional score-difference support (TARGET coords) = {tt}  "
        f"({len(tt)}/{D_LATENT} coords)")

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
        log("  WARNING: shift attenuation alpha = "
            f"{data['shift_alpha']:.4f} < 0.1. The requested shift was almost entirely "
            "scaled away to keep env B stationary, so the environments barely differ. "
            "Lower the shift magnitude in shift_var, or lower EDGES_PER_NODE / the base "
            "coefficient range so there is stability headroom for a real shift.")
    if len(tt) >= D_LATENT:
        log("  NOTE: the shift touches every coordinate. Nothing is left rotationally free,")
        log("  which makes this an easier problem than the sparse-shift regime of interest.")
    log("  --- END SANITY ---")

    return dict(cov_diff=cov_diff, lagged_cov_diff={str(k): v for k, v in lag_diffs.items()},
                kurtosis=kurt.tolist(), kurtosis_mean=float(kurt.mean()),
                n_target_coords=len(tt), shift_alpha=float(data["shift_alpha"]),
                rho_a=float(data["rho_a"]), rho_b=float(data["rho_b"]), void=bool(void))


# ----------------------------------------------------- flow model (UNCHANGED — reused)


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

    Applied per timepoint (the true mixing is instantaneous), but trained on whole
    windows: the temporal structure enters through the conditional score in the loss.
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
    targets = z[:, max_lag:, :].reshape(-1, d)                      # (B*(L-max_lag), d)
    preds = [z[:, max_lag - l - 1 : L - l - 1, :] for l in range(max_lag)]
    P = torch.cat(preds, dim=-1).reshape(-1, d * max_lag)           # (B*(L-max_lag), d*max_lag)
    G = P.T @ P + ridge * P.shape[0] * torch.eye(P.shape[1], device=z.device, dtype=z.dtype)
    return torch.linalg.solve(G, P.T @ targets)                     # (d*max_lag, d)


def standardize_coords(z_a, z_b):
    """Unit-variance per latent coordinate, pooled over both environments, IN-GRAPH.

    Anti-gaming guard: scaling coordinate k by c leaves this output unchanged, so no
    sparsity penalty downstream can be reduced by shrinking a coordinate.
    """
    pooled = torch.cat([z_a.reshape(-1, z_a.shape[-1]), z_b.reshape(-1, z_b.shape[-1])], 0)
    sd = pooled.std(0, unbiased=False) + 1e-6
    return z_a / sd, z_b / sd


def sparsity_ratio(M):
    """L1/L2 of a tensor: minimized (=1) when one entry carries everything, max sqrt(numel)."""
    return M.abs().sum() / (M.pow(2).sum().sqrt() + 1e-8)


# --------------------------------------------------------- THE OBJECTIVE (the one change)


def build_pairs(z, max_lag):
    """(n_win, L, d) contiguous -> (z_t: (M, d), past: (M, d*max_lag)).

    Lagged pairs never cross a window boundary.
    """
    n, L, d = z.shape
    z_t = z[:, max_lag:, :].reshape(-1, d)
    past = torch.cat([z[:, max_lag - l - 1 : L - l - 1, :] for l in range(max_lag)],
                     dim=-1).reshape(-1, d * max_lag)
    return z_t, past


class ConditionalScoreNet(nn.Module):
    """Noise-prediction net for DSM on the CONDITIONAL p_e(z_t | past).

    Inputs : noised z_t, clean past, log sigma, one-hot environment.
    Output : eps_hat. The score is s_e(z_t; past) = -eps_hat / sigma.

    Shared trunk with a one-hot environment input, so both environments get identical
    capacity. A per-environment net could underfit one side and manufacture a fake sparse
    score difference; that failure mode would be invisible in the penalty.
    """

    def __init__(self, d, max_lag, hidden, layers):
        super().__init__()
        in_dim = d + d * max_lag + 1 + 2          # z_t, past, log sigma, env one-hot
        mods, prev = [], in_dim
        for _ in range(layers):
            mods += [nn.Linear(prev, hidden), nn.SiLU()]
            prev = hidden
        mods += [nn.Linear(prev, d)]
        self.net = nn.Sequential(*mods)

    def forward(self, z_t, past, sigma, env):
        """z_t (M,d), past (M,d*max_lag), sigma (M,1), env int 0|1 -> eps_hat (M,d)."""
        e = torch.zeros(z_t.shape[0], 2, device=z_t.device, dtype=z_t.dtype)
        e[:, env] = 1.0
        h = torch.cat([z_t, past, torch.log(sigma), e], dim=-1)
        return self.net(h)

    def score(self, z_t, past, sigma_val, env):
        """s_e(z_t; past) = -eps_hat/sigma, evaluated at (near-)clean z_t."""
        sig = torch.full((z_t.shape[0], 1), sigma_val, device=z_t.device, dtype=z_t.dtype)
        return -self.forward(z_t, past, sig, env) / sigma_val


def dsm_step_loss(score_net, z_t, past, env, sigmas):
    """Multi-level denoising score matching, noise-prediction parametrization.

    z_tilde = z_t + sigma*eps. True conditional score of the perturbed density is
    -eps/sigma, so with s = -eps_hat/sigma the sigma^2-weighted DSM objective collapses to
    ||eps_hat - eps||^2, which is O(1) at every noise level. That is the whole reason for
    this parametrization: no noise level dominates the gradient.
    """
    idx = torch.randint(0, len(sigmas), (z_t.shape[0],), device=z_t.device)
    sig = torch.tensor(sigmas, device=z_t.device, dtype=z_t.dtype)[idx].unsqueeze(-1)
    eps = torch.randn_like(z_t)
    z_tilde = z_t + sig * eps
    eps_hat = score_net(z_tilde, past, sig, env)
    return ((eps_hat - eps) ** 2).sum(-1).mean()


SQRT_EPS = 1e-8   # see score_diff_profile_t: sqrt'(0) is infinite and the penalty aims at 0


def score_diff_profile_t(score_net, z_t_a, past_a, z_t_b, past_b):
    """Per-coordinate RMS of d(z) = s_a(z) - s_b(z), evaluated on BOTH environments' points.

    Returns (profile: (d,), raw_mean_norm: scalar). The score difference is a FUNCTION, so
    it is evaluated on the union of both environments' support, not just one side's.

    The +SQRT_EPS inside both sqrts is NOT cosmetic. d/dx sqrt(x) = 1/(2 sqrt(x)) -> inf as
    x -> 0, and driving per-coordinate score differences to zero on the unshifted
    coordinates is precisely what the sparsity penalty is FOR. So the objective walks its
    own gradient into a singularity: the better it works, the closer prof gets to 0 and the
    larger the gradient, until it overflows to NaN. Observed live — the encoder went NaN
    between step 0 and step 500 of stage C, and grad clipping cannot rescue it because
    clip_grad_norm_ on an already-NaN gradient just scales NaN by NaN.
    """
    z_t = torch.cat([z_t_a, z_t_b], 0)
    past = torch.cat([past_a, past_b], 0)
    s_a = score_net.score(z_t, past, DSM_SIGMA_EVAL, 0)
    s_b = score_net.score(z_t, past, DSM_SIGMA_EVAL, 1)
    d = s_a - s_b
    prof = (d.pow(2).mean(0) + SQRT_EPS).sqrt()
    mag = (d.pow(2).sum(-1) + SQRT_EPS).sqrt().mean()
    return prof, mag


def fit_score_net(z_a, z_b, device, steps, log=None, tag=""):
    """Fit a ConditionalScoreNet by DSM on the given latents. Returns (net, dsm_a, dsm_b).

    Used for BOTH arms and for the TRUE/RANDOM references, with the same architecture and
    the same step count, so the three-way comparison is apples to apples. If the
    architecture systematically under-detects the environment difference, TRUE shows it too
    and the comparison stays valid.
    """
    z_a = z_a.to(device).float()
    z_b = z_b.to(device).float()
    za_s, zb_s = standardize_coords(z_a, z_b)
    ta, pa = build_pairs(za_s, MAX_LAG)
    tb, pb = build_pairs(zb_s, MAX_LAG)

    net = ConditionalScoreNet(D_LATENT, MAX_LAG, SCORE_HIDDEN, SCORE_LAYERS).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=SCORE_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    bs = min(4096, ta.shape[0])

    la = lb = float("nan")
    for step in range(steps):
        i = torch.randint(0, ta.shape[0], (bs,), device=device)
        j = torch.randint(0, tb.shape[0], (bs,), device=device)
        loss_a = dsm_step_loss(net, ta[i], pa[i], 0, DSM_SIGMAS)
        loss_b = dsm_step_loss(net, tb[j], pb[j], 1, DSM_SIGMAS)
        loss = loss_a + loss_b
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        la, lb = loss_a.item(), loss_b.item()
        if not np.isfinite(la) or not np.isfinite(lb):
            raise RuntimeError(
                f"[{tag}] DSM fit went non-finite at step {step} (a={la}, b={lb}). The "
                "latents handed to fit_score_net are probably already NaN.")
        if log is not None and step % 1000 == 0:
            log(f"      [{tag}] dsm step {step:5d}  loss_a {la:7.4f}  loss_b {lb:7.4f}")
    return net, la, lb


def measure_score_diff(z_a, z_b, device, steps=SCORE_FIT_STEPS, log=None, tag=""):
    """Fit a score net on these latents, then report the score-difference structure.

    Everything here is in the latents' OWN coordinates: no permutation, no MCC.

    dsm_a/dsm_b are the achieved DSM losses — the estimator-convergence check. The loss is
    ||eps_hat - eps||^2 summed over d coordinates and averaged over the batch, so a DEAD net
    (eps_hat = 0) scores E||eps||^2 = d = D_LATENT. A converged net is clearly below that.
    Read these before reading any sparsity number.
    """
    net, dsm_a, dsm_b = fit_score_net(z_a, z_b, device, steps, log=log, tag=tag)
    net.eval()
    with torch.no_grad():
        za_s, zb_s = standardize_coords(z_a.to(device).float(), z_b.to(device).float())
        ta, pa = build_pairs(za_s, MAX_LAG)
        tb, pb = build_pairs(zb_s, MAX_LAG)
        n = min(20000, ta.shape[0], tb.shape[0])
        prof, mag = score_diff_profile_t(net, ta[:n], pa[:n], tb[:n], pb[:n])
        prof = prof.cpu().numpy()
    return dict(
        profile=prof.tolist(),
        l1l2=float(np.abs(prof).sum() / (np.linalg.norm(prof) + 1e-12)),
        gini=gini(prof),
        magnitude=float(mag.item()),
        dsm_a=float(dsm_a), dsm_b=float(dsm_b),
    )


def train_score_fn_model(data, device, log):
    """Encoder trained under the CROSS-ENVIRONMENT SCORE-FUNCTION-DIFFERENCE penalty.

    loss = reconstruction + LAMBDA_PRIOR * heavy_tail_prior + LAMBDA_SCORE * score_sparsity

    Staging (see header): A recon warmup, B score warmup on frozen encoder, C alternate.
    """
    x_a = torch.from_numpy(data["x_a"]).to(device)   # (n_win, L, N)
    x_b = torch.from_numpy(data["x_b"]).to(device)
    n = x_a.shape[0]

    model = FlowEncoder(N_OBS, D_LATENT, FLOW_COUPLINGS, FLOW_HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    score_net = ConditionalScoreNet(D_LATENT, MAX_LAG, SCORE_HIDDEN, SCORE_LAYERS).to(device)
    s_opt = torch.optim.Adam(score_net.parameters(), lr=SCORE_LR)

    def sample():
        ia = torch.randint(0, n, (BATCH_WINDOWS,), device=device)
        ib = torch.randint(0, n, (BATCH_WINDOWS,), device=device)
        return x_a[ia], x_b[ib]

    def encode_pairs(xa, xb):
        za, lda = model.encode(xa)
        zb, ldb = model.encode(xb)
        za_s, zb_s = standardize_coords(za, zb)
        ta, pa = build_pairs(za_s, MAX_LAG)
        tb, pb = build_pairs(zb_s, MAX_LAG)
        return za, zb, lda, ldb, ta, pa, tb, pb

    def recon_and_prior(xa, xb, za, zb, lda, ldb):
        rec = (((model.reconstruct(xa) - xa) ** 2).mean()
               + ((model.reconstruct(xb) - xb) ** 2).mean())
        nll = (za.abs().sum(-1) - lda).mean() + (zb.abs().sum(-1) - ldb).mean()
        return rec, nll

    # --- Stage A: reconstruction + prior only. Gives the score net something stable to fit.
    log(f"    [stage A] recon warmup, {RECON_WARMUP} steps")
    for step in range(RECON_WARMUP):
        xa, xb = sample()
        za, zb, lda, ldb, *_ = encode_pairs(xa, xb)
        rec, nll = recon_and_prior(xa, xb, za, zb, lda, ldb)
        loss = rec + LAMBDA_PRIOR * nll
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 250 == 0:
            log(f"      step {step:5d}  rec {rec.item():7.4f}")

    # --- Stage B: encoder FROZEN, fit the score net by DSM on its latents.
    log(f"    [stage B] score warmup on frozen encoder, {SCORE_WARMUP} steps")
    for step in range(SCORE_WARMUP):
        xa, xb = sample()
        with torch.no_grad():
            _, _, _, _, ta, pa, tb, pb = encode_pairs(xa, xb)
        loss_a = dsm_step_loss(score_net, ta, pa, 0, DSM_SIGMAS)
        loss_b = dsm_step_loss(score_net, tb, pb, 1, DSM_SIGMAS)
        s_opt.zero_grad(set_to_none=True)
        (loss_a + loss_b).backward()
        s_opt.step()
        if step % 500 == 0:
            log(f"      step {step:5d}  dsm_a {loss_a.item():7.4f}  dsm_b {loss_b.item():7.4f}")

    # --- Stage C: alternate. SCORE_INNER_STEPS DSM steps on DETACHED latents (so the score
    #     net tracks the moving latent distribution), then one encoder step through the
    #     penalty. The encoder gradient flows only via the score net's EVALUATION POINTS.
    log(f"    [stage C] alternating, {TRAIN_STEPS} encoder steps "
        f"({SCORE_INNER_STEPS} dsm steps each)")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_STEPS)
    dsm_a = dsm_b = float("nan")
    nonfinite_grad_steps = 0
    for step in range(TRAIN_STEPS):
        for _ in range(SCORE_INNER_STEPS):
            xa, xb = sample()
            with torch.no_grad():
                _, _, _, _, ta, pa, tb, pb = encode_pairs(xa, xb)
            loss_a = dsm_step_loss(score_net, ta, pa, 0, DSM_SIGMAS)
            loss_b = dsm_step_loss(score_net, tb, pb, 1, DSM_SIGMAS)
            s_opt.zero_grad(set_to_none=True)
            (loss_a + loss_b).backward()
            s_opt.step()
            dsm_a, dsm_b = loss_a.item(), loss_b.item()

        if not np.isfinite(dsm_a) or not np.isfinite(dsm_b):
            raise RuntimeError(
                f"stage C step {step}: DSM loss went non-finite (a={dsm_a}, b={dsm_b}). "
                "The score net or the encoder it reads from has diverged. Nothing "
                "downstream is meaningful.")

        xa, xb = sample()
        za, zb, lda, ldb, ta, pa, tb, pb = encode_pairs(xa, xb)
        rec, nll = recon_and_prior(xa, xb, za, zb, lda, ldb)

        # group sparsity over LATENT COORDINATES of the score difference
        prof, _mag = score_diff_profile_t(score_net, ta, pa, tb, pb)
        l_score = sparsity_ratio(prof)      # L1/L2 over the d per-coordinate norms

        loss = rec + LAMBDA_PRIOR * nll + LAMBDA_SCORE * l_score

        # FAIL FAST. Previously a NaN here just propagated: 5000 silent NaN steps, then a
        # crash in sklearn's LinearRegression with a traceback pointing nowhere near the
        # cause. Report WHICH term died, at the step it died, and stop.
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"stage C step {step}: non-finite loss. Component diagnostic — "
                f"rec={rec.item()}, nll={nll.item()}, l_score={l_score.item()}, "
                f"prof finite={bool(torch.isfinite(prof).all())}, "
                f"prof min={float(prof.min())}, prof max={float(prof.max())}, "
                f"za finite={bool(torch.isfinite(za).all())}, "
                f"latent sd min={float(torch.cat([za, zb], 0).reshape(-1, D_LATENT).std(0).min())}"
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        # clip_grad_norm_ CANNOT rescue a non-finite gradient: it computes the norm (inf or
        # NaN) and scales by it, so NaN stays NaN and silently poisons every later step.
        # Skip the update instead, and keep count — a few skips are survivable, many mean
        # the objective is sitting on a singularity and the run is not trustworthy.
        if not torch.isfinite(gnorm):
            nonfinite_grad_steps += 1
            if nonfinite_grad_steps == 1:
                log(f"      !! step {step}: non-finite gradient (norm={gnorm}); skipping "
                    f"update. rec={rec.item():.4f} l_score={l_score.item():.4f} "
                    f"prof_min={float(prof.min()):.3e}")
            opt.zero_grad(set_to_none=True)
            sched.step()
            continue
        opt.step()
        sched.step()

        if step % 500 == 0:
            # sd_min watches the OTHER plausible divergence route, so it is visible rather
            # than inferred. The prior term is (|za| - logdet_couplings), and proj's
            # Jacobian is NOT in that logdet (proj is 50->10, so there is no proper density
            # here — the term is a heuristic, not a normalized likelihood). It is therefore
            # unbounded below: shrink proj toward 0, grow decoder by the reciprocal, and
            # reconstruction is untouched while the prior falls forever. If that is what is
            # happening, sd_min slides toward 0 and standardize_coords (which divides by
            # sd + 1e-6) starts amplifying. A steady sd_min means this is not the problem.
            with torch.no_grad():
                sd_min = float(torch.cat([za, zb], 0).reshape(-1, D_LATENT).std(0).min())
            log(f"      step {step:5d}  loss {loss.item():9.3f}  rec {rec.item():7.4f}  "
                f"score_l1l2 {l_score.item():6.3f}  dsm_a {dsm_a:6.4f}  dsm_b {dsm_b:6.4f}  "
                f"sd_min {sd_min:7.4f}")

    if nonfinite_grad_steps:
        frac = nonfinite_grad_steps / TRAIN_STEPS
        log(f"    WARNING: {nonfinite_grad_steps}/{TRAIN_STEPS} ({frac:.1%}) encoder steps "
            "had a non-finite gradient and were skipped.")
        if frac > 0.05:
            raise RuntimeError(
                f"{frac:.1%} of encoder steps had non-finite gradients. The optimization is "
                "sitting on a singularity (most likely the score-difference profile being "
                "driven to exactly zero); results from this run are not trustworthy.")

    log(f"    final DSM loss: a {dsm_a:.4f}  b {dsm_b:.4f}   "
        f"(near {float(D_LATENT):.1f} = score net learned nothing; well below = converged)")
    return model, dict(dsm_a=float(dsm_a), dsm_b=float(dsm_b),
                       nonfinite_grad_steps=int(nonfinite_grad_steps))


def train_mse_baseline(data, device, log):
    """Plain pointwise linear autoencoder. Rotationally blind by construction. UNCHANGED."""
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

    return Wrap(enc, dec), dict(dsm_a=float("nan"), dsm_b=float("nan"))


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
    perm = np.array([true_of_hat[i] for i in range(d)])   # recovered i -> true perm[i]
    out = np.zeros_like(D)
    for l in range(max_lag):
        blk = D[l * d : (l + 1) * d, :]                   # rows = source (lagged), cols = target
        out[l * d : (l + 1) * d, :] = blk[np.ix_(np.argsort(perm), np.argsort(perm))]
    return out


def true_support_mask(shift_support, d, max_lag):
    """Mask over the (d*max_lag, d) mechanism matrix. Row = lagged source, col = target."""
    m = np.zeros((d * max_lag, d), dtype=bool)
    for (l, i, j) in shift_support:
        m[l * d + j, i] = True     # z_i[t] <- z_j[t-l-1]
    return m


def permute_vector(v, true_of_hat, d):
    """Remap a per-coordinate profile from recovered into ground-truth coordinate order."""
    perm = np.array([true_of_hat[i] for i in range(d)])
    out = np.zeros_like(v)
    out[perm] = v
    return out


def var_fit_quality(za, zb):
    """One-step-ahead prediction R2 of the fitted latent VAR, per environment, on a
    held-out tail of the windows. Answers: are these latents a FAITHFUL dynamical model,
    or did the encoder degrade the fit to buy sparsity? za, zb: (n_win, L, d) torch.

    A VAR is closed under rotation, so a RANDOM rotation of the true latents should keep
    R2 ~ TRUE's R2. If the TRAINED model's R2 is clearly below TRUE's while its
    score-difference is sparser, the encoder degraded the dynamics to buy sparsity
    (degenerate / specification failure), not identified the rotation.
    """
    def r2_env(z):
        n = z.shape[0]
        cut = max(1, int(0.8 * n))
        z_tr, z_te = z[:cut], z[cut:]
        za_s, _ = standardize_coords(z_tr, z_tr)          # standardize on train coords
        B = fit_latent_var(za_s, MAX_LAG, RIDGE)          # (d*max_lag, d)
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


def reference_score_diffs(seed, dist, device, log):
    """TRUE and RANDOM score-difference references for a (seed, dist). Cached by the caller.

    These depend only on the data, not on the objective, so both arms share them.
    """
    data = make_dataset(seed, dist)
    n_test = int(TEST_FRAC * N_WINDOWS)
    tr = slice(0, N_WINDOWS - n_test)
    za_true = torch.from_numpy(data["z_a"][tr])
    zb_true = torch.from_numpy(data["z_b"][tr])

    log(f"    [ref] fitting TRUE-latent score net (seed {seed}, {dist})")
    true_sd = measure_score_diff(za_true, zb_true, device, log=log, tag="TRUE")
    ra_t, rb_t = var_fit_quality(za_true.double(), zb_true.double())
    true_sd["var_r2"] = [ra_t, rb_t]

    rng = np.random.default_rng(10_000 + seed)
    Q, _ = np.linalg.qr(rng.normal(size=(D_LATENT, D_LATENT)))
    za_rot = (za_true.numpy().reshape(-1, D_LATENT) @ Q).reshape(za_true.shape)
    zb_rot = (zb_true.numpy().reshape(-1, D_LATENT) @ Q).reshape(zb_true.shape)
    log(f"    [ref] fitting RANDOM-rotation score net (seed {seed}, {dist})")
    rand_sd = measure_score_diff(torch.from_numpy(za_rot), torch.from_numpy(zb_rot),
                                 device, log=log, tag="RANDOM")
    ra_r, rb_r = var_fit_quality(torch.from_numpy(za_rot).double(),
                                 torch.from_numpy(zb_rot).double())
    rand_sd["var_r2"] = [ra_r, rb_r]

    return dict(true=true_sd, random=rand_sd,
                target_coords=true_target_coords(data["shift_support"]))


def sparsity_diagnostic(seed, device, log):
    """Detailed standalone printout of the key readout, on Laplace data. --sparsity mode."""
    log("=" * 72)
    log(f"SCORE-DIFFERENCE DIAGNOSTIC (seed {seed}) — permutation-free, oracle-referenced")
    log("=" * 72)

    data = make_dataset(seed, "laplace")
    n_test = int(TEST_FRAC * N_WINDOWS)
    tr = slice(0, N_WINDOWS - n_test)
    train = {k: data[k][tr] for k in ("x_a", "x_b", "z_a", "z_b")}

    refs = reference_score_diffs(seed, "laplace", device, log)
    t, r = refs["true"], refs["random"]

    log("    [trained] training the score-fn encoder")
    model, _ = train_score_fn_model(train, device, log)
    model.eval()
    with torch.no_grad():
        za = model.encode(torch.from_numpy(train["x_a"]).to(device))[0].cpu()
        zb = model.encode(torch.from_numpy(train["x_b"]).to(device))[0].cpu()
    m_sd = measure_score_diff(za, zb, device, log=log, tag="TRAINED")
    ra_m, rb_m = var_fit_quality(za.double(), zb.double())

    log("-" * 72)
    log(f"  TRUE unmixing   : score L1/L2 = {t['l1l2']:.3f}  Gini = {t['gini']:.3f}  "
        f"|d| = {t['magnitude']:.3f}  VAR R2 = [{t['var_r2'][0]:.3f}, {t['var_r2'][1]:.3f}]  "
        f"dsm = [{t['dsm_a']:.3f}, {t['dsm_b']:.3f}]")
    log(f"  TRAINED model   : score L1/L2 = {m_sd['l1l2']:.3f}  Gini = {m_sd['gini']:.3f}  "
        f"|d| = {m_sd['magnitude']:.3f}  VAR R2 = [{ra_m:.3f}, {rb_m:.3f}]  "
        f"dsm = [{m_sd['dsm_a']:.3f}, {m_sd['dsm_b']:.3f}]")
    log(f"  RANDOM rotation : score L1/L2 = {r['l1l2']:.3f}  Gini = {r['gini']:.3f}  "
        f"|d| = {r['magnitude']:.3f}  VAR R2 = [{r['var_r2'][0]:.3f}, {r['var_r2'][1]:.3f}]  "
        f"dsm = [{r['dsm_a']:.3f}, {r['dsm_b']:.3f}]")
    log(f"  true score-difference support (target coords) = {refs['target_coords']}")
    log("-" * 72)
    log("  L1/L2 is over the d per-coordinate norms: 1.0 = one coordinate carries the whole")
    log(f"  difference, {np.sqrt(D_LATENT):.2f} = perfectly uniform across all {D_LATENT}.")
    log("  SPARSITY: TRUE = floor (identified). RANDOM = scrambled ceiling (not identified).")
    log("  VAR R2  : a VAR is rotation-closed, so RANDOM R2 should ~ TRUE R2 (control).")
    log("  |d|     : if TRAINED |d| << TRUE |d|, the score net simply failed to see the")
    log("            environment difference and the sparsity number is vacuous.")
    log("  dsm     : near {:.1f} = the score net learned nothing. Check before reading L1/L2."
        .format(float(D_LATENT)))
    log("  VERDICT:")
    log("    TRAINED sparser-than-TRUE AND VAR R2 << TRUE  -> DEGENERATE: bought sparsity by")
    log("      degrading the dynamics. Specification failure, not identification.")
    log("    TRAINED ~ TRUE on BOTH sparsity and VAR R2     -> genuinely identified.")
    log("    TRAINED ~ RANDOM sparsity                      -> objective did not act.")
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
        ref_cache[key] = reference_score_diffs(seed, dist, device, log)
    refs = ref_cache[key]

    torch.manual_seed(seed)
    if objective == "score_fn":
        model, train_dsm = train_score_fn_model(train, device, log)
    else:
        model, train_dsm = train_mse_baseline(train, device, log)
    model.eval()

    with torch.no_grad():
        # HELD-OUT windows: recovery metrics (MCC, subspace R2, recon r).
        xa = torch.from_numpy(test["x_a"]).to(device)
        xb = torch.from_numpy(test["x_b"]).to(device)
        za, _ = model.encode(xa)
        zb, _ = model.encode(xb)
        x_all = torch.cat([xa, xb], 0)
        x_rec = model.reconstruct(x_all).cpu().numpy()

        z_hat = torch.cat([za, zb], 0).reshape(-1, D_LATENT).cpu().numpy()
        z_true = np.concatenate([test["z_a"], test["z_b"]], 0).reshape(-1, D_LATENT)

        pooled = torch.cat([za.reshape(-1, D_LATENT), zb.reshape(-1, D_LATENT)], 0)
        sds = pooled.std(0).cpu().numpy()

        # TRAIN windows: the score-difference readout. This MUST use the same windows as the
        # TRUE/RANDOM references in reference_score_diffs, or the three-way comparison comes
        # from score nets fit on ~4x different sample counts and the L1/L2 numbers are not
        # comparable. The comparison asks where the OBJECTIVE'S OPTIMUM sits, which is a
        # question about the training data, not about generalization. MCC above stays
        # held-out.
        za_tr, _ = model.encode(torch.from_numpy(train["x_a"]).to(device))
        zb_tr, _ = model.encode(torch.from_numpy(train["x_b"]).to(device))

    # Guard the readout boundary. Without this a NaN model reaches sklearn and dies there,
    # with a traceback pointing at LinearRegression instead of at the training divergence
    # that actually caused it. Fail where the problem is.
    if not np.isfinite(z_hat).all():
        raise RuntimeError(
            f"{condition} seed {seed}: encoder produced non-finite latents "
            f"({np.isnan(z_hat).sum()} NaN of {z_hat.size}). Training diverged; the "
            "per-step guards in stage C should have caught this earlier.")

    m, true_of_hat, per_true = mcc_and_matching(z_true, z_hat)
    r2 = subspace_r2(z_true, z_hat)
    rr = recon_r(np.concatenate([test["x_a"], test["x_b"]], 0), x_rec)

    # score-difference structure of THIS model's latents, in its own coordinates.
    # Fitted post-hoc for both arms, so the MSE baseline gets the same measurement.
    trained_sd = measure_score_diff(za_tr.cpu(), zb_tr.cpu(), device, log=log, tag="readout")
    ra_m, rb_m = var_fit_quality(za_tr.double().cpu(), zb_tr.double().cpu())

    # localization: is the score difference on the coordinates the shift actually targeted?
    prof = np.array(trained_sd["profile"])
    prof_true_order = permute_vector(prof, true_of_hat, D_LATENT)
    prof_norm = prof_true_order / (prof_true_order.max() + 1e-12)
    targets = refs["target_coords"]
    mask = np.zeros(D_LATENT, dtype=bool)
    mask[targets] = True
    k = int(mask.sum())
    hit = float(mask[np.argsort(prof_norm)[::-1][:k]].sum()) / max(k, 1)
    on_m = float(prof_norm[mask].mean()) if k else float("nan")
    off_m = float(prof_norm[~mask].mean()) if k < D_LATENT else float("nan")

    touched = sorted({c for (_, i, j) in data["shift_support"] for c in (i, j)})
    untouched = [c for c in range(D_LATENT) if c not in touched]
    mcc_touched = float(per_true[touched].mean()) if touched else float("nan")
    mcc_untouched = float(per_true[untouched].mean()) if untouched else float("nan")

    res = dict(
        condition=condition, objective=objective, dist=dist, seed=seed,
        mcc=m, mcc_shift_touched=mcc_touched, mcc_shift_untouched=mcc_untouched,
        n_touched=len(touched), subspace_r2=r2, recon_r=rr,
        # the key readout, permutation-free
        score_l1l2_trained=trained_sd["l1l2"],
        score_l1l2_true=refs["true"]["l1l2"],
        score_l1l2_random=refs["random"]["l1l2"],
        score_gini_trained=trained_sd["gini"],
        score_gini_true=refs["true"]["gini"],
        score_gini_random=refs["random"]["gini"],
        score_mag_trained=trained_sd["magnitude"],
        score_mag_true=refs["true"]["magnitude"],
        score_mag_random=refs["random"]["magnitude"],
        var_r2_trained=[ra_m, rb_m],
        var_r2_true=refs["true"]["var_r2"],
        var_r2_random=refs["random"]["var_r2"],
        # score estimator convergence
        dsm_trained=[trained_sd["dsm_a"], trained_sd["dsm_b"]],
        dsm_true=[refs["true"]["dsm_a"], refs["true"]["dsm_b"]],
        dsm_inloop=[train_dsm["dsm_a"], train_dsm["dsm_b"]],
        # localization
        score_profile=prof_norm.tolist(),
        score_topk_hit_rate=hit,
        score_on_support_mean=on_m,
        score_off_support_mean=off_m,
        target_coords=targets,
        true_support=[list(s) for s in data["shift_support"]],
        scale_gaming_check=dict(
            latent_sd_min=float(sds.min()), latent_sd_max=float(sds.max()),
            latent_sd_ratio=float(sds.max() / (sds.min() + 1e-12)),
        ),
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


def fig_score_diff(condition, results):
    """Per-coordinate score difference at true target coords vs elsewhere, across seeds."""
    if not results:
        return None
    on, off = [], []
    for r in results:
        p = np.array(r["score_profile"])
        mask = np.zeros(D_LATENT, dtype=bool)
        mask[r["target_coords"]] = True
        on.extend(p[mask].tolist())
        off.extend(p[~mask].tolist())
    if not on or not off:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.boxplot([on, off], showfliers=False)   # `labels=` kwarg moved in mpl 3.9
    ax1.set_xticklabels(["true target coords", "other coords"])
    jrng = np.random.default_rng(0)
    ax1.scatter(jrng.normal(1, 0.05, len(on)), on, s=14, alpha=0.6, color="#C44E52", zorder=3)
    ax1.set_ylabel("normalized per-coordinate |s_a - s_b|")
    ax1.set_title(f"{condition}: score-function difference\nvs ground-truth shift targets "
                  f"(n={len(results)} seeds)")
    ax1.grid(alpha=0.3, axis="y")

    hits = [r["score_topk_hit_rate"] for r in results]
    ax2.hist(hits, bins=np.linspace(0, 1, 11), color="#4C72B0", alpha=0.85)
    ax2.set_xlabel("top-k hit rate (k = #true target coords)")
    ax2.set_ylabel("seeds")
    ax2.set_title(f"support recovery\nmean hit {np.mean(hits):.2f}")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"fig_score_diff_{condition}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_true_trained_random(condition, results):
    """THE key figure: score-difference sparsity for TRUE vs TRAINED vs RANDOM, per seed."""
    if not results:
        return None
    rs = sorted(results, key=lambda r: r["seed"])
    seeds = [r["seed"] for r in rs]
    tru = [r["score_l1l2_true"] for r in rs]
    trn = [r["score_l1l2_trained"] for r in rs]
    rnd = [r["score_l1l2_random"] for r in rs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    xs = np.arange(len(seeds))
    w = 0.27
    ax1.bar(xs - w, tru, w, label="TRUE (floor / identified)", color="#55A868")
    ax1.bar(xs, trn, w, label="TRAINED", color="#4C72B0")
    ax1.bar(xs + w, rnd, w, label="RANDOM (ceiling / not identified)", color="#C44E52")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(seeds)
    ax1.set_xlabel("seed")
    ax1.set_ylabel("score-difference L1/L2  (lower = sparser)")
    ax1.set_title(f"{condition}: permutation-free key readout\n"
                  "TRAINED below TRUE = degenerate, not identified")
    ax1.axhline(1.0, color="k", ls=":", lw=1, label="1.0 = single coordinate")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, axis="y")

    r2t = [np.mean(r["var_r2_true"]) for r in rs]
    r2m = [np.mean(r["var_r2_trained"]) for r in rs]
    r2r = [np.mean(r["var_r2_random"]) for r in rs]
    ax2.bar(xs - w, r2t, w, label="TRUE", color="#55A868")
    ax2.bar(xs, r2m, w, label="TRAINED", color="#4C72B0")
    ax2.bar(xs + w, r2r, w, label="RANDOM", color="#C44E52")
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
            s_true=ms("score_l1l2_true"), s_trained=ms("score_l1l2_trained"),
            s_random=ms("score_l1l2_random"),
            hit=ms("score_topk_hit_rate"),
            var_true=ms(None, lambda r: np.mean(r["var_r2_true"])),
            var_trained=ms(None, lambda r: np.mean(r["var_r2_trained"])),
            var_random=ms(None, lambda r: np.mean(r["var_r2_random"])),
            dsm_trained=ms(None, lambda r: np.mean(r["dsm_trained"])),
            dsm_true=ms(None, lambda r: np.mean(r["dsm_true"])),
            mag_trained=ms("score_mag_trained"), mag_true=ms("score_mag_true"),
        ))
    return rows


def write_report(rows, all_results, figs, meta):
    L, verdict = [], []
    L.append("# Score-function-difference gate — ceiling probe\n")
    L.append(f"- torch {meta['torch']}, numpy {meta['numpy']}, scipy {meta['scipy']}, "
             f"sklearn {meta['sklearn']}")
    L.append(f"- device: {meta['device']} ({meta['gpu']})")
    L.append(f"- d={D_LATENT}, N={N_OBS}, max_lag={MAX_LAG}, N_SHIFT={N_SHIFT}, "
             f"windows={N_WINDOWS}x{WINDOW_L}, seeds up to {N_SEEDS}")
    L.append("- ceiling probe: no HRF, no observation noise")
    L.append("- identifying signal: group sparsity, over latent coordinates, of the "
             "cross-environment CONDITIONAL score difference "
             "grad_zt log p_a(z_t|past) - grad_zt log p_b(z_t|past), on unit-variance latents")
    L.append(f"- estimator: multi-level DSM, sigmas {DSM_SIGMAS}, evaluated at "
             f"sigma={DSM_SIGMA_EVAL}\n")

    voids = [r for r in all_results if r["sanity"]["void"]]
    if voids:
        L.append(f"> **{len(voids)} runs had a VOID data sanity check** (lagged cross-covariance "
                 f"difference below {LAGGED_COV_WARN}). Their numbers are meaningless.\n")

    L.append("## THE KEY READOUT — permutation-free score-difference sparsity\n")
    L.append("L1/L2 over the d per-coordinate norms of the score difference, each measured in "
             "that model's OWN coordinates. No MCC, no permutation. "
             f"1.0 = one coordinate carries everything; {np.sqrt(D_LATENT):.2f} = uniform "
             f"across all {D_LATENT}.\n")
    L.append("**TRAINED ≈ TRUE → identified. TRAINED ≈ RANDOM → objective did not act. "
             "TRAINED < TRUE → DEGENERATE (out-sparsed reality; this is how the previous "
             "coefficient objective died).**\n")
    L.append("| condition | score L1/L2 TRUE | score L1/L2 TRAINED | score L1/L2 RANDOM | "
             "VAR R² TRUE | VAR R² TRAINED | VAR R² RANDOM |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(
            f"| {r['condition']} "
            f"| {r['s_true'][0]:.3f} ± {r['s_true'][1]:.3f} "
            f"| {r['s_trained'][0]:.3f} ± {r['s_trained'][1]:.3f} "
            f"| {r['s_random'][0]:.3f} ± {r['s_random'][1]:.3f} "
            f"| {r['var_true'][0]:.3f} | {r['var_trained'][0]:.3f} | {r['var_random'][0]:.3f} |"
        )
    L.append("")

    L.append("### Score-estimator convergence (check BEFORE reading anything above)\n")
    L.append(f"DSM loss near {float(D_LATENT):.1f} means the score net predicted nothing and "
             "every sparsity number above is vacuous. Well below means it converged. "
             "`|d|` is the raw magnitude of the score difference: if TRAINED `|d|` is far "
             "below TRUE `|d|`, the net simply never saw the environment difference.\n")
    L.append("| condition | DSM TRUE | DSM TRAINED | \\|d\\| TRUE | \\|d\\| TRAINED |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['condition']} | {r['dsm_true'][0]:.3f} | {r['dsm_trained'][0]:.3f} "
                 f"| {r['mag_true'][0]:.3f} | {r['mag_trained'][0]:.3f} |")
    L.append("")

    L.append("## Recovery metrics (mean ± std over seeds)\n")
    L.append("| condition | objective | data | n | MCC | subspace R² | recon r | "
             "score top-k hit | MCC touched | MCC untouched |")
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

    # --- 1. the real verdict: TRUE vs TRAINED vs RANDOM
    for dist in ("laplace", "gaussian"):
        cond = f"score_fn_{dist}"
        if cond not in by:
            continue
        r = by[cond]
        t, m_, rd = r["s_true"][0], r["s_trained"][0], r["s_random"][0]
        span = abs(rd - t)
        tol = max(0.05 * max(span, 1e-9), 0.02)
        if m_ < t - tol:
            call = ("DEGENERATE — TRAINED out-sparsed TRUE. The objective's optimum is NOT at "
                    "the truth. Same failure class as the falsified coefficient objective.")
        elif abs(m_ - t) <= tol:
            call = "IDENTIFIED — TRAINED matches TRUE sparsity."
        elif abs(m_ - rd) <= tol:
            call = "NOT IDENTIFIED — TRAINED sits at RANDOM. The objective did not act."
        else:
            frac = (rd - m_) / (span + 1e-12)
            call = (f"PARTIAL — TRAINED is {frac:.0%} of the way from RANDOM to TRUE. "
                    "Between the two references; read the per-seed JSON.")
        line = (f"**score_fn + {dist} — {call}** "
                f"(TRUE {t:.3f}, TRAINED {m_:.3f}, RANDOM {rd:.3f})")
        L.append(line + "\n")
        verdict.append(line)

        vt, vm = r["var_true"][0], r["var_trained"][0]
        if vm < vt - 0.1:
            line = (f"  - VAR R² dropped from {vt:.3f} (TRUE) to {vm:.3f} (TRAINED): the "
                    "encoder degraded the dynamics. Any sparsity gain is bought, not earned.")
            L.append(line + "\n")
            verdict.append(line)

        dt, dm = r["dsm_true"][0], r["dsm_trained"][0]
        if dm > 0.9 * D_LATENT or dt > 0.9 * D_LATENT:
            line = (f"  - **SCORE ESTIMATOR DID NOT CONVERGE** (DSM TRUE {dt:.3f}, TRAINED "
                    f"{dm:.3f}, dead ≈ {float(D_LATENT):.1f}). The sparsity numbers above are "
                    "vacuous. Raise SCORE_FIT_STEPS / SCORE_WARMUP before reading anything.")
            L.append(line + "\n")
            verdict.append(line)

    # --- 2. MCC against the MSE floor on the SAME data, not against the absolute gate
    for dist in ("laplace", "gaussian"):
        sc, ms_ = f"score_fn_{dist}", f"mse_{dist}"
        if sc not in by or ms_ not in by:
            continue
        a, sa = by[sc]["mcc"]
        b, sb = by[ms_]["mcc"]
        delta = a - b
        pooled = float(np.sqrt(sa ** 2 + sb ** 2))
        line = (f"**MCC vs the MSE floor ({dist}): score_fn {a:.3f} ± {sa:.3f}, "
                f"MSE {b:.3f} ± {sb:.3f}, DELTA {delta:+.3f}** (pooled sd {pooled:.3f})")
        L.append(line + "\n")
        verdict.append(line)
        if delta <= pooled:
            line = ("  - The score objective does not clear the rotationally-blind baseline by "
                    "more than one pooled sd on this data. Whatever the absolute MCC is, that "
                    "is not identification.")
            L.append(line + "\n")
            verdict.append(line)

    if "score_fn_laplace" in by:
        m, s = by["score_fn_laplace"]["mcc"]
        L.append(f"Legacy absolute gate, reported but NOT the verdict: score_fn+laplace MCC "
                 f"{m:.3f} ± {s:.3f} vs {MCC_GATE}. See caveat 1 — at d={D_LATENT} the MSE "
                 "floor is already near 0.5, so this threshold cannot separate anything.\n")

    L.append("## Caveats — read before quoting these numbers\n")
    L.append(f"1. **The absolute MCC>{MCC_GATE} gate is not informative at d={D_LATENT}.** "
             "A rotationally blind linear autoencoder scores around 0.5 here, purely from "
             "Hungarian matching on a 10x10 correlation matrix. The verdict above therefore "
             "uses the score-minus-MSE delta on identical data and the permutation-free "
             "TRUE/TRAINED/RANDOM comparison instead.")
    L.append("2. **The Gaussian arm is not a clean negative control.** The conditional score "
             "difference is sparse for Gaussian innovations too: with psi(eps) = -eps the "
             "difference at the shifted target is still exactly (B_b - B_a)·past, supported "
             "on the target coordinate alone. A Gaussian VAR with a sparse lagged shift is "
             "identifiable from this signal without any non-Gaussianity, so score_fn+gaussian "
             "passing is most likely CORRECT rather than a harness bug. What it would show is "
             "that the power comes from the mechanism/score structure, not from heavy tails.")
    L.append("3. **The Laplace prior is a confound.** The loss contains both an ICA-style "
             "non-Gaussian prior (LAMBDA_PRIOR) and the score-difference term (LAMBDA_SCORE), "
             "either of which can break rotation alone. This 2x2 cannot attribute the result "
             "to the score term. The disambiguating runs are LAMBDA_SCORE=0 (prior only) and "
             "LAMBDA_PRIOR=0 (score only); neither is in this sweep. If score_fn+gaussian and "
             "score_fn+laplace land in the same place, the prior is probably not the driver.")
    L.append("4. **The encoder gradient is an approximation.** The score net's dependence on "
             "the encoder is not differentiated through; only its evaluation points are. This "
             "is the standard bilevel approximation. Stage B and the inner loop keep the score "
             "estimate near its optimum, and the DSM losses are reported, but the gradient is "
             "not the exact total derivative and the optimum found may reflect that.")
    L.append("5. **The score-difference support is TARGET coordinates only.** DSM on z_t "
             "estimates grad_zt log p(z_t|past), whose environment difference is supported on "
             "the shifted TARGETS. Shifted SOURCES live in the score w.r.t. past, which is not "
             "estimated here. `score_topk_hit_rate` is scored against targets only; do not "
             "read it as full mechanism-support recovery.")
    L.append(f"6. **Coordinates the shift never touches are not pinned by the score term.** "
             f"With N_SHIFT={N_SHIFT} the score-difference support is the shifted target set "
             "(printed per seed in the sanity block). Any rotation acting only on untouched "
             "coordinates leaves the penalty unchanged, so those coordinates are free under "
             "this term alone. Read the MCC touched/untouched split before the aggregate MCC.")
    L.append("7. **Ceiling probe only.** No HRF, no noise, linear instantaneous mixing, "
             "correctly specified lag order, and a score estimator fit on the same data it is "
             "evaluated on. A pass here is necessary, not sufficient.\n")

    if figs:
        L.append("## Figures\n")
        for f in figs:
            L.append(f"- `{os.path.basename(f)}`")
        L.append("")

    atomic_write(os.path.join(OUT_DIR, "results.md"), "\n".join(L), is_json=False)
    return verdict


# ---------------------------------------------------------------------- oracle test


def run_oracle(log, device):
    """Harness self-test. No encoder training: feed the TRUE unmixing (plus a known
    permutation) straight into the readout.

    Checks 1-4 are carried over unchanged from the trusted file. Check 5 is new and covers
    the new machinery: on TRUE latents, the estimated conditional score difference must
    localize on the shifted TARGET coordinates. If it does not, the score estimator or the
    support convention is wrong and the whole sweep is noise.
    """
    log("=" * 72)
    log("ORACLE HARNESS TEST (no training; true unmixing fed straight into the readout)")
    log("=" * 72)
    fails = []

    # 0. SIMULATOR INTEGRITY, across every seed the sweep will actually use.
    #    This check exists because the previous simulator silently produced non-stationary
    #    env-B latents (inf/NaN) and a DENSE env A vs env B coefficient difference, on seeds
    #    the oracle never looked at. Checking only seed 0 hid both faults: seed 0 happened to
    #    need no rescale. Never trust seed 0 to speak for the sweep.
    bad_alpha = []
    for seed in range(max(N_SEEDS, 3)):
        for dist in ("laplace", "gaussian"):
            try:
                data = make_dataset(seed, dist)     # raises on rho >= 1, off-support diff, inf
            except RuntimeError as e:
                log(f"[FAIL] seed {seed} {dist}: simulator integrity — {e}")
                fails.append(f"simulator integrity (seed {seed}, {dist}): {e}")
                continue
            if data["shift_alpha"] < 0.1:
                bad_alpha.append((seed, dist, data["shift_alpha"]))
    log(f"[{'ok  ' if not fails else 'FAIL'}] simulator integrity over "
        f"{max(N_SEEDS, 3)} seeds x 2 dists: stationary, finite, "
        f"and env A vs env B differ ONLY on the shift support")
    if bad_alpha:
        log(f"[warn] {len(bad_alpha)} (seed,dist) had shift attenuation alpha < 0.1 — the "
            f"requested shift was mostly scaled away to keep env B stationary:")
        for (s, d_, a) in bad_alpha[:6]:
            log(f"         seed {s} {d_}: alpha = {a:.4f}")
        log("       These seeds carry a much weaker mechanism difference than the config "
            "asks for. Not fatal, but the signal is not what N_SHIFT/1.5-2.5 implies.")

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
        contrast = D[mask].mean() / D[~mask].mean()
        ok = hit == 1.0
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: mech diff on true latents, "
            f"top-{k} hit {hit:.2f}, on/off contrast {contrast:.1f}")
        if not ok:
            fails.append(f"support localization (seed {seed}, hit {hit})")

    # 3. does permute_mechanism invert a known coordinate permutation?
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

    # 3b. does permute_vector invert a known coordinate permutation? (new: score profile)
    v_true = rng.random(D_LATENT)
    perm2 = rng.permutation(D_LATENT)
    v_rec = np.zeros_like(v_true)
    for i in range(D_LATENT):
        v_rec[i] = v_true[perm2[i]]
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
        A_eff = A / (flat.std(0) + 1e-8)[:, None]        # mixing into standardized observations
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
        za_s, zb_s = standardize_coords(za, zb)
        D = np.abs(permute_mechanism(fit_latent_var(za_s, MAX_LAG, RIDGE).numpy()
                                     - fit_latent_var(zb_s, MAX_LAG, RIDGE).numpy(),
                                     true_of_hat, D_LATENT, MAX_LAG))
        D /= D.max()
        mask = true_support_mask(data["shift_support"], D_LATENT, MAX_LAG)
        k = int(mask.sum())
        hit = float(mask.ravel()[np.argsort(D.ravel())[::-1][:k]].sum()) / k
        found_perm = true_of_hat == {i: int(perm[i]) for i in range(D_LATENT)}
        ok = (m > 0.95) and (hit == 1.0)
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: ORACLE end to end, MCC {m:.4f}, "
            f"top-{k} hit {hit:.2f}, permutation recovered: {found_perm}")
        if not ok:
            fails.append(f"oracle end to end (seed {seed}, MCC {m:.3f}, hit {hit})")

    # 5. NEW — the score machinery itself. On TRUE latents the estimated conditional score
    #    difference must (a) have a converged DSM loss, (b) localize on the shifted TARGETS,
    #    and (c) be clearly sparser than under a random rotation of the same latents.
    #    (c) is the load-bearing one: it asserts the objective can tell truth from rotation
    #    AT ALL. If TRUE is not sparser than RANDOM, the penalty has no signal to give the
    #    encoder and the sweep cannot succeed for any reason other than the Laplace prior.
    for seed in range(2):
        data = make_dataset(seed, "laplace")
        za = torch.from_numpy(data["z_a"])
        zb = torch.from_numpy(data["z_b"])
        sd_true = measure_score_diff(za, zb, device, steps=SCORE_FIT_STEPS)
        targets = true_target_coords(data["shift_support"])
        prof = np.array(sd_true["profile"])
        prof = prof / (prof.max() + 1e-12)
        mask = np.zeros(D_LATENT, dtype=bool)
        mask[targets] = True
        k = int(mask.sum())
        hit = float(mask[np.argsort(prof)[::-1][:k]].sum()) / max(k, 1)

        rng2 = np.random.default_rng(10_000 + seed)
        Q, _ = np.linalg.qr(rng2.normal(size=(D_LATENT, D_LATENT)))
        za_r = torch.from_numpy((za.numpy().reshape(-1, D_LATENT) @ Q).reshape(za.shape))
        zb_r = torch.from_numpy((zb.numpy().reshape(-1, D_LATENT) @ Q).reshape(zb.shape))
        sd_rand = measure_score_diff(za_r, zb_r, device, steps=SCORE_FIT_STEPS)

        converged = max(sd_true["dsm_a"], sd_true["dsm_b"]) < 0.9 * D_LATENT
        sparser = sd_true["l1l2"] < sd_rand["l1l2"] - 0.02
        ok_hit = hit >= 0.5
        ok = converged and sparser and ok_hit
        log(f"[{'ok  ' if ok else 'FAIL'}] seed {seed}: SCORE machinery on true latents — "
            f"L1/L2 true {sd_true['l1l2']:.3f} vs random {sd_rand['l1l2']:.3f}, "
            f"top-{k} target hit {hit:.2f}, dsm [{sd_true['dsm_a']:.3f}, "
            f"{sd_true['dsm_b']:.3f}]")
        if not converged:
            fails.append(f"score DSM did not converge on true latents (seed {seed})")
        if not sparser:
            fails.append(f"score diff on TRUE not sparser than RANDOM (seed {seed}): "
                         f"{sd_true['l1l2']:.3f} vs {sd_rand['l1l2']:.3f} — the objective "
                         f"has no signal to give")
        if not ok_hit:
            fails.append(f"score diff on TRUE does not localize on targets (seed {seed}, "
                         f"hit {hit:.2f})")

    log("=" * 72)
    if fails:
        log("ORACLE TEST FAILED. Do not trust any sweep numbers until this is fixed:")
        for f in fails:
            log("  - " + f)
        return False
    log("ORACLE TEST PASSED — index conventions, readout and score machinery are sound.")
    return True


# ------------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true",
                    help="harness self-test only (no encoder training); run this first")
    ap.add_argument("--smoke", action="store_true",
                    help="oracle, then one seed of score_fn+laplace and mse+laplace")
    ap.add_argument("--sparsity", action="store_true",
                    help="detailed TRUE/TRAINED/RANDOM score-difference printout, seed 0")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    os.makedirs(SEED_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "n/a"

    def log(msg):
        print(msg, flush=True)

    log("=" * 72)
    log("score-function-difference gate — ceiling probe")
    log(f"python  {platform.python_version()}  ({sys.platform})")
    log(f"torch   {torch.__version__}   cuda_available={torch.cuda.is_available()}")
    log(f"numpy   {np.__version__}   scipy {scipy.__version__}   sklearn {sklearn.__version__}")
    log(f"device  {device}   gpu: {gpu}")
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
        conditions = [c for c in CONDITIONS if c[0] in ("score_fn_laplace", "mse_laplace")]
        n_seeds = 1
        log("SMOKE MODE: 1 seed, score_fn_laplace + mse_laplace only\n")

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
                f"recon_r {res['recon_r']:.3f}  "
                f"score L1/L2 true/trained/random "
                f"{res['score_l1l2_true']:.3f}/{res['score_l1l2_trained']:.3f}/"
                f"{res['score_l1l2_random']:.3f}  ({res['seconds']:.0f}s)")

    figs = []
    for cond, _, _ in conditions:
        rs = [r for r in all_results if r["condition"] == cond]
        for fn in (fig_score_diff, fig_true_trained_random):
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
        log("SMOKE MODE — 1 seed only. Numbers are not the sweep. "
            "Check the sanity block above, then run without --smoke.")
    log(f"wrote {os.path.join(OUT_DIR, 'results.md')}")


if __name__ == "__main__":
    main()