#!/usr/bin/env python
"""
run_score_gate.py — synthetic ceiling-probe gate for a mechanism-based CRL objective.

QUESTION
    Does an objective that penalizes NON-SPARSE cross-environment MECHANISM differences
    recover latents up to rotation (MCC > 0.4) on clean two-environment VAR data, where
    a plain pointwise MSE autoencoder is rotationally blind and fails (MCC ~ 0.19)?

DESIGN
    Data   d=10 latent VAR, sparse (~1.5 edges/node), max_lag 2, Laplace innovations
           (Gaussian variant = negative control). Two environments differ by N_SHIFT
           off-diagonal LAGGED coefficients; the rest is identical. Contiguous windows
           of length L, time order preserved. Linear mixing to N=50 observed channels.
           Ceiling probe: no HRF, no observation noise.

    Model  The true mixing is instantaneous and linear, so the encoder is applied
           pointwise (linear projection N->d, then a RealNVP flow) but is TRAINED ON
           WHOLE CONTIGUOUS WINDOWS: a latent VAR is fit inside the latent space by a
           differentiable ridge solve across each window. This is option (b) of the
           corrected design — the encoder is pointwise, the OBJECTIVE is temporal.

    Signal Under a rotation M, a latent VAR with coefficients B becomes M B M^-1, so the
           cross-environment mechanism difference becomes M (B_a - B_b) M^-1. That is
           sparse only when M is close to a permutation. Penalizing non-sparsity of
           (B_a - B_b) therefore breaks the rotational symmetry. The difference is taken
           on the LAGGED VAR COEFFICIENTS, not on the marginal density of z[t].

    Guard  Latent coordinates are standardized to unit variance IN-GRAPH (pooled over
           both environments) before the VAR fit, so rescaling a coordinate leaves the
           penalty exactly unchanged and cannot buy sparsity. See scale_gaming_check in
           the per-seed JSON.

    Base   Plain pointwise linear MSE autoencoder, no mechanism term. Reference failure.

READ THIS BEFORE TRUSTING THE VERDICT
    The Gaussian arm is NO LONGER a clean negative control under this design, and is
    expected to PASS. Mechanism-difference sparsity is a second-order identifying signal:
    a Gaussian VAR with a sparse lagged shift is identifiable up to rotation without any
    non-Gaussianity. The old (marginal-score) design needed heavy tails; this one does
    not. If score+Gaussian passes, that is most likely CORRECT, not a harness bug. See
    the note printed in results.md.

RUN — in this order
    tmux new -s gate
    python run_score_gate.py --oracle    # harness self-test, no training, ~1 min
    python run_score_gate.py --smoke     # oracle + 1 seed of score/mse on Laplace, ~2 min
    python run_score_gate.py             # full 2x2 sweep

    --oracle feeds the TRUE unmixing into the readout and asserts MCC ~ 1.0 and perfect
    support localization. If it fails, the sweep numbers are meaningless — fix that first.
    --smoke additionally prints the data sanity block and gives you the mse+laplace
    reference point (expected near MCC 0.19).

RUNTIME
    ~20-35 min for the full 2x2 at 12 seeds on an A100-80GB (48 runs, ~25 s/run).
    Resumable: per-condition per-seed JSON, atomic write, finished seeds are skipped.

OUTPUT
    ./results_score_gate/
        results.md                      summary table + verdict + caveats
        seeds/<condition>_seed<k>.json  per-run metrics
        fig_mechanism_<condition>.png   |B_a - B_b|: true support vs off-support
        fig_mechanism_heatmap_<c>.png   seed-0 mechanism difference, true support boxed
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

OUT_DIR = "./results_score_gate"
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
MCC_GATE = 0.4

FLOW_COUPLINGS = 6
FLOW_HIDDEN = 256
TRAIN_STEPS = 5000
LR = 1e-3
BATCH_WINDOWS = 32
RIDGE = 1e-3               # ridge on the in-graph latent VAR solve
LAMBDA_MECH = 50.0          # weight on mechanism-difference sparsity
LAMBDA_PRIOR = 0.1         # weight on the Laplace prior (rotation-breaking, higher-order)

AE_STEPS = 3000
AE_LR = 1e-3

CONDITIONS = [
    ("score_laplace", "score", "laplace"),
    ("score_gaussian", "score", "gaussian"),
    ("mse_laplace", "mse", "laplace"),
    ("mse_gaussian", "mse", "gaussian"),
]

# ||C_a(lag) - C_b(lag)||_F below this => environments indistinguishable in the mechanism.
# Measured range over 12 seeds at these settings: min 2.13, mean 3.78. 0.1 is ~20x below the
# observed floor, so it fires only on a real break (e.g. a shift that stabilize() scaled away).
LAGGED_COV_WARN = 0.1

# ------------------------------------------------------------------------- simulator


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


def stabilize(coefs, target_rho=0.9):
    """Rescale so the companion matrix has spectral radius < target_rho."""
    d = coefs[0].shape[0]
    max_lag = len(coefs)
    comp = np.zeros((d * max_lag, d * max_lag))
    comp[:d, :] = np.concatenate(coefs, axis=1)
    if max_lag > 1:
        comp[d:, : d * (max_lag - 1)] = np.eye(d * (max_lag - 1))
    rho = np.max(np.abs(np.linalg.eigvals(comp)))
    if rho > target_rho:
        return [c * (target_rho / rho) for c in coefs]
    return [c.copy() for c in coefs]


def shift_var(rng, coefs, n_shift):
    """Change n_shift existing off-diagonal LAGGED coefficients. Returns (coefs_b, support)."""
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
    new = [c.copy() for c in coefs]
    support = []
    for k in rng.choice(len(offdiag), size=n_shift, replace=False):
        l, i, j = offdiag[k]
        old = new[l][i, j]
        new[l][i, j] = -np.sign(old) * rng.uniform(1.5, 2.5)  # sign flip + magnitude change
        support.append((int(l), int(i), int(j)))
    return stabilize(new), support


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
    coefs_b, shift_support = shift_var(rng_struct, coefs_a, N_SHIFT)
    z_a = simulate_windows(rng_innov, coefs_a, N_WINDOWS, WINDOW_L, dist)
    z_b = simulate_windows(rng_innov, coefs_b, N_WINDOWS, WINDOW_L, dist)
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
        shift_support=shift_support,
    )


# ------------------------------------------------------------------- data sanity block


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
    log("    into the stationary covariance). The environments are NOT distinguishable only")
    log("    in the lags; what is specific to the lags is the SPARSE structure of the shift.")
    log(f"  excess kurtosis of z_a per coord         = [{', '.join(f'{k:.2f}' for k in kurt)}]")
    log(f"    mean {kurt.mean():.3f}  (Laplace innovations => expect clearly > 0; Gaussian => ~0)")
    log("  ground-truth shifted coefficients (lag, target i, source j): a -> b")
    for (l, i, j) in data["shift_support"]:
        va = data["coefs_a"][l][i, j]
        vb = data["coefs_b"][l][i, j]
        log(f"    lag {l+1}  z{i} <- z{j}   {va:+.3f} -> {vb:+.3f}   |delta| = {abs(vb - va):.3f}")

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
    log("  --- END SANITY ---")

    return dict(cov_diff=cov_diff, lagged_cov_diff={str(k): v for k, v in lag_diffs.items()},
                kurtosis=kurt.tolist(), kurtosis_mean=float(kurt.mean()), void=bool(void))


# ------------------------------------------------------------------------ flow model


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
    windows: the temporal structure enters through the latent VAR fit in the loss.
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


# ------------------------------------------------------------------ latent VAR + loss


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

    Anti-gaming guard: scaling coordinate k by c leaves this output unchanged, so the
    sparsity penalty cannot be reduced by shrinking a coordinate.
    """
    pooled = torch.cat([z_a.reshape(-1, z_a.shape[-1]), z_b.reshape(-1, z_b.shape[-1])], 0)
    sd = pooled.std(0, unbiased=False) + 1e-6
    return z_a / sd, z_b / sd


def sparsity_ratio(M):
    """L1/L2 of a matrix: minimized (=1) when one entry carries everything, max sqrt(numel)."""
    return M.abs().sum() / (M.pow(2).sum().sqrt() + 1e-8)


def train_score_model(data, device, log):
    x_a = torch.from_numpy(data["x_a"]).to(device)   # (n_win, L, N)
    x_b = torch.from_numpy(data["x_b"]).to(device)

    model = FlowEncoder(N_OBS, D_LATENT, FLOW_COUPLINGS, FLOW_HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_STEPS)

    n = x_a.shape[0]
    for step in range(TRAIN_STEPS):
        ia = torch.randint(0, n, (BATCH_WINDOWS,), device=device)
        ib = torch.randint(0, n, (BATCH_WINDOWS,), device=device)
        xa, xb = x_a[ia], x_b[ib]

        za, lda = model.encode(xa)
        zb, ldb = model.encode(xb)

        rec = ((model.reconstruct(xa) - xa) ** 2).mean() + ((model.reconstruct(xb) - xb) ** 2).mean()

        # heavy-tailed prior: sensitivity to higher-order structure, not just 2nd order
        nll = (za.abs().sum(-1) - lda).mean() + (zb.abs().sum(-1) - ldb).mean()

        # MECHANISM difference on the LAGGED VAR coefficients, on scale-invariant latents
        za_s, zb_s = standardize_coords(za, zb)
        Ba = fit_latent_var(za_s, MAX_LAG, RIDGE)
        Bb = fit_latent_var(zb_s, MAX_LAG, RIDGE)
        l_mech = sparsity_ratio(Ba - Bb)

        loss = rec + LAMBDA_PRIOR * nll + LAMBDA_MECH * l_mech
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

        if step % 500 == 0:
            log(f"    step {step:5d}  loss {loss.item():9.3f}  rec {rec.item():7.4f}  "
                f"mech_l1l2 {l_mech.item():7.3f}")

    return model


def train_mse_baseline(data, device, log):
    """Plain pointwise linear autoencoder. Rotationally blind by construction."""
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

    return Wrap(enc, dec)


# --------------------------------------------------------------------------- metrics


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


# ------------------------------------------------------------------------- single run


def run_one(condition, objective, dist, seed, device, log, sanity_cache):
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

    torch.manual_seed(seed)
    if objective == "score":
        model = train_score_model(train, device, log)
    else:
        model = train_mse_baseline(train, device, log)
    model.eval()

    with torch.no_grad():
        xa = torch.from_numpy(test["x_a"]).to(device)
        xb = torch.from_numpy(test["x_b"]).to(device)
        za, _ = model.encode(xa)
        zb, _ = model.encode(xb)
        x_all = torch.cat([xa, xb], 0)
        x_rec = model.reconstruct(x_all).cpu().numpy()

        z_hat = torch.cat([za, zb], 0).reshape(-1, D_LATENT).cpu().numpy()
        z_true = np.concatenate([test["z_a"], test["z_b"]], 0).reshape(-1, D_LATENT)

        # mechanism difference in RECOVERED coordinates, held-out windows, scale-normalized
        za_s, zb_s = standardize_coords(za, zb)
        Ba = fit_latent_var(za_s, MAX_LAG, RIDGE).cpu().numpy()
        Bb = fit_latent_var(zb_s, MAX_LAG, RIDGE).cpu().numpy()

        # scale-gaming check: coordinate std spread before standardization
        pooled = torch.cat([za.reshape(-1, D_LATENT), zb.reshape(-1, D_LATENT)], 0)
        sds = pooled.std(0).cpu().numpy()

    m, true_of_hat, per_true = mcc_and_matching(z_true, z_hat)
    r2 = subspace_r2(z_true, z_hat)
    rr = recon_r(np.concatenate([test["x_a"], test["x_b"]], 0), x_rec)

    # The mechanism term can only pin coordinates the shift actually touches. Coordinates
    # untouched by the shift stay rotationally free under that term alone, so split the MCC
    # by whether a coordinate is in the shift support. See caveat 5 in results.md.
    touched = sorted({c for (_, i, j) in data["shift_support"] for c in (i, j)})
    untouched = [c for c in range(D_LATENT) if c not in touched]
    mcc_touched = float(per_true[touched].mean()) if touched else float("nan")
    mcc_untouched = float(per_true[untouched].mean()) if untouched else float("nan")

    D = np.abs(permute_mechanism(Ba - Bb, true_of_hat, D_LATENT, MAX_LAG))
    D = D / (D.max() + 1e-12)
    mask = true_support_mask(data["shift_support"], D_LATENT, MAX_LAG)
    on_sup = D[mask]
    off_sup = D[~mask]
    n_sup = int(mask.sum())
    topk = np.argsort(D.ravel())[::-1][:n_sup]
    hit = float(mask.ravel()[topk].sum()) / max(n_sup, 1)

    res = dict(
        condition=condition, objective=objective, dist=dist, seed=seed,
        mcc=m, mcc_shift_touched=mcc_touched, mcc_shift_untouched=mcc_untouched,
        n_touched=len(touched), subspace_r2=r2, recon_r=rr,
        mech_gini=gini(D.ravel()),
        mech_on_support_mean=float(on_sup.mean()),
        mech_off_support_mean=float(off_sup.mean()),
        mech_contrast=float(on_sup.mean() / (off_sup.mean() + 1e-12)),
        mech_topk_hit_rate=hit,
        mech_diff=D.tolist(),
        true_support=[list(s) for s in data["shift_support"]],
        scale_gaming_check=dict(
            latent_sd_min=float(sds.min()), latent_sd_max=float(sds.max()),
            latent_sd_ratio=float(sds.max() / (sds.min() + 1e-12)),
        ),
        sanity=sanity,
        seconds=time.time() - t0,
    )
    return res




def var_fit_quality(za, zb):
    """One-step-ahead prediction R2 of the fitted latent VAR, per environment, on a
    held-out tail of the windows. Answers: are these latents a FAITHFUL dynamical model,
    or did the encoder degrade the fit to buy sparsity? za, zb: (n_win, L, d) torch.

    A VAR is closed under rotation, so a RANDOM rotation of the true latents should keep
    R2 ~ TRUE's R2. If the TRAINED model's R2 is clearly below TRUE's while its
    mechanism-difference is sparser, the encoder degraded the dynamics to buy sparsity
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


def sparsity_diagnostic(seed, device, log):
    """Permutation-free, MCC-free check: is the TRAINED model's mechanism-difference
    as sparse (in its own coordinates) as the TRUE unmixing's mechanism-difference?

    Both numbers computed the same way: standardize coords, fit latent VAR per env,
    take |B_a - B_b|, measure sparsity (L1/L2 ratio and Gini). No permutation, no MCC.
    This is what training actually optimizes, so it is not contaminated by the
    MCC-based permute_mechanism alignment used in mech_hit.
    """
    log("=" * 72)
    log(f"SPARSITY DIAGNOSTIC (seed {seed}) — permutation-free, oracle-referenced")
    log("=" * 72)

    data = make_dataset(seed, "laplace")
    n_test = int(TEST_FRAC * N_WINDOWS)
    tr = slice(0, N_WINDOWS - n_test)
    train = {k: data[k][tr] for k in ("x_a", "x_b", "z_a", "z_b")}

    def mech_sparsity(za, zb):
        za_s, zb_s = standardize_coords(za, zb)
        Ba = fit_latent_var(za_s, MAX_LAG, RIDGE)
        Bb = fit_latent_var(zb_s, MAX_LAG, RIDGE)
        Dm = (Ba - Bb)
        l1l2 = float(sparsity_ratio(Dm))
        g = gini(np.abs(Dm.cpu().numpy()).ravel())
        return l1l2, g

    # 1. TRUE unmixing: encode observations with the exact inverse mixing (the oracle latents)
    #    Use the ground-truth latents directly — that IS the true unmixing's output.
    za_true = torch.from_numpy(train["z_a"]).double()
    zb_true = torch.from_numpy(train["z_b"]).double()
    l1l2_true, gini_true = mech_sparsity(za_true, zb_true)
    ra_t, rb_t = var_fit_quality(za_true, zb_true)
    log(f"  TRUE unmixing     : mech L1/L2 = {l1l2_true:.3f}   Gini = {gini_true:.3f}   "
        f"VAR R2 = [{ra_t:.3f}, {rb_t:.3f}]")

    # 2. TRAINED score model: encode the same windows, measure the same sparsity
    model = train_score_model(train, device, log)
    model.eval()
    with torch.no_grad():
        za = model.encode(torch.from_numpy(train["x_a"]).to(device))[0].double().cpu()
        zb = model.encode(torch.from_numpy(train["x_b"]).to(device))[0].double().cpu()
    l1l2_tr, gini_tr = mech_sparsity(za, zb)
    ra_m, rb_m = var_fit_quality(za, zb)
    log(f"  TRAINED model     : mech L1/L2 = {l1l2_tr:.3f}   Gini = {gini_tr:.3f}   "
        f"VAR R2 = [{ra_m:.3f}, {rb_m:.3f}]")

    # 3. reference: a RANDOM rotation of the true latents (what "not identified" looks like)
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(D_LATENT, D_LATENT)))
    za_rot = (za_true.numpy().reshape(-1, D_LATENT) @ Q).reshape(za_true.shape)
    zb_rot = (zb_true.numpy().reshape(-1, D_LATENT) @ Q).reshape(zb_true.shape)
    l1l2_rot, gini_rot = mech_sparsity(torch.from_numpy(za_rot), torch.from_numpy(zb_rot))
    ra_r, rb_r = var_fit_quality(torch.from_numpy(za_rot), torch.from_numpy(zb_rot))
    log(f"  RANDOM rotation   : mech L1/L2 = {l1l2_rot:.3f}   Gini = {gini_rot:.3f}   "
        f"VAR R2 = [{ra_r:.3f}, {rb_r:.3f}]")

    log("-" * 72)
    log("  SPARSITY: TRUE = floor (identified). RANDOM = scrambled ceiling (not identified).")
    log("  VAR R2 : a VAR is rotation-closed, so RANDOM R2 should ~ TRUE R2 (control).")
    log("  VERDICT:")
    log("    TRAINED sparser-than-TRUE AND VAR R2 << TRUE  -> DEGENERATE: bought sparsity by")
    log("      degrading the dynamics. Specification failure, not identification.")
    log("    TRAINED ~ TRUE on BOTH sparsity and VAR R2     -> genuinely identified.")
    log("    TRAINED ~ RANDOM sparsity                      -> objective did not act.")
    log("=" * 72)
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


def fig_mechanism(condition, results):
    """Recovered |B_a - B_b| at true-support entries vs everywhere else, across seeds."""
    if not results:
        return None
    on, off = [], []
    for r in results:
        D = np.array(r["mech_diff"])
        mask = true_support_mask([tuple(s) for s in r["true_support"]], D_LATENT, MAX_LAG)
        on.extend(D[mask].tolist())
        off.extend(D[~mask].tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.boxplot([on, off], showfliers=False)   # set_xticklabels: `labels=` kwarg moved in mpl 3.9
    ax1.set_xticklabels(["true shift support", "off-support"])
    jrng = np.random.default_rng(0)
    ax1.scatter(jrng.normal(1, 0.05, len(on)), on, s=14, alpha=0.6, color="#C44E52", zorder=3)
    ax1.set_ylabel("normalized |B_a - B_b| entry")
    ax1.set_title(f"{condition}: mechanism difference\nrecovered vs ground-truth support "
                  f"(n={len(results)} seeds)")
    ax1.grid(alpha=0.3, axis="y")

    hits = [r["mech_topk_hit_rate"] for r in results]
    ginis = [r["mech_gini"] for r in results]
    ax2.hist(hits, bins=np.linspace(0, 1, 11), color="#4C72B0", alpha=0.85)
    ax2.set_xlabel("top-k hit rate (k = #true shifted coefficients)")
    ax2.set_ylabel("seeds")
    ax2.set_title(f"support recovery\nmean hit {np.mean(hits):.2f}, mean Gini {np.mean(ginis):.3f}")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"fig_mechanism_{condition}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_mechanism_heatmap(condition, results):
    """Seed-0 mechanism difference heatmap with the true shift support boxed."""
    r0 = next((r for r in results if r["seed"] == 0), None)
    if r0 is None:
        return None
    D = np.array(r0["mech_diff"])
    mask = true_support_mask([tuple(s) for s in r0["true_support"]], D_LATENT, MAX_LAG)

    fig, ax = plt.subplots(figsize=(6.5, 8))
    im = ax.imshow(D, cmap="viridis", aspect="auto")
    for (r_, c_) in zip(*np.where(mask)):
        ax.add_patch(plt.Rectangle((c_ - 0.5, r_ - 0.5), 1, 1, fill=False,
                                   edgecolor="red", linewidth=2.2))
    for l in range(1, MAX_LAG):
        ax.axhline(l * D_LATENT - 0.5, color="white", linewidth=1.5)
    ax.set_xlabel("target coordinate z_i[t]  (ground-truth order)")
    ax.set_ylabel("lagged source  (blocks: lag 1, lag 2)")
    ax.set_title(f"{condition}, seed 0\nrecovered |B_a - B_b|; red = true shift support\n"
                 f"MCC {r0['mcc']:.3f}, top-k hit {r0['mech_topk_hit_rate']:.2f}")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"fig_mechanism_heatmap_{condition}.png")
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

        def ms(key):
            v = np.array([r[key] for r in rs], dtype=np.float64)
            return float(np.nanmean(v)), float(np.nanstd(v))

        rows.append(dict(condition=cond, objective=obj, dist=dist, n=len(rs),
                         mcc=ms("mcc"), r2=ms("subspace_r2"), rr=ms("recon_r"),
                         gini=ms("mech_gini"), hit=ms("mech_topk_hit_rate"),
                         contrast=ms("mech_contrast"),
                         mcc_on=ms("mcc_shift_touched"), mcc_off=ms("mcc_shift_untouched")))
    return rows


def write_report(rows, all_results, figs, meta):
    L, verdict = [], []
    L.append("# Mechanism-gate ceiling probe\n")
    L.append(f"- torch {meta['torch']}, numpy {meta['numpy']}, scipy {meta['scipy']}, "
             f"sklearn {meta['sklearn']}")
    L.append(f"- device: {meta['device']} ({meta['gpu']})")
    L.append(f"- d={D_LATENT}, N={N_OBS}, max_lag={MAX_LAG}, windows={N_WINDOWS}x{WINDOW_L}, "
             f"seeds up to {N_SEEDS}, gate MCC>{MCC_GATE}")
    L.append("- ceiling probe: no HRF, no observation noise")
    L.append("- identifying signal: sparsity of the cross-environment latent VAR "
             "coefficient difference (B_a - B_b), on unit-variance latents\n")

    voids = [r for r in all_results if r["sanity"]["void"]]
    if voids:
        L.append(f"> **{len(voids)} runs had a VOID data sanity check** (lagged cross-covariance "
                 f"difference below {LAGGED_COV_WARN}). Their numbers are meaningless.\n")

    L.append("## Results (mean ± std over seeds)\n")
    L.append("| condition | objective | data | n | MCC | subspace R² | recon r | "
             "mech Gini | top-k hit | on/off contrast |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(
            f"| {r['condition']} | {r['objective']} | {r['dist']} | {r['n']} "
            f"| {r['mcc'][0]:.3f} ± {r['mcc'][1]:.3f} "
            f"| {r['r2'][0]:.3f} ± {r['r2'][1]:.3f} "
            f"| {r['rr'][0]:.3f} ± {r['rr'][1]:.3f} "
            f"| {r['gini'][0]:.3f} ± {r['gini'][1]:.3f} "
            f"| {r['hit'][0]:.2f} ± {r['hit'][1]:.2f} "
            f"| {r['contrast'][0]:.1f} ± {r['contrast'][1]:.1f} |"
        )
    L.append("")

    L.append("### MCC split by shift support\n")
    L.append("The mechanism term can only pin coordinates the shift touches. If the mechanism "
             "term is doing the work, `MCC touched` should be well above `MCC untouched`.\n")
    L.append("| condition | MCC (shift-touched coords) | MCC (untouched coords) |")
    L.append("|---|---|---|")
    for r in rows:
        L.append(f"| {r['condition']} | {r['mcc_on'][0]:.3f} ± {r['mcc_on'][1]:.3f} "
                 f"| {r['mcc_off'][0]:.3f} ± {r['mcc_off'][1]:.3f} |")
    L.append("")

    by = {r["condition"]: r for r in rows}
    L.append("## Verdict\n")

    if "score_laplace" in by:
        m, s = by["score_laplace"]["mcc"]
        status = "PASS" if m > MCC_GATE else "FAIL"
        line = f"**GATE — score + Laplace: {status}** (MCC = {m:.3f} ± {s:.3f}, threshold {MCC_GATE})"
        L.append(line + "\n")
        verdict.append(line)

    if "mse_laplace" in by:
        m, s = by["mse_laplace"]["mcc"]
        if m > MCC_GATE:
            line = ("**WARNING — HARNESS SUSPECT: MSE + Laplace PASSED the gate "
                    f"(MCC = {m:.3f} ± {s:.3f} > {MCC_GATE}). A pointwise linear MSE "
                    "autoencoder is rotationally blind and must not identify latents. "
                    "Do not trust the score result until this is explained.**")
        else:
            line = (f"Control OK — MSE + Laplace failed as expected "
                    f"(MCC = {m:.3f} ± {s:.3f} ≤ {MCC_GATE}).")
        L.append(line + "\n")
        verdict.append(line)

    if "score_gaussian" in by:
        m, s = by["score_gaussian"]["mcc"]
        line = f"Score + Gaussian: MCC = {m:.3f} ± {s:.3f} ({'PASS' if m > MCC_GATE else 'FAIL'})."
        L.append(line + "\n")
        verdict.append(line)

    if "mse_gaussian" in by:
        m, s = by["mse_gaussian"]["mcc"]
        L.append(f"Reference — MSE + Gaussian: MCC = {m:.3f} ± {s:.3f}.\n")

    L.append("## Caveats — read before quoting these numbers\n")
    L.append("1. **The Gaussian arm is not a clean negative control under this design.** "
             "Mechanism-difference sparsity is a second-order signal: a Gaussian VAR with a "
             "sparse lagged shift is identifiable up to rotation with no non-Gaussianity at "
             "all. Under the previous marginal-score design, Gaussian data was expected to "
             "fail; here score+Gaussian passing is most likely the CORRECT result and not a "
             "bug. What it would show is that the objective's power comes from the mechanism "
             "term, not from heavy tails.")
    L.append("2. **The Laplace prior is a confound in the Laplace arm.** The loss contains "
             "both an ICA-style non-Gaussian prior and the mechanism-sparsity term, either of "
             "which can break rotation alone. This sweep cannot attribute the score+Laplace "
             "result to the mechanism term. The disambiguating runs are LAMBDA_MECH=0 "
             "(prior only) and LAMBDA_PRIOR=0 (mechanism only); neither is in this 2x2.")
    L.append("3. **Ceiling probe only.** No HRF, no noise, linear instantaneous mixing, "
             "correctly specified lag order. A pass here is a necessary condition, not "
             "evidence the objective survives realistic data.")
    L.append("4. `scale_gaming_check` in each seed JSON reports the spread of latent "
             "coordinate standard deviations. The penalty is computed on in-graph "
             "unit-variance latents, so scale cannot buy sparsity, but a large "
             "`latent_sd_ratio` still means the flow is doing something worth a look.")
    L.append(f"5. **The mechanism term cannot identify every coordinate.** With N_SHIFT="
             f"{N_SHIFT} the shift touches only ~2-4 of the {D_LATENT} coordinates. Any "
             "rotation acting only on the untouched coordinates leaves (B_a - B_b) equally "
             "sparse, so those coordinates are not pinned by the mechanism term at all. This "
             "caps the achievable MCC well below 1 and puts the 0.4 gate uncomfortably close "
             "to the arithmetic: roughly (n_touched * ~0.9 + n_untouched * ~0.2) / "
             f"{D_LATENT}. Read the MCC-split table above before reading the gate — a pass "
             "driven by the untouched coordinates is the Laplace prior's doing, not the "
             "mechanism term's.\n")

    if figs:
        L.append("## Figures\n")
        for f in figs:
            L.append(f"- `{os.path.basename(f)}`")
        L.append("")

    atomic_write(os.path.join(OUT_DIR, "results.md"), "\n".join(L), is_json=False)
    return verdict


# -------------------------------------------------------------------------- oracle test


def run_oracle(log):
    """Harness self-test. No training: encode with the TRUE unmixing plus a known permutation
    and push it through the same readout the real runs use.

    If the index conventions are right this must give MCC ~ 1.0 and top-k hit == 1.0. If it
    does not, the harness is mislabelled and every number the sweep produces is noise.
    """
    log("=" * 72)
    log("ORACLE HARNESS TEST (no training; true unmixing fed straight into the readout)")
    log("=" * 72)
    fails = []

    # 1. does the latent VAR fit recover the ground-truth coefficients on true latents?
    for seed in range(3):
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
    for seed in range(3):
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

    # 4. end to end: oracle encoder through the real readout
    for seed in range(3):
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

    log("=" * 72)
    if fails:
        log("ORACLE TEST FAILED. Do not trust any sweep numbers until this is fixed:")
        for f in fails:
            log("  - " + f)
        return False
    log("ORACLE TEST PASSED — index conventions and readout are sound.")
    return True


# ------------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true",
                    help="harness self-test only (no training, ~1 min); run this first")
    if "--sparsity" in sys.argv:
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        sparsity_diagnostic(0, _dev, lambda m: print(m, flush=True))
        return
    ap.add_argument("--smoke", action="store_true",
                    help="oracle test, then one seed of score+laplace and mse+laplace")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    os.makedirs(SEED_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "n/a"

    def log(msg):
        print(msg, flush=True)

    log("=" * 72)
    log("mechanism-gate ceiling probe")
    log(f"python  {platform.python_version()}  ({sys.platform})")
    log(f"torch   {torch.__version__}   cuda_available={torch.cuda.is_available()}")
    log(f"numpy   {np.__version__}   scipy {scipy.__version__}   sklearn {sklearn.__version__}")
    log(f"device  {device}   gpu: {gpu}")
    log("=" * 72)

    meta = dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, device=device, gpu=gpu)

    if args.oracle or args.smoke:
        if not run_oracle(log):
            sys.exit(1)
        log("")
        if args.oracle:
            return

    conditions = CONDITIONS
    n_seeds = args.seeds
    if args.smoke:
        conditions = [c for c in CONDITIONS if c[0] in ("score_laplace", "mse_laplace")]
        n_seeds = 1
        log("SMOKE MODE: 1 seed, score_laplace + mse_laplace only\n")

    sanity_cache = {}
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
            res = run_one(cond, obj, dist, seed, device, log, sanity_cache)
            atomic_write(p, res, is_json=True)
            all_results.append(res)
            log(f"       MCC {res['mcc']:.3f}  R2 {res['subspace_r2']:.3f}  "
                f"recon_r {res['recon_r']:.3f}  mech_hit {res['mech_topk_hit_rate']:.2f}  "
                f"({res['seconds']:.0f}s)")

    figs = []
    for cond, _, _ in conditions:
        rs = [r for r in all_results if r["condition"] == cond]
        for fn in (fig_mechanism, fig_mechanism_heatmap):
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