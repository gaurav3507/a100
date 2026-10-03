"""E0 oracle harness for GRD (milestone M1).

E0 is the blocking gate: it must be green on the Mac CPU before any experiment or
any A100 use (Lesson 1). It exercises the simulator and the P4 rank readout end to
end on a known-answer construction, across seeds 0..9 (never seed 0 alone,
Lesson 9), and it includes a negative control that must FAIL (Lesson 8).

Known-answer construction
  A hard intervention on a SOURCE node has no incoming edges to cut, so it changes
  only that node's noise variance and produces a clean rank-1 direction in the
  latent covariance difference. A hard intervention on ALL source nodes therefore
  builds a covariance difference of known rank equal to the number of source
  nodes, with every signal direction well separated from the finite-sample noise
  floor. This gives an exact, robust known answer on every seed (the number of
  sources, and hence the rank, varies by seed, which broadens coverage across
  ranks 1..4). The construction is multi-node whenever a seed has two or more
  sources. iv_scale is pinned so the exact population covariance matches the data.

E0-A construction check
  The empirical covariance difference between the observational and the
  intervention environment, in the control-fitted PCA space, has numerical rank
  equal to the constructed rank. The readout sees what was built.

E0-B rank readout check
  The P4 readout, given that covariance difference, returns the constructed rank:
  the deterministic spectral rank equals it, and the LFC bootstrap test rejects
  H0(rank <= r) for every r below the constructed rank (the informative, high
  power direction). The LFC sequential estimate and its nominal-alpha behaviour at
  the true rank are reported per seed.

E0-C blocking blind-baseline gate
  A rotation-blind linear MSE autoencoder (PCA, the closed-form optimum of a
  linear reconstruction autoencoder) uses no environment or intervention labels.
  It recovers the latent SUBSPACE (linear-reconstruction R^2 near 1) but not the
  latent AXES, so its MCC against the ground-truth latents is well below 1, near
  the random-rotation chance floor. If that MCC ever reaches the recovery
  threshold the metric or harness is lying: the report is marked HARNESS_INVALID
  and the script exits non-zero.

Writes results/e0/e0_report.json (tracked) and results/e0/e0_spectra.npz (ignored
binary). Prints a PASS/FAIL table and exits non-zero if any check fails or if the
harness is invalid (Lesson 13).
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# The macOS Accelerate BLAS shipped with numpy 2.x raises spurious "divide by
# zero / overflow / invalid value encountered in matmul" RuntimeWarnings on
# perfectly finite operands (it sets FPU exception flags while processing SIMD
# padding lanes). Verified benign here: every generated array is finite (the
# simulator asserts this), and a plain finite matmul reproduces the warning.
# Only this exact message is suppressed; all other warnings stay visible.
warnings.filterwarnings("ignore", message=r".*encountered in matmul",
                        category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results" / "e0"

SEEDS = list(range(10))
D_LATENT = 5
D_OBS = 200
N_PER_ENV = 4000
D_PROJ = 5                 # PCA projection dimension (equals d_latent, keeps the latent subspace)
EDGE_PROB = 0.4
IV_SCALE = 3.0             # pinned hard noise variance, so population matches data
ALPHA = 0.05
B_BOOT = 500
RANK_RTOL = 0.05           # numerical-rank tolerance; safe range on this construction is ~0.013..0.26
RECOVERY_MCC = 0.90        # a method at or above this "recovered" the latents; the blind baseline must stay below


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SIM = _load(REPO / "sim" / "simulator.py", "grd_sim_simulator")
RR = _load(REPO / "src" / "gate" / "rank_readout.py", "grd_gate_rank_readout")


# ------------------------------------------------------------------ helpers
def fit_pca(Xc, d):
    """PCA fitted on control cells only. Returns (mu, W) with W of shape (D, d)."""
    mu = Xc.mean(0)
    _, _, Vt = np.linalg.svd(Xc - mu, full_matrices=False)
    return mu, Vt[:d].T


def project(X, mu, W):
    return (X - mu) @ W


def _zscore(A):
    A = A - A.mean(0)
    sd = A.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return A / sd


def mcc(Z_hat, Z):
    """Mean correlation coefficient: best permutation/sign matching of components
    by absolute Pearson correlation (Hungarian). No linear realignment, so a
    rotation-blind estimate is correctly penalised.
    """
    n = Z_hat.shape[0]
    C = np.abs((_zscore(Z_hat).T @ _zscore(Z)) / n)
    row, col = linear_sum_assignment(-C)
    return float(C[row, col].mean())


def subspace_r2(Z_hat, Z):
    """R^2 of the best linear reconstruction of the true latents from Z_hat.
    Near 1 means Z_hat spans the latent subspace even when its axes are wrong.
    """
    Zc = Z - Z.mean(0)
    Zh = Z_hat - Z_hat.mean(0)
    coef, _, _, _ = np.linalg.lstsq(Zh, Zc, rcond=None)
    resid = Zc - Zh @ coef
    ss_tot = float(np.sum(Zc ** 2))
    if ss_tot == 0:
        return 0.0
    return float(1.0 - np.sum(resid ** 2) / ss_tot)


def chance_mcc(Z, rng, n_rot=20):
    """Chance-floor MCC: MCC of random orthogonal rotations of Z against Z."""
    d = Z.shape[1]
    vals = []
    for _ in range(n_rot):
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        vals.append(mcc(Z @ Q, Z))
    return float(np.mean(vals))


def _fingerprint(X):
    sl = np.ascontiguousarray(np.round(X[:50, :10], 6))
    return {"shape": [int(s) for s in X.shape],
            "slice_sha256_16": hashlib.sha256(sl.tobytes()).hexdigest()[:16]}


def _code_commit():
    try:
        h = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            stderr=subprocess.DEVNULL).decode().strip() != ""
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# --------------------------------------------------------------------- E0
def run_e0():
    report = {
        "milestone": "M1",
        "code_commit": _code_commit(),
        "config": dict(seeds=SEEDS, d_latent=D_LATENT, D=D_OBS, n_per_env=N_PER_ENV,
                       d_proj=D_PROJ, edge_prob=EDGE_PROB, iv_scale=IV_SCALE,
                       alpha=ALPHA, B=B_BOOT, rank_rtol=RANK_RTOL,
                       recovery_mcc_threshold=RECOVERY_MCC, mixing="linear",
                       intervention="hard on all source nodes"),
        "data_fingerprint": {},
        "E0A": {"per_seed": []},
        "E0B": {"per_seed": []},
        "E0C": {"per_seed": []},
    }
    spectra = {}

    for seed in SEEDS:
        rng_scm = np.random.default_rng(seed)
        B, nv, is_source = SIM.make_scm(D_LATENT, rng_scm, edge_prob=EDGE_PROB)
        nodes = tuple(int(i) for i in np.where(is_source)[0])
        r_c = SIM.constructed_rank(B, nv, "hard", nodes, IV_SCALE)

        specs = [
            SIM.EnvSpec("basis", None, ()),
            SIM.EnvSpec("obs", None, ()),
            SIM.EnvSpec("iv", "hard", nodes, IV_SCALE),
        ]
        ds = SIM.simulate(D_LATENT, D_OBS, N_PER_ENV, specs, seed,
                          mixing="linear", edge_prob=EDGE_PROB)
        report["data_fingerprint"][str(seed)] = _fingerprint(ds.environments["iv"].X)

        mu, W = fit_pca(ds.environments["basis"].X, D_PROJ)
        Y0 = project(ds.environments["obs"].X, mu, W)
        Ye = project(ds.environments["iv"].X, mu, W)
        Delta = RR.covariance_difference(Ye, Y0)
        svals = RR.singular_values(Delta)
        spectra[f"seed{seed}"] = svals

        # ---- E0-A: empirical numerical rank equals the constructed rank.
        emp_rank = RR.spectral_rank(Delta, rtol=RANK_RTOL)
        gap = RR.gap_ratio(Delta, r_c)
        a_pass = bool(emp_rank == r_c)
        report["E0A"]["per_seed"].append(dict(
            seed=seed, nodes=list(nodes), constructed_rank=int(r_c),
            empirical_rank=int(emp_rank), gap_ratio=round(float(gap), 3),
            multi_node=bool(len(nodes) >= 2), passed=a_pass))

        # ---- E0-B: the P4 readout returns the constructed rank.
        spec_rank = RR.spectral_rank(Delta, rtol=RANK_RTOL)
        rng_boot = np.random.default_rng(70_000 + seed)
        est = RR.estimate_rank_lfc(Ye, Y0, r_max=min(r_c + 1, D_PROJ - 1),
                                   B=B_BOOT, alpha=ALPHA, rng=rng_boot)
        rejects_below = True
        for r in range(r_c):
            rng_r = np.random.default_rng(71_000 + seed * 100 + r)
            res = RR.lfc_rank_test(Ye, Y0, r, B=B_BOOT, alpha=ALPHA, rng=rng_r)
            rejects_below = rejects_below and bool(res["reject"])
        b_pass = bool(spec_rank == r_c and rejects_below)
        report["E0B"]["per_seed"].append(dict(
            seed=seed, constructed_rank=int(r_c), spectral_rank=int(spec_rank),
            lfc_rejects_below_rank=bool(rejects_below),
            lfc_sequential_k_hat=int(est["k_hat"]),
            lfc_k_hat_matches=bool(est["k_hat"] == r_c),
            passed=b_pass))

        # ---- E0-C: rotation-blind linear autoencoder must FAIL recovery.
        X_obs = ds.environments["obs"].X
        Z_true = ds.environments["obs"].Z
        mu_ae, W_ae = fit_pca(X_obs, D_LATENT)          # blind linear AE optimum = PCA
        Z_hat = project(X_obs, mu_ae, W_ae)
        blind_mcc = mcc(Z_hat, Z_true)
        r2 = subspace_r2(Z_hat, Z_true)
        rng_ch = np.random.default_rng(80_000 + seed)
        floor = chance_mcc(Z_true, rng_ch)
        fails = bool(blind_mcc < RECOVERY_MCC)
        # Metric self-test: a known relabel (permutation + sign) of the true
        # latents MUST score MCC ~ 1. Without this the negative control could
        # pass vacuously with an always-low metric (Lesson 8).
        rng_perm = np.random.default_rng(90_000 + seed)
        perm = rng_perm.permutation(D_LATENT)
        signs = rng_perm.choice([-1.0, 1.0], D_LATENT)
        mcc_selftest = mcc(Z_true[:, perm] * signs, Z_true)
        report["E0C"]["per_seed"].append(dict(
            seed=seed, blind_mcc=round(blind_mcc, 4), subspace_r2=round(r2, 4),
            chance_mcc=round(floor, 4), fails_recovery=fails,
            metric_selftest_mcc=round(mcc_selftest, 4)))

    # ---- verdicts
    a_all = all(r["passed"] for r in report["E0A"]["per_seed"])
    b_all = all(r["passed"] for r in report["E0B"]["per_seed"])
    c_fails_all = all(r["fails_recovery"] for r in report["E0C"]["per_seed"])
    c_metric_ok = all(r["metric_selftest_mcc"] > 0.999 for r in report["E0C"]["per_seed"])
    report["E0A"]["verdict"] = "PASS" if a_all else "FAIL"
    report["E0B"]["verdict"] = "PASS" if b_all else "FAIL"
    # E0-C is a negative control: PASS means the metric can score a true recovery
    # (self-test ~ 1) AND the blind baseline still failed. Either half broken is
    # a lying harness.
    report["E0C"]["verdict"] = ("PASS" if (c_fails_all and c_metric_ok)
                                else "HARNESS_INVALID")

    if not (c_fails_all and c_metric_ok):
        report["status"] = "HARNESS_INVALID"
    elif a_all and b_all:
        report["status"] = "PASS"
    else:
        report["status"] = "FAIL"
    return report, spectra


def _print_table(report):
    print("\nE0 ORACLE SUMMARY  (status: %s)" % report["status"], flush=True)
    print("=" * 78, flush=True)
    print(f"{'seed':>4} {'nodes':>10} {'rc':>3} {'E0A rank':>9} {'gap':>9} "
          f"{'E0B spec':>9} {'lfc_k':>6} {'blindMCC':>9} {'chance':>7} {'r2':>6}",
          flush=True)
    a = {r["seed"]: r for r in report["E0A"]["per_seed"]}
    b = {r["seed"]: r for r in report["E0B"]["per_seed"]}
    c = {r["seed"]: r for r in report["E0C"]["per_seed"]}
    for s in SEEDS:
        print(f"{s:>4} {str(a[s]['nodes']):>10} {a[s]['constructed_rank']:>3} "
              f"{a[s]['empirical_rank']:>9} {a[s]['gap_ratio']:>9.1f} "
              f"{b[s]['spectral_rank']:>9} {b[s]['lfc_sequential_k_hat']:>6} "
              f"{c[s]['blind_mcc']:>9.3f} {c[s]['chance_mcc']:>7.3f} "
              f"{c[s]['subspace_r2']:>6.3f}", flush=True)
    print("-" * 78, flush=True)
    print(f"E0-A construction check     : {report['E0A']['verdict']}", flush=True)
    print(f"E0-B rank readout check     : {report['E0B']['verdict']}", flush=True)
    st = [r["metric_selftest_mcc"] for r in report["E0C"]["per_seed"]]
    print(f"E0-C blind-baseline control : {report['E0C']['verdict']} "
          f"(must fail recovery; blind MCC < {RECOVERY_MCC})", flush=True)
    print(f"     metric self-test MCC (true relabel, must be ~1): "
          f"min={min(st):.4f} max={max(st):.4f}", flush=True)
    print(f"OVERALL STATUS              : {report['status']}", flush=True)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    report, spectra = run_e0()
    (RESULTS / "e0_report.json").write_text(json.dumps(report, indent=2))
    np.savez(RESULTS / "e0_spectra.npz", **spectra)
    _print_table(report)
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
