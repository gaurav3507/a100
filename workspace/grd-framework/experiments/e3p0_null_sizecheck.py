"""E3.0 regression test: the precision gate's null must be sample-size matched (Lesson 2).

The bug: null_threshold estimated its bootstrap at Y_obs's (large) sample size, but
precision_signal is called on perturbation environments that may have far fewer rows.
A smaller environment has a noisier covariance, so its precision difference against
Y_obs is inflated by finite-sample noise alone; scored against the tight full-size
null, a pure-control environment is then "detected" as a recoverable direction (a false
positive). The fix computes the null at the SAME sample size as the environment.

This check encodes the bug-and-fix so it cannot silently return. On pure synthetic
control data (no intervention anywhere), fake "environments" are fresh small control
draws that carry NO signal. It measures the false-positive detection rate of such fake
environments against:
  * the size-MATCHED null (n_env = small size): must sit near alpha (~5%);
  * the old FULL-size null (n_env = None -> len(Y_obs)): inflated, near 100%.

Passes when matched-null false-positive rate is near alpha and full-size-null rate is
high (the bug reproduced). Imports the gate readout and the simulator; reimplements
nothing. Writes results/e3p0/null_sizecheck_report.json; exits non-zero if the
regression property does not hold.
"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=r".*encountered in matmul",
                        category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results" / "e3p0"

D_LATENT = 5
D_OBS = 200
D_PROJ = 5
EDGE_PROB = 0.4
ALPHA = 0.05
B_BOOT = 500
N_OBS = 4000
N_SMALL = 40           # perturbation-environment size, much smaller than N_OBS
K_FAKE = 300           # number of fake (pure-control, small) environments
SEED = 0
MATCHED_MAX = 0.15     # matched-null false-positive rate must be near alpha
FULLSIZE_MIN = 0.80    # full-size-null false-positive rate must be high (bug reproduced)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SIM = _load(REPO / "sim" / "simulator.py", "grd_sim_simulator")
PR = _load(REPO / "src" / "gate" / "precision_readout.py", "grd_gate_precision_readout")
BK = _load(REPO / "src" / "recover" / "backbone.py", "grd_recover_backbone")
E0 = _load(REPO / "experiments" / "e0_oracle.py", "grd_e0_oracle")


def run():
    rng = np.random.default_rng(SEED)
    B, nv, is_src = SIM.make_scm(D_LATENT, rng, edge_prob=EDGE_PROB)
    A = rng.standard_normal((D_OBS, D_LATENT))
    sd = None

    def control_X(n):
        nonlocal sd
        Z = SIM.sample_latent(B, nv, n, rng)         # observational latents, no intervention
        Xsig = SIM.mix_linear(Z, A)
        if sd is None:
            sd = 0.1 * float(np.mean(Xsig.std(0)))
        return SIM.add_obs_noise(Xsig, sd, rng)

    X_basis = control_X(N_OBS)
    X_obs = control_X(N_OBS)
    mu, Wp = BK.fit_pca(X_basis, D_PROJ)
    Y_obs = BK.project(X_obs, mu, Wp)

    # Two nulls, computed once each: size-matched (n_env=N_SMALL) and full-size (old).
    matched_crit = PR.null_threshold(Y_obs, alpha=ALPHA, B=B_BOOT,
                                     rng=np.random.default_rng(100), n_env=N_SMALL)
    fullsize_crit = PR.null_threshold(Y_obs, alpha=ALPHA, B=B_BOOT,
                                      rng=np.random.default_rng(100), n_env=None)

    det_matched = det_fullsize = 0
    for _ in range(K_FAKE):
        Y_fake = BK.project(control_X(N_SMALL), mu, Wp)   # pure control, small size, NO signal
        s = PR.precision_signal(Y_fake, Y_obs)
        det_matched += int(s > matched_crit)
        det_fullsize += int(s > fullsize_crit)

    rate_matched = det_matched / K_FAKE
    rate_fullsize = det_fullsize / K_FAKE
    passed = bool(rate_matched <= MATCHED_MAX and rate_fullsize >= FULLSIZE_MIN)

    return dict(
        milestone="E3.0",
        code_commit=E0._code_commit(),
        config=dict(d_latent=D_LATENT, D=D_OBS, d_proj=D_PROJ, alpha=ALPHA, B=B_BOOT,
                    n_obs=N_OBS, n_small=N_SMALL, k_fake=K_FAKE, seed=SEED,
                    matched_max=MATCHED_MAX, fullsize_min=FULLSIZE_MIN),
        matched_null_threshold=round(float(matched_crit), 5),
        fullsize_null_threshold=round(float(fullsize_crit), 5),
        false_positive_rate_matched=round(rate_matched, 4),
        false_positive_rate_fullsize=round(rate_fullsize, 4),
        interpretation=("size-matched null keeps false positives near alpha; the full-size "
                        "null (the bug) flags nearly all pure-control small environments"),
        status="PASS" if passed else "FAIL")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = run()
    (RESULTS / "null_sizecheck_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nE3.0 NULL SIZE CHECK  (status: {report['status']})", flush=True)
    print(f"  N_obs={N_OBS}, fake-env size={N_SMALL}, K={K_FAKE}, alpha={ALPHA}", flush=True)
    print(f"  false-positive rate  size-matched null : {report['false_positive_rate_matched']} "
          f"(must be <= {MATCHED_MAX}, near alpha {ALPHA})", flush=True)
    print(f"  false-positive rate  full-size null(bug): {report['false_positive_rate_fullsize']} "
          f"(must be >= {FULLSIZE_MIN})", flush=True)
    print(f"  thresholds: matched={report['matched_null_threshold']} "
          f"full-size={report['fullsize_null_threshold']}", flush=True)
    print(f"  STATUS: {report['status']}", flush=True)
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
