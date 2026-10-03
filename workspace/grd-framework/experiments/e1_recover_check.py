"""E1 recovery check for the RECOVER backbone (milestone M2).

Confirms the RECOVER backbone (src/recover/backbone.py, the R2 covariance-based
linear estimator) recovers the latent variables on OUR simulator's clean data,
where the E0-C rotation-blind baseline failed. This does NOT wire the estimator
into the Gate (that is M3/E1-proper). It does not modify the E0 code; it imports
the E0 MCC metric so the comparison is apples-to-apples.

Construction (clean, identifiable by construction)
  For each seed: an observational environment plus one PERFECT (hard) intervention
  per latent (atomic, target-noise variance reduced to iv_scale=0.1). This is the
  regime where "hard gives perfect identifiability" (design doc, Bing/Varici
  lineage). Targets are known (one environment per latent). The observational
  environment is byte-identical to E0's, so the blind baseline reproduces E0-C.

Assertions
  RECOVER succeeds where blind failed: MCC(Z_hat, true latents) is near 1 and
  above the 0.90 recovery threshold on ALL seeds 0..9 (Lesson 9), well above the
  E0-C blind ceiling (~0.75) which is recomputed here per seed for contrast.

Writes results/e1/e1_recover_report.json (tracked). Prints a PASS/FAIL table and
exits non-zero if any seed's recovery MCC is below threshold (Lesson 13).
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
RESULTS = REPO / "results" / "e1"

SEEDS = list(range(10))
D_LATENT = 5
D_OBS = 200
N_PER_ENV = 4000
D_PROJ = 5
EDGE_PROB = 0.4
IV_SCALE = 0.1                 # reduced-variance perfect interventions (the identifiable regime)
RECOVERY_MCC = 0.90


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SIM = _load(REPO / "sim" / "simulator.py", "grd_sim_simulator")
E0 = _load(REPO / "experiments" / "e0_oracle.py", "grd_e0_oracle")   # reuse mcc, fingerprint, commit
BK = _load(REPO / "src" / "recover" / "backbone.py", "grd_recover_backbone")


def run_e1():
    report = {
        "milestone": "M2",
        "code_commit": E0._code_commit(),
        "config": dict(seeds=SEEDS, d_latent=D_LATENT, D=D_OBS, n_per_env=N_PER_ENV,
                       d_proj=D_PROJ, edge_prob=EDGE_PROB, iv_scale=IV_SCALE,
                       recovery_mcc_threshold=RECOVERY_MCC, mixing="linear",
                       intervention="atomic perfect (hard) interventions, one per latent, reduced target-noise variance",
                       targets="known",
                       estimator="R2 covariance backbone: precision-difference top eigenvector",
                       blind_baseline="E0-C rotation-blind linear autoencoder (PCA), recomputed on the observational env"),
        "data_fingerprint": {},
        "per_seed": [],
    }

    for seed in SEEDS:
        # Build: basis (control), obs, and one atomic perfect intervention per latent.
        specs = [SIM.EnvSpec("basis", None, ()), SIM.EnvSpec("obs", None, ())]
        for i in range(D_LATENT):
            specs.append(SIM.EnvSpec(f"iv{i}", "hard", (i,), IV_SCALE))
        ds = SIM.simulate(D_LATENT, D_OBS, N_PER_ENV, specs, seed,
                          mixing="linear", edge_prob=EDGE_PROB)
        report["data_fingerprint"][str(seed)] = E0._fingerprint(ds.environments["obs"].X)

        envs = {k: e.X for k, e in ds.environments.items()}
        targets = {f"iv{i}": i for i in range(D_LATENT)}
        out = BK.recover(envs, targets, D_LATENT, basis_key="basis", obs_key="obs")

        Z_true = ds.environments["obs"].Z
        recover_mcc = E0.mcc(out["Z_hat"], Z_true)

        # Blind baseline, same metric, same observational data as E0-C.
        mu_ae, W_ae = BK.fit_pca(ds.environments["obs"].X, D_LATENT)
        Z_blind = BK.project(ds.environments["obs"].X, mu_ae, W_ae)
        blind_mcc = E0.mcc(Z_blind, Z_true)

        report["per_seed"].append(dict(
            seed=seed, recover_mcc=round(recover_mcc, 4),
            blind_mcc=round(blind_mcc, 4),
            recovers=bool(recover_mcc >= RECOVERY_MCC),
            beats_blind=bool(recover_mcc > blind_mcc)))

    all_recover = all(r["recovers"] for r in report["per_seed"])
    report["verdict"] = "PASS" if all_recover else "FAIL"
    report["status"] = report["verdict"]
    return report


def _print_table(report):
    print("\nE1 RECOVERY CHECK  (status: %s)" % report["status"], flush=True)
    print("=" * 60, flush=True)
    print(f"{'seed':>4} {'recover MCC':>12} {'blind MCC (E0-C)':>18} {'recovers?':>10}",
          flush=True)
    for r in report["per_seed"]:
        print(f"{r['seed']:>4} {r['recover_mcc']:>12.4f} {r['blind_mcc']:>18.4f} "
              f"{'yes' if r['recovers'] else 'NO':>10}", flush=True)
    print("-" * 60, flush=True)
    rec = [r["recover_mcc"] for r in report["per_seed"]]
    bl = [r["blind_mcc"] for r in report["per_seed"]]
    print(f"recover MCC: min={min(rec):.4f} mean={np.mean(rec):.4f}   "
          f"blind MCC: min={min(bl):.4f} max={max(bl):.4f}", flush=True)
    print(f"threshold {RECOVERY_MCC}; all seeds recover: {report['verdict']}", flush=True)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = run_e1()
    (RESULTS / "e1_recover_report.json").write_text(json.dumps(report, indent=2))
    _print_table(report)
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
