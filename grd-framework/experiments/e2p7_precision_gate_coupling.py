"""M2.7 / Path A: per-node precision gate readout + coupling recheck.

M2.6 found the pooled-covariance gate and the backbone disagree on anisotropic seed 7
(gate read aggregate rank 4, backbone recovered all 5) because they read different
signals. src/gate/precision_readout.py reads the SAME per-node precision differences
the backbone uses to isolate directions. This recheck asks: with that readout, do the
gate count and the backbone agree with the true recoverable dimension on ALL seeds 0-9?

For each seed (REDUCING regime, iv_scale = 0.1, the regime the backbone recovers on):
  1. Generate E1's atomic all-node construction plus one all-node aggregate env (so the
     OLD pooled-covariance gate count can be shown beside the NEW per-node count).
  2. NEW gate: precision_readout.count_recoverable over the per-node atomic envs.
  3. OLD gate: the existing pooled aggregate reading (M2.6): estimate_rank_lfc on the
     single all-node covariance difference.
  4. Backbone recovery MCC on the atomic envs.
Assert across ALL seeds: NEW precision-gate count == true recoverable dimension d
(Lesson 9). Print the OLD pooled count beside it so the report shows whether the switch
fixed seed 7.

Backbone direction isolation (provenance, one line): backbone._unmixing_row takes the
top-eigenvalue eigenvector of the per-node precision difference
inv(cov(Y_env)) - inv(cov(Y_obs)); the new gate counts detectability of that same
top eigenvalue per node.

Read-only in spirit: imports the simulator, existing pooled readout, the new precision
readout, the backbone and the E0 mcc; modifies none of them. Writes
results/e2p7/precision_gate_report.json (tracked), prints a PASS/FAIL table, exits
non-zero if any seed's new-gate count != d (Lesson 13).
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
RESULTS = REPO / "results" / "e2p7"

SEEDS = list(range(10))
D_LATENT = 5
D_OBS = 200
N_PER_ENV = 4000
D_PROJ = 5
EDGE_PROB = 0.4
ALPHA = 0.05
B_BOOT = 500
IV_REDUCING = 0.1
RECOVERY_MCC = 0.90


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SIM = _load(REPO / "sim" / "simulator.py", "grd_sim_simulator")
RR = _load(REPO / "src" / "gate" / "rank_readout.py", "grd_gate_rank_readout")
PR = _load(REPO / "src" / "gate" / "precision_readout.py", "grd_gate_precision_readout")
BK = _load(REPO / "src" / "recover" / "backbone.py", "grd_recover_backbone")
E0 = _load(REPO / "experiments" / "e0_oracle.py", "grd_e0_oracle")


def run():
    report = {
        "milestone": "M2.7",
        "code_commit": E0._code_commit(),
        "backbone_isolation": (
            "backbone._unmixing_row takes the top-eigenvalue eigenvector of the per-node "
            "precision difference inv(cov(Y_env)) - inv(cov(Y_obs)); the new gate counts "
            "detectability of that same top eigenvalue per node vs a control-vs-control null."),
        "config": dict(seeds=SEEDS, d_latent=D_LATENT, D=D_OBS, n_per_env=N_PER_ENV,
                       d_proj=D_PROJ, edge_prob=EDGE_PROB, alpha=ALPHA, B=B_BOOT,
                       iv_scale=IV_REDUCING, regime="REDUCING",
                       recovery_mcc_threshold=RECOVERY_MCC,
                       new_gate="src/gate/precision_readout.py count_recoverable (per-node precision)",
                       old_gate="src/gate/rank_readout.py estimate_rank_lfc on the pooled all-node covariance difference (M2.6)"),
        "data_fingerprint": {},
        "per_seed": [],
    }

    for seed in SEEDS:
        rng_scm = np.random.default_rng(seed)
        B, nv, is_source = SIM.make_scm(D_LATENT, rng_scm, edge_prob=EDGE_PROB)
        all_nodes = tuple(range(D_LATENT))
        true_dim = SIM.constructed_rank(B, nv, "hard", all_nodes, IV_REDUCING)

        specs = [SIM.EnvSpec("basis", None, ()), SIM.EnvSpec("obs", None, ())]
        for i in range(D_LATENT):
            specs.append(SIM.EnvSpec(f"iv{i}", "hard", (i,), IV_REDUCING))
        specs.append(SIM.EnvSpec("ivall", "hard", all_nodes, IV_REDUCING))
        ds = SIM.simulate(D_LATENT, D_OBS, N_PER_ENV, specs, seed,
                          mixing="linear", edge_prob=EDGE_PROB)
        report["data_fingerprint"][str(seed)] = E0._fingerprint(ds.environments["iv0"].X)

        mu, Wp = BK.fit_pca(ds.environments["basis"].X, D_PROJ)
        Y_obs = BK.project(ds.environments["obs"].X, mu, Wp)
        Y_int = [BK.project(ds.environments[f"iv{i}"].X, mu, Wp) for i in range(D_LATENT)]

        # NEW gate: per-node precision detectability count.
        new = PR.count_recoverable(Y_int, Y_obs, alpha=ALPHA, B=B_BOOT,
                                   rng=np.random.default_rng(910_000 + seed))
        new_count = new["count"]

        # OLD gate: pooled all-node covariance LFC (M2.6 method).
        Y_all = BK.project(ds.environments["ivall"].X, mu, Wp)
        old_count = RR.estimate_rank_lfc(Y_all, Y_obs, r_max=D_PROJ - 1, B=B_BOOT,
                                         alpha=ALPHA,
                                         rng=np.random.default_rng(700_000 + seed))["k_hat"]

        # Backbone recovery on the atomic envs.
        envs = {k: e.X for k, e in ds.environments.items() if k != "ivall"}
        targets = {f"iv{i}": i for i in range(D_LATENT)}
        out = BK.recover(envs, targets, D_LATENT, basis_key="basis", obs_key="obs")
        recover_mcc = E0.mcc(out["Z_hat"], ds.environments["obs"].Z)

        report["per_seed"].append(dict(
            seed=seed, true_dim=int(true_dim),
            new_precision_gate_count=int(new_count),
            new_gate_pass=bool(new_count == true_dim),
            new_gate_min_ratio=round(float(min(new["ratios"])), 2),
            old_pooled_cov_count=int(old_count),
            old_gate_pass=bool(old_count == true_dim),
            recover_mcc=round(float(recover_mcc), 4),
            recover_pass=bool(recover_mcc >= RECOVERY_MCC)))

    all_new_pass = all(r["new_gate_pass"] for r in report["per_seed"])
    old_fail_seeds = [r["seed"] for r in report["per_seed"] if not r["old_gate_pass"]]
    new_fail_seeds = [r["seed"] for r in report["per_seed"] if not r["new_gate_pass"]]
    report["verdict"] = "PASS" if all_new_pass else "FAIL"
    report["status"] = report["verdict"]
    report["old_gate_fail_seeds"] = old_fail_seeds
    report["new_gate_fail_seeds"] = new_fail_seeds
    seed7 = next(r for r in report["per_seed"] if r["seed"] == 7)
    report["seed7_fixed"] = bool((not seed7["old_gate_pass"]) and seed7["new_gate_pass"])
    report["conclusion"] = (
        f"Path A works: the per-node precision gate count == true dimension on all seeds "
        f"and now agrees with the backbone. The old pooled-covariance gate failed on seeds "
        f"{old_fail_seeds}; the new gate fixes them (seed 7 specifically: "
        f"{'fixed' if report['seed7_fixed'] else 'NOT fixed'}). Gate and backbone now read "
        f"the same per-node precision signal."
        if all_new_pass else
        f"Path A incomplete: the per-node precision gate still misreads on seeds "
        f"{new_fail_seeds}.")
    return report


def _print(report):
    print(f"\nM2.7 PRECISION-GATE COUPLING  (status: {report['status']})", flush=True)
    print("=" * 74, flush=True)
    print(f"{'seed':>4} {'true dim':>9} {'NEW prec count':>15} {'pass':>5} "
          f"{'min ratio':>10} {'OLD pooled':>11} {'recover MCC':>12}", flush=True)
    for r in report["per_seed"]:
        print(f"{r['seed']:>4} {r['true_dim']:>9} {r['new_precision_gate_count']:>15} "
              f"{'yes' if r['new_gate_pass'] else 'NO':>5} {r['new_gate_min_ratio']:>10.1f} "
              f"{r['old_pooled_cov_count']:>11} {r['recover_mcc']:>12.4f}", flush=True)
    print("-" * 74, flush=True)
    print(f"backbone isolation: {report['backbone_isolation']}", flush=True)
    print(f"old pooled-covariance gate failed on seeds: {report['old_gate_fail_seeds']}", flush=True)
    print(f"seed 7 specifically fixed by the switch: {report['seed7_fixed']}", flush=True)
    print(f"VERDICT: {report['verdict']}", flush=True)
    print(f"CONCLUSION: {report['conclusion']}", flush=True)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = run()
    (RESULTS / "precision_gate_report.json").write_text(json.dumps(report, indent=2))
    _print(report)
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
