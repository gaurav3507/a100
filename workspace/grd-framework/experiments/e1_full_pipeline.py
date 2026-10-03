"""E1-full: first end-to-end GATE -> RECOVER -> DISCOVER pipeline test (milestone M3).

Chains the validated stages on clean reducing-regime data (iv_scale 0.1), seeds 0-9:
  GATE   src/gate/precision_readout.py  -> n_recoverable
  RECOVER src/recover/backbone.py       -> Z_hat (node-aligned recovered latents)
  DISCOVER src/discover/discover.py     -> latent graph + per-edge status codes

Asserts across ALL seeds 0-9 (Lesson 9):
  (a) gate n_recoverable == true dimension (re-confirm Path A in-pipeline);
  (b) recovery MCC >= 0.90 (re-confirm end to end, same E0 mcc);
  (c) the DECIDED edges match the true latent graph. Bar: SHD over decided edges == 0
      (Discover commits only to correct edges) AND at least one edge is DECIDED_PRESENT
      (abstention is not degenerate). Rationale: in the clean perfect-known-target regime
      Discover should be trustworthy on every edge it decides and abstain rather than err;
      the edges it cannot resolve (weak or order-uncertain) are reported via the abstention
      rate, not forced. SHD over decided, decided precision/recall, and abstention rate are
      all reported.

Read-only in spirit: imports the simulator, precision gate readout, backbone, discover, and
the E0 mcc; modifies none of them. Writes results/e1_full/e1_full_report.json (tracked),
prints a PASS/FAIL table, exits non-zero if any seed fails (a), (b), or the (c) bar (Lesson 13).
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
RESULTS = REPO / "results" / "e1_full"

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
PR = _load(REPO / "src" / "gate" / "precision_readout.py", "grd_gate_precision_readout")
BK = _load(REPO / "src" / "recover" / "backbone.py", "grd_recover_backbone")
DISC = _load(REPO / "src" / "discover" / "discover.py", "grd_discover")
E0 = _load(REPO / "experiments" / "e0_oracle.py", "grd_e0_oracle")


def _true_edges(B):
    d = B.shape[0]
    return set((int(k), int(j)) for j in range(d) for k in range(d)
               if abs(B[j, k]) > 1e-9)


def run():
    report = {
        "milestone": "M3",
        "code_commit": E0._code_commit(),
        "discovery_method": (
            "interventional order identification (perfect interventions reveal descendants, "
            "Hauser and Buhlmann 2012 style) + ordered least-squares linear-SEM regression, with "
            "orientation-checked abstention on unresolved edges. Not a new algorithm."),
        "config": dict(seeds=SEEDS, d_latent=D_LATENT, D=D_OBS, n_per_env=N_PER_ENV,
                       d_proj=D_PROJ, edge_prob=EDGE_PROB, alpha=ALPHA, B=B_BOOT,
                       iv_scale=IV_REDUCING, regime="REDUCING",
                       recovery_mcc_threshold=RECOVERY_MCC,
                       c_bar="SHD over decided edges == 0 per seed AND >=1 edge decided present"),
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
        ds = SIM.simulate(D_LATENT, D_OBS, N_PER_ENV, specs, seed,
                          mixing="linear", edge_prob=EDGE_PROB)
        report["data_fingerprint"][str(seed)] = E0._fingerprint(ds.environments["obs"].X)

        mu, Wp = BK.fit_pca(ds.environments["basis"].X, D_PROJ)
        Y_obs = BK.project(ds.environments["obs"].X, mu, Wp)
        Y_int = [BK.project(ds.environments[f"iv{i}"].X, mu, Wp) for i in range(D_LATENT)]

        # (1) GATE
        gate = PR.count_recoverable(Y_int, Y_obs, alpha=ALPHA, B=B_BOOT,
                                    rng=np.random.default_rng(910_000 + seed))
        n_recoverable = gate["count"]

        # (2) RECOVER
        envs = {k: e.X for k, e in ds.environments.items()}
        targets = {f"iv{i}": i for i in range(D_LATENT)}
        out = BK.recover(envs, targets, D_LATENT, basis_key="basis", obs_key="obs")
        recover_mcc = E0.mcc(out["Z_hat"], ds.environments["obs"].Z)
        W = out["W"]

        # (3) DISCOVER on node-aligned recovered latents per environment.
        Z_obs = Y_obs @ W.T
        Z_int = {i: (Y_int[i] @ W.T) for i in range(D_LATENT)}
        dsc = DISC.discover(Z_obs, Z_int)

        # Evaluate (c) against the true SCM.
        te = _true_edges(B)
        decided_present = {tuple(map(int, e.split("->")))
                           for e, s in dsc["status"].items() if s == DISC.DECIDED_PRESENT}
        decided_absent = {tuple(map(int, e.split("->")))
                          for e, s in dsc["status"].items() if s == DISC.DECIDED_ABSENT}
        decided = decided_present | decided_absent
        shd_decided = sum(1 for e in decided if (e in decided_present) != (e in te))
        tp = len(decided_present & te)
        fp = len(decided_present - te)
        fn_decided = len({e for e in decided_absent if e in te})   # true edges wrongly absent
        dec_precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        dec_recall = (tp / (tp + fn_decided)) if (tp + fn_decided) > 0 else 1.0
        n_pairs = D_LATENT * (D_LATENT - 1)
        abstention_rate = dsc["n_undecided"] / n_pairs

        gate_ok = bool(n_recoverable == true_dim)
        recover_ok = bool(recover_mcc >= RECOVERY_MCC)
        c_ok = bool(shd_decided == 0 and dsc["n_decided_present"] >= 1)

        report["per_seed"].append(dict(
            seed=seed, true_dim=int(true_dim), gate_n_recoverable=int(n_recoverable),
            gate_ok=gate_ok, recover_mcc=round(float(recover_mcc), 4), recover_ok=recover_ok,
            shd_decided=int(shd_decided),
            decided_precision=round(float(dec_precision), 3),
            decided_recall=round(float(dec_recall), 3),
            n_true_edges=len(te), n_decided_present=dsc["n_decided_present"],
            n_decided_absent=dsc["n_decided_absent"], n_undecided=dsc["n_undecided"],
            abstention_rate=round(float(abstention_rate), 3),
            c_ok=c_ok,
            status_codes=dsc["status"],
            seed_pass=bool(gate_ok and recover_ok and c_ok)))

    all_pass = all(r["seed_pass"] for r in report["per_seed"])
    report["verdict"] = "PASS" if all_pass else "FAIL"
    report["status"] = report["verdict"]
    return report


def _print(report):
    print(f"\nE1-FULL PIPELINE  (status: {report['status']})", flush=True)
    print("=" * 84, flush=True)
    print(f"{'seed':>4} {'true dim':>8} {'gate n_rec':>10} {'MCC':>7} "
          f"{'SHD_dec':>8} {'dec prec':>9} {'dec rec':>8} {'abstain':>8} {'pass':>5}",
          flush=True)
    for r in report["per_seed"]:
        print(f"{r['seed']:>4} {r['true_dim']:>8} {r['gate_n_recoverable']:>10} "
              f"{r['recover_mcc']:>7.4f} {r['shd_decided']:>8} "
              f"{r['decided_precision']:>9.2f} {r['decided_recall']:>8.2f} "
              f"{r['abstention_rate']:>8.2f} {'yes' if r['seed_pass'] else 'NO':>5}",
              flush=True)
    print("-" * 84, flush=True)
    print(f"discovery: {report['discovery_method']}", flush=True)
    print(f"(c) bar: {report['config']['c_bar']}", flush=True)
    print(f"VERDICT: {report['verdict']}", flush=True)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = run()
    (RESULTS / "e1_full_report.json").write_text(json.dumps(report, indent=2))
    _print(report)
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
