"""GATE 1 --- a pure MEASUREMENT shift must load diagonal-only.

Simulate an environment pair whose only real change is per-channel (diagonal
autocorrelation + noise scale), fit VAR(1) in each, and check that AXIS flags the
change as diagonal and NOT off-diagonal. If this gate fails, the test cannot be
trusted on real data.

Run on the A100 (or anywhere with numpy/scipy). Imports only from ideas/axis/.

    python experiments/01_oracle_measurement.py
"""

import os
import sys
import json
import hashlib
import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IDEA_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(IDEA_ROOT, "src"))

import yaml  # noqa: E402
import var_fit  # noqa: E402
import dissociation  # noqa: E402
import oracle_sim  # noqa: E402

MODE = "measurement"
# Allow up to this many rejections in the group that should be UNCHANGED
# (multiple-testing false positives). Expected null rejections are well below 1.
FP_BUDGET = 2


def array_hash(a):
    """Deterministic fingerprint of an array's exact contents."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def load_config():
    path = os.path.join(IDEA_ROOT, "configs", "default.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg, path


def run_one(seed, magnitude, cfg):
    """One environment-pair comparison. Returns a result record."""
    sim = oracle_sim.simulate(
        MODE, seed=seed, d=cfg["d"], T=cfg["T"], n_env=cfg["n_env"],
        magnitude=magnitude,
    )
    x0, x1 = sim["series"][0], sim["series"][1]
    n_boot = cfg["n_boot"]
    fit0 = var_fit.fit_var1(x0, bootstrap=True, n_boot=n_boot, seed=seed)
    fit1 = var_fit.fit_var1(x1, bootstrap=True, n_boot=n_boot, seed=seed + 10_000)

    ratio, summary = dissociation.dissociate(
        fit0["A"], fit0["A_se"], fit0["n"],
        fit1["A"], fit1["A_se"], fit1["n"],
        threshold=cfg["threshold"],
    )

    # GATE 1: the diagonal change must be detected, off-diagonal must stay clean.
    passed = (summary["diag_reject"] > 0) and (summary["offdiag_reject"] <= FP_BUDGET)

    return {
        "seed": int(seed),
        "magnitude": float(magnitude),
        "n": int(fit0["n"]),
        "env0_hash": array_hash(x0),
        "env1_hash": array_hash(x1),
        "summary": summary,
        "offdiag_share": ratio,
        "pass": bool(passed),
    }


def main():
    cfg, cfg_path = load_config()
    records = []
    print(f"GATE 1 (measurement -> diagonal-only), threshold={cfg['threshold']}, "
          f"FP_BUDGET={FP_BUDGET}")
    print(f"{'mag':>5} {'seed':>5} {'diag':>5} {'offdiag':>8} {'share':>7} {'pass':>5}")
    for magnitude in cfg["magnitudes"]:
        for seed in cfg["seeds"]:
            r = run_one(seed, magnitude, cfg)
            records.append(r)
            share = r["offdiag_share"]
            share_s = "None" if share is None else f"{share:.3f}"
            print(f"{magnitude:>5} {seed:>5} {r['summary']['diag_reject']:>5} "
                  f"{r['summary']['offdiag_reject']:>8} {share_s:>7} "
                  f"{'PASS' if r['pass'] else 'FAIL':>5}")

    n_pass = sum(r["pass"] for r in records)
    n_total = len(records)
    overall = n_pass == n_total
    print(f"\n{'='*48}")
    print(f"GATE 1 overall: {'PASS' if overall else 'FAIL'}  "
          f"({n_pass}/{n_total} comparisons passed)")

    results_dir = os.path.join(IDEA_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out = {
        "gate": "01_oracle_measurement",
        "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "overall_pass": bool(overall),
        "n_pass": n_pass,
        "n_total": n_total,
        "fp_budget": FP_BUDGET,
        "config": cfg,
        "config_path": cfg_path,
        "records": records,
    }
    out_path = os.path.join(results_dir, "01_oracle_measurement.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
