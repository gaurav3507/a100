#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from crlcd_audit.method_adapters import included_method_names
from crlcd_audit.provenance import gcarl_commit, git_commit, write_json_atomic
from crlcd_audit.runner import run_oracle_cell

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", default=included_method_names())
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()
    sim = yaml.safe_load((ROOT / "config" / "sim1.yaml").read_text(encoding="utf-8"))
    method_cfg = yaml.safe_load((ROOT / "config" / "methods.yaml").read_text(encoding="utf-8"))
    seeds = args.seeds or sim["scm_seeds"]
    producing_commit = git_commit()
    producing_gcarl = gcarl_commit()
    cells = []
    global_methods = included_method_names()
    for method in args.methods:
        method_index = global_methods.index(method)
        for seed in seeds:
            path = ROOT / "results" / "oracle" / f"seed_{seed}_{method.lower().replace(' ', '-')}.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                expected_seed = 200000 + 1000 * seed + method_index
                if payload.get("project_commit") != producing_commit:
                    raise RuntimeError(f"Stale oracle artifact from another project commit: {path}")
                if payload.get("gcarl_submodule_commit") != producing_gcarl:
                    raise RuntimeError(f"Stale oracle artifact from another G-CaRL commit: {path}")
                if payload.get("method") != method or payload.get("scm_seed") != seed or payload.get("method_seed") != expected_seed:
                    raise RuntimeError(f"Oracle artifact identity mismatch: {path}")
            else:
                payload = run_oracle_cell(method, seed, 200000 + 1000 * seed + method_index)
                write_json_atomic(path, payload)
            cells.append(payload)
    methods = {}
    for method in args.methods:
        selected = [cell for cell in cells if cell["method"] == method]
        values = [cell["inter_group"]["f1"] for cell in selected]
        orientations = [cell["orientation"]["f1"] for cell in selected if cell["orientation"] is not None]
        kind = selected[0]["method_type"]
        threshold = method_cfg["oracle_thresholds"]["skeleton_f1" if kind == "pc" else "directed_f1"]
        methods[method] = {
            "method_type": kind,
            "primary_metric": selected[0]["primary_metric_type"],
            "individual_primary_f1": values,
            "mean_primary_f1": float(np.mean(values)),
            "std_primary_f1": float(np.std(values)),
            "orientation_individual_f1": orientations or None,
            "mean_orientation_f1": float(np.mean(orientations)) if orientations else None,
            "threshold": threshold,
            "eligible": bool(np.mean(values) >= threshold),
            "orientation_eligible": bool(orientations and np.mean(orientations) >= method_cfg["oracle_thresholds"]["orientation_f1"]),
        }
    write_json_atomic(ROOT / "results" / "oracle_summary.json", {"aggregation_rule": "eligibility uses mean F1 across all five preregistered SCM seeds", "scm_seeds": seeds, "methods": methods})
    print(json.dumps(methods, indent=2))


if __name__ == "__main__":
    main()
