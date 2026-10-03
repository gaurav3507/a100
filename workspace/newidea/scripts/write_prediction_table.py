#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from crlcd_audit.method_adapters import included_method_names, method_type

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS = {
    "PC-Pearson": {
        "A1": ("NOMINALLY_INVARIANT", "Variable relabeling preserves Fisher-z PC output up to the same relabeling", "Permutation equivariance"),
        "B": ("NOMINALLY_INVARIANT", "Pearson correlations and frozen per-cell standardization remove positive component scaling", "Correlation scale invariance"),
        "C": ("NOMINALLY_NON_INVARIANT", "Pearson partial correlations need not survive nonlinear marginal warps", "Gaussian partial-correlation test restriction"),
        "D": ("NOMINALLY_NON_INVARIANT", "Within-group coordinate mixing changes the node-level CI graph", "Node semantics and conditional independence"),
        "E": ("NOMINALLY_NON_INVARIANT", "Measurement error can alter conditional independences", "Causal sufficiency and measurement assumptions"),
    },
    "PC-nonparanormal-preproc": {
        "A1": ("NOMINALLY_INVARIANT", "Variable relabeling preserves the CPDAG up to relabeling", "Permutation equivariance"),
        "B": ("NOMINALLY_INVARIANT", "Positive scaling preserves ranks and the normal-score data exactly", "Rank invariance"),
        "C": ("NOMINALLY_INVARIANT", "Strictly increasing component warps preserve ranks and the normal-score data exactly", "Nonparanormal marginal invariance"),
        "D": ("NOMINALLY_NON_INVARIANT", "Coordinate mixing is not a marginal monotone transform and can change CI structure", "Nonparanormal model restriction"),
        "E": ("NOMINALLY_NON_INVARIANT", "Independent measurement error changes ranks and can alter CI structure", "Measurement assumptions"),
    },
    "DirectLiNGAM": {
        "A1": ("NOMINALLY_INVARIANT", "Node relabeling preserves the fitted graph after truth remapping", "Permutation equivariance"),
        "B": ("NOMINALLY_INVARIANT", "Frozen per-cell standardization removes positive component scaling", "Scale normalization and linear SEM"),
        "C": ("NOMINALLY_NON_INVARIANT", "Nonlinear marginal warps generally destroy the linear additive SEM form", "Linearity and non-Gaussian independent errors"),
        "D": ("NOMINALLY_NON_INVARIANT", "Coordinate mixing changes the node-level linear SEM and graph", "Linear SEM node semantics"),
        "E": ("NOMINALLY_NON_INVARIANT", "Measurement noise violates the error and measurement model", "Independent non-Gaussian error assumptions"),
    },
    "NOTEARS-linear": {
        "A1": ("NOMINALLY_INVARIANT", "The score and acyclicity constraint are permutation equivariant", "Permutation equivariance"),
        "B": ("NOMINALLY_INVARIANT", "Frozen per-cell standardization removes positive component scaling", "Standardized linear least-squares score"),
        "C": ("NOMINALLY_NON_INVARIANT", "Nonlinear marginal warps leave the linear SEM score class", "Linear functional-form restriction"),
        "D": ("NOMINALLY_NON_INVARIANT", "Coordinate mixing changes node semantics and the linear graph", "Linear SEM node semantics"),
        "E": ("NOMINALLY_NON_INVARIANT", "Measurement error changes the least-squares causal model", "Measurement assumptions"),
    },
    "NOTEARS-MLP": {
        "A1": ("NOMINALLY_INVARIANT", "The nonlinear score and acyclicity constraint are permutation equivariant", "Permutation equivariance"),
        "B": ("NOMINALLY_INVARIANT", "Frozen per-cell standardization removes positive component scaling", "Scale normalization"),
        "C": ("THEORETICALLY_UNCLEAR", "Model flexibility helps but marginal reparameterization can change additive noise and optimization behavior", "Nonlinear additive model and score restrictions"),
        "D": ("NOMINALLY_NON_INVARIANT", "Coordinate mixing changes the node-level graph target", "Node semantics"),
        "E": ("NOMINALLY_NON_INVARIANT", "Measurement error is outside the fitted nonlinear SEM", "Measurement assumptions"),
    },
    "GOLEM": {
        "A1": ("NOMINALLY_INVARIANT", "The likelihood score is permutation equivariant", "Permutation equivariance"),
        "B": ("NOMINALLY_INVARIANT", "Frozen per-cell standardization removes positive component scaling", "Scale normalization"),
        "C": ("NOMINALLY_NON_INVARIANT", "Nonlinear marginal warps violate the linear Gaussian score model", "Linear Gaussian likelihood restriction"),
        "D": ("NOMINALLY_NON_INVARIANT", "Coordinate mixing changes the node-level linear graph", "Linear SEM node semantics"),
        "E": ("NOMINALLY_NON_INVARIANT", "Measurement error is outside the likelihood model", "Measurement assumptions"),
    },
}


def main() -> None:
    if any((ROOT / "results" / "cells").glob("*.json")):
        raise SystemExit("Refusing to preregister after transformed-grid results exist")
    oracle_path = ROOT / "results" / "oracle_summary.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    expected = set(included_method_names())
    if set(oracle["methods"]) != expected:
        raise SystemExit("Oracle summary does not contain the complete smoke-passed method set")
    if oracle.get("scm_seeds") != [0, 1, 2, 3, 4]:
        raise SystemExit("Oracle summary does not contain the five preregistered SCM seeds")
    for method in expected:
        if len(oracle["methods"][method]["individual_primary_f1"]) != 5:
            raise SystemExit(f"Oracle method {method} does not contain five seed scores")
        for seed in oracle["scm_seeds"]:
            path = ROOT / "results" / "oracle" / f"seed_{seed}_{method.lower().replace(' ', '-')}.json"
            if not path.exists() or path.stat().st_size == 0:
                raise SystemExit(f"Missing oracle artifact: {path}")
    lines = [
        "# Preregistered estimator predictions",
        "",
        "Generated after the complete oracle gate and before any transformed-grid result. Predictions are encoded in the committed generator script and were not derived from transformed results.",
        "",
        "| method | transformation | prediction | short reason | theoretical assumption | load-bearing metric |",
        "|---|---|---|---|---|---|",
    ]
    for method in included_method_names():
        oracle_item = oracle["methods"][method]
        primary = "skeleton F1" if method_type(method) == "pc" else "directed F1"
        if not oracle_item["eligible"]:
            primary += " (oracle-ineligible, descriptive only)"
        for family in ["A1", "B", "C", "D", "E"]:
            prediction, reason, assumption = PREDICTIONS[method][family]
            lines.append(f"| {method} | {family} | {prediction} | {reason} | {assumption} | {primary} |")
    lines.extend(["", "PC orientation is load-bearing only for a PC method whose separately preregistered mean oracle orientation F1 is at least 0.80. `THEORETICALLY_UNCLEAR` cells are exploratory and cannot alone establish an observed-vs-nominal contradiction.", ""])
    (ROOT / "prediction_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote prediction_table.md with no transformed-grid results present")


if __name__ == "__main__":
    main()
