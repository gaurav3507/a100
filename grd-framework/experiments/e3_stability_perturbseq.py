"""Five-seed Perturb-seq stability for both gate null geometries."""

import importlib.util
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np

from e3_gate_compare import (compare_geometries, comparison_summary,
                             decision_records, shift_alignment)
from data_paths import perturbseq_path


spec = importlib.util.spec_from_file_location("pr", "src/gate/precision_readout.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

D = 10
NMIN = 200
SEEDS = range(5)
ALPHA = 0.05
Q = 0.05
B_BOOT = 500


def screen_once(X, g, ctrl, cand, seed):
    Xc = X[g == ctrl]
    mu = Xc.mean(0)
    _, _, Vt = np.linalg.svd(Xc - mu, full_matrices=False)
    Bp = Vt[:D].T
    proj = lambda M: (M - mu) @ Bp
    Yobs = proj(Xc)
    perts = [p for p in cand if (g == p).sum() >= NMIN]
    Yperts = [proj(X[g == p]) for p in perts]
    sizes = [int(len(Y)) for Y in Yperts]

    pert_cmp = compare_geometries(
        pr, Yperts, Yobs, seed=seed, alpha=ALPHA, B=B_BOOT, q=Q)

    selection_rng = np.random.default_rng(seed)
    nfake = 50
    Yfake = []
    for _ in range(nfake):
        n = int(selection_rng.choice(sizes))
        idx = selection_rng.choice(len(Xc), n, replace=False)
        Yfake.append(proj(Xc[idx]))
    fake_cmp = compare_geometries(
        pr, Yfake, Yobs, seed=10_000 + seed, alpha=ALPHA, B=B_BOOT, q=Q)

    align = {}
    for geometry, result in pert_cmp.items():
        align[f"{geometry}_raw"] = shift_alignment(
            Yperts, Yobs, result["raw_detect"])
        align[f"{geometry}_bh"] = shift_alignment(
            Yperts, Yobs, result["bh_detect"])

    fake_labels = [f"random_control_{i:02d}" for i in range(nfake)]
    return dict(
        seed=seed,
        n_perts=len(perts),
        perturbation_screen=comparison_summary(pert_cmp),
        negative_control_screen=comparison_summary(fake_cmp),
        confound_alignment=align,
        per_perturbation=decision_records(perts, Yperts, pert_cmp),
        negative_control_records=decision_records(fake_labels, Yfake, fake_cmp),
    )


def _aggregate(rows, screen_name):
    aggregate = {}
    for geometry in ("old_two_bootstrap", "corrected_disjoint"):
        for decision in ("raw", "bh"):
            values = np.asarray([
                row[screen_name][geometry][f"{decision}_fraction"] for row in rows
            ], dtype=float)
            prefix = f"{geometry}_{decision}_fraction"
            aggregate[f"{prefix}_mean"] = round(float(values.mean()), 6)
            aggregate[f"{prefix}_sd"] = round(float(values.std()), 6)
            aggregate[f"{prefix}_per_seed"] = [round(float(v), 6) for v in values]
    return aggregate


def run(name, path, ctrl, single):
    A = ad.read_h5ad(path)
    X = (A.X.toarray().astype(np.float64) if hasattr(A.X, "toarray")
         else np.asarray(A.X, np.float64))
    g = A.obs["guide_ids"].astype(str).values
    del A
    if single:
        def is_single(label):
            return (label != ctrl and label != "" and "," not in label
                    and "+" not in label and "_" not in label)

        cand = sorted({label for label in np.unique(g) if is_single(label)})
    else:
        cand = [label for label in np.unique(g) if label != ctrl]

    rows = [screen_once(X, g, ctrl, cand, seed) for seed in SEEDS]
    report = dict(
        dataset=name,
        n_perts=rows[0]["n_perts"],
        seeds=list(SEEDS),
        alpha=ALPHA,
        q=Q,
        B=B_BOOT,
        primary_decision="corrected_disjoint BH-FDR at q=0.05",
        bh_families=(
            "Within each seed, BH applied separately to powered perturbations "
            "and random controls"),
        perturbation_stability=_aggregate(rows, "perturbation_screen"),
        negative_control_stability=_aggregate(rows, "negative_control_screen"),
        per_seed=rows,
    )
    out = Path(f"results/e3_stability/{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in (
        "dataset", "n_perts", "seeds", "primary_decision",
        "perturbation_stability", "negative_control_stability")}, indent=2))
    return report


def main():
    kind = sys.argv[1]
    if kind == "k562":
        run("K562", str(perturbseq_path("causalbench_k562.h5ad")), "", False)
    elif kind == "rpe1":
        run("RPE1", str(perturbseq_path("causalbench_rpe1.h5ad")), "", False)
    elif kind == "norman":
        run("Norman", str(perturbseq_path("Norman2019_raw.h5ad")), "", True)
    else:
        raise SystemExit(f"unknown dataset: {kind}")


if __name__ == "__main__":
    main()
