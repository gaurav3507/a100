"""RPE1 control-heterogeneity diagnostics under both gate null geometries.

This reports association, not causal attribution: random and structured control
splits test whether the gate reacts to control heterogeneity, while alignment
measures whether detected perturbation mean shifts point along the leading control
PCs. Raw and BH-FDR decisions are retained for both null constructions.
"""

import importlib.util
import json
from pathlib import Path

import anndata as ad
import numpy as np

from e3_gate_compare import (compare_geometries, comparison_summary,
                             decision_records, shift_alignment)
from data_paths import perturbseq_path


spec = importlib.util.spec_from_file_location("pr", "src/gate/precision_readout.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

H5 = str(perturbseq_path("causalbench_rpe1.h5ad"))
CTRL = ""
NMIN = 200
D = 10
SEED = 0
ALPHA = 0.05
Q = 0.05
B_BOOT = 500


def main():
    A = ad.read_h5ad(H5)
    X = (A.X.toarray().astype(np.float64) if hasattr(A.X, "toarray")
         else np.asarray(A.X, np.float64))
    g = A.obs["guide_ids"].astype(str).values
    del A

    Xc = X[g == CTRL]
    mu = Xc.mean(0)
    _, _, Vt = np.linalg.svd(Xc - mu, full_matrices=False)
    Bp = Vt[:D].T
    proj = lambda M: (M - mu) @ Bp
    Yobs = proj(Xc)

    perts = [p for p in np.unique(g)
             if p != CTRL and (g == p).sum() >= NMIN]
    Yperts = [proj(X[g == p]) for p in perts]
    sizes = [len(Y) for Y in Yperts]
    pert_cmp = compare_geometries(
        pr, Yperts, Yobs, seed=SEED, alpha=ALPHA, B=B_BOOT, q=Q)

    m = int(np.median(sizes))
    struct_labels, Ystruct = [], []
    for j in range(D):
        order = np.argsort(Yobs[:, j])
        struct_labels.extend([f"pc{j}_low", f"pc{j}_high"])
        Ystruct.extend([proj(Xc[order[:m]]), proj(Xc[order[-m:]])])
    struct_cmp = compare_geometries(
        pr, Ystruct, Yobs, seed=20_000 + SEED, alpha=ALPHA, B=B_BOOT, q=Q)

    selection_rng = np.random.default_rng(SEED)
    nrandom = 20
    Yrandom = []
    for _ in range(nrandom):
        n = int(selection_rng.choice(sizes))
        idx = selection_rng.choice(len(Xc), n, replace=False)
        Yrandom.append(proj(Xc[idx]))
    random_labels = [f"random_control_{i:02d}" for i in range(nrandom)]
    random_cmp = compare_geometries(
        pr, Yrandom, Yobs, seed=10_000 + SEED, alpha=ALPHA, B=B_BOOT, q=Q)

    align = {}
    for geometry, result in pert_cmp.items():
        align[f"{geometry}_raw"] = shift_alignment(
            Yperts, Yobs, result["raw_detect"])
        align[f"{geometry}_bh"] = shift_alignment(
            Yperts, Yobs, result["bh_detect"])

    corrected = pert_cmp["corrected_disjoint"]
    report = dict(
        dataset="causalbench_rpe1",
        n_control=int(len(Xc)),
        n_powered_perts=len(perts),
        d_proj=D,
        nmin=NMIN,
        alpha=ALPHA,
        q=Q,
        B=B_BOOT,
        primary_decision="corrected_disjoint BH-FDR at q=0.05",
        bh_families=(
            "BH applied separately to powered perturbations, random controls, "
            "and structured controls"),
        n_detected_perts=int(corrected["bh_count"]),
        perturbation_screen=comparison_summary(pert_cmp),
        random_control_screen=comparison_summary(random_cmp),
        structured_control_screen=comparison_summary(struct_cmp),
        perturbation_shift_alignment=align,
        interpretation=(
            "Structured-control firing and leading-PC alignment diagnose association "
            "with control heterogeneity; they do not prove that heterogeneity caused "
            "the perturbation detections."),
        per_perturbation=decision_records(perts, Yperts, pert_cmp),
        random_control_records=decision_records(
            random_labels, Yrandom, random_cmp),
        structured_control_records=decision_records(
            struct_labels, Ystruct, struct_cmp),
    )
    out = Path("results/e3/e3_rpe1_confound_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in (
        "dataset", "n_control", "n_powered_perts", "primary_decision",
        "n_detected_perts", "perturbation_screen", "random_control_screen",
        "structured_control_screen", "perturbation_shift_alignment",
        "interpretation")}, indent=2))


if __name__ == "__main__":
    main()
