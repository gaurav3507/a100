"""E3 Perturb-seq panel with historical-vs-corrected gate decisions.

Usage:
    python experiments/e3_perturbseq_panel.py NAME H5AD CONTROL SINGLE_GENE_ONLY [OUTPUT_STEM]

Use CONTROL=EMPTY for the empty-string control label and SINGLE_GENE_ONLY=1 for
the Norman single-gene screen. The output includes raw alpha-level and BH-FDR
decisions for both null geometries, plus per-perturbation audit records.
"""

import importlib.util
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np

from e3_gate_compare import (compare_geometries, comparison_summary,
                             decision_records, shift_alignment)


spec = importlib.util.spec_from_file_location("pr", "src/gate/precision_readout.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

D = 10
NMIN = 200
SEED = 0
ALPHA = 0.05
Q = 0.05
B_BOOT = 500


def main():
    name, path, ctrl, single_arg = sys.argv[1:5]
    output_stem = sys.argv[5] if len(sys.argv) > 5 else name
    single = single_arg == "1"
    if ctrl == "EMPTY":
        ctrl = ""

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
        dropped = sorted({label for label in np.unique(g)
                          if label != ctrl and label not in cand})
        print("KEPT examples:", cand[:6])
        print("DROPPED examples:", dropped[:6])
        print(f"single-gene labels {len(cand)} | dropped {len(dropped)}")
    else:
        cand = [label for label in np.unique(g) if label != ctrl]

    Xc = X[g == ctrl]
    mu = Xc.mean(0)
    _, _, Vt = np.linalg.svd(Xc - mu, full_matrices=False)
    Bp = Vt[:D].T
    proj = lambda M: (M - mu) @ Bp
    Yobs = proj(Xc)

    perts = [p for p in cand if (g == p).sum() >= NMIN]
    Yperts = [proj(X[g == p]) for p in perts]
    sizes = [int(len(Y)) for Y in Yperts]
    print(f"{name}: X {X.shape} | control {len(Xc)} | powered perts {len(perts)}")

    pert_cmp = compare_geometries(
        pr, Yperts, Yobs, seed=SEED, alpha=ALPHA, B=B_BOOT, q=Q)

    selection_rng = np.random.default_rng(SEED)
    nfake = 50
    Yfake = []
    for _ in range(nfake):
        n = int(selection_rng.choice(sizes))
        idx = selection_rng.choice(len(Xc), n, replace=False)
        Yfake.append(proj(Xc[idx]))
    fake_labels = [f"random_control_{i:02d}" for i in range(nfake)]
    fake_cmp = compare_geometries(
        pr, Yfake, Yobs, seed=10_000 + SEED, alpha=ALPHA, B=B_BOOT, q=Q)

    m = int(np.median(sizes))
    struct_labels, Ystruct = [], []
    for j in range(D):
        order = np.argsort(Yobs[:, j])
        struct_labels.extend([f"pc{j}_low", f"pc{j}_high"])
        Ystruct.extend([proj(Xc[order[:m]]), proj(Xc[order[-m:]])])
    struct_cmp = compare_geometries(
        pr, Ystruct, Yobs, seed=20_000 + SEED, alpha=ALPHA, B=B_BOOT, q=Q)

    align = {}
    for geometry, result in pert_cmp.items():
        align[f"{geometry}_raw"] = shift_alignment(
            Yperts, Yobs, result["raw_detect"])
        align[f"{geometry}_bh"] = shift_alignment(
            Yperts, Yobs, result["bh_detect"])

    corrected = pert_cmp["corrected_disjoint"]
    report = dict(
        dataset=name,
        result_stem=output_stem,
        n_cells=int(X.shape[0]),
        n_genes=int(X.shape[1]),
        control_label=repr(ctrl),
        n_control=int(len(Xc)),
        single_gene_only=single,
        d_proj=D,
        nmin=NMIN,
        n_powered_perts=len(perts),
        alpha=ALPHA,
        q=Q,
        B=B_BOOT,
        primary_decision="corrected_disjoint BH-FDR at q=0.05",
        bh_families=(
            "BH applied separately to powered perturbations, random controls, "
            "and structured controls"),
        n_detectable=int(corrected["bh_count"]),
        frac_detectable=round(corrected["bh_count"] / len(perts), 6),
        perturbation_screen=comparison_summary(pert_cmp),
        negative_control_screen=comparison_summary(fake_cmp),
        structured_control_screen=comparison_summary(struct_cmp),
        confound_alignment=align,
        per_perturbation=decision_records(perts, Yperts, pert_cmp),
        negative_control_records=decision_records(fake_labels, Yfake, fake_cmp),
        structured_control_records=decision_records(
            struct_labels, Ystruct, struct_cmp),
    )
    out = Path(f"results/e3/e3_{output_stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in (
        "dataset", "n_control", "n_powered_perts", "primary_decision",
        "n_detectable", "frac_detectable", "perturbation_screen",
        "negative_control_screen", "structured_control_screen",
        "confound_alignment")}, indent=2))


if __name__ == "__main__":
    main()
