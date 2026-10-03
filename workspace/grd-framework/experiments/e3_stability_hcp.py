"""Five-seed HCP task-fMRI stability for both gate null geometries."""

import glob
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from e3_gate_compare import (compare_geometries, comparison_summary,
                             decision_records, shift_alignment)
from data_paths import hcp_ts_root


spec = importlib.util.spec_from_file_location("pr", "src/gate/precision_readout.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

D = 10
TS = str(hcp_ts_root())
TASKS = ["WM", "GAMBLING", "MOTOR", "LANGUAGE", "SOCIAL", "RELATIONAL", "EMOTION"]
SEEDS = range(5)
ALPHA = 0.05
Q = 0.05
B_BOOT = 500


def conn(ts):
    C = np.corrcoef(ts, rowvar=False)
    iu = np.triu_indices(C.shape[0], k=1)
    return C[iu]


def build_groups():
    subs = sorted({os.path.basename(f).split("_")[0]
                   for f in glob.glob(f"{TS}/*.npy")})
    groups = {}
    for task in TASKS:
        values = []
        for subject in subs:
            arrays = [
                np.load(f"{TS}/{subject}_{task}_{encoding}.npy").astype(float)
                for encoding in ("LR", "RL")
                if os.path.exists(f"{TS}/{subject}_{task}_{encoding}.npy")
            ]
            if arrays:
                values.append(conn(np.concatenate(arrays, 0)))
        if values:
            groups[task] = np.asarray(values)
    return groups


def once(groups, baseline, seed):
    Yb = groups[baseline]
    mu = Yb.mean(0)
    _, _, Vt = np.linalg.svd(Yb - mu, full_matrices=False)
    Bp = Vt[:D].T
    proj = lambda M: (M - mu) @ Bp
    Yobs = proj(Yb)
    labels = [label for label in groups if label != baseline]
    Yenvs = [proj(groups[label]) for label in labels]

    env_cmp = compare_geometries(
        pr, Yenvs, Yobs, seed=seed, alpha=ALPHA, B=B_BOOT, q=Q)

    selection_rng = np.random.default_rng(seed)
    sizes = [len(Y) for Y in Yenvs]
    nfake = 50
    Yfake = []
    for _ in range(nfake):
        n = min(int(selection_rng.choice(sizes)), len(Yb) // 2)
        idx = selection_rng.choice(len(Yb), n, replace=False)
        Yfake.append(proj(Yb[idx]))
    fake_cmp = compare_geometries(
        pr, Yfake, Yobs, seed=10_000 + seed, alpha=ALPHA, B=B_BOOT, q=Q)

    align = {}
    for geometry, result in env_cmp.items():
        align[f"{geometry}_raw"] = shift_alignment(
            Yenvs, Yobs, result["raw_detect"])
        align[f"{geometry}_bh"] = shift_alignment(
            Yenvs, Yobs, result["bh_detect"])

    fake_labels = [f"random_control_{i:02d}" for i in range(nfake)]
    return dict(
        seed=seed,
        environment_screen=comparison_summary(env_cmp),
        negative_control_screen=comparison_summary(fake_cmp),
        confound_alignment=align,
        per_environment=decision_records(labels, Yenvs, env_cmp),
        negative_control_records=decision_records(fake_labels, Yfake, fake_cmp),
    )


def aggregate(rows, screen_name):
    report = {}
    for geometry in ("old_two_bootstrap", "corrected_disjoint"):
        for decision in ("raw", "bh"):
            values = np.asarray([
                row[screen_name][geometry][f"{decision}_fraction"] for row in rows
            ], dtype=float)
            prefix = f"{geometry}_{decision}_fraction"
            report[f"{prefix}_mean"] = round(float(values.mean()), 6)
            report[f"{prefix}_sd"] = round(float(values.std()), 6)
            report[f"{prefix}_per_seed"] = [round(float(v), 6) for v in values]
    return report


def main():
    groups = build_groups()
    baseline = max(groups, key=lambda label: len(groups[label]))
    rows = [once(groups, baseline, seed) for seed in SEEDS]
    report = dict(
        dataset="fMRI_HCP_task",
        baseline=baseline,
        n_environments=len(groups) - 1,
        seeds=list(SEEDS),
        alpha=ALPHA,
        q=Q,
        B=B_BOOT,
        primary_decision="corrected_disjoint BH-FDR at q=0.05",
        bh_families=(
            "Within each seed, BH applied separately to task environments "
            "and random controls"),
        environment_stability=aggregate(rows, "environment_screen"),
        negative_control_stability=aggregate(rows, "negative_control_screen"),
        per_seed=rows,
    )
    out = Path("results/e3_stability/HCP.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in (
        "dataset", "baseline", "n_environments", "seeds",
        "primary_decision", "environment_stability",
        "negative_control_stability")}, indent=2))


if __name__ == "__main__":
    main()
