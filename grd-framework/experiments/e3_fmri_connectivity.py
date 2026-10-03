"""E3 fMRI connectivity panel with historical-vs-corrected gate decisions."""

import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

from e3_gate_compare import (compare_geometries, comparison_summary,
                             decision_records, shift_alignment)
from data_paths import abide_npz_path, hcp_ts_root


spec = importlib.util.spec_from_file_location("pr", "src/gate/precision_readout.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

SEED = 0
ALPHA = 0.05
Q = 0.05
B_BOOT = 500


def subject_connectivity(ts):
    C = np.corrcoef(ts, rowvar=False)
    iu = np.triu_indices(C.shape[0], k=1)
    return C[iu]


def build_subject_vectors(mode):
    """Return group connectivity matrices, baseline label, and group type."""
    if mode == "abide":
        z = np.load(abide_npz_path(), allow_pickle=True)
        X = z["X"].astype(float)
        group_ids = z["site_ids"]
        features = {}
        for i in range(len(X)):
            features.setdefault(str(group_ids[i]), []).append(
                subject_connectivity(X[i]))
        groups = {label: np.asarray(values) for label, values in features.items()}
        baseline = max(groups, key=lambda label: len(groups[label]))
        group_type = "site (measurement shift)"
    elif mode == "hcp":
        ts_root = str(hcp_ts_root())
        tasks = ["WM", "GAMBLING", "MOTOR", "LANGUAGE", "SOCIAL",
                 "RELATIONAL", "EMOTION"]
        subjects = sorted({os.path.basename(f).split("_")[0]
                           for f in glob.glob(f"{ts_root}/*.npy")})
        groups = {}
        for task in tasks:
            values = []
            for subject in subjects:
                arrays = [
                    np.load(f"{ts_root}/{subject}_{task}_{encoding}.npy").astype(float)
                    for encoding in ("LR", "RL")
                    if os.path.exists(f"{ts_root}/{subject}_{task}_{encoding}.npy")
                ]
                if arrays:
                    values.append(subject_connectivity(np.concatenate(arrays, 0)))
            if values:
                groups[task] = np.asarray(values)
        baseline = max(groups, key=lambda label: len(groups[label]))
        group_type = "task (mechanism shift)"
    else:
        raise SystemExit(f"unknown mode: {mode}")

    n_features_raw = next(iter(groups.values())).shape[1]
    finite = np.ones(n_features_raw, dtype=bool)
    for values in groups.values():
        finite &= np.all(np.isfinite(values), axis=0)
    groups = {label: values[:, finite] for label, values in groups.items()}
    return groups, baseline, group_type, n_features_raw, int(finite.sum())


def main():
    mode = sys.argv[1]
    groups, baseline, group_type, n_features_raw, n_features_used = (
        build_subject_vectors(mode))
    sizes = {label: len(values) for label, values in groups.items()}
    print(f"{mode}: {len(groups)} groups, baseline={baseline}")
    print("subjects per group:", sizes)

    D = min(10, min(sizes.values()) - 2)
    if D < 3:
        raise SystemExit(
            f"[stop] smallest group {min(sizes.values())} too small for a gate")
    print(f"projection D={D} (min group size {min(sizes.values())})")

    Yb = groups[baseline]
    mu = Yb.mean(0)
    _, _, Vt = np.linalg.svd(Yb - mu, full_matrices=False)
    Bp = Vt[:D].T
    proj = lambda M: (M - mu) @ Bp
    Yobs = proj(Yb)
    labels = [label for label in groups if label != baseline]
    Yenvs = [proj(groups[label]) for label in labels]
    env_sizes = [len(Y) for Y in Yenvs]

    env_cmp = compare_geometries(
        pr, Yenvs, Yobs, seed=SEED, alpha=ALPHA, B=B_BOOT, q=Q)

    selection_rng = np.random.default_rng(SEED)
    nfake = 50
    Yfake = []
    for _ in range(nfake):
        n = min(int(selection_rng.choice(env_sizes)), len(Yb) // 2)
        idx = selection_rng.choice(len(Yb), n, replace=False)
        Yfake.append(proj(Yb[idx]))
    fake_labels = [f"random_control_{i:02d}" for i in range(nfake)]
    fake_cmp = compare_geometries(
        pr, Yfake, Yobs, seed=10_000 + SEED, alpha=ALPHA, B=B_BOOT, q=Q)

    m = min(int(np.median(env_sizes)), len(Yb) // 2)
    struct_labels, Ystruct = [], []
    for j in range(D):
        order = np.argsort(Yobs[:, j])
        struct_labels.extend([f"pc{j}_low", f"pc{j}_high"])
        Ystruct.extend([proj(Yb[order[:m]]), proj(Yb[order[-m:]])])
    struct_cmp = compare_geometries(
        pr, Ystruct, Yobs, seed=20_000 + SEED, alpha=ALPHA, B=B_BOOT, q=Q)

    align = {}
    for geometry, result in env_cmp.items():
        align[f"{geometry}_raw"] = shift_alignment(
            Yenvs, Yobs, result["raw_detect"])
        align[f"{geometry}_bh"] = shift_alignment(
            Yenvs, Yobs, result["bh_detect"])

    corrected = env_cmp["corrected_disjoint"]
    dataset = "fMRI_ABIDE_site" if mode == "abide" else "fMRI_HCP_task"
    report = dict(
        dataset=dataset,
        grouping=group_type,
        feature="per-subject functional connectivity (upper-triangle correlation)",
        sample_unit="subject",
        d_proj=D,
        baseline=str(baseline),
        n_features_raw=n_features_raw,
        n_features_used=n_features_used,
        n_environments=len(Yenvs),
        environment_subjects={str(label): int(len(Y))
                              for label, Y in zip(labels, Yenvs)},
        alpha=ALPHA,
        q=Q,
        B=B_BOOT,
        primary_decision="corrected_disjoint BH-FDR at q=0.05",
        bh_families=(
            "BH applied separately to observed environments, random controls, "
            "and structured controls"),
        n_detectable=int(corrected["bh_count"]),
        frac_detectable=round(corrected["bh_count"] / len(Yenvs), 6),
        environment_screen=comparison_summary(env_cmp),
        negative_control_screen=comparison_summary(fake_cmp),
        structured_control_screen=comparison_summary(struct_cmp),
        confound_alignment=align,
        per_environment=decision_records(labels, Yenvs, env_cmp),
        negative_control_records=decision_records(fake_labels, Yfake, fake_cmp),
        structured_control_records=decision_records(
            struct_labels, Ystruct, struct_cmp),
    )
    out = Path(f"results/e3/e3_{dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in (
        "dataset", "baseline", "n_environments", "primary_decision",
        "n_detectable", "frac_detectable", "environment_screen",
        "negative_control_screen", "structured_control_screen",
        "confound_alignment")}, indent=2))


if __name__ == "__main__":
    main()
