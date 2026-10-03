"""Panel-wide real-data subspace attribution after the corrected BH gate.

Datasets: K562, RPE1, Norman, and HCP task fMRI. Upstream detectability is read
from the committed Task 1b result JSONs and uses the corrected-disjoint BH
decision. Only upstream-detected environments require a subspace calculation;
the others retain NO_DETECTABLE_SHIFT without loading their raw dataset.

The attribution statistic is the largest principal angle between top-d sample
subspaces in the FULL observed feature space. Its null geometry exactly matches
precision_readout.subspace_null:

* When n_env <= n_control // 2, a control permutation is split into an n_env
  pseudo-environment and its disjoint complement.
* Otherwise, two size-n_env bootstrap samples are drawn with replacement.

The implementation accelerates the same statistic without approximation. For
D <= N it updates centered feature-space scatter matrices from sufficient
statistics. For D > N it evaluates principal angles from centered Gram blocks,
avoiding repeated operations in the 19,900-dimensional HCP feature space.

Neutral verdict language is deliberate:

* NO_DETECTABLE_SHIFT: upstream corrected BH gate did not reject.
* SHARED_SUBSPACE_REJECTED: the upstream gate rejected and the secondary
  subspace test rejects after BH across the full dataset family. This is
  evidence against shared mixing within the linear top-d model and is
  consistent with measurement contamination; it does not identify its cause.
* SHARED_SUBSPACE_NOT_REJECTED: the upstream gate rejected but the secondary
  test did not. This is non-rejection, not confirmation of a mechanism shift.

Undetected environments receive attribution p=1 for the dataset-wide BH family.
The readout remains a secondary, post-gate diagnostic rather than a selective-
inference theorem. No dataset-specific outcome is assumed in advance.

Writes results/e3_attribution/panel_attribution.json.
"""

import hashlib
import importlib.util
import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from e3_fmri_connectivity import build_subject_vectors
from data_paths import perturbseq_path


# NumPy 2.0 on macOS Accelerate can emit spurious matmul RuntimeWarnings while
# returning finite products. Public entry points below reject non-finite inputs.
warnings.filterwarnings("ignore", message=r".*encountered in matmul",
                        category=RuntimeWarning)


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "e3_attribution"
ALPHA = 0.05
Q = 0.05
B_BOOT = 500
D_SUBSPACE = 10
SEED = 0


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PR = _load(REPO / "src" / "gate" / "precision_readout.py",
           "grd_gate_precision_readout")


DATASETS = [
    dict(
        name="K562",
        kind="perturbseq",
        data_path=str(perturbseq_path("causalbench_k562.h5ad")),
        gate_path=REPO / "results" / "e3" / "e3_K562_CRISPRi.json",
        control="",
    ),
    dict(
        name="RPE1",
        kind="perturbseq",
        data_path=str(perturbseq_path("causalbench_rpe1.h5ad")),
        gate_path=REPO / "results" / "e3" / "e3_rpe1_gate_fixed.json",
        control="",
    ),
    dict(
        name="Norman",
        kind="perturbseq",
        data_path=str(perturbseq_path("Norman2019_raw.h5ad")),
        gate_path=REPO / "results" / "e3" / "e3_Norman_CRISPRa_singlegene.json",
        control="",
    ),
    dict(
        name="HCP",
        kind="hcp",
        gate_path=REPO / "results" / "e3" / "e3_fMRI_HCP_task.json",
    ),
]


def _finite_array(X, name):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.all(np.isfinite(X)):
        raise ValueError(f"{name} must be a finite two-dimensional array")
    return X


def _top_eigenvectors(scatter, d):
    """Top-d eigenvectors of a symmetric scatter matrix, in ascending order."""
    D = scatter.shape[0]
    values, vectors = eigh(
        scatter, subset_by_index=[D - d, D - 1], driver="evr",
        check_finite=False)
    if values[0] <= 0:
        raise np.linalg.LinAlgError(
            f"top-{d} scatter subspace is not positive rank")
    return vectors


def _scatter_from_rows(X):
    X = np.asarray(X, dtype=float)
    total = X.sum(axis=0)
    return X.T @ X - np.outer(total, total) / len(X)


def _angle_from_bases(U, V):
    singular = np.linalg.svd(U.T @ V, compute_uv=False)
    return float(np.arccos(np.clip(singular.min(), -1.0, 1.0)))


def _primal_angle(X_left, X_right, d):
    left = _top_eigenvectors(_scatter_from_rows(X_left), d)
    right = _top_eigenvectors(_scatter_from_rows(X_right), d)
    return _angle_from_bases(left, right)


def _center_gram(K):
    return K - K.mean(axis=0, keepdims=True) - K.mean(
        axis=1, keepdims=True) + K.mean()


def _center_cross_gram(K):
    return K - K.mean(axis=0, keepdims=True) - K.mean(
        axis=1, keepdims=True) + K.mean()


def _gram_angle_blocks(K_left, K_right, K_cross, d):
    """Principal angle from within- and cross-sample raw Gram blocks."""
    G_left = _center_gram(np.asarray(K_left, dtype=float))
    G_right = _center_gram(np.asarray(K_right, dtype=float))
    G_cross = _center_cross_gram(np.asarray(K_cross, dtype=float))

    nl, nr = len(G_left), len(G_right)
    eval_left, U_left = eigh(
        G_left, subset_by_index=[nl - d, nl - 1], driver="evr",
        check_finite=False)
    eval_right, U_right = eigh(
        G_right, subset_by_index=[nr - d, nr - 1], driver="evr",
        check_finite=False)
    if eval_left[0] <= 0 or eval_right[0] <= 0:
        raise np.linalg.LinAlgError("Gram subspace is not positive rank")

    cross = U_left.T @ G_cross @ U_right
    cross /= np.sqrt(eval_left)[:, None]
    cross /= np.sqrt(eval_right)[None, :]
    singular = np.linalg.svd(cross, compute_uv=False)
    return float(np.arccos(np.clip(singular.min(), -1.0, 1.0)))


def _gram_angle(X_left, X_right, d):
    X_left = np.asarray(X_left, dtype=float)
    X_right = np.asarray(X_right, dtype=float)
    return _gram_angle_blocks(
        X_left @ X_left.T,
        X_right @ X_right.T,
        X_left @ X_right.T,
        d,
    )


def fast_subspace_angle(X_env, X_obs, d):
    """Exact principal-angle statistic with a dimension-adaptive backend."""
    X_env = _finite_array(X_env, "X_env")
    X_obs = _finite_array(X_obs, "X_obs")
    if X_obs.shape[1] <= max(len(X_env), len(X_obs)):
        return _primal_angle(X_env, X_obs, d)
    return _gram_angle(X_env, X_obs, d)


def _primal_null_values(X_obs, d, n_env, B, rng, disjoint):
    """Exact null draws using feature-scatter sufficient statistics."""
    X_obs = np.asarray(X_obs, dtype=float)
    n, D = X_obs.shape
    total_sum = X_obs.sum(axis=0)
    total_cross = X_obs.T @ X_obs
    values = np.empty(B)

    if disjoint:
        for b in range(B):
            idx = rng.permutation(n)[:n_env]
            sample = X_obs[idx]
            sample_sum = sample.sum(axis=0)
            sample_cross = sample.T @ sample
            sample_scatter = sample_cross - np.outer(
                sample_sum, sample_sum) / n_env
            n_ref = n - n_env
            ref_sum = total_sum - sample_sum
            ref_cross = total_cross - sample_cross
            ref_scatter = ref_cross - np.outer(ref_sum, ref_sum) / n_ref
            values[b] = _angle_from_bases(
                _top_eigenvectors(sample_scatter, d),
                _top_eigenvectors(ref_scatter, d))
    else:
        for b in range(B):
            left = X_obs[rng.integers(0, n, n_env)]
            right = X_obs[rng.integers(0, n, n_env)]
            values[b] = _primal_angle(left, right, d)
    return values


def _gram_null_values(X_obs, d, n_env, B, rng, disjoint):
    """Exact null draws using a precomputed raw control Gram matrix."""
    X_obs = np.asarray(X_obs, dtype=float)
    n = len(X_obs)
    raw_gram = X_obs @ X_obs.T
    values = np.empty(B)
    for b in range(B):
        if disjoint:
            idx = rng.permutation(n)
            left_idx, right_idx = idx[:n_env], idx[n_env:]
        else:
            left_idx = rng.integers(0, n, n_env)
            right_idx = rng.integers(0, n, n_env)
        values[b] = _gram_angle_blocks(
            raw_gram[np.ix_(left_idx, left_idx)],
            raw_gram[np.ix_(right_idx, right_idx)],
            raw_gram[np.ix_(left_idx, right_idx)],
            d,
        )
    return values


def fast_subspace_null_values(X_obs, d, n_env, B, rng):
    """Exact values from the same branch rule as PR.subspace_null."""
    X_obs = _finite_array(X_obs, "X_obs")
    n, D = X_obs.shape
    use_disjoint = bool(PR._use_disjoint_null(n, int(n_env), d, True))
    if D <= n:
        values = _primal_null_values(
            X_obs, d, int(n_env), B, rng, use_disjoint)
        backend = "feature_scatter"
    else:
        values = _gram_null_values(
            X_obs, d, int(n_env), B, rng, use_disjoint)
        backend = "sample_gram"
    return values, use_disjoint, backend


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return None


def _load_gate(config):
    report = json.loads(Path(config["gate_path"]).read_text())
    key = "per_perturbation" if config["kind"] == "perturbseq" else "per_environment"
    rows = report[key]
    return report, rows


def _load_perturbseq_selected(config, selected_labels):
    import anndata as ad

    A = ad.read_h5ad(config["data_path"])
    labels = A.obs["guide_ids"].astype(str).values
    keep = (labels == config["control"]) | np.isin(labels, selected_labels)
    matrix = A.X[keep]
    X = (matrix.toarray().astype(np.float64) if hasattr(matrix, "toarray")
         else np.asarray(matrix, dtype=np.float64))
    kept_labels = labels[keep]
    del A
    X_obs = X[kept_labels == config["control"]]
    environments = {
        str(label): X[kept_labels == label] for label in selected_labels
    }
    return X_obs, environments, dict(
        n_control=int(len(X_obs)), n_features=int(X.shape[1]))


def _load_hcp_selected(selected_labels):
    groups, baseline, _, n_features_raw, n_features_used = (
        build_subject_vectors("hcp"))
    X_obs = groups[baseline]
    environments = {str(label): groups[label] for label in selected_labels}
    return X_obs, environments, dict(
        baseline=str(baseline), n_control=int(len(X_obs)),
        n_features_raw=int(n_features_raw), n_features=int(n_features_used))


def run_dataset(config):
    gate_report, gate_rows = _load_gate(config)
    selected_rows = [
        row for row in gate_rows
        if bool(row["corrected_disjoint"]["bh_detect"])
    ]
    selected_labels = [str(row["environment"]) for row in selected_rows]
    print(f"[{config['name']}] upstream BH selected "
          f"{len(selected_rows)}/{len(gate_rows)}", flush=True)

    output_rows = []
    selected_set = set(selected_labels)
    for gate_row in gate_rows:
        label = str(gate_row["environment"])
        output_rows.append(dict(
            environment=label,
            n_samples=int(gate_row["n_samples"]),
            upstream_corrected_bh=label in selected_set,
            upstream_pvalue=float(
                gate_row["corrected_disjoint"]["pvalue"]),
            subspace_angle_deg=None,
            subspace_threshold_deg=None,
            subspace_pvalue=1.0,
            subspace_raw_reject=False,
            subspace_bh_reject=False,
            subspace_disjoint_applied=None,
            subspace_backend=None,
            verdict="NO_DETECTABLE_SHIFT",
        ))

    metadata = dict(raw_data_loaded=False, n_unique_null_sizes=0,
                    null_size_cache=[])
    if selected_rows:
        if config["kind"] == "perturbseq":
            X_obs, environments, loaded = _load_perturbseq_selected(
                config, selected_labels)
        else:
            X_obs, environments, loaded = _load_hcp_selected(selected_labels)
        metadata.update(loaded)
        metadata["raw_data_loaded"] = True
        cache = {}
        by_label = {row["environment"]: row for row in output_rows}
        pvalues_all = np.ones(len(output_rows), dtype=float)

        for gate_row in selected_rows:
            label = str(gate_row["environment"])
            X_env = environments[label]
            n_env = len(X_env)
            if n_env not in cache:
                null, applied, backend = fast_subspace_null_values(
                    X_obs, D_SUBSPACE, n_env, B_BOOT,
                    np.random.default_rng(
                        np.random.SeedSequence([SEED, int(n_env)])))
                cache[n_env] = (null, applied, backend)
                print(f"[{config['name']}] null n={n_env} "
                      f"backend={backend} disjoint={applied}", flush=True)
            null, applied, backend = cache[n_env]
            angle = fast_subspace_angle(X_env, X_obs, D_SUBSPACE)
            threshold = float(np.quantile(null, 1.0 - ALPHA))
            pvalue = float(
                (1 + np.count_nonzero(null >= angle)) / (B_BOOT + 1))
            row = by_label[label]
            row.update(
                subspace_angle_deg=round(float(np.degrees(angle)), 6),
                subspace_threshold_deg=round(
                    float(np.degrees(threshold)), 6),
                subspace_pvalue=pvalue,
                subspace_raw_reject=bool(angle > threshold),
                subspace_disjoint_applied=bool(applied),
                subspace_backend=backend,
                verdict="SHARED_SUBSPACE_NOT_REJECTED",
            )

        metadata["n_unique_null_sizes"] = len(cache)
        metadata["null_size_cache"] = [
            dict(n_env=int(n_env), disjoint_applied=bool(values[1]),
                 backend=values[2])
            for n_env, values in sorted(cache.items())
        ]

        for i, row in enumerate(output_rows):
            pvalues_all[i] = float(row["subspace_pvalue"])
        bh = PR.bh_fdr(pvalues_all, q=Q)
        for row, reject in zip(output_rows, bh):
            if not row["upstream_corrected_bh"]:
                continue
            row["subspace_bh_reject"] = bool(reject)
            row["verdict"] = (
                "SHARED_SUBSPACE_REJECTED" if reject
                else "SHARED_SUBSPACE_NOT_REJECTED")

    summary = dict(
        n_environments=len(output_rows),
        n_upstream_bh_detected=sum(
            row["upstream_corrected_bh"] for row in output_rows),
        n_subspace_raw_rejected=sum(
            row["subspace_raw_reject"] for row in output_rows),
        n_subspace_bh_rejected=sum(
            row["subspace_bh_reject"] for row in output_rows),
        n_shared_subspace_not_rejected=sum(
            row["verdict"] == "SHARED_SUBSPACE_NOT_REJECTED"
            for row in output_rows),
        verdict_counts=dict(
            NO_DETECTABLE_SHIFT=sum(
                row["verdict"] == "NO_DETECTABLE_SHIFT"
                for row in output_rows),
            SHARED_SUBSPACE_REJECTED=sum(
                row["verdict"] == "SHARED_SUBSPACE_REJECTED"
                for row in output_rows),
            SHARED_SUBSPACE_NOT_REJECTED=sum(
                row["verdict"] == "SHARED_SUBSPACE_NOT_REJECTED"
                for row in output_rows),
        ),
    )
    return dict(
        dataset=config["name"],
        gate_result=str(Path(config["gate_path"]).relative_to(REPO)),
        gate_result_sha256=_sha256(config["gate_path"]),
        gate_primary_decision=gate_report["primary_decision"],
        metadata=metadata,
        summary=summary,
        environments=output_rows,
    )


def main():
    started = time.time()
    reports = [run_dataset(config) for config in DATASETS]
    report = dict(
        experiment="e3_panel_attribution",
        code_commit=_code_commit(),
        datasets=[config["name"] for config in DATASETS],
        d_subspace=D_SUBSPACE,
        alpha=ALPHA,
        q=Q,
        B=B_BOOT,
        min_attainable_pvalue=round(1.0 / (B_BOOT + 1), 9),
        upstream_decision="corrected-disjoint BH-FDR from Task 1b",
        attribution_bh_family=(
            "all environments within each dataset; upstream-undetected "
            "environments assigned subspace p=1"),
        null_geometry=(
            "exact disjoint pseudo-environment/complement when feasible; "
            "historical two-bootstrap fallback otherwise"),
        verdict_semantics=dict(
            NO_DETECTABLE_SHIFT="upstream corrected BH gate did not reject",
            SHARED_SUBSPACE_REJECTED=(
                "secondary BH test rejects shared top-d subspace; evidence "
                "against shared mixing, not identification of a cause"),
            SHARED_SUBSPACE_NOT_REJECTED=(
                "secondary test does not reject shared top-d subspace; "
                "non-rejection, not mechanism confirmation"),
        ),
        limitation=(
            "Secondary post-gate diagnostic; no selective-inference guarantee. "
            "The linear top-d subspace test cannot detect scalar gains or nonlinear "
            "measurement changes."),
        reports=reports,
        wall_seconds=round(time.time() - started, 3),
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "panel_attribution.json"
    out.write_text(json.dumps(report, indent=2))

    print("\nDATASET  UPSTREAM  SUBSPACE-RAW  SUBSPACE-BH  NOT-REJECTED", flush=True)
    for dataset in reports:
        summary = dataset["summary"]
        print(f"{dataset['dataset']:8s} "
              f"{summary['n_upstream_bh_detected']:8d} "
              f"{summary['n_subspace_raw_rejected']:13d} "
              f"{summary['n_subspace_bh_rejected']:12d} "
              f"{summary['n_shared_subspace_not_rejected']:12d}", flush=True)
    print(f"written {out} ({report['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
