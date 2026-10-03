from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .gcarl_adapter import generate_sim1, load_sim1_config
from .graph_metrics import directed_metrics, orientation_metrics, skeleton_from_directed, skeleton_metrics
from .graph_truth import inter_group_mask, intra_group_mask, remap_truth_for_permutation
from .method_adapters import fit_method, method_type, versions
from .method_adapters.discovery import load_method_config
from .provenance import canonical_array_fingerprint, gcarl_commit, git_commit, utc_timestamp, write_json_atomic
from .representation_metrics import rank_aware_mcc, standard_mcc
from .transforms import apply_transform

ROOT = Path(__file__).resolve().parents[2]


def _load_or_generate(seed: int) -> dict:
    path = ROOT / "results" / "data" / f"sim1_seed_{seed}.npz"
    if path.exists():
        raw = np.load(path, allow_pickle=False)
        cfg = load_sim1_config()
        return {**{key: raw[key] for key in raw.files}, "config": cfg}
    data = generate_sim1(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, z=data["z"], lam1=data["lam1"], lam2=data["lam2"], lamin1=data["lamin1"], lamin2=data["lamin2"], inter_truth=data["inter_truth"], intra_truth=data["intra_truth"])
    return data


def _score(method: str, fit: dict, inter_truth: np.ndarray, intra_truth: np.ndarray, groups: int, dim: int) -> dict:
    inter_mask = inter_group_mask(groups, dim)
    intra_mask = intra_group_mask(groups, dim)
    if method_type(method) == "pc":
        primary = skeleton_metrics(inter_truth, fit["skeleton"], inter_mask)
        orient = orientation_metrics(inter_truth, fit["directed"], fit["skeleton"], inter_mask)
        intra = skeleton_metrics(intra_truth, fit["skeleton"], intra_mask)
        return {"primary_metric_type": "skeleton_f1", "primary": primary, "orientation": orient, "intra": intra}
    primary = directed_metrics(inter_truth, fit["directed"], inter_mask)
    intra = directed_metrics(intra_truth, fit["directed"], intra_mask)
    return {"primary_metric_type": "directed_f1", "primary": primary, "orientation": None, "intra": intra}


def run_oracle_cell(method: str, scm_seed: int, method_seed: int) -> dict:
    data = _load_or_generate(scm_seed)
    fit = fit_method(method, data["z"], method_seed)
    cfg = data["config"]
    scores = _score(method, fit, data["inter_truth"], data["intra_truth"], cfg["num_group"], cfg["num_dim"])
    return {
        "method": method,
        "method_type": method_type(method),
        "scm_seed": scm_seed,
        "method_seed": method_seed,
        "primary_metric_type": scores["primary_metric_type"],
        "inter_group": scores["primary"],
        "orientation": scores["orientation"],
        "intra_group": scores["intra"],
        "intra_group_status": "EXPLORATORY_ONLY",
        "method_versions": versions(),
        "generator_source_path": "vendor/GCaRL/subfunc/generate_dataset.py",
        "gcarl_submodule_commit": gcarl_commit(),
        "project_commit": git_commit(),
        "resolved_config": {"sim1": cfg, "method_procedure": load_method_config()},
        "timestamp": utc_timestamp(),
        "data_fingerprint": canonical_array_fingerprint(data["z"], cfg),
    }


def run_manifest_cell(row: dict, oracle_summary: dict) -> dict:
    scm_seed = int(row["scm_seed"])
    transform_seed = int(row["transformation_seed"])
    method_seed = int(row["method_seed"])
    method = row["method"]
    family = row["transform_family"]
    level_raw = row["level"]
    level = "seeded" if level_raw == "seeded" else float(level_raw)
    data = _load_or_generate(scm_seed)
    transformed = apply_transform(data["z"], family, level, transform_seed)
    inter_truth = data["inter_truth"]
    intra_truth = data["intra_truth"]
    if transformed.permutations is not None:
        inter_truth = remap_truth_for_permutation(inter_truth, transformed.permutations, data["config"]["num_dim"])
        intra_truth = remap_truth_for_permutation(intra_truth, transformed.permutations, data["config"]["num_dim"])
    fit = fit_method(method, transformed.values, method_seed)
    cfg = data["config"]
    scores = _score(method, fit, inter_truth, intra_truth, cfg["num_group"], cfg["num_dim"])
    oracle = oracle_summary["methods"][method]
    oracle_path = ROOT / "results" / "oracle" / f"seed_{scm_seed}_{method.lower().replace(' ', '-')}.json"
    if not oracle_path.exists():
        raise RuntimeError(f"Missing matching-seed oracle result: {oracle_path}")
    oracle_cell = json.loads(oracle_path.read_text(encoding="utf-8"))
    ceiling = float(oracle_cell["inter_group"]["f1"])
    primary_f1 = float(scores["primary"]["f1"])
    degradation = (ceiling - primary_f1) / ceiling if ceiling else None
    return {
        "cell_id": row["cell_id"],
        "method": method,
        "method_implementation_version": versions(),
        "method_type": method_type(method),
        "scm_seed": scm_seed,
        "transformation_seed": transform_seed,
        "method_seed": method_seed,
        "transform_family": family,
        "transform_class": row["transform_class"],
        "transform_parameters": transformed.metadata,
        "exact_transform_metadata": transformed.metadata,
        "standard_mcc": standard_mcc(data["z"], transformed.values),
        "rank_aware_mcc": rank_aware_mcc(data["z"], transformed.values),
        "primary_inter_group_metric_type": scores["primary_metric_type"],
        "inter_group_precision": scores["primary"]["precision"],
        "inter_group_recall": scores["primary"]["recall"],
        "inter_group_f1": primary_f1,
        "inter_group_shd": scores["primary"]["shd"],
        "pc_skeleton_metrics": scores["primary"] if method_type(method) == "pc" else None,
        "pc_orientation_metrics": scores["orientation"],
        "pc_orientation_load_bearing": bool(oracle["orientation_eligible"]) if method_type(method) == "pc" else None,
        "intra_group_metrics": scores["intra"],
        "intra_group_status": "EXPLORATORY_ONLY",
        "oracle_ceiling": ceiling,
        "normalized_degradation": degradation,
        "inapplicable_metrics_reason": None if method_type(method) == "pc" else "PC skeleton and orientation metrics are not applicable to a directed DAG estimator",
        "generator_source_path": "vendor/GCaRL/subfunc/generate_dataset.py",
        "gcarl_submodule_commit": gcarl_commit(),
        "project_commit": git_commit(),
        "resolved_config": {"sim1": cfg, "method_procedure": load_method_config()},
        "timestamp": utc_timestamp(),
        "data_fingerprint": canonical_array_fingerprint(data["z"], cfg),
        "transformed_fingerprint": canonical_array_fingerprint(transformed.values, cfg),
    }


def save_result(row: dict, payload: dict) -> None:
    write_json_atomic(ROOT / row["expected_output_path"], payload)
