from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _levels(transform_cfg: dict) -> list[tuple[str, str, list]]:
    rows = []
    for key in ["permutation", "scaling", "cubic", "entanglement", "noise"]:
        spec = transform_cfg[key]
        rows.append((spec["family"], spec["class"], spec["levels"]))
    return rows


def build_manifest(methods: list[str], output_path: Path, seeds: list[int] | None = None) -> list[dict]:
    sim = yaml.safe_load((ROOT / "config" / "sim1.yaml").read_text(encoding="utf-8"))
    transforms = yaml.safe_load((ROOT / "config" / "transforms.yaml").read_text(encoding="utf-8"))
    method_config = yaml.safe_load((ROOT / "config" / "methods.yaml").read_text(encoding="utf-8"))
    global_methods = list(method_config["methods"])
    unknown_methods = sorted(set(methods) - set(global_methods))
    if unknown_methods:
        raise RuntimeError(f"Unknown manifest methods: {unknown_methods}")
    seeds = seeds or sim["scm_seeds"]
    rows = []
    for scm_seed in seeds:
        for family, transform_class, levels in _levels(transforms):
            for level_index, level in enumerate(levels):
                transform_seed = 100000 + 1000 * scm_seed + level_index
                for method in methods:
                    method_index = global_methods.index(method)
                    method_seed = 200000 + 1000 * scm_seed + method_index
                    level_token = str(level).replace(".", "p")
                    cell_id = f"s{scm_seed}-{family}-{level_token}-{method.lower().replace(' ', '-')}"
                    rows.append({
                        "cell_id": cell_id,
                        "scm_seed": scm_seed,
                        "transform_family": family,
                        "transform_class": transform_class,
                        "level": level,
                        "transformation_seed": transform_seed,
                        "method": method,
                        "method_seed": method_seed,
                        "expected_output_path": f"results/cells/{cell_id}.json",
                    })
    ids = [row["cell_id"] for row in rows]
    if not rows:
        raise RuntimeError("No oracle-eligible methods remain; transformed audit is globally blocked")
    if len(ids) != len(set(ids)):
        raise RuntimeError("Manifest contains duplicate cell IDs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_results(manifest_path: Path, allow_incomplete: bool = False) -> dict:
    rows = read_manifest(manifest_path)
    manifest_ids = [row["cell_id"] for row in rows]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise RuntimeError("Duplicate manifest cell IDs")
    completed = []
    unknown = []
    required = {"cell_id", "method", "standard_mcc", "rank_aware_mcc", "inter_group_f1", "normalized_degradation"}
    for path in (ROOT / "results" / "cells").glob("*.json"):
        if path.stat().st_size == 0:
            raise RuntimeError(f"Empty result file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing = required - payload.keys()
        if missing:
            raise RuntimeError(f"Missing metrics in {path}: {sorted(missing)}")
        if payload["cell_id"] not in manifest_ids:
            unknown.append(payload["cell_id"])
        completed.append(payload["cell_id"])
    if len(completed) != len(set(completed)):
        raise RuntimeError("Duplicate completed cell IDs")
    if unknown:
        raise RuntimeError(f"Unknown completed cell IDs: {unknown}")
    missing_ids = sorted(set(manifest_ids) - set(completed))
    if missing_ids and not allow_incomplete:
        raise RuntimeError(f"Missing {len(missing_ids)} result cells")
    if not allow_incomplete and set(completed) != set(manifest_ids):
        raise RuntimeError("Completed and manifest cell ID sets differ")
    return {"manifest": len(manifest_ids), "completed": len(completed), "missing": len(missing_ids)}
