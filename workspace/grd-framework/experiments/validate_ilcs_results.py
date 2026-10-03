"""Validate the complete three-seed iLCS result set before committing it."""

import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results" / "ilcs_baseline"
EXPECTED_SEEDS = [0, 1, 2]
DATASETS = ["K562", "RPE1", "Norman"]
RATE_KEYS = [
    "naive_fake_rate",
    "calib_fake_rate",
    "naive_random_rate",
    "calib_random_rate",
    "naive_struct_rate",
    "calib_struct_rate",
    "naive_pert_rate",
    "calib_pert_rate",
    "jaccard_calib_vs_gate",
]


def load(path):
    if not path.is_file():
        raise AssertionError(f"missing {path}")
    return json.loads(path.read_text())


def finite_rate(value, label):
    if value is None or not math.isfinite(float(value)):
        raise AssertionError(f"{label} is missing or non-finite: {value}")
    if not 0.0 <= float(value) <= 1.0:
        raise AssertionError(f"{label} is outside [0,1]: {value}")


def validate_seed(seed):
    directory = ROOT / f"seed{seed}"
    summary = load(directory / "summary.json")
    if summary["seed"] != seed:
        raise AssertionError(f"seed label mismatch in {directory}")
    if set(summary["datasets_complete"]) != set(DATASETS):
        raise AssertionError(
            f"seed {seed} incomplete: {summary['datasets_complete']}")
    if summary["datasets_failed"] or summary["errors"]:
        raise AssertionError(f"seed {seed} has errors")
    for name in DATASETS:
        per_dataset = load(directory / f"{name}.json")
        if per_dataset["dataset"] != name:
            raise AssertionError(f"dataset mismatch in seed {seed} {name}")
        records = per_dataset.get("records", [])
        if not records or not any(row["kind"] == "pert" for row in records):
            raise AssertionError(f"seed {seed} {name} has no perturbation records")
        dataset_summary = summary["per_dataset"][name]
        for key in RATE_KEYS:
            finite_rate(dataset_summary.get(key), f"seed {seed} {name} {key}")
    return summary


def validate_aggregate():
    report = load(ROOT / "aggregate.json")
    if report["seeds"] != EXPECTED_SEEDS:
        raise AssertionError(f"aggregate seeds are {report['seeds']}")
    for name in DATASETS:
        dataset = report["per_dataset"].get(name)
        if dataset is None or dataset.get("n_seeds") != len(EXPECTED_SEEDS):
            raise AssertionError(f"aggregate {name} does not contain three seeds")
        for key in RATE_KEYS:
            values = dataset.get(key)
            if values is None or len(values) != 3 or values[2] != 3:
                raise AssertionError(f"aggregate {name} {key} is incomplete: {values}")
            finite_rate(values[0], f"aggregate {name} {key} mean")
            if not math.isfinite(float(values[1])) or float(values[1]) < 0.0:
                raise AssertionError(f"aggregate {name} {key} SD invalid: {values[1]}")
    return report


def main():
    for seed in EXPECTED_SEEDS:
        validate_seed(seed)
    aggregate = validate_aggregate()
    print("DATASET  NAIVE-FAKE  CALIB-FAKE  NAIVE-STRUCT  CALIB-STRUCT  JACCARD")
    for name in DATASETS:
        row = aggregate["per_dataset"][name]
        print(
            f"{name:7s} "
            f"{row['naive_fake_rate'][0]:10.4f} "
            f"{row['calib_fake_rate'][0]:10.4f} "
            f"{row['naive_struct_rate'][0]:12.4f} "
            f"{row['calib_struct_rate'][0]:12.4f} "
            f"{row['jaccard_calib_vs_gate'][0]:8.4f}"
        )
    print("VALIDATION PASS: seeds 0,1,2; all datasets complete; Jaccard populated")


if __name__ == "__main__":
    main()
