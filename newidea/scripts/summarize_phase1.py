#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from crlcd_audit.manifest import read_manifest, validate_results
from crlcd_audit.provenance import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest_path = ROOT / "results" / "manifest.csv"
    validate_results(manifest_path)
    groups = defaultdict(list)
    for row in read_manifest(manifest_path):
        payload = json.loads((ROOT / row["expected_output_path"]).read_text(encoding="utf-8"))
        key = (payload["transform_class"], payload["transform_family"], str(row["level"]), payload["method"], payload["primary_inter_group_metric_type"])
        groups[key].append(payload)
    summaries = []
    for key, cells in sorted(groups.items()):
        f1 = [float(cell["inter_group_f1"]) for cell in cells]
        degradation = [float(cell["normalized_degradation"]) for cell in cells]
        standard = [float(cell["standard_mcc"]) for cell in cells]
        rank = [float(cell["rank_aware_mcc"]) for cell in cells]
        summaries.append({
            "transform_class": key[0],
            "transform_family": key[1],
            "level": key[2],
            "method": key[3],
            "primary_metric_type": key[4],
            "n_seeds": len(cells),
            "mean_f1": float(np.mean(f1)),
            "std_f1": float(np.std(f1)),
            "individual_f1": f1,
            "mean_normalized_degradation": float(np.mean(degradation)),
            "std_normalized_degradation": float(np.std(degradation)),
            "individual_normalized_degradation": degradation,
            "mean_standard_mcc": float(np.mean(standard)),
            "mean_rank_aware_mcc": float(np.mean(rank)),
            "cell_ids": [cell["cell_id"] for cell in cells],
        })
    write_json_atomic(ROOT / "results" / "phase1_summary.json", summaries)
    csv_rows = []
    for item in summaries:
        csv_rows.append({**item, "individual_f1": json.dumps(item["individual_f1"]), "individual_normalized_degradation": json.dumps(item["individual_normalized_degradation"]), "cell_ids": json.dumps(item["cell_ids"])})
    with open(ROOT / "results" / "phase1_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote {len(summaries)} aggregate rows with individual seed values")


if __name__ == "__main__":
    main()
