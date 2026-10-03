#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from crlcd_audit.manifest import validate_results
from crlcd_audit.provenance import gcarl_commit, git_commit
from crlcd_audit.runner import run_manifest_cell, save_result

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "manifest.csv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with open(args.manifest, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[:args.limit]
    oracle = json.loads((ROOT / "results" / "oracle_summary.json").read_text(encoding="utf-8"))
    producing_commit = git_commit()
    producing_gcarl = gcarl_commit()
    for index, row in enumerate(rows, 1):
        path = ROOT / row["expected_output_path"]
        if path.exists() and not args.overwrite:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("project_commit") != producing_commit:
                raise RuntimeError(f"Stale transformed artifact from another project commit: {path}")
            if existing.get("gcarl_submodule_commit") != producing_gcarl:
                raise RuntimeError(f"Stale transformed artifact from another G-CaRL commit: {path}")
            if existing.get("cell_id") != row["cell_id"]:
                raise RuntimeError(f"Transformed artifact identity mismatch: {path}")
            continue
        save_result(row, run_manifest_cell(row, oracle))
        print(f"[{index}/{len(rows)}] {row['cell_id']}", flush=True)
    status = validate_results(args.manifest, allow_incomplete=args.limit is not None)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
