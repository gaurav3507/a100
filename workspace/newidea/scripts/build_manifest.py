#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crlcd_audit.manifest import build_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "manifest.csv")
    args = parser.parse_args()
    oracle = json.loads((ROOT / "results" / "oracle_summary.json").read_text(encoding="utf-8"))
    eligible = [name for name, item in oracle["methods"].items() if item["eligible"]]
    rows = build_manifest(eligible, args.output, args.seeds)
    print(f"Wrote {len(rows)} manifest cells for {len(eligible)} eligible methods")


if __name__ == "__main__":
    main()

