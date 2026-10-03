"""Generate and persist the v2 three-tier fold files (plan §3), so every
model in every tier consumes byte-identical splits:

  tier1_folds.json  window-level stratified 5-fold (the UL-DD paper's own
                    protocol; all 19 subjects pooled, C/F/L included)
  tier2_folds.json  contiguous 4-minute-block folds, boundary-straddling
                    windows purged to fold=-1 (excluded from train AND test)
  folds_v2.json     LOSO with awake-only subjects (C/F/L) in every training
                    pool (16 eval folds, 18-subject train pools)

Run AFTER scripts/build_feature_table.py (fold files are keyed to the
table's (subject, session, window_id) rows and validated against it at
merge time -- a stale fold file fails loudly, never trains silently).

Usage: python scripts/build_v2_folds.py [--seed 0]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from common.dataset import DEFAULT_FEATURES_PATH
from common.splits import (
    LOSO_V2_FOLDS_PATH, TIER1_FOLDS_PATH, TIER2_FOLDS_PATH,
    block_folds, loso_folds, save_window_folds, stratified_window_folds,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_parquet(DEFAULT_FEATURES_PATH)
    print(f"Feature table: {len(df)} windows, {df['subject'].nunique()} subjects")

    tier1 = stratified_window_folds(df, n_splits=args.n_splits, seed=args.seed)
    save_window_folds(tier1, TIER1_FOLDS_PATH)
    print(f"Tier 1: {TIER1_FOLDS_PATH.name} -- "
          f"{tier1['fold'].value_counts().sort_index().to_dict()} windows/fold")

    tier2 = block_folds(df, n_splits=args.n_splits)
    n_purged = int((tier2["fold"] == -1).sum())
    save_window_folds(tier2, TIER2_FOLDS_PATH)
    print(f"Tier 2: {TIER2_FOLDS_PATH.name} -- "
          f"{n_purged} boundary windows purged ({n_purged / len(tier2):.1%})")

    loso_v2 = loso_folds(include_awake_only_in_train=True)
    LOSO_V2_FOLDS_PATH.write_text(json.dumps(loso_v2, indent=2))
    sizes = {f["fold"]: len(f["train_subjects"]) for f in loso_v2}
    print(f"Tier 3: {LOSO_V2_FOLDS_PATH.name} -- {len(loso_v2)} folds, "
          f"train-pool sizes {sorted(set(sizes.values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
