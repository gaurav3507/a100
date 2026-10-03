"""Track A ablation study (tasks.md Task 3.4): full model, no-GNN
(concatenation instead), no-Transformer (last-window features only),
no-MGAF (simple average fusion), no-reconstruction-loss. Reports accuracy
delta per component vs. the full model, same structure as FatigueNet's
Table 5.

Uses a subset of LOSO folds by default (--folds) since 5 variants x 16
folds is expensive; pass --folds with no args for the full LOSO ablation
once compute allows it (tasks.md: full LOSO is required for final reported
numbers, but a subset is a reasonable dev-time check).

Usage:
  .venv/Scripts/python.exe track_a_heavy/ablation.py --folds 0 1 2 --epochs 15
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common.dataset import ULDDSequenceDataset
from common.fatiguenet_model import FatigueNetModel, fatiguenet_loss
from common.metrics import aggregate
from common.splits import load_folds
from track_a_heavy.model import MODALITY_DIMS, NUM_CLASSES
from track_a_heavy.train import collate, run_epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "track_a_ablation.json"

VARIANTS = {
    "full": dict(model_kwargs={}, lambda2=0.05),
    "no_gnn": dict(model_kwargs=dict(use_gnn=False), lambda2=0.05),
    "no_transformer": dict(model_kwargs=dict(use_transformer=False), lambda2=0.05),
    "no_mgaf": dict(model_kwargs=dict(use_mgaf=False), lambda2=0.05),
    "no_reconstruction_loss": dict(model_kwargs={}, lambda2=0.0),
}


def train_variant_fold(variant_name, held_out_subject, dataset, args):
    spec = VARIANTS[variant_name]
    subjects = dataset.df["subject"].to_numpy()
    # Fold-local impute/z-score statistics: training subjects only (leak fix).
    dataset.set_stats_subjects(sorted(set(subjects) - {held_out_subject}))
    train_idx = np.where(subjects != held_out_subject)[0]
    test_idx = np.where(subjects == held_out_subject)[0]

    torch.manual_seed(args.seed)
    model = FatigueNetModel(
        MODALITY_DIMS, num_classes=NUM_CLASSES, hidden_dim=args.hidden_dim,
        max_context_len=args.context_len, dropout=args.dropout, **spec["model_kwargs"],
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    for _ in range(args.epochs):
        run_epoch(model, train_loader, optimizer, lambda2=spec["lambda2"])
    _, test_metrics = run_epoch(model, test_loader, optimizer=None, lambda2=spec["lambda2"])
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--context-len", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"device: {DEVICE}")
    all_folds = load_folds()
    folds = [f for f in all_folds if f["fold"] in args.folds] if args.folds else all_folds
    # See track_a_heavy/train.py: always load ALL foldable subjects so
    # each fold's "train" set is every other subject, not just the other
    # requested folds' subjects.
    dataset = ULDDSequenceDataset(context_len=args.context_len, folds=[f["fold"] for f in all_folds])

    results = {}
    for variant_name in VARIANTS:
        print(f"\n=== variant: {variant_name} ===")
        fold_metrics = []
        for f in folds:
            t0 = time.time()
            metrics = train_variant_fold(variant_name, f["held_out_subject"], dataset, args)
            fold_metrics.append(metrics)
            print(f"  fold {f['fold']} (held out {f['held_out_subject']}): "
                  f"acc={metrics.accuracy:.4f} ({time.time()-t0:.1f}s)")
        results[variant_name] = aggregate(fold_metrics)

    full_acc = results["full"]["accuracy"]["mean"]
    print(f"\n{'Variant':24s} {'Accuracy':>10s} {'Delta (pp)':>12s}")
    print("-" * 48)
    for name, agg in results.items():
        acc = agg["accuracy"]["mean"]
        delta = (acc - full_acc) * 100
        print(f"{name:24s} {acc*100:9.2f}% {delta:+11.2f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
