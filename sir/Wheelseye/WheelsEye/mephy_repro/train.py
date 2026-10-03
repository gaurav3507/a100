"""Train/evaluate the shared FatigueNet architecture on MePhy, as a
reimplementation sanity check before adapting it to UL-DD for Track A
(tasks.md 1.2/3.1).

Important methodology note, found while reading the FatigueNet paper: its
own reported 90.2% test accuracy comes from a **random 80/20 window-level
split**, not a subject-independent one -- the paper states "the dataset
was randomly partitioned into training and testing sets." tasks.md Section
0 explicitly forbids random splits for this project's own pipeline because
they leak within-subject correlation across train/test and inflate
accuracy. So this script runs BOTH protocols:

  --split random  replicates the paper's own (leaky) methodology, as the
                  actual "does our reimplementation hit ~90.2%" check
  --split loso    subject-independent leave-one-user-out, the rigorous
                  number, expected to be lower than 90.2% precisely
                  because it removes the leakage the paper's protocol
                  allowed. A lower LOSO number is not a bug -- it's the
                  point of comparing the two.

Usage:
  .venv/Scripts/python.exe mephy_repro/train.py --split random
  .venv/Scripts/python.exe mephy_repro/train.py --split loso
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

from common.fatiguenet_model import FatigueNetModel, fatiguenet_loss
from common.metrics import aggregate, compute_metrics
from mephy_repro.dataset import MODALITY_DIMS, MePhySequenceDataset, collate
from mephy_repro.paths import CONDITION_LABEL, FULL_COVERAGE_USERS, LABEL_NAMES

MEPHY_LABELS = tuple(sorted(CONDITION_LABEL.values()))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "mephy_repro_results"


def build_model(context_len: int) -> FatigueNetModel:
    return FatigueNetModel(
        MODALITY_DIMS, num_classes=len(LABEL_NAMES), hidden_dim=256,
        gnn_layers=4, gnn_hops=3, gnn_heads=4,
        transformer_layers=4, transformer_heads=4, transformer_head_dim=64,
        dropout=0.4, max_context_len=context_len,
    ).to(DEVICE)


def run_epoch(model, loader, optimizer=None, lambda1=0.01, lambda2=0.05):
    training = optimizer is not None
    model.train() if training else model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            nodes = {m: v.to(DEVICE) for m, v in batch["nodes"].items()}
            labels = batch["label"].to(DEVICE)
            if training:
                optimizer.zero_grad()
            out = model(nodes)
            loss, _ = fatiguenet_loss(
                out["logits"], labels, out["recon_loss"], model.classifier.weight, lambda1, lambda2
            )
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            all_preds.append(out["logits"].argmax(dim=-1).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    metrics = compute_metrics(y_true, y_pred, labels=MEPHY_LABELS, class_names=LABEL_NAMES)
    return total_loss / max(n_batches, 1), metrics


def train_one_split(train_idx, test_idx, full_dataset, context_len, epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    model = build_model(context_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    train_loader = DataLoader(
        Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx), batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    for epoch in range(epochs):
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer)
        if epoch == epochs - 1 or epoch % 5 == 0:
            print(f"    epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  train_acc={train_metrics.accuracy:.4f}")

    test_loss, test_metrics = run_epoch(model, test_loader, optimizer=None)
    return train_metrics, test_metrics


def run_random_split(context_len=12, epochs=25, batch_size=64, lr=1e-4, seed=0, test_frac=0.2):
    print("=== Random 80/20 window-level split (replicates the paper's own protocol) ===")
    ds = MePhySequenceDataset(context_len=context_len)
    n = len(ds)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    print(f"  n_train={len(train_idx)} n_test={len(test_idx)}")

    train_metrics, test_metrics = train_one_split(
        train_idx, test_idx, ds, context_len, epochs, batch_size, lr, seed
    )
    print(f"  TRAIN: acc={train_metrics.accuracy:.4f} f1_macro={train_metrics.f1_macro:.4f}")
    print(f"  TEST:  acc={test_metrics.accuracy:.4f} f1_macro={test_metrics.f1_macro:.4f}")
    print(f"  (paper reports 95.0% train / 90.2% test under this same random-split protocol)")
    return {"train": train_metrics.to_dict(), "test": test_metrics.to_dict()}


def run_loso(context_len=12, epochs=15, batch_size=32, lr=1e-4, seed=0):
    print("=== Leave-one-user-out (subject-independent, rigorous number) ===")
    ds = MePhySequenceDataset(context_len=context_len)
    users = ds.df["user"].to_numpy()
    fold_metrics = []

    for i, held_out in enumerate(FULL_COVERAGE_USERS):
        train_idx = np.where(users != held_out)[0]
        test_idx = np.where(users == held_out)[0]
        t0 = time.time()
        _, test_metrics = train_one_split(train_idx, test_idx, ds, context_len, epochs, batch_size, lr, seed)
        fold_metrics.append(test_metrics)
        print(f"  fold {i+1}/{len(FULL_COVERAGE_USERS)} held_out={held_out}: "
              f"acc={test_metrics.accuracy:.4f} f1_macro={test_metrics.f1_macro:.4f} ({time.time()-t0:.1f}s)")

    agg = aggregate(fold_metrics)
    print(f"\n  LOSO aggregate: acc={agg['accuracy']['mean']:.4f}+/-{agg['accuracy']['std']:.4f} "
          f"f1_macro={agg['f1_macro']['mean']:.4f}+/-{agg['f1_macro']['std']:.4f}")
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["random", "loso", "both"], default="both")
    parser.add_argument("--context-len", type=int, default=12)
    parser.add_argument("--epochs-random", type=int, default=25)
    parser.add_argument("--epochs-loso", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"device: {DEVICE}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    if args.split in ("random", "both"):
        results["random_split"] = run_random_split(
            context_len=args.context_len, epochs=args.epochs_random, seed=args.seed
        )
        (RESULTS_DIR / "random_split.json").write_text(json.dumps(results["random_split"], indent=2))

    if args.split in ("loso", "both"):
        results["loso"] = run_loso(context_len=args.context_len, epochs=args.epochs_loso, seed=args.seed)
        (RESULTS_DIR / "loso.json").write_text(json.dumps(results["loso"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
