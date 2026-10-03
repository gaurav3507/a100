"""Track A training (tasks.md Task 3.2): train per LOSO fold using
common/splits.py's fold assignments (identical to Track B's), log
per-fold and averaged accuracy/F1/confusion matrix, save the best
checkpoint per fold.

Runs entirely off-device (workstation/Colab GPU) per tasks.md Section 0 --
Track A never touches the Jetson.

Usage:
  .venv/Scripts/python.exe track_a_heavy/train.py --folds 0 1 2 --epochs 15
  .venv/Scripts/python.exe track_a_heavy/train.py --epochs 15   # all 16 folds
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
from common.fatiguenet_model import fatiguenet_loss
from common.metrics import Metrics, aggregate, compute_metrics
from common.splits import load_folds
from track_a_heavy.model import build_track_a_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "track_a_checkpoints"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "track_a_results.json"


def collate(batch: list[dict]) -> dict:
    modalities = batch[0]["nodes"].keys()
    nodes = {m: torch.stack([b["nodes"][m] for b in batch]) for m in modalities}
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"nodes": nodes, "label": labels}


def run_epoch(model, loader, optimizer=None, lambda1=0.01, lambda2=0.05):
    training = optimizer is not None
    model.train() if training else model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

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

    y_pred, y_true = np.concatenate(all_preds), np.concatenate(all_labels)
    return total_loss / max(n_batches, 1), compute_metrics(y_true, y_pred)


def train_fold(fold_id: int, held_out_subject: str, dataset, args) -> Metrics:
    subjects = dataset.df["subject"].to_numpy()
    train_idx = np.where(subjects != held_out_subject)[0]
    test_idx = np.where(subjects == held_out_subject)[0]

    torch.manual_seed(args.seed)
    model = build_track_a_model(
        hidden_dim=args.hidden_dim, context_len=args.context_len, dropout=args.dropout
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    best_test_acc, best_state = -1.0, None
    for epoch in range(args.epochs):
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer)
        _, test_metrics = run_epoch(model, test_loader, optimizer=None)
        if test_metrics.accuracy > best_test_acc:
            best_test_acc = test_metrics.accuracy
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch == args.epochs - 1 or epoch % 5 == 0:
            print(f"    epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f} "
                  f"train_acc={train_metrics.accuracy:.4f}  test_acc={test_metrics.accuracy:.4f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, CHECKPOINT_DIR / f"fold{fold_id}_{held_out_subject}.pt")

    model.load_state_dict(best_state)
    _, final_metrics = run_epoch(model, test_loader, optimizer=None)
    return final_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="subset of fold ids; default = all")
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
    folds = [f for f in all_folds if f["fold"] in args.folds] if args.folds is not None else all_folds

    # Always load ALL foldable subjects, not just the folds being iterated
    # this run -- each fold's "train" set is "every subject except the one
    # held out", which must include all other subjects regardless of which
    # subset of folds --folds restricts *evaluation* to. Filtering the
    # dataset itself to args.folds would silently shrink training data to
    # just the other requested folds' subjects (e.g. --folds 0 1 would
    # train fold 0 on only subject B instead of all 15 other subjects).
    dataset = ULDDSequenceDataset(context_len=args.context_len, folds=[f["fold"] for f in all_folds])

    fold_metrics = []
    for f in folds:
        print(f"Fold {f['fold']} (held out: {f['held_out_subject']})")
        t0 = time.time()
        metrics = train_fold(f["fold"], f["held_out_subject"], dataset, args)
        fold_metrics.append(metrics)
        print(f"  -> acc={metrics.accuracy:.4f} f1_macro={metrics.f1_macro:.4f} ({time.time()-t0:.1f}s)")

    agg = aggregate(fold_metrics)
    print(f"\nAggregate over {len(fold_metrics)} folds: "
          f"acc={agg['accuracy']['mean']:.4f}+/-{agg['accuracy']['std']:.4f} "
          f"f1_macro={agg['f1_macro']['mean']:.4f}+/-{agg['f1_macro']['std']:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
