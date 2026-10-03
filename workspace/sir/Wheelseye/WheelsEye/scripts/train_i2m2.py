"""I2M2 (NeurIPS 2024) adapted to UL-DD LOSO: inter- + intra-modality fusion.

Following Madaan et al.'s framework: one unimodal expert per modality
(intra-modality), one early-fusion multimodal expert (inter-modality), and a
product-of-experts combination of their class probabilities at inference.
Experts here are small MLPs matching this repo's Track B capacity class.

Protocol matches the patched Track A/B pipeline exactly:
  - 18-subject training cohort (C/F/L rows train, never validate/test)
  - per-fold impute/z-score statistics from training subjects only
  - epoch selection on 2 held-out TRAINING subjects, never the test subject
  - fold-dependent validation RNG (seed*1000+fold) so folds do not share
    training sets (the bug found in Track A's first patched run)

Usage:
    python scripts/train_i2m2.py                 # all 16 folds
    python scripts/train_i2m2.py --folds 0 1 2   # subset
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.dataset import DEFAULT_FEATURES_PATH, feature_groups
from common.metrics import Metrics, aggregate, compute_metrics
from common.paths import NO_DROWSY_SUBJECTS
from common.splits import load_folds

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "i2m2_results.json"
NUM_CLASSES = 3


class ExpertMLP(nn.Module):
    """One expert: same capacity class as Track B's MLP (64-32-16)."""

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def poe_log_probs(logit_list: list[torch.Tensor]) -> torch.Tensor:
    """Product-of-experts in log space.

    Multiplying raw probabilities lets any single expert's ~0 kill the
    product (found in Tier-3 testing); summing log-softmaxes is the
    numerically safe equivalent, renormalized at the end.
    """
    log_p = torch.stack([F.log_softmax(l, dim=-1) for l in logit_list]).sum(dim=0)
    return log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)


def prepare_fold(df, features_by_group, held_out_subject, seed, fold_id):
    """Split + fold-local impute/standardize. Returns tensors per group."""
    subjects = df["subject"].to_numpy()

    train_subjects = sorted(set(subjects) - {held_out_subject})
    val_pool = [s for s in train_subjects if s not in NO_DROWSY_SUBJECTS]
    rng = np.random.RandomState(seed * 1000 + fold_id)   # fold-dependent!
    val_subjects = set(rng.choice(val_pool, size=2, replace=False))
    stats_subjects = [s for s in train_subjects if s not in val_subjects]

    stats_df = df[df["subject"].isin(stats_subjects)]
    masks = {
        "train": df["subject"].isin(stats_subjects).to_numpy(),
        "val": df["subject"].isin(val_subjects).to_numpy(),
        "test": (subjects == held_out_subject),
    }

    tensors = {split: {} for split in masks}
    for gname, cols in features_by_group.items():
        med = stats_df[cols].median()
        fill = {c: (0.0 if pd.isna(med[c]) else float(med[c])) for c in cols}
        mean = stats_df[cols].mean().fillna(0.0)
        std = stats_df[cols].std().replace(0, 1.0).fillna(1.0)
        X = df[cols].fillna(value=fill)
        X = ((X - mean) / std).to_numpy(dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        for split, m in masks.items():
            tensors[split][gname] = torch.from_numpy(X[m])

    y = df["label"].to_numpy()
    labels = {split: torch.from_numpy(y[m].astype(np.int64)) for split, m in masks.items()}
    return tensors, labels, sorted(val_subjects)


def train_one_expert(X_train, y_train, X_val, y_val, input_dim, args) -> nn.Module:
    """Train a single expert with val-based epoch selection."""
    torch.manual_seed(args.seed)
    model = ExpertMLP(input_dim, dropout=args.dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ds = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    best_acc, best_state = -1.0, None
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X_val.to(DEVICE)).argmax(-1).cpu() == y_val).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


def run_fold(fold_id, held_out_subject, df, groups, args):
    tensors, labels, val_subjects = prepare_fold(df, groups, held_out_subject, args.seed, fold_id)

    concat = {s: torch.cat([tensors[s][g] for g in groups], dim=1) for s in tensors}
    experts, names = [], []

    for g, cols in groups.items():                      # intra-modality experts
        experts.append(train_one_expert(tensors["train"][g], labels["train"],
                                        tensors["val"][g], labels["val"], len(cols), args))
        names.append(g)
    experts.append(train_one_expert(concat["train"], labels["train"],   # inter-modality expert
                                    concat["val"], labels["val"],
                                    concat["train"].shape[1], args))
    names.append("multimodal")

    with torch.no_grad():
        test_inputs = [tensors["test"][g].to(DEVICE) for g in groups] + [concat["test"].to(DEVICE)]
        logits = [m(x) for m, x in zip(experts, test_inputs)]
        y_pred = poe_log_probs(logits).argmax(-1).cpu().numpy()
        per_expert = {n: compute_metrics(labels["test"].numpy(), l.argmax(-1).cpu().numpy()).accuracy
                      for n, l in zip(names, logits)}

    metrics = compute_metrics(labels["test"].numpy(), y_pred)
    return metrics, per_expert, val_subjects


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    args = p.parse_args()

    print(f"device: {DEVICE}")
    all_folds = load_folds()
    folds = [f for f in all_folds if f["fold"] in args.folds] if args.folds is not None else all_folds
    all_ids = [f["fold"] for f in all_folds]

    df = pd.read_parquet(args.features_path)
    df = df[df["fold"].isin(all_ids + [-1])]            # Option A cohort
    groups = feature_groups(include_telemetry=False)
    print(f"rows: {len(df)}  subjects: {df['subject'].nunique()}  "
          f"nodes: {[f'{g}({len(c)})' for g, c in groups.items()]}")

    fold_metrics, expert_log = [], {}
    for f in folds:
        t0 = time.time()
        m, per_expert, vs = run_fold(f["fold"], f["held_out_subject"], df, groups, args)
        fold_metrics.append(m)
        expert_log[f["held_out_subject"]] = per_expert
        pe = "  ".join(f"{k}={v:.3f}" for k, v in per_expert.items())
        print(f"  fold {f['fold']} (held out {f['held_out_subject']}): "
              f"I2M2 acc={m.accuracy:.4f} f1={m.f1_macro:.4f}  [{pe}]  "
              f"val={vs} ({time.time()-t0:.1f}s)")

    agg = aggregate(fold_metrics)
    print(f"\nI2M2 aggregate over {len(fold_metrics)} folds: "
          f"acc={agg['accuracy']['mean']:.4f}+/-{agg['accuracy']['std']:.4f} "
          f"f1_macro={agg['f1_macro']['mean']:.4f}+/-{agg['f1_macro']['std']:.4f}")

    agg["per_fold_expert_accuracy"] = expert_log
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(agg, indent=2))
    print(f"Saved {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
