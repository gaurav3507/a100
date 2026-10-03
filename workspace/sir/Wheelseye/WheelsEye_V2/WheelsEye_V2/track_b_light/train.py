"""Track B training (tasks.md Task 4.2): trains whichever of MLP/LightGBM/
GRU is selected, on the identical LOSO folds as Track A (common/splits.py),
so Phase 7's comparison isn't undermined by different splits.

Usage:
  .venv/Scripts/python.exe track_b_light/train.py --model mlp --folds 0 1 2
  .venv/Scripts/python.exe track_b_light/train.py --model lightgbm
  .venv/Scripts/python.exe track_b_light/train.py --model gru --context-len 8
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

from common.dataset import CORE_FEATURES, ULDDSequenceDataset, ULDDWindowDataset
from common.metrics import Metrics, aggregate, compute_metrics
from common.splits import load_folds
from track_b_light.model import GRUClassifier, INPUT_DIM, MLPClassifier, build_lightgbm_params

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "track_b_results"


def train_mlp_fold(held_out_subject, dataset, args):
    subjects = dataset.df["subject"].to_numpy()
    train_idx = np.where(subjects != held_out_subject)[0]
    test_idx = np.where(subjects == held_out_subject)[0]

    torch.manual_seed(args.seed)
    model = MLPClassifier(input_dim=INPUT_DIM, num_classes=NUM_CLASSES, dropout=args.dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            x, y = batch["flat"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["flat"].to(DEVICE)
            preds.append(model(x).argmax(-1).cpu().numpy())
            labels.append(batch["label"].numpy())
    return compute_metrics(np.concatenate(labels), np.concatenate(preds)), model.num_params()


def train_gru_fold(held_out_subject, dataset, args):
    subjects = dataset.df["subject"].to_numpy()
    train_idx = np.where(subjects != held_out_subject)[0]
    test_idx = np.where(subjects == held_out_subject)[0]

    torch.manual_seed(args.seed)
    model = GRUClassifier(
        input_dim=INPUT_DIM, hidden_size=args.gru_hidden, num_classes=NUM_CLASSES, dropout=args.dropout
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            x, y = batch["sequence"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["sequence"].to(DEVICE)
            preds.append(model(x).argmax(-1).cpu().numpy())
            labels.append(batch["label"].numpy())
    return compute_metrics(np.concatenate(labels), np.concatenate(preds)), model.num_params()


def train_lightgbm_fold(held_out_subject, df, args):
    import lightgbm as lgb

    train_df = df[df["subject"] != held_out_subject]
    test_df = df[df["subject"] == held_out_subject]

    medians = train_df[CORE_FEATURES].median()
    X_train = train_df[CORE_FEATURES].fillna(medians).to_numpy()
    y_train = train_df["label"].to_numpy()
    X_test = test_df[CORE_FEATURES].fillna(medians).to_numpy()
    y_test = test_df["label"].to_numpy()

    params = build_lightgbm_params(num_classes=NUM_CLASSES, n_estimators=args.lgbm_n_estimators)
    model = lgb.LGBMClassifier(**params, verbosity=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return compute_metrics(y_test, y_pred), model.booster_.num_trees()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "gru", "lightgbm"], required=True)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--context-len", type=int, default=8)
    parser.add_argument("--gru-hidden", type=int, default=48)
    parser.add_argument("--lgbm-n-estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"device: {DEVICE}  model: {args.model}")
    all_folds = load_folds()
    folds = [f for f in all_folds if f["fold"] in args.folds] if args.folds else all_folds
    all_fold_ids = [f["fold"] for f in all_folds]

    # Always load ALL foldable subjects (see track_a_heavy/train.py for
    # why): each fold's "train" set must be every other subject, not just
    # the other requested folds' subjects, or --folds subsets silently
    # starve training data.
    if args.model == "mlp":
        dataset = ULDDWindowDataset(folds=all_fold_ids)
    elif args.model == "gru":
        dataset = ULDDSequenceDataset(context_len=args.context_len, folds=all_fold_ids)
    else:
        import pandas as pd
        from common.dataset import DEFAULT_FEATURES_PATH
        df = pd.read_parquet(DEFAULT_FEATURES_PATH)
        df = df[df["fold"].isin(all_fold_ids)]

    fold_metrics = []
    for f in folds:
        t0 = time.time()
        if args.model == "mlp":
            metrics, n_params = train_mlp_fold(f["held_out_subject"], dataset, args)
        elif args.model == "gru":
            metrics, n_params = train_gru_fold(f["held_out_subject"], dataset, args)
        else:
            metrics, n_params = train_lightgbm_fold(f["held_out_subject"], df, args)
        fold_metrics.append(metrics)
        print(f"  fold {f['fold']} (held out {f['held_out_subject']}): "
              f"acc={metrics.accuracy:.4f} f1_macro={metrics.f1_macro:.4f} "
              f"params/trees={n_params} ({time.time()-t0:.1f}s)")

    agg = aggregate(fold_metrics)
    print(f"\n{args.model} aggregate over {len(fold_metrics)} folds: "
          f"acc={agg['accuracy']['mean']:.4f}+/-{agg['accuracy']['std']:.4f} "
          f"f1_macro={agg['f1_macro']['mean']:.4f}+/-{agg['f1_macro']['std']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.model}.json").write_text(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
