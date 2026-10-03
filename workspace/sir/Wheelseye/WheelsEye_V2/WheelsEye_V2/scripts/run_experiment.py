"""v2 experiment runner (plan §3, §7, Phase 3/4): one entry point for every
model x tier x calibration combination, so every number in the results pack
comes from the same fold files, the same metrics code, and the same
train-fold-only statistics.

  python scripts/run_experiment.py --tier 1 --model lightgbm
  python scripts/run_experiment.py --tier 1 --model i2m2
  python scripts/run_experiment.py --tier 1 --model gnn_v2
  python scripts/run_experiment.py --tier 1 --model gnn_v2_nognn   # fair-trial arm
  python scripts/run_experiment.py --tier 3 --model stacked --calibrate
  python scripts/run_experiment.py --tier 3 --model i2m2 --calibrate --smooth 3

Leakage discipline carried over from the v1 defect register:
  #1 no test-set model selection: early stopping picks the best epoch on a
     held-out VALIDATION slice of the training windows; the test fold is
     evaluated exactly once, after the model is frozen.
  #2 no full-table statistics: imputation medians and z-score mean/std come
     from the training rows only (dataset stats_source).
  #3 fold-independent randomness: every seeded operation folds the fold id
     into the seed (seed * 1000 + fold).
  #4 no silent cohort exclusion: Tier 1/2 pool all 19 subjects; Tier 3
     trains on 18 subjects per fold (C/F/L awake-only rows included).
  #12/#13 no overwritten or aggregate-only output: one JSON per run config,
     carrying per-fold metrics alongside the aggregate.

Sequence-model context honesty note: sequence datasets are built over the
full table so each window's trailing context is its real session history;
context windows contribute FEATURES only (labels never enter the input).
Under Tier 3 sequences never cross subjects, so the strictest tier is fully
clean; under Tier 1 this matches the published protocol being reproduced;
under Tier 2 a training window's context can reach into an adjacent purged/
foreign block -- stated in the report rather than hidden.

Smoothing (--smooth K) evaluates the causal majority-smoothed stream
(plan §6) and is refused for Tier 1, whose test sets are window-scattered.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from common.calibration import apply_alert_baseline
from common.calibration_enroll import apply_enrollment_baseline
from common.dataset import (
    CONTEXT_FEATURES, CORE_FEATURE_GROUPS, CORE_FEATURES, DEFAULT_FEATURES_PATH,
    ULDDSequenceDataset, ULDDWindowDataset, V2_FEATURE_GROUPS, V2_FEATURES,
)
from common.metrics import aggregate, compute_metrics
from common.smoothing import smooth_by_session
from common.splits import (
    LOSO_V2_FOLDS_PATH, TIER1_FOLDS_PATH, TIER2_FOLDS_PATH,
    load_window_folds, merge_window_folds,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "results"

TABULAR_MODELS = ("lightgbm", "stacked")
WINDOW_MODELS = ("mlp", "i2m2")
SEQUENCE_MODELS = ("gru", "transformer", "gnn_v2", "gnn_v2_nognn")
ALL_MODELS = TABULAR_MODELS + WINDOW_MODELS + SEQUENCE_MODELS

NUM_CLASSES = 3


# --------------------------------------------------------------------------
# fold plans: list of (fold_id, fold_label, train_mask, test_mask) over df
# --------------------------------------------------------------------------

def fold_plan(df: pd.DataFrame, tier: int) -> tuple[pd.DataFrame, list[dict]]:
    if tier == 1:
        df = merge_window_folds(df, load_window_folds(TIER1_FOLDS_PATH), "exp_fold")
    elif tier == 2:
        df = merge_window_folds(df, load_window_folds(TIER2_FOLDS_PATH), "exp_fold")
        df = df[df["exp_fold"] >= 0].reset_index(drop=True)  # boundary-purged windows
    elif tier == 3:
        loso = json.loads(LOSO_V2_FOLDS_PATH.read_text())
        plans = []
        for f in loso:
            plans.append({
                "fold": f["fold"],
                "label": f"held_out={f['held_out_subject']}",
                "train_mask": df["subject"].isin(f["train_subjects"]).to_numpy(),
                "test_mask": (df["subject"] == f["held_out_subject"]).to_numpy(),
            })
        return df, plans
    else:
        raise ValueError(f"unknown tier {tier}")

    plans = []
    for f in sorted(df["exp_fold"].unique()):
        plans.append({
            "fold": int(f),
            "label": f"fold={int(f)}",
            "train_mask": (df["exp_fold"] != f).to_numpy(),
            "test_mask": (df["exp_fold"] == f).to_numpy(),
        })
    return df, plans


# --------------------------------------------------------------------------
# shared torch helpers
# --------------------------------------------------------------------------

def stratified_val_split(labels: np.ndarray, frac: float, seed: int):
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(labels))
    fit_idx, val_idx = train_test_split(
        idx, test_size=frac, random_state=seed, stratify=labels
    )
    return fit_idx, val_idx


def class_weights(labels: np.ndarray) -> torch.Tensor:
    """Balanced class weights from the TRAINING distribution (plan §6)."""
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def train_torch(model, forward_fn, loss_fn, fit_loader, val_loader, test_loader, epochs, lr):
    """Generic loop: select the best epoch on the VALIDATION loader, then
    evaluate the frozen best state on the test loader exactly once."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def evaluate(loader):
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                logits = forward_fn(model, batch)
                preds.append(logits.argmax(-1).cpu().numpy())
                labels.append(batch["label"].numpy())
        return np.concatenate(labels), np.concatenate(preds)

    best_val, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        for batch in fit_loader:
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimizer.step()
        vy, vp = evaluate(val_loader)
        val_acc = float(np.mean(vy == vp))
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    y_true, y_pred = evaluate(test_loader)
    return y_true, y_pred, {"best_val_acc": best_val}


def window_collate(batch):
    modalities = batch[0]["nodes"].keys()
    return {
        "nodes": {m: torch.stack([b["nodes"][m] for b in batch]) for m in modalities},
        "flat": torch.stack([b["flat"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def sequence_collate(batch):
    modalities = batch[0]["nodes"].keys()
    return {
        "nodes": {m: torch.stack([b["nodes"][m] for b in batch]) for m in modalities},
        "sequence": torch.stack([b["sequence"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


# --------------------------------------------------------------------------
# per-model fold runners: (train_df, test_df, full_df, args, seed) -> result
# --------------------------------------------------------------------------

def run_tabular_fold(model_name, train_df, test_df, flat_features, groups, args, seed):
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    if model_name == "lightgbm":
        import lightgbm as lgb

        from track_b_light.model import build_lightgbm_params

        medians = train_df[flat_features].median()
        params = build_lightgbm_params(num_classes=NUM_CLASSES, n_estimators=args.lgbm_n_estimators)
        clf = lgb.LGBMClassifier(**params, random_state=seed, verbosity=-1)
        clf.fit(train_df[flat_features].fillna(medians).to_numpy(), y_train)
        y_pred = clf.predict(test_df[flat_features].fillna(medians).to_numpy())
        return y_test, y_pred, {}

    from track_b_light.stacked_fusion import StackedFusionClassifier

    stack_groups = dict(groups)
    if args.features == "v2":
        stack_groups["context"] = list(CONTEXT_FEATURES)  # tabular-only extra group
    clf = StackedFusionClassifier(stack_groups, num_classes=NUM_CLASSES, seed=seed)
    clf.fit(train_df, y_train)
    y_pred = clf.predict(test_df)
    return y_test, y_pred, {}


def run_window_fold(model_name, train_df, test_df, groups, args, seed):
    torch.manual_seed(seed)
    train_ds = ULDDWindowDataset(train_df, groups=groups, stats_source=train_df)
    test_ds = ULDDWindowDataset(test_df, groups=groups, stats_source=train_df)

    labels = train_ds.df["label"].to_numpy()
    fit_idx, val_idx = stratified_val_split(labels, args.val_frac, seed)
    weights = class_weights(labels[fit_idx])

    loaders = dict(batch_size=args.batch_size, collate_fn=window_collate,
                    num_workers=args.num_workers,
                    pin_memory=(args.num_workers > 0),
                    persistent_workers=(args.num_workers > 0),
                    prefetch_factor=(args.prefetch if args.num_workers > 0 else None))
    fit_loader = DataLoader(Subset(train_ds, fit_idx), shuffle=True, **loaders)
    val_loader = DataLoader(Subset(train_ds, val_idx), shuffle=False, **loaders)
    test_loader = DataLoader(test_ds, shuffle=False, **loaders)

    if model_name == "mlp":
        from track_b_light.model import MLPClassifier

        model = MLPClassifier(
            input_dim=len(train_ds.all_features), num_classes=NUM_CLASSES, dropout=args.dropout
        ).to(DEVICE)
        forward = lambda m, b: m(b["flat"].to(DEVICE))
        loss = lambda m, b: torch.nn.functional.cross_entropy(
            m(b["flat"].to(DEVICE)), b["label"].to(DEVICE), weight=weights
        )
        y_true, y_pred, extra = train_torch(
            model, forward, loss, fit_loader, val_loader, test_loader, args.epochs, args.lr
        )
        return y_true, y_pred, extra

    from track_a_heavy.i2m2 import I2M2Model, i2m2_loss

    dims = {name: len(cols) for name, cols in train_ds.feature_groups.items()}
    model = I2M2Model(dims, num_classes=NUM_CLASSES, dropout=args.dropout).to(DEVICE)

    def to_device(b):
        return {m: v.to(DEVICE) for m, v in b["nodes"].items()}

    forward = lambda m, b: m(to_device(b))["logits"]
    loss = lambda m, b: i2m2_loss(m(to_device(b)), b["label"].to(DEVICE))[0]
    y_true, y_pred, extra = train_torch(
        model, forward, loss, fit_loader, val_loader, test_loader, args.epochs, args.lr
    )

    # free per-modality readout: each expert evaluated alone on the test fold
    model.eval()
    expert_preds = {m: [] for m in model.modalities}
    with torch.no_grad():
        for batch in test_loader:
            out = model(to_device(batch))
            for m in model.modalities:
                expert_preds[m].append(out["expert_logits"][m].argmax(-1).cpu().numpy())
    extra["expert_test_acc"] = {
        m: float(np.mean(np.concatenate(p) == y_true)) for m, p in expert_preds.items()
    }
    return y_true, y_pred, extra


def run_sequence_fold(model_name, full_df, train_mask, test_mask, groups, args, seed):
    torch.manual_seed(seed)
    train_df = full_df[train_mask]
    seq_ds = ULDDSequenceDataset(
        full_df, groups=groups, stats_source=train_df, context_len=args.context_len
    )
    # the dataset sorts and reindexes internally -- recover membership by key
    key = ["subject", "session", "window_id"]
    train_keys = set(map(tuple, train_df[key].itertuples(index=False)))
    test_keys = set(map(tuple, full_df[test_mask][key].itertuples(index=False)))
    ds_keys = list(map(tuple, seq_ds.df[key].itertuples(index=False)))
    train_idx = np.array([i for i, k in enumerate(ds_keys) if k in train_keys])
    test_idx = np.array([i for i, k in enumerate(ds_keys) if k in test_keys])

    labels = seq_ds.df["label"].to_numpy()[train_idx]
    fit_rel, val_rel = stratified_val_split(labels, args.val_frac, seed)
    weights = class_weights(labels[fit_rel])

    loaders = dict(batch_size=args.batch_size, collate_fn=sequence_collate,
                    num_workers=args.num_workers,
                    pin_memory=(args.num_workers > 0),
                    persistent_workers=(args.num_workers > 0),
                    prefetch_factor=(args.prefetch if args.num_workers > 0 else None))
    fit_loader = DataLoader(Subset(seq_ds, train_idx[fit_rel]), shuffle=True, **loaders)
    val_loader = DataLoader(Subset(seq_ds, train_idx[val_rel]), shuffle=False, **loaders)
    test_loader = DataLoader(Subset(seq_ds, test_idx), shuffle=False, **loaders)

    input_dim = len(seq_ds.all_features)
    if model_name == "gru":
        from track_b_light.model import GRUClassifier

        model = GRUClassifier(
            input_dim=input_dim, hidden_size=args.gru_hidden,
            num_classes=NUM_CLASSES, dropout=args.dropout,
        ).to(DEVICE)
        forward = lambda m, b: m(b["sequence"].to(DEVICE))
    elif model_name == "transformer":
        from track_a_heavy.transformer_fusion import TransformerFusion

        model = TransformerFusion(
            input_dim=input_dim, num_classes=NUM_CLASSES, dropout=args.dropout,
            max_context_len=args.context_len,
        ).to(DEVICE)
        forward = lambda m, b: m(b["sequence"].to(DEVICE))
    else:  # gnn_v2 / gnn_v2_nognn
        from track_a_heavy.gnn_v2 import build_gnn_v2

        model = build_gnn_v2(
            dropout=args.dropout, context_len=args.context_len,
            use_gnn=(model_name == "gnn_v2"), groups=groups,
        ).to(DEVICE)
        forward = lambda m, b: m({k: v.to(DEVICE) for k, v in b["nodes"].items()})["logits"]

    if model_name in ("gnn_v2", "gnn_v2_nognn"):
        # weighted CE + L2 on the classifier head; NO reconstruction term
        # (GNN_V2_LAMBDA2 = 0 by design -- measured harmful in v1, +2.78 pp)
        def loss(m, b):
            logits = forward(m, b)
            ce = torch.nn.functional.cross_entropy(logits, b["label"].to(DEVICE), weight=weights)
            return ce + 0.01 * m.classifier.weight.pow(2).sum()
    else:
        def loss(m, b):
            return torch.nn.functional.cross_entropy(
                forward(m, b), b["label"].to(DEVICE), weight=weights
            )

    y_true, y_pred, extra = train_torch(
        model, forward, loss, fit_loader, val_loader, test_loader, args.epochs, args.lr
    )
    test_meta = seq_ds.df.iloc[test_idx][["subject", "session", "window_id"]]
    return y_true, y_pred, extra, test_meta


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--model", choices=ALL_MODELS, required=True)
    parser.add_argument("--features", choices=["core", "v2"], default="v2")
    parser.add_argument("--calibrate", action="store_true",
                        help="Tier-3 alert-baseline calibration (plan §5)")
    parser.add_argument("--smooth", type=int, default=0,
                        help="causal majority smoothing width (tiers 2/3 only)")
    parser.add_argument("--folds", type=int, nargs="*", default=None,
                        help="subset of fold ids to EVALUATE (training pools unaffected)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--context-len", type=int, default=12)
    parser.add_argument("--gru-hidden", type=int, default=48)
    parser.add_argument("--lgbm-n-estimators", type=int, default=200)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes; 0 keeps the original single-process behaviour")
    parser.add_argument("--prefetch", type=int, default=4,
                        help="batches prefetched per worker (ignored when --num-workers 0)")
    parser.add_argument("--cal-mean-only", action="store_true",
                        help="enrollment calibration subtracts the enrollment mean only (no std division)")
    parser.add_argument("--n-enroll", type=int, default=0,
                        help="strict enrollment calibration: baseline from the first K Awake windows only (0 = whole session)")
    args = parser.parse_args()

    if args.smooth and args.tier == 1:
        raise SystemExit("--smooth is meaningless for Tier 1 (window-scattered test sets); "
                         "use it with tiers 2/3, whose test sets are contiguous streams.")

    if args.features == "v2":
        flat_features, groups = list(V2_FEATURES), V2_FEATURE_GROUPS
        tabular_flat = flat_features + list(CONTEXT_FEATURES)
    else:
        flat_features, groups = list(CORE_FEATURES), CORE_FEATURE_GROUPS
        tabular_flat = list(flat_features)

    df = pd.read_parquet(DEFAULT_FEATURES_PATH)
    if args.calibrate:
        # sensor features only -- context drift columns are already relative
        if args.n_enroll:
            df, _enroll_mask = apply_enrollment_baseline(df, flat_features, n_enroll=args.n_enroll, mean_only=args.cal_mean_only)
        else:
            df = apply_alert_baseline(df, flat_features, mode="replace")

    df, plans = fold_plan(df, args.tier)
    if args.folds is not None:
        plans = [p for p in plans if p["fold"] in args.folds]

    print(f"device: {DEVICE}  tier: {args.tier}  model: {args.model}  "
          f"features: {args.features}  calibrate: {args.calibrate}  folds: {len(plans)}")

    per_fold, fold_metrics, smoothed_metrics = [], [], []
    for plan in plans:
        seed = args.seed * 1000 + plan["fold"]  # defect #3: fold-decoupled RNG
        t0 = time.time()
        test_meta = None

        if args.model in TABULAR_MODELS:
            train_df, test_df = df[plan["train_mask"]], df[plan["test_mask"]]
            y_true, y_pred, extra = run_tabular_fold(
                args.model, train_df, test_df, tabular_flat, groups, args, seed
            )
            test_meta = test_df[["subject", "session", "window_id"]]
        elif args.model in WINDOW_MODELS:
            train_df, test_df = df[plan["train_mask"]], df[plan["test_mask"]]
            y_true, y_pred, extra = run_window_fold(
                args.model, train_df, test_df, groups, args, seed
            )
            test_meta = test_df[["subject", "session", "window_id"]]
        else:
            y_true, y_pred, extra, test_meta = run_sequence_fold(
                args.model, df, plan["train_mask"], plan["test_mask"], groups, args, seed
            )

        m = compute_metrics(y_true, y_pred)
        fold_metrics.append(m)
        record = {"fold": plan["fold"], "label": plan["label"],
                  "metrics": m.to_dict(), **extra}

        if args.smooth:
            y_smooth = smooth_by_session(test_meta.reset_index(drop=True), y_pred, k=args.smooth)
            ms = compute_metrics(y_true, y_smooth)
            smoothed_metrics.append(ms)
            record["metrics_smoothed"] = ms.to_dict()

        per_fold.append(record)
        print(f"  {plan['label']}: acc={m.accuracy:.4f} f1={m.f1_macro:.4f} "
              f"({time.time() - t0:.1f}s)")

    agg = aggregate(fold_metrics)
    print(f"\nAggregate ({len(fold_metrics)} folds): "
          f"acc={agg['accuracy']['mean']:.4f}+/-{agg['accuracy']['std']:.4f} "
          f"f1={agg['f1_macro']['mean']:.4f}+/-{agg['f1_macro']['std']:.4f}")

    out = {
        "config": {k: v for k, v in vars(args).items()},
        "per_fold": per_fold,          # defect #13: per-fold values persisted
        "aggregate": agg,
    }
    if smoothed_metrics:
        out["aggregate_smoothed"] = aggregate(smoothed_metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = (f"tier{args.tier}_{args.model}_{args.features}"
           + ("_cal" if args.calibrate else "")
           + (f"_smooth{args.smooth}" if args.smooth else "")
           + (f"_enroll{args.n_enroll}" if args.n_enroll else "")
           + ("_mean" if getattr(args, "cal_mean_only", False) else "")
           + f"_seed{args.seed}")
    path = RESULTS_DIR / f"{tag}.json"      # defect #12: unique path per config
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
