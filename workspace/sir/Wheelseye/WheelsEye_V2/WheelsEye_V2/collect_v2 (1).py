"""Collect WheelsEye_V2 results: tier x model matrix + calibration class-loss check.

Reads every data/processed/results/*.json, recomputes aggregates from the
per_fold entries (so nothing depends on how each driver summarized itself),
pools confusion matrices across folds, and reports per-class support so we
can see whether tier-3 calibration removed Low-class windows from scoring.

Usage:
    python collect_v2.py                       # default results dir
    python collect_v2.py --dir path/to/results
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = ["Low", "Medium", "High"]


def load(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"  ! unreadable: {path.name}")
        return None
    if "per_fold" not in d or not d["per_fold"]:
        return None
    cfg = d.get("config", {})
    folds = d["per_fold"]
    acc = np.array([f["metrics"]["accuracy"] for f in folds])
    f1 = np.array([f["metrics"]["f1_macro"] for f in folds])
    has_sm = all("metrics_smoothed" in f for f in folds)
    acc_sm = np.array([f["metrics_smoothed"]["accuracy"] for f in folds]) if has_sm else None
    f1_sm = np.array([f["metrics_smoothed"]["f1_macro"] for f in folds]) if has_sm else None
    cm = np.zeros((3, 3), dtype=int)
    for f in folds:
        m = f["metrics"].get("confusion_matrix")
        if m is not None:
            cm += np.asarray(m, dtype=int)
    support = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, np.diag(cm) / support, np.nan)
    pooled_acc = np.trace(cm) / cm.sum() if cm.sum() else float("nan")
    folds_with_no_low = sum(1 for f in folds
                            if f["metrics"].get("confusion_matrix") and sum(f["metrics"]["confusion_matrix"][0]) == 0)
    return dict(name=path.stem, tier=cfg.get("tier", "?"), model=cfg.get("model", path.stem),
                cal=cfg.get("calibrate", False), smooth=cfg.get("smooth", 0), seed=cfg.get("seed"),
                n=len(folds), acc=acc.mean(), acc_sd=acc.std(), f1=f1.mean(), f1_sd=f1.std(),
                acc_sm=(acc_sm.mean() if has_sm else None), f1_sm=(f1_sm.mean() if has_sm else None),
                pooled_acc=pooled_acc, support=support, recall=recall, total=int(cm.sum()),
                folds_no_low=folds_with_no_low)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=Path("data/processed/results"))
    args = p.parse_args()
    files = sorted(args.dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no json in {args.dir}")
    rows = [r for r in (load(f) for f in files) if r]
    print(f"parsed {len(rows)} result files from {args.dir}\n")

    # ---- Table 1: tier x model matrix ----
    tiers = sorted({r["tier"] for r in rows}, key=str)
    models = sorted({r["model"] for r in rows})
    print("TABLE 1 -- accuracy % (macro-F1 %) by model x tier, mean over folds")
    print(f"{'model':<22}" + "".join(f"{'tier '+str(t):>22}" for t in tiers))
    print("-" * (22 + 22 * len(tiers)))
    for m in models:
        line = f"{m:<22}"
        for t in tiers:
            hit = [r for r in rows if r["model"] == m and r["tier"] == t]
            line += f"{'':>22}" if not hit else f"{hit[0]['acc']*100:>9.2f} ({hit[0]['f1']*100:5.2f})   "
        print(line)

    # ---- Table 1b: RAW vs SMOOTHED for every result that carries metrics_smoothed ----
    sm = [r for r in rows if r["acc_sm"] is not None]
    if sm:
        print("\nTABLE 1b -- raw vs causal-smoothed stream (acc % / F1 %), from metrics_smoothed already in the JSONs")
        print(f"{'result':<46}{'raw acc':>9}{'smooth':>9}{'delta':>8} | {'raw F1':>8}{'smooth':>9}{'delta':>8}")
        print("-" * 100)
        for r in sorted(sm, key=lambda r: (str(r["tier"]), r["model"])):
            print(f"{r['name']:<46}{r['acc']*100:>9.2f}{r['acc_sm']*100:>9.2f}{(r['acc_sm']-r['acc'])*100:>+8.2f} | "
                  f"{r['f1']*100:>8.2f}{r['f1_sm']*100:>9.2f}{(r['f1_sm']-r['f1'])*100:>+8.2f}")

    # ---- Table 2: calibration class-loss diagnostic ----
    print("\nTABLE 2 -- pooled test support per class and pooled recall (checks whether "
          "calibration/smoothing removed Low windows from scoring)")
    print(f"{'result':<44}{'total':>7}{'Low':>7}{'Med':>7}{'High':>7} | {'R_Low':>6}{'R_Med':>6}{'R_High':>7} | {'no-Low folds':>12}")
    print("-" * 112)
    for r in sorted(rows, key=lambda r: (str(r["tier"]), r["model"])):
        s, rc = r["support"], r["recall"]
        fmt = lambda v: "  nan" if np.isnan(v) else f"{v*100:5.1f}"
        print(f"{r['name']:<44}{r['total']:>7}{s[0]:>7}{s[1]:>7}{s[2]:>7} | "
              f"{fmt(rc[0]):>6}{fmt(rc[1]):>6}{fmt(rc[2]):>7} | {r['folds_no_low']:>6}/{r['n']}")

    # ---- Table 3: same model across tiers -> support delta ----
    by_model = defaultdict(dict)
    for r in rows:
        by_model[r["model"]][r["tier"]] = r
    print("\nTABLE 3 -- Low-class test support: tier1 vs tier3 (windows lost to calibration)")
    for m, d in sorted(by_model.items()):
        if 1 in d and 3 in d:
            a, b = d[1]["support"][0], d[3]["support"][0]
            print(f"  {m:<22} Low support {a:>6} -> {b:>6}   ({(b-a)/a*100 if a else float('nan'):+.1f}%)"
                  f"   total {d[1]['total']} -> {d[3]['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
