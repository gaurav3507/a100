"""Paired per-fold statistical tests for WheelsEye_V2 results.

Every V2 result JSON stores per-fold metrics keyed by held-out subject, so any
two LOSO results can be compared as 16 paired observations. Reports, per
metric: mean paired difference, 95% bootstrap CI, Wilcoxon signed-rank p,
exact sign-test p, and win/tie/loss counts. Pairing is by held_out label,
never by fold index, so results with different fold orderings still align.

Usage:
    python paired_tests.py --ref RESULTS/tier3_lightgbm_v2_cal_seed0.json \
        RESULTS/tier3_i2m2_v2_cal_smooth3_seed0.json RESULTS/tier3_stacked_v2_cal_seed0.json
    python paired_tests.py --ref RESULTS/tier3_lightgbm_v2_cal_seed0.json --all RESULTS
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

METRICS = ("accuracy", "f1_macro")


def per_fold(path: Path) -> dict[str, dict[str, float]]:
    d = json.loads(path.read_text())
    return {f["label"]: {m: float(f["metrics"][m]) for m in METRICS} for f in d["per_fold"]}


def bootstrap_ci(diff: np.ndarray, n=20000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(ref: dict, other: dict, metric: str) -> dict:
    keys = sorted(set(ref) & set(other))
    a = np.array([ref[k][metric] for k in keys]); b = np.array([other[k][metric] for k in keys])
    diff = a - b                                   # positive = ref better
    nz = diff[diff != 0]
    w_p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 5 else float("nan")
    wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
    s_p = float(stats.binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else float("nan")
    lo, hi = bootstrap_ci(diff)
    d_z = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("nan")   # paired Cohen's d
    gt = sum((x > y) for x in a for y in b); lt = sum((x < y) for x in a for y in b)
    cliff = float((gt - lt) / (len(a) * len(b)))                                                # Cliff's delta
    return dict(d_z=d_z, cliff=cliff, n=len(keys), ref_mean=a.mean(), other_mean=b.mean(), mean_diff=diff.mean(),
                ci=(lo, hi), wilcoxon_p=w_p, sign_p=s_p, wins=wins, ties=int((diff == 0).sum()), losses=losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", type=Path, required=True, help="reference result JSON (e.g. LightGBM-cal)")
    p.add_argument("others", nargs="*", type=Path)
    p.add_argument("--all", type=Path, help="compare ref against every tier3 JSON in this dir")
    args = p.parse_args()

    others = list(args.others)
    if args.all:
        others += [q for q in sorted(args.all.glob("tier3_*.json")) if q.resolve() != args.ref.resolve()]
    if not others:
        raise SystemExit("nothing to compare: pass result JSONs or --all DIR")

    ref = per_fold(args.ref)
    print(f"reference: {args.ref.name}  ({len(ref)} folds)\n")
    for metric in METRICS:
        print(f"=== {metric} ===  (diff = reference − other, in percentage points)")
        print(f"{'other':<46}{'n':>3}{'ref':>7}{'other':>7}{'Δ':>7}{'95% CI':>17}{'Wilcoxon p':>12}{'sign p':>9}{'W/T/L':>8}{'d_z':>7}{'Cliff':>7}")
        print("-" * 130)
        for q in others:
            r = compare(ref, per_fold(q), metric)
            star = " *" if (not np.isnan(r["wilcoxon_p"]) and r["wilcoxon_p"] < 0.05) else ""
            print(f"{q.stem[:46]:<46}{r['n']:>3}{r['ref_mean']*100:>7.2f}{r['other_mean']*100:>7.2f}"
                  f"{r['mean_diff']*100:>+7.2f}{'[%+.1f, %+.1f]' % (r['ci'][0]*100, r['ci'][1]*100):>17}"
                  f"{r['wilcoxon_p']:>12.4f}{r['sign_p']:>9.4f}{r['wins']:>4}/{r['ties']}/{r['losses']}{r['d_z']:>7.2f}{r['cliff']:>7.2f}{star}")
        print()
    print("* = Wilcoxon p < 0.05 (two-sided). d_z = paired Cohen's d (0.2 small / 0.5 medium / 0.8 large); Cliff's delta on fold values (0.147/0.33/0.474).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
