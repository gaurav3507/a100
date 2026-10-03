"""Collect sweep results: per-fold values from logs + seed-aggregated table.

The entry points save only across-fold aggregates, but paired statistical
comparisons need per-fold accuracies. Those appear in the captured stdout in
two stable formats, both parsed here:

  track_a / track_b torch:   "  -> acc=0.6829 f1_macro=0.4545 (801.8s)"
      (fold identity from the preceding "Fold N (held out: X)" line)
  track_b lightgbm:  "  fold 3 (held out E): acc=0.4906 f1_macro=0.4725 ..."
  ablation:          "  fold 0 (held out A): acc=0.5845 (112.0s)"
      (variant from the preceding "=== variant: NAME ===" line)

Outputs:
  sweep_results/per_fold.csv        long table: job,variant,seed,fold_subject,acc,f1
  sweep_results/summary.csv         mean+/-std over folds, then over seeds
  printed summary table
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SWEEP_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "sweep_results"
LOG_DIR = SWEEP_DIR / "logs"

FOLD_HDR = re.compile(r"^Fold (\d+) \(held out: (\w)\)")
ARROW = re.compile(r"^\s*-> acc=([0-9.]+) f1_macro=([0-9.]+)")
INLINE = re.compile(r"^\s*fold (\d+) \(held out (\w)\): .*?acc=([0-9.]+)(?: f1_macro=([0-9.]+))?")
VARIANT = re.compile(r"^=== variant: (\w+) ===")


def parse_log(path: Path) -> list[dict]:
    """One row per completed fold found in a unit's log."""
    m = re.match(r"(.+)_seed(\d+)\.log$", path.name)
    if not m:
        return []
    job, seed = m.group(1), int(m.group(2))

    rows, variant, pending_fold = [], "", None
    for line in path.read_text(errors="replace").splitlines():
        v = VARIANT.match(line)
        if v:
            variant, pending_fold = v.group(1), None
            continue
        h = FOLD_HDR.match(line)
        if h:
            pending_fold = h.group(2)
            continue
        a = ARROW.match(line)
        if a and pending_fold is not None:
            rows.append(dict(job=job, variant=variant, seed=seed,
                             fold_subject=pending_fold,
                             acc=float(a.group(1)), f1=float(a.group(2))))
            pending_fold = None
            continue
        i = INLINE.match(line)
        if i:
            rows.append(dict(job=job, variant=variant, seed=seed,
                             fold_subject=i.group(2), acc=float(i.group(3)),
                             f1=float(i.group(4)) if i.group(4) else float("nan")))
    return rows


def main():
    logs = sorted(LOG_DIR.glob("*.log"))
    if not logs:
        print(f"no logs in {LOG_DIR} -- run sweep_runner.py first")
        return 1

    rows = [r for p in logs for r in parse_log(p)]
    print(f"parsed {len(rows)} fold-results from {len(logs)} logs")

    with open(SWEEP_DIR / "per_fold.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["job", "variant", "seed", "fold_subject", "acc", "f1"])
        w.writeheader()
        w.writerows(rows)

    # group: (job, variant) -> seed -> [fold accs]
    grouped: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[(r["job"], r["variant"])][r["seed"]].append(r["acc"])

    print(f"\n{'job/variant':<28}{'acc mean':>10}{'fold sd':>9}{'seed sd':>9}{'seeds':>7}")
    print("-" * 63)
    summary = []
    for (job, variant), by_seed in sorted(grouped.items()):
        name = f"{job}/{variant}" if variant else job
        seed_means = [float(np.mean(a)) for a in by_seed.values()]
        fold_sds = [float(np.std(a)) for a in by_seed.values()]
        mean, seed_sd, fold_sd = float(np.mean(seed_means)), float(np.std(seed_means)), float(np.mean(fold_sds))
        n_folds = {len(a) for a in by_seed.values()}
        flag = "" if n_folds == {16} else f"  (!folds={sorted(n_folds)})"
        print(f"{name:<28}{mean*100:>9.2f}%{fold_sd*100:>8.1f} {seed_sd*100:>8.2f} {len(by_seed):>6}{flag}")
        summary.append(dict(name=name, acc_mean=mean, fold_sd=fold_sd,
                            seed_sd=seed_sd, n_seeds=len(by_seed)))

    with open(SWEEP_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"\nwrote {SWEEP_DIR/'per_fold.csv'} and {SWEEP_DIR/'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
