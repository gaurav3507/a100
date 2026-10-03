"""Enrollment curve for LightGBM under Tier-3 (LOSO + per-driver calibration).

Runs every enrollment length (windows of 5 s: 12=1 min ... 240=20 min, plus
the whole Awake session), then seed repeats for the two headline configs,
skipping any result file that already exists (resume-safe). Finally prints a
summary table, writes enrollment_curve.csv and enrollment_curve.png.

Usage (from the WheelsEye_V2 root, venv active):
    python run_enrollment_curve.py             # run everything missing, then summarize
    python run_enrollment_curve.py --dry-run   # show the plan only
    python run_enrollment_curve.py --summary   # skip running; summarize what exists
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

RESULTS = Path("data/processed/results")
RUNNER = ["python", "scripts/run_experiment.py", "--tier", "3", "--model", "lightgbm", "--calibrate"]
STEP_SECONDS = 5

ENROLL_WINDOWS = [12, 24, 48, 96, 144, 240, 360, 420, 460]   # 1..20 min, then 30/35/38 min (overlapping most of the alert session)
SEED_REPEAT_CONFIGS = [0, 48]                    # whole-session and 4-minute get 3 seeds
SEEDS = [0, 1, 2]


def result_path(k: int, seed: int) -> Path:
    tag = "tier3_lightgbm_v2_cal" + (f"_enroll{k}" if k else "") + f"_seed{seed}"
    return RESULTS / f"{tag}.json"


def plan() -> list[tuple[int, int]]:
    units = [(k, 0) for k in [0] + ENROLL_WINDOWS]                    # all lengths, seed 0
    units += [(k, s) for k in SEED_REPEAT_CONFIGS for s in SEEDS[1:]]  # extra seeds on headline configs
    return units


def run_unit(k: int, seed: int) -> bool:
    cmd = RUNNER + (["--n-enroll", str(k)] if k else []) + ["--seed", str(seed)]
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0 or not result_path(k, seed).exists():
        print(f"!!! unit k={k} seed={seed} failed (rc={rc}) -- continuing")
        return False
    return True


def load_agg(p: Path) -> tuple[float, float] | None:
    try:
        d = json.loads(p.read_text())
        pf = d["per_fold"]
        acc = sum(f["metrics"]["accuracy"] for f in pf) / len(pf)
        f1 = sum(f["metrics"]["f1_macro"] for f in pf) / len(pf)
        return acc, f1
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return None


def summarize() -> list[dict]:
    rows = []
    for k in [0] + ENROLL_WINDOWS:
        accs, f1s = [], []
        for s in SEEDS:
            r = load_agg(result_path(k, s)) if result_path(k, s).exists() else None
            if r:
                accs.append(r[0]); f1s.append(r[1])
        if not accs:
            continue
        mean = lambda v: sum(v) / len(v)
        sd = lambda v: (sum((x - mean(v)) ** 2 for x in v) / len(v)) ** 0.5 if len(v) > 1 else 0.0
        rows.append(dict(enroll_windows=k, enroll_minutes=(k * STEP_SECONDS / 60 if k else None),
                         n_seeds=len(accs), acc=mean(accs), acc_sd=sd(accs), f1=mean(f1s), f1_sd=sd(f1s)))
    return rows


def print_and_save(rows: list[dict]) -> None:
    print("\nENROLLMENT CURVE -- Tier-3 LOSO, LightGBM + per-driver calibration")
    print(f"{'enrollment':<22}{'seeds':>6}{'accuracy %':>14}{'macro-F1 %':>14}")
    print("-" * 56)
    for r in sorted(rows, key=lambda r: (r["enroll_windows"] == 0, r["enroll_windows"])):
        label = "whole Awake session" if r["enroll_windows"] == 0 else f"{r['enroll_minutes']:g} min ({r['enroll_windows']} win)"
        print(f"{label:<22}{r['n_seeds']:>6}{r['acc']*100:>9.2f} ± {r['acc_sd']*100:<4.1f}{r['f1']*100:>9.2f} ± {r['f1_sd']*100:<4.1f}")
    with open("enrollment_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\nwrote enrollment_curve.csv")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipped figure"); return
    curve = sorted([r for r in rows if r["enroll_windows"]], key=lambda r: r["enroll_windows"])
    whole = next((r for r in rows if r["enroll_windows"] == 0), None)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    x = [r["enroll_minutes"] for r in curve]
    ax.errorbar(x, [r["acc"] * 100 for r in curve], yerr=[r["acc_sd"] * 100 for r in curve],
                marker="o", capsize=3, label="Accuracy")
    ax.errorbar(x, [r["f1"] * 100 for r in curve], yerr=[r["f1_sd"] * 100 for r in curve],
                marker="s", capsize=3, label="Macro-F1")
    if whole:
        ax.axhline(whole["acc"] * 100, ls="--", lw=1, color="gray")
        ax.text(x[-1], whole["acc"] * 100 + 0.8, f"whole-session calibration ({whole['acc']*100:.1f}%)",
                ha="right", fontsize=9, color="gray")
    ax.axvspan(20, max(x) * 1.05, color="gray", alpha=.10)
    ax.text(21, 47.5, "baseline overlaps >50% of\nscored alert session\n(increasingly transductive)", fontsize=8, color="dimgray")
    ax.axhline(45.83, ls=":", lw=1, color="black")
    ax.text(x[0], 45.83 + 0.8, "no calibration (45.8%)", fontsize=9)
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{m:g}" for m in x])
    ax.set_xlabel("Enrollment length (minutes of alert driving)"); ax.set_ylabel("LOSO score (%)")
    ax.set_title("How much enrollment does a new driver need?  (16-fold LOSO, LightGBM)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=.3)
    fig.savefig("enrollment_curve.png", bbox_inches="tight"); print("wrote enrollment_curve.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary", action="store_true", help="only summarize existing results")
    args = ap.parse_args()

    if not args.summary:
        units = plan()
        todo = [(k, s) for k, s in units if not result_path(k, s).exists()]
        print(f"plan: {len(units)} units, {len(units) - len(todo)} already done, {len(todo)} to run")
        for k, s in todo:
            print(f"  k={k or 'whole'} seed={s} -> {result_path(k, s).name}")
        if args.dry_run:
            return 0
        ok = sum(run_unit(k, s) for k, s in todo)
        print(f"\nfinished: {ok}/{len(todo)} units succeeded")

    rows = summarize()
    if not rows:
        print("no results found"); return 1
    print_and_save(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
