"""Multi-seed neural runs for the WheelsEye revision (reviewer objection: single-seed
comparison against a deterministic tree model).

Runs each neural architecture under the headline configuration
(tier 3, --calibrate --n-enroll 96 --cal-mean-only) for three seeds, then reports
mean +/- seed s.d. beside the deterministic LightGBM reference, and writes a LaTeX
row set for Table IV.

Resume-safe: a (model, seed) whose result JSON already exists and parses is skipped,
so an interrupted run is restarted with the same command and loses nothing.

Usage (WheelsEye_V2 root, venv active, GPU free):
    python run_multiseed_v1.py --dry-run          # show the plan
    python run_multiseed_v1.py                    # run everything missing
    python run_multiseed_v1.py --summary          # aggregate what exists, no runs
    python run_multiseed_v1.py --models i2m2 stacked --seeds 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import signal
try:                                    # allow piping into head/grep without a traceback
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except (AttributeError, ValueError):
    pass

RESULTS = Path("data/processed/results")
RUNNER = ["scripts/run_experiment.py", "--tier", "3", "--calibrate",
          "--n-enroll", "96", "--cal-mean-only", "--num-workers", "8"]

# neural models needing seeds; lightgbm is deterministic and used as the reference
MODELS = ["i2m2", "stacked", "gnn_v2_nognn", "transformer"]
SEEDS = [0, 1, 2]
REFERENCE = ("lightgbm", "tier3_lightgbm_v2_cal_enroll96_mean_seed0")


def result_path(model: str, seed: int) -> Path:
    return RESULTS / f"tier3_{model}_v2_cal_enroll96_mean_seed{seed}.json"


def read_agg(p: Path):
    """Return (acc_mean, f1_mean, n_folds) recomputed from per_fold, or None."""
    try:
        d = json.loads(p.read_text())
        pf = d["per_fold"]
        if not pf:
            return None
        acc = sum(f["metrics"]["accuracy"] for f in pf) / len(pf)
        f1 = sum(f["metrics"]["f1_macro"] for f in pf) / len(pf)
        return acc, f1, len(pf)
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return None


def done(model: str, seed: int) -> bool:
    p = result_path(model, seed)
    if not p.exists():
        return False
    if read_agg(p) is None:                      # partial write from a crash
        print(f"  ! {p.name} unreadable -- will re-run")
        p.unlink(missing_ok=True)
        return False
    return True


def preflight(models):
    problems = []
    if not Path("scripts/run_experiment.py").exists():
        problems.append("scripts/run_experiment.py not found -- run from the WheelsEye_V2 root")
    src = Path("scripts/run_experiment.py").read_text() if Path("scripts/run_experiment.py").exists() else ""
    if "cal_mean_only" not in src:
        problems.append("run_experiment.py lacks --cal-mean-only; apply patch_meanonly_exact.py first")
    if "apply_enrollment_baseline" not in src:
        problems.append("run_experiment.py lacks --n-enroll; apply patch_enroll_exact.py first")
    mod = Path("common/calibration_enroll.py")
    if not mod.exists():
        problems.append("common/calibration_enroll.py missing")
    elif "mirror_cols" not in mod.read_text():
        problems.append("common/calibration_enroll.py is an old version (no mirror_cols); "
                        "copy calibration_enroll_v3.py over it or the sequence models will crawl")
    if not (RESULTS.parent / "uldd_features.parquet").exists():
        problems.append("data/processed/uldd_features.parquet missing")
    return problems


def run_unit(model: str, seed: int, gpu: str, log_dir: Path) -> bool:
    cmd = [sys.executable] + RUNNER + ["--model", model, "--seed", str(seed)]
    log = log_dir / f"{model}_seed{seed}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu                     # never inherit a stale ""
    print(f"\n>>> {model} seed={seed}  (GPU {gpu})  -> {log.name}", flush=True)
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    dt = (time.time() - t0) / 60
    if rc != 0 or not result_path(model, seed).exists():
        print(f"    FAIL rc={rc} after {dt:.1f} min -- see {log}", flush=True)
        return False
    a = read_agg(result_path(model, seed))
    print(f"    OK  acc={a[0]*100:.2f} f1={a[1]*100:.2f} folds={a[2]}  ({dt:.1f} min)", flush=True)
    return True


def summarize(models, seeds):
    import statistics as st
    rows = []
    print(f"\n{'model':<16}{'seeds':>6}{'acc mean':>10}{'seed sd':>9}{'f1 mean':>9}{'seed sd':>9}")
    print("-" * 59)
    ref = read_agg(RESULTS / f"{REFERENCE[1]}.json")
    if ref:
        print(f"{'lightgbm (det.)':<16}{1:>6}{ref[0]*100:>10.2f}{0.0:>9.2f}{ref[1]*100:>9.2f}{0.0:>9.2f}")
        rows.append(("LightGBM (deterministic)", ref[0] * 100, 0.0, ref[1] * 100, 0.0, 1))
    for m in models:
        got = [read_agg(result_path(m, s)) for s in seeds]
        got = [g for g in got if g]
        if not got:
            print(f"{m:<16}{0:>6}{'--':>10}")
            continue
        a = [g[0] * 100 for g in got]; f = [g[1] * 100 for g in got]
        asd = st.stdev(a) if len(a) > 1 else 0.0
        fsd = st.stdev(f) if len(f) > 1 else 0.0
        print(f"{m:<16}{len(got):>6}{st.mean(a):>10.2f}{asd:>9.2f}{st.mean(f):>9.2f}{fsd:>9.2f}")
        rows.append((m, st.mean(a), asd, st.mean(f), fsd, len(got)))
    if ref and len(rows) > 1:
        print("\nmargins of LightGBM over each neural model (accuracy pp):")
        for name, acc, sd, _, _, n in rows[1:]:
            print(f"  vs {name:<16} {ref[0]*100 - acc:+7.2f} pp   (neural seed s.d. {sd:.2f}, n={n})")
    tex = RESULTS / "multiseed_table_rows.tex"
    with open(tex, "w") as fh:
        fh.write("% Table IV -- mean-only column with seed dispersion\n")
        for name, acc, sd, f1, fsd, n in rows:
            disp = "deterministic" if n == 1 and "LightGBM" in name else f"$\\pm$ {sd:.2f}"
            fh.write(f"{name} & {acc:.2f} {disp} & {f1:.2f} \\\\\n")
    print(f"\nwrote {tex}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS, choices=MODELS)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    if args.summary:
        summarize(args.models, args.seeds)
        return 0

    units = [(m, s) for m in args.models for s in args.seeds]
    todo = [(m, s) for m, s in units if not done(m, s)]
    print(f"plan: {len(units)} units, {len(units)-len(todo)} already done, {len(todo)} to run")
    for m, s in todo:
        print(f"  {m} seed={s}")
    if args.dry_run:
        return 0

    problems = preflight(args.models)
    if problems:
        print("\nPREFLIGHT FAILED:")
        for p in problems:
            print("  -", p)
        return 1

    log_dir = RESULTS / "multiseed_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    t0, ok = time.time(), 0
    for m, s in todo:
        ok += run_unit(m, s, args.gpu, log_dir)
    print(f"\nfinished {ok}/{len(todo)} units in {(time.time()-t0)/3600:.1f} h")
    summarize(args.models, args.seeds)
    return 0 if ok == len(todo) else 2


if __name__ == "__main__":
    raise SystemExit(main())
