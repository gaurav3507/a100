"""Multi-seed LOSO sweep (Task 3.5): Track A + ablation variants + Track B.

Drives the existing entry points as subprocesses; never imports their code.
Each unit = one (job, seed) pair. Handles the failure modes that would
otherwise corrupt an overnight run:

  - fixed output paths     -> each unit's JSON is copied to sweep_results/
                              immediately after success (entry points would
                              otherwise overwrite across seeds)
  - crash / restart        -> resume: a unit with a valid saved JSON is skipped
  - partial JSON on crash  -> resume validates JSON; invalid -> re-run
  - one unit failing       -> logged, queue continues
  - env contamination      -> CUDA_VISIBLE_DEVICES set explicitly per job
  - missing fold-RNG fix   -> preflight refuses to start Track A jobs until
                              train.py uses fold-dependent validation RNG
  - disk exhaustion        -> preflight free-space check

Per-fold values (needed for paired tests) are recovered from captured stdout
by scripts/collect_sweep.py, which is tested against real log formats.

Usage:
    python scripts/sweep_runner.py --dry-run          # print plan only
    python scripts/sweep_runner.py                    # run everything
    python scripts/sweep_runner.py --seeds 0 1        # fewer seeds
    python scripts/sweep_runner.py --jobs track_a mlp # subset of jobs
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROCESSED = REPO / "data" / "processed"
SWEEP_DIR = PROCESSED / "sweep_results"
LOG_DIR = SWEEP_DIR / "logs"

# job name -> (cmd builder, entry-point output path, needs_gpu)
JOBS = {
    "track_a": (
        lambda seed: [sys.executable, "track_a_heavy/train.py", "--epochs", "15", "--seed", str(seed)],
        PROCESSED / "track_a_results.json", True),
    "ablation": (
        lambda seed: [sys.executable, "track_a_heavy/ablation.py", "--epochs", "15", "--seed", str(seed)],
        PROCESSED / "track_a_ablation.json", True),
    "mlp": (
        lambda seed: [sys.executable, "track_b_light/train.py", "--model", "mlp", "--seed", str(seed)],
        PROCESSED / "track_b_results" / "mlp.json", True),
    "gru": (
        lambda seed: [sys.executable, "track_b_light/train.py", "--model", "gru", "--seed", str(seed)],
        PROCESSED / "track_b_results" / "gru.json", True),
    "lightgbm": (
        lambda seed: [sys.executable, "track_b_light/train.py", "--model", "lightgbm", "--seed", str(seed)],
        PROCESSED / "track_b_results" / "lightgbm.json", False),
}


def preflight(jobs: list[str]) -> list[str]:
    """Abort-worthy problems, checked BEFORE any compute is spent."""
    problems = []

    if not (PROCESSED / "uldd_features.parquet").exists():
        problems.append("uldd_features.parquet missing -- run build_feature_table.py")

    free_gb = shutil.disk_usage(REPO).free / 1e9
    if free_gb < 2:
        problems.append(f"only {free_gb:.1f} GB free -- clear space first")

    if any(j in jobs for j in ("track_a", "ablation")):
        train_src = (REPO / "track_a_heavy" / "train.py").read_text()
        if "args.seed * 1000 + fold_id" not in train_src:
            problems.append(
                "track_a_heavy/train.py still uses a fold-independent validation RNG.\n"
                "    Fix (one line inside train_fold):\n"
                "      rng = np.random.RandomState(args.seed * 1000 + fold_id)\n"
                "    Without it, folds share training sets and every seed repeats that bug.")

    return problems


def unit_paths(job: str, seed: int) -> tuple[Path, Path]:
    return SWEEP_DIR / f"{job}_seed{seed}.json", LOG_DIR / f"{job}_seed{seed}.log"


def already_done(job: str, seed: int) -> bool:
    out, _ = unit_paths(job, seed)
    if not out.exists():
        return False
    try:
        json.loads(out.read_text())
        return True
    except json.JSONDecodeError:
        out.unlink()          # partial write from a crash -- redo this unit
        return False


def run_unit(job: str, seed: int, gpu: str) -> bool:
    cmd_builder, src_json, needs_gpu = JOBS[job]
    cmd = cmd_builder(seed)
    out_json, log_path = unit_paths(job, seed)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu if needs_gpu else ""
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "8")

    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    dt = time.time() - t0

    if proc.returncode != 0:
        print(f"  FAIL {job} seed={seed} rc={proc.returncode} ({dt:.0f}s) -- see {log_path}")
        return False
    if not src_json.exists():
        print(f"  FAIL {job} seed={seed}: exited 0 but {src_json.name} missing ({dt:.0f}s)")
        return False

    shutil.copy2(src_json, out_json)     # preserve before the next seed overwrites
    print(f"  OK   {job} seed={seed} ({dt:.0f}s) -> {out_json.name}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--jobs", nargs="*", default=list(JOBS), choices=list(JOBS))
    p.add_argument("--gpu", default="0", help="CUDA device index for GPU jobs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    units = [(j, s) for j in args.jobs for s in args.seeds]
    pending = [(j, s) for j, s in units if not already_done(j, s)]
    done = len(units) - len(pending)

    print(f"sweep: {len(units)} units total, {done} already done, {len(pending)} to run")
    for j, s in pending:
        print(f"  plan: {j} seed={s}" + ("" if JOBS[j][2] else " (cpu)"))
    if args.dry_run:
        return 0

    problems = preflight(args.jobs)
    if problems:
        print("\nPREFLIGHT FAILED:")
        for pr in problems:
            print(f"  - {pr}")
        return 1

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    t0, ok = time.time(), 0
    for j, s in pending:
        ok += run_unit(j, s, args.gpu)
    print(f"\nfinished: {ok}/{len(pending)} units succeeded "
          f"in {(time.time()-t0)/3600:.1f}h. Results in {SWEEP_DIR}")
    print("Next: python scripts/collect_sweep.py")
    return 0 if ok == len(pending) else 2


if __name__ == "__main__":
    raise SystemExit(main())
