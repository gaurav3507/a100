#!/usr/bin/env python3
"""CAFE: fast per-video FID using the C-MET pytorch-fid/custom.py code itself (v1).

Run from the C-MET repository root with the C_MET environment active:

    python3 ../cafe_fid_fast_v1.py --validate            # both backends vs the FID column of ours.csv
    python3 ../cafe_fid_fast_v1.py --backend exact       # compute only

Why: custom.py handles one video pair at a time, and the per-video matrix square
root (scipy.linalg.sqrtm of a 2048 x 2048 product) runs on the CPU. On the DGX this
takes about 6 h for 1141 videos while the GPU sits idle.

Exact backend: custom.py is imported, and its own statistics_for_dirs_grouped() is
called with the same chunking as its main() (groups of --video-batch-size pairs,
--batch-size images, first path then second path), so the activations come from the
same call sequence. The distance is custom.py's own calculate_frechet_distance(),
run in a pool of worker processes with one BLAS thread each. Multi-threaded BLAS
can change the last bits of a result, so the output is validated against the
reference values, not assumed identical.

GPU backend: the same formula, with Tr(sqrt(S1 S2)) computed in float64 from the
eigenvalues of sqrt(S1) S2 sqrt(S1) on the GPU. Mathematically equal, numerically
not identical: screening only, unless validation shows negligible differences.

Nothing is written into the evaluation tree. Per-row results go to
reports/fid_fast_<stamp>.csv next to this script.
Exit code: 0 clean, 1 validation failure or errors, 2 bad usage.
"""

import argparse
import contextlib
import csv
import datetime
import importlib.util
import io
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

SCRIPT = "cafe_fid_fast_v1"
CUSTOM = "evaluation/pytorch-fid/custom.py"
DEFAULT_PATHS = ["runs/mead_ours/frames/ours", "runs/mead_ours/frames/ours_GT"]  # custom.py call order
DEFAULT_CSV = "evaluation/runs/mead_ours/ours.csv"
EQUIV_TOL = 1e-6  # absolute FID difference accepted as numerically equivalent

_custom = None


def load_custom(path):
    spec = importlib.util.spec_from_file_location("cmet_pytorch_fid_custom", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _worker_init(custom_path):
    global _custom
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except Exception:
        pass
    _custom = load_custom(custom_path)


def _fid_task(idx, m1, s1, m2, s2):
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            v = float(_custom.calculate_frechet_distance(m1, s1, m2, s2))
        return idx, v, buf.getvalue().strip()
    except Exception as e:
        return idx, None, "%s: %s" % (type(e).__name__, str(e)[:120])


def fid_gpu(m1, s1, m2, s2, device):
    import torch
    S1 = torch.as_tensor(s1, dtype=torch.float64, device=device)
    S2 = torch.as_tensor(s2, dtype=torch.float64, device=device)
    w, V = torch.linalg.eigh((S1 + S1.T) / 2)
    R = (V * w.clamp(min=0).sqrt()) @ V.T  # sqrt(S1)
    M = R @ S2 @ R
    ev = torch.linalg.eigvalsh((M + M.T) / 2).clamp(min=0)
    d = torch.as_tensor(m1 - m2, dtype=torch.float64, device=device)
    return float(d @ d + torch.trace(S1) + torch.trace(S2) - 2 * ev.sqrt().sum())


def prep_running(here):
    try:
        pid = int(open(os.path.join(here, "reports", "fullrun.lock")).read().split()[0])
        return pid if b"cafe_eval_prep" in open("/proc/%d/cmdline" % pid, "rb").read() else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", nargs=2, default=DEFAULT_PATHS,
                    help="frame parents relative to evaluation/, in the order custom.py was called")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="CSV with the reference FID column (for --validate)")
    ap.add_argument("--backend", choices=["exact", "gpu", "both"], default=None)
    ap.add_argument("--validate", action="store_true", help="compare with the CSV's FID column")
    ap.add_argument("--workers", type=int, default=32, help="processes for the exact backend")
    ap.add_argument("--batch-size", type=int, default=50, help="as custom.py")
    ap.add_argument("--video-batch-size", type=int, default=16, help="as custom.py")
    ap.add_argument("--num-workers", type=int, default=None, help="dataloader workers (custom.py: min(cpus, 8))")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N pairs only (tests)")
    ap.add_argument("--allow-concurrent", action="store_true", help="run even while a full run is active")
    args = ap.parse_args()
    backend = args.backend or ("both" if args.validate else "exact")

    repo, here = os.getcwd(), os.path.dirname(os.path.abspath(__file__))
    custom_path = os.path.join(repo, CUSTOM)
    if not (os.path.isfile(os.path.join(repo, "inference.py")) and os.path.isfile(custom_path)):
        print("Run this from the C-MET repository root (cd /workspace/sir/CAFE/C-MET).")
        return 2
    pid = prep_running(here)
    if pid and not args.allow_concurrent:
        print("A cafe_eval_prep run is active (pid %d); it would compete for CPU. Wait for it, or pass "
              "--allow-concurrent." % pid)
        return 2

    ev = os.path.join(repo, "evaluation")
    parents = [os.path.join(ev, p) for p in args.paths]
    lists = []
    for p in parents:
        if not os.path.isdir(p):
            print("Missing frame parent: %s" % p)
            return 2
        lists.append(sorted(e.name for e in os.scandir(p) if e.is_dir()))
    if lists[0] != lists[1]:
        a, b = set(lists[0]), set(lists[1])
        print("FAIL: the two frame parents hold different video directories (%d only in the first, %d only in "
              "the second); custom.py would pair them by position and misalign rows." % (len(a - b), len(b - a)))
        return 1
    names = lists[0][:args.limit] if args.limit else lists[0]

    tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print("%s | %s | backend %s | %d pairs | paths %s" % (SCRIPT, tag, backend, len(names), " , ".join(args.paths)),
          flush=True)
    custom = load_custom(custom_path)
    import torch
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.num_workers is None:
        try:
            ncpu = len(os.sched_getaffinity(0))
        except AttributeError:
            ncpu = os.cpu_count()
        args.num_workers = min(ncpu, 8) if ncpu else 0
    block = custom.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = custom.InceptionV3([block]).to(device)
    model.eval()

    pool = None
    if backend in ("exact", "both"):
        import multiprocessing as mp
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                   initializer=_worker_init, initargs=(custom_path,))
    exact, gpu, notes, pending = {}, {}, {}, set()
    t_stats = t_gpu = 0.0
    t0 = time.time()
    for start in range(0, len(names), args.video_batch_size):
        chunk = names[start:start + args.video_batch_size]
        t = time.time()
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            st0 = custom.statistics_for_dirs_grouped([os.path.join(parents[0], n) for n in chunk], model,
                                                     args.batch_size, 2048, device, args.num_workers)
            st1 = custom.statistics_for_dirs_grouped([os.path.join(parents[1], n) for n in chunk], model,
                                                     args.batch_size, 2048, device, args.num_workers)
        t_stats += time.time() - t
        for n, (m1, s1), (m2, s2) in zip(chunk, st0, st1):
            if m1 is None or m2 is None:
                notes[n] = "no frames found"
                continue
            if pool is not None:
                while len(pending) >= 4 * args.workers:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        i, v, note = f.result()
                        exact[i] = v
                        if note:
                            notes[i] = note
                pending.add(pool.submit(_fid_task, n, m1, s1, m2, s2))
            if backend in ("gpu", "both"):
                t = time.time()
                gpu[n] = fid_gpu(m1, s1, m2, s2, device)
                t_gpu += time.time() - t
        print("  %d/%d pairs through InceptionV3 (%.0f s)" % (min(start + args.video_batch_size, len(names)),
                                                               len(names), time.time() - t0), flush=True)
    for f in pending:
        i, v, note = f.result()
        exact[i] = v
        if note:
            notes[i] = note
    if pool is not None:
        pool.shutdown()
    wall = time.time() - t0

    ref = {}
    if args.validate:
        import pandas as pd
        df = pd.read_csv(os.path.join(repo, args.csv))
        if "FID" not in df.columns:
            print("FAIL: %s has no FID column to validate against" % args.csv)
            return 1
        for n in names:
            i = int(n)
            if i in df.index and pd.notna(df.at[i, "FID"]):
                ref[n] = float(df.at[i, "FID"])

    os.makedirs(os.path.join(here, "reports"), exist_ok=True)
    out = os.path.join(here, "reports", "fid_fast_%s.csv" % tag)
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["dir", "fid_exact", "fid_gpu", "fid_ref", "d_exact", "d_gpu", "note"])
        for n in names:
            e, g, r = exact.get(n), gpu.get(n), ref.get(n)
            wr.writerow([n, e, g, r, (e - r) if e is not None and r is not None else None,
                         (g - r) if g is not None and r is not None else None, notes.get(n, "")])

    def mean(d):
        v = [x for x in d.values() if x is not None]
        return (sum(v) / len(v), len(v)) if v else (None, 0)

    code = 0
    print("timing: InceptionV3 stats %.0f s, GPU distances %.1f s, wall %.0f s" % (t_stats, t_gpu, wall))
    for label, d in (("exact", exact), ("gpu", gpu)):
        if backend not in (label, "both"):
            continue
        m, k = mean(d)
        line = "%-5s: mean FID %s over %d/%d pairs" % (label, "%.6f" % m if m is not None else "-", k, len(names))
        if args.validate:
            common = [n for n in names if d.get(n) is not None and n in ref]
            if not common:
                print(line + "; no reference values to compare")
                code = 1
                continue
            diffs = [abs(d[n] - ref[n]) for n in common]
            rm = sum(ref[n] for n in common) / len(common)
            dm = sum(d[n] for n in common) / len(common) - rm
            worst = max(common, key=lambda n: abs(d[n] - ref[n]))
            ok = max(diffs) <= EQUIV_TOL and len(common) == len(names)
            verdict = ("EQUIVALENT (max |diff| <= %g)" % EQUIV_TOL) if ok else (
                "NOT EQUIVALENT" if label == "exact" else "screening only")
            line += ("; vs reference on %d pairs: max |diff| %.3e (row %s), mean |diff| %.3e, "
                     "difference of mean FID %+.3e; %s" % (len(common), max(diffs), worst,
                                                           sum(diffs) / len(diffs), dm, verdict))
            if label == "exact" and not ok:
                code = 1
        print(line)
    bad = {n: v for n, v in notes.items() if "singular" not in v}
    if notes:
        print("notes: %d rows (%s)" % (len(notes), "; ".join("%s: %s" % kv for kv in list(notes.items())[:4])))
    if bad:
        code = 1
    print("RESULT: %s" % out)
    return code


if __name__ == "__main__":
    sys.exit(main())
