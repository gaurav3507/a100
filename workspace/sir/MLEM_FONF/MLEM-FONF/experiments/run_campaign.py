"""Overnight classical-baseline campaign (JBHI statistics in one run).

Runs ALL classical baselines (incl. supplement tier) + frozen A-FONF over
full seed counts, across staged dataset x geometry grids, with:
  * process-pool parallelism (cells run concurrently; per-cell methods stay
    sequential and deterministic),
  * crash-proof progress: every result row is appended + flushed to the
    stage CSV the moment it is computed,
  * resume: on restart, completed (slice, incident, seed, method) cells are
    detected from the CSVs and skipped.

Typical overnight launch (ant-pc):
  nohup python experiments/run_campaign.py --root /media/ant-pc/HDD2/datasets \
      --stages mayo_parallel lodopab_parallel mayo_fan \
      --realizations 10 --slices 20 \
      > results/campaign_night1.log 2>&1 &
  tail -f results/campaign_night1.log
"""
import argparse
import csv
import os
import pathlib
import sys
import time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np

_G = {}



def _mlem_iters(geometry, inc, default):
    """Early-stopped MLEM protocol: iteration count frozen on the validation
    patient (results/mlem_iters.json); falls back to `default` if absent."""
    import json as _json
    p = pathlib.Path(__file__).resolve().parents[1] / "results" / "mlem_iters.json"
    if p.exists():
        try:
            return int(_json.load(open(p)).get(f"{geometry}:{int(inc)}", default))
        except Exception:
            return default
    return default

def _init_worker(geometry: str):
    """Build the (geometry-specific) system matrix once per worker."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"          # avoid BLAS oversubscription
    from mlem_fonf import build_system_matrix, build_fanbeam_matrix
    if geometry == "parallel":
        _G["A"] = build_system_matrix(256, 64, 90)
    else:
        _G["A"] = build_fanbeam_matrix(256, 64, 90, sod=600.0, odd=600.0)
    _G["geometry"] = geometry


def _run_cell(task):
    """Compute every requested method for one (slice, incident, seed) cell."""
    (dataset, geometry, tag, idx, ref, inc, seed, iters, sel_iters,
     with_supp, methods_filter) = task
    from mlem_fonf import (poisson_sinogram, mlem, mlem_spatial, pnp_admm_tv,
                           reconstruct, all_metrics)
    from mlem_fonf.fbp import fbp
    A = _G["A"]
    ref = np.asarray(ref, dtype=np.float64)
    g, counts = poisson_sinogram(A, ref, inc, np.random.default_rng(100 * idx + seed),
                                 return_counts=True)
    scale = (A @ ref.ravel()).max() / inc
    methods = {
        "FBP": lambda: fbp(g, A, 256, 64, 90, geometry),
        "MLEM": lambda: mlem(g, A, 256, _mlem_iters(geometry, inc, iters)),
        "MLEM+TV": lambda: mlem_spatial(g, A, 256, "tv", iters),
        "MLEM+AD": lambda: mlem_spatial(g, A, 256, "ad", iters),
        "PnP-ADMM(TV)": lambda: pnp_admm_tv(g, A, 256, iters),
        "A-FONF": lambda: reconstruct(counts, scale, A, 256, n_bins=64,
                                      n_iter=iters, select_iter=sel_iters)[0],
    }
    if with_supp:
        methods["MLEM+FuzzyAD"] = lambda: mlem_spatial(g, A, 256, "fuzzyad",
                                                       iters)
    if methods_filter:
        methods = {k: v for k, v in methods.items() if k in methods_filter}
    rows = []
    for name, fn in methods.items():
        t0 = time.time()
        try:
            rec = np.clip(np.asarray(fn(), dtype=float), 0, None)
            if not np.isfinite(rec).all():
                raise FloatingPointError("non-finite reconstruction")
            met = all_metrics(ref, rec)
            if not all(np.isfinite(v) for v in met.values()):
                raise FloatingPointError("non-finite metric")
            rows.append([dataset, geometry, inc, tag, idx, seed, name,
                         met["SNR"], met["MSE"], met["RMSE"], met["PSNR"],
                         met["CP"], met["MSSIM"], time.time() - t0])
        except Exception as exc:  # noqa: BLE001 — isolate, log loudly, go on
            print(f"!!! FAILED {name} @ {dataset}/{geometry} inc={inc} "
                  f"{tag}#{idx} seed={seed}: {type(exc).__name__}: {exc}",
                  flush=True)
            rows.append([dataset, geometry, inc, tag, idx, seed, name,
                         float("nan")] * 1 + [float("nan")] * 6
                        + [time.time() - t0])
            rows[-1] = [dataset, geometry, inc, tag, idx, seed, name,
                        float("nan"), float("nan"), float("nan"),
                        float("nan"), float("nan"), float("nan"),
                        time.time() - t0]
    return rows


def _load_slices(dataset: str, root, limit: int):
    if dataset == "phantom":
        from mlem_fonf.phantoms import PHANTOMS
        names = ["shepp_logan", "ct_sim", "elliptical", "thorax_like"]
        return [(nm, i, PHANTOMS[nm](256)) for i, nm in enumerate(names[:limit])]
    from mlem_fonf import data as D
    if dataset == "mayo":
        return [(p, i, im) for p, i, im in
                D.mayo_slices(root, "test", "full", every=4, limit=limit)]
    if dataset == "lodopab":
        return [(f"f{fi}", si, im) for fi, si, im in
                D.lodopab_slices(root, "test", limit=limit)]
    raise ValueError(dataset)


def _done_keys(csvp: pathlib.Path):
    done = set()
    if csvp.exists():
        with open(csvp) as fh:
            for row in csv.DictReader(fh):
                done.add((row["dataset"], row["geometry"],
                          float(row["incident"]), row["slice_tag"],
                          int(row["slice_idx"]), int(row["seed"]),
                          row["method"]))
    return done


STAGES = {
    "mayo_parallel": ("mayo", "parallel"),
    "mayo_fan": ("mayo", "fan"),
    "lodopab_parallel": ("lodopab", "parallel"),
    "lodopab_fan": ("lodopab", "fan"),
    "phantom_parallel": ("phantom", "parallel"),
    "phantom_fan": ("phantom", "fan"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--stages", nargs="+", default=["mayo_parallel",
                                                    "lodopab_parallel"],
                    choices=list(STAGES))
    ap.add_argument("--incidents", type=float, nargs="+",
                    default=[60, 300, 1500, 3000])
    ap.add_argument("--realizations", type=int, default=10)
    ap.add_argument("--slices", type=int, default=20)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--select-iters", type=int, default=800,
                    help="A-FONF Morozov stage-1 iterations")
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = auto (cpu_count - 2, capped at 10)")
    ap.add_argument("--no-supplement", action="store_true")
    ap.add_argument("--methods", nargs="+", default=None)
    a = ap.parse_args()
    from mlem_fonf.version import banner
    banner("run_campaign")
    nw = a.workers or max(2, min(cpu_count() - 2, 10))
    outdir = pathlib.Path(__file__).resolve().parents[1] / "results"
    outdir.mkdir(exist_ok=True)
    method_names = ["FBP", "MLEM", "MLEM+TV", "MLEM+AD", "PnP-ADMM(TV)",
                    "A-FONF"] + ([] if a.no_supplement else ["MLEM+FuzzyAD"])
    if a.methods:
        method_names = [m for m in method_names if m in a.methods]
    header = ["dataset", "geometry", "incident", "slice_tag", "slice_idx",
              "seed", "method", "SNR", "MSE", "RMSE", "PSNR", "CP", "MSSIM",
              "time_s"]
    grand_t0 = time.time()
    for stage in a.stages:
        dataset, geometry = STAGES[stage]
        print(f"\n===== STAGE {stage} ({nw} workers) =====", flush=True)
        slices = _load_slices(dataset, a.root, a.slices)
        csvp = outdir / f"benchmark_{dataset}_{geometry}.csv"
        done = _done_keys(csvp)
        tasks = []
        for tag, idx, ref in slices:
            for inc in a.incidents:
                for seed in range(a.realizations):
                    missing = [m for m in method_names
                               if (dataset, geometry, float(inc), str(tag),
                                   int(idx), int(seed), m) not in done]
                    if missing:
                        tasks.append((dataset, geometry, str(tag), int(idx),
                                      ref, float(inc), int(seed), a.iters,
                                      a.select_iters, not a.no_supplement,
                                      missing))
        total = len(tasks)
        print(f"slices={len(slices)} incidents={a.incidents} "
              f"seeds={a.realizations} -> {total} cells to run "
              f"({len(done)} rows already done)", flush=True)
        if total == 0:
            print("stage already complete — skipping", flush=True)
            continue
        new_file = not csvp.exists()
        with open(csvp, "a", newline="") as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(header)
                fh.flush()
            t0, ncells = time.time(), 0
            with Pool(nw, initializer=_init_worker,
                      initargs=(geometry,)) as pool:
                for rows in pool.imap_unordered(_run_cell, tasks, chunksize=1):
                    for r in rows:
                        w.writerow(r)
                    fh.flush()
                    os.fsync(fh.fileno())
                    ncells += 1
                    if ncells % 10 == 0 or ncells == total:
                        el = time.time() - t0
                        eta = el / ncells * (total - ncells)
                        print(f"  [{stage}] {ncells}/{total} cells | "
                              f"elapsed {el/60:.1f} min | ETA {eta/60:.1f} min",
                              flush=True)
        print(f"stage {stage} DONE -> {csvp}", flush=True)
    print(f"\nCAMPAIGN COMPLETE in {(time.time()-grand_t0)/3600:.2f} h",
          flush=True)
