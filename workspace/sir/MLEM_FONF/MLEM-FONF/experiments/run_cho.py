"""Task-based evaluation (CHO) for every reconstruction method.

SKE/BKS detection of a low-contrast lesion inserted into real anatomy:
for each (method, dose, lesion contrast x size) we reconstruct N
signal-present and N signal-absent realizations, extract the ROI, project on
Laguerre-Gauss channels and report split-half d' with a bootstrap CI.

  python experiments/run_cho.py --root <datasets-root> --workers 32 \
      --realizations 100 --slices 4
"""
import argparse, csv, itertools, os, pathlib, sys, time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np

_G = {}


def _init(geometry, refs, lesion, stripe=0.0, obj_stripe=0.0):
    """Workers hold the slices and lesion ONCE (tasks carry only indices)."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    from mlem_fonf import build_system_matrix, build_fanbeam_matrix
    _G["A"] = (build_system_matrix(256, 64, 90) if geometry == "parallel"
               else build_fanbeam_matrix(256, 64, 90, sod=600, odd=600))
    _G["geom"] = geometry
    _G["refs"] = refs
    _G["les"] = lesion
    _G["stripe"] = stripe
    _G["obj_stripe"] = obj_stripe


def _one(task):
    """One (method, dose, lesion, slice, seed, present/absent) reconstruction ROI.

    Failures are isolated: a NaN ROI is returned and the campaign continues."""
    (method, inc, si, present, center, roi, seed, iters, geometry, cell,
     sel_iters) = task
    ref, les = _G["refs"][si], _G["les"]
    A = _G["A"]
    try:
        return _one_inner(task, ref, les, A, method, inc, si, present, center, roi,
                          seed, iters, geometry, cell, _G.get("stripe", 0.0),
                          _G.get("obj_stripe", 0.0), sel_iters)
    except Exception as exc:                      # isolate: log, return NaN ROI
        print(f"!!! CHO cell failed {method}@{inc} si={si} seed={seed}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return (method, inc, present, si, cell,
                np.full((roi, roi), np.nan, dtype=np.float32))


def _detector_gain(n_bins, n_angles, amp, seed):
    """Multiplicative detector-gain drift: a few miscalibrated channels give
    stripes in the sinogram (rings after reconstruction)."""
    rng = np.random.default_rng(seed)
    g = np.ones(n_bins)
    bad = rng.choice(n_bins, size=max(1, n_bins // 16), replace=False)
    g[bad] = 1.0 + amp * rng.choice([-1.0, 1.0], size=len(bad))
    return np.tile(g, (n_angles, 1)).ravel()


def _object_stripe(n, amp, seed):
    """One realization of a narrowband directional interference pattern:
    frequency drawn inside the measurable band (r <= n_bins/2 - 2 at the
    campaign geometry) and a random phase, so the artifact is part of the
    BACKGROUND STATISTICS the observer must contend with."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(8.0, 28.0) / n          # cycles/pixel, in-band
    th = rng.uniform(0, np.pi)
    ph = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:n, 0:n]
    return amp * np.sin(2 * np.pi * r * (np.cos(th) * xx + np.sin(th) * yy) + ph)


def _one_inner(task, ref, les, A, method, inc, si, present, center, roi, seed,
               iters, geometry, cell, stripe=0.0, obj_stripe=0.0, sel_iters=None):
    from mlem_fonf import (poisson_sinogram, mlem, mlem_spatial, pnp_admm_tv,
                           reconstruct)
    from mlem_fonf.fbp import fbp
    from mlem_fonf.cho import extract_roi
    tgt = ref + les if present else ref
    if obj_stripe > 0:
        # the SAME artifact realization for both classes of a pair (seed//10),
        # so it is a background nuisance, not a cue for the signal
        tgt = np.clip(tgt + _object_stripe(tgt.shape[0], obj_stripe, seed // 10), 0, None)
    g, counts = poisson_sinogram(A, tgt, inc, np.random.default_rng(seed),
                                 return_counts=True)
    if stripe > 0:                     # same gain pattern for both classes
        gain = _detector_gain(64, 90, stripe, 12345)
        counts = counts * gain
        g = g * gain
    scale = (A @ tgt.ravel()).max() / inc
    if method == "FBP":
        rec = fbp(g, A, 256, 64, 90, geometry)
    elif method == "MLEM":
        rec = mlem(g, A, 256, iters)
    elif method == "MLEM+TV":
        rec = mlem_spatial(g, A, 256, "tv", iters)
    elif method == "MLEM+AD":
        rec = mlem_spatial(g, A, 256, "ad", iters)
    elif method == "PnP-ADMM(TV)":
        rec = pnp_admm_tv(g, A, 256, iters)
    elif method == "A-FONF":
        kw = {"select_iter": sel_iters} if sel_iters else {}   # None -> shipped default (800)
        rec = reconstruct(counts, scale, A, 256, n_bins=64, n_iter=iters, **kw)[0]
    else:
        raise ValueError(method)
    return (method, inc, present, si, cell, extract_roi(np.asarray(rec), center, roi))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--geometry", default="parallel", choices=["parallel", "fan"])
    ap.add_argument("--methods", nargs="+",
                    default=["FBP", "MLEM", "MLEM+TV", "MLEM+AD", "PnP-ADMM(TV)", "A-FONF"])
    ap.add_argument("--incidents", type=float, nargs="+", default=[300, 1500, 3000])
    ap.add_argument("--contrasts", type=float, nargs="+", default=[0.03, 0.06])
    ap.add_argument("--radii", type=float, nargs="+", default=[3.0, 6.0])
    ap.add_argument("--realizations", type=int, default=100)
    ap.add_argument("--slices", type=int, default=4)
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--roi", type=int, default=48)
    ap.add_argument("--object-stripe", type=float, default=0.0,
                    help="amplitude of a RANDOM in-band directional stripe added\n"
                         "to the acquisition (frequency and phase redrawn per\n"
                         "realization): the narrowband artifact class the notch\n"
                         "rule targets and the 263/263 gate validated")
    ap.add_argument("--stripe", type=float, default=0.0,
                    help="detector-gain drift amplitude (0 = clean acquisition); "
                         "the artifact-present task our notch rule targets")
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--channel-width", type=float, default=None,
                    help="LG channel width a (default: 2 x lesion radius, "
                         "the standard signal-matched choice)")
    ap.add_argument("--cache-dir", default=None,
                    help="ROI cache location (default: derived from --out); "
                         "point at an existing cache to re-score it")
    ap.add_argument("--stats-only", action="store_true",
                    help="recompute d' from cached ROIs without reconstructing")
    ap.add_argument("--zero-mean-channels", action="store_true",
                    help="ablation only: non-standard zero-mean channels")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--select-iters", type=int, default=None,
                    help="iterations for the Morozov selection stage; default "
                         "keeps the shipped pipeline value (800), which costs "
                         "6x that per A-FONF reconstruction")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from mlem_fonf.version import banner
    from mlem_fonf import data as D
    from mlem_fonf.cho import (laguerre_gauss_channels, gaussian_lesion,
                               channel_responses, bootstrap_ci, auc_from_dprime)
    banner("run_cho")
    nw = a.workers or max(2, min(cpu_count() - 2, 64))
    refs = [im for _, _, im in D.mayo_slices(a.root, a.split, "full", every=8,
                                             limit=a.slices)]
    assert refs, f"no slices found for split={a.split} under {a.root}"
    center = (150, 128)                       # soft-tissue location, off-centre
    tag = (f"cho_{a.geometry}" + (f"_stripe{a.stripe:g}" if a.stripe > 0 else "")
           + (f"_objstripe{a.object_stripe:g}" if a.object_stripe > 0 else ""))
    out = pathlib.Path(a.out) if a.out else (pathlib.Path(__file__).resolve().parents[1]
                                             / "results" / f"{tag}.csv")
    lock = pathlib.Path(str(out).replace(".csv", "")) if not a.cache_dir else pathlib.Path(a.cache_dir)
    lock.mkdir(parents=True, exist_ok=True)
    lock = lock / ".campaign.lock"
    if lock.exists() and not a.stats_only:
        try:
            other = int(lock.read_text().split()[0])
            alive = pathlib.Path(f"/proc/{other}").exists()
        except Exception:
            alive = False
        if alive:
            raise SystemExit(
                f"another campaign (pid {other}) is writing this cache: {lock.parent}\n"
                "run it with a different --out/--cache-dir, or wait for it to finish "
                "(concurrent writers corrupt the ROI cache).")
        print(f"[note] stale lock from pid {other if 'other' in dir() else '?'} removed",
              flush=True)
    if not a.stats_only:
        lock.write_text(f"{os.getpid()}\n")
        import atexit
        atexit.register(lambda: lock.exists() and lock.unlink())
    header = ["geometry", "method", "incident", "contrast", "radius_px", "n_pairs",
              "d_prime", "ci_lo", "ci_hi", "auc"]
    rows, done_blocks = [header], set()
    if out.exists():                      # resume: keep finished lesion blocks
        import csv as _csv
        with open(out) as fh:
            for r in _csv.DictReader(fh):
                rows.append([r[k] for k in header])
                # a block only counts as done if it already has at least as
                # many pairs as the current request (otherwise recompute)
                if int(r["n_pairs"]) >= a.slices * a.realizations:
                    done_blocks.add((float(r["contrast"]), float(r["radius_px"])))
                else:
                    done_blocks.discard((float(r["contrast"]), float(r["radius_px"])))
        print(f"resume: {len(done_blocks)} lesion block(s) already complete", flush=True)
    t0 = time.time()
    for contrast, radius in itertools.product(a.contrasts, a.radii):
        if (contrast, radius) in done_blocks:
            # a finished CSV block is only trustworthy if the ROI cache still
            # holds every requested method (cells can be deleted deliberately,
            # e.g. after an algorithm fix)
            cdir = ((pathlib.Path(a.cache_dir) if a.cache_dir
                     else pathlib.Path(str(out).replace(".csv", "")))
                    / f"c{contrast}_r{radius}")
            missing = [m for m in a.methods
                       if not any(cdir.glob(f"{m.replace('/', '_')}_*_1.npy"))]
            if missing:
                print(f"[c={contrast}, r={radius}] CSV says done but cache is "
                      f"missing {missing} — recomputing", flush=True)
                done_blocks.discard((contrast, radius))
            else:
                print(f"[c={contrast}, r={radius}] already done — skipping", flush=True)
                continue
        rows = [r for r in rows if r is header or not (
            str(r[3]) == str(contrast) and str(r[4]) == str(radius))]
        les = gaussian_lesion(256, radius_px=radius, contrast=contrast, center=center)
        ch = laguerre_gauss_channels(a.roi, a.channels,
                                     a_u=a.channel_width or max(4.0, 2.0 * radius),
                                     zero_mean=a.zero_mean_channels)
        cache = (pathlib.Path(a.cache_dir) if a.cache_dir
                 else pathlib.Path(str(out).replace(".csv", ""))) / f"c{contrast}_r{radius}"
        cache.mkdir(parents=True, exist_ok=True)
        idx_path = cache / "_done.json"
        import json as _json
        done_cells = set(map(tuple, _json.load(open(idx_path)))) if idx_path.exists() else set()
        if done_cells:
            print(f"  cache: {len(done_cells)} cells already computed", flush=True)
        store, new_cells = {}, {}

        def _flush(key, arr_list):
            """Append ROIs AND their cell ids so the ensemble can be sorted
            deterministically (imap_unordered returns in completion order)."""
            m, inc, pres = key
            stem = f"{m.replace('/', '_')}_{int(inc)}_{int(pres)}"
            f, fk = cache / f"{stem}.npy", cache / f"{stem}_keys.npy"
            cells = new_cells.pop(key, [])
            keys_new = np.asarray([[c[2], c[3]] for c in cells], dtype=np.int64)
            old = np.load(f) if f.exists() else np.empty((0, a.roi, a.roi), dtype=np.float32)
            oldk = np.load(fk) if fk.exists() else np.empty((0, 2), dtype=np.int64)
            np.save(f, np.concatenate([old, np.asarray(arr_list, dtype=np.float32)]))
            np.save(fk, np.concatenate([oldk, keys_new]))
            done_cells.update(cells)
            _json.dump([list(c) for c in done_cells], open(idx_path, "w"))

        tasks = []
        expected = len(a.methods) * len(a.incidents) * len(refs) * a.realizations * 2
        for method in a.methods:
            for inc in a.incidents:
                for si, ref in enumerate(refs):
                    for k in range(a.realizations):
                        for present in (True, False):
                            # unique, reproducible seed per (method-agnostic) cell:
                            # noise depends only on (slice, realization, class)
                            seed = 7_000_000 + si * 100_000 + k * 10 + int(present)
                            cell = (method, float(inc), si, k, int(present))
                            if cell in done_cells:
                                continue
                            tasks.append((method, inc, si, present, center,
                                          a.roi, seed, a.iters, a.geometry, cell,
                                          a.select_iters))
        print(f"[c={contrast}, r={radius}] {len(tasks)} of {expected} reconstructions "
              f"to run on {nw} workers", flush=True)
        # ROIs are streamed to a memmap-backed cache per (method, inc, class):
        # the parent never holds the full ensemble, and a crash keeps progress.
        if a.stats_only:
            tasks = []
            print("  --stats-only: recomputing d' from cached ROIs", flush=True)
        if not tasks:
            print("  all cells cached — computing statistics only", flush=True)
        last_flush = [time.time()]
        with Pool(nw, initializer=_init, initargs=(a.geometry, refs, les, a.stripe, a.object_stripe),
                  maxtasksperchild=200) as pool:
            done = 0
            for m, inc, pres, si, cell, roi_img in pool.imap_unordered(_one, tasks, chunksize=4):
                store.setdefault((m, inc, pres), []).append(roi_img.astype(np.float32))
                new_cells.setdefault((m, inc, pres), []).append(cell)
                done += 1
                now = time.time()
                if (len(store[(m, inc, pres)]) >= 50
                        or now - last_flush[0] > 60):            # size OR time
                    for key in list(store):
                        if store[key]:
                            _flush(key, store.pop(key))
                    last_flush[0] = now
                if done % 200 == 0:
                    el = time.time() - t0
                    print(f"  {done}/{len(tasks)} | {el/60:.1f} min", flush=True)
        for key, lst in list(store.items()):
            _flush(key, lst)
        store = {}
        for method in a.methods:
            for inc in a.incidents:
                fp = cache / f"{method.replace('/', '_')}_{int(inc)}_1.npy"
                fa = cache / f"{method.replace('/', '_')}_{int(inc)}_0.npy"
                if not (fp.exists() and fa.exists()):
                    print(f"  [warn] missing ROIs for {method}@{inc:g} — skipped")
                    continue
                P, Ab = np.load(fp), np.load(fa)
                kp = cache / f"{method.replace('/', '_')}_{int(inc)}_1_keys.npy"
                ka = cache / f"{method.replace('/', '_')}_{int(inc)}_0_keys.npy"
                if kp.exists() and ka.exists():          # deterministic order
                    P = P[np.lexsort(np.load(kp).T[::-1])]
                    Ab = Ab[np.lexsort(np.load(ka).T[::-1])]
                okP = np.isfinite(P).all(axis=(1, 2))
                okA = np.isfinite(Ab).all(axis=(1, 2))
                if not (okP.all() and okA.all()):
                    print(f"  [warn] dropping {int((~okP).sum() + (~okA).sum())} "
                          f"failed ROIs for {method}@{inc:g}")
                P, Ab = P[okP], Ab[okA]
                if min(len(P), len(Ab)) < 10:
                    print(f"  [warn] too few usable pairs for {method}@{inc:g}")
                    continue
                if len(P) < 30:
                    print(f"  [warn] only {len(P)} pairs for {method}@{inc:g} — "
                          "d' estimates below ~30 pairs are unreliable", flush=True)
                d, lo, hi = bootstrap_ci(channel_responses(P, ch),
                                         channel_responses(Ab, ch), n_boot=400)
                rows.append([a.geometry, method, inc, contrast, radius, len(P),
                             f"{d:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                             f"{auc_from_dprime(d):.4f}"])
                print(f"  {method:<14} inc={inc:<6g} c={contrast} r={radius}: "
                      f"d'={d:.2f} [{lo:.2f},{hi:.2f}]", flush=True)
        with open(out, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
    print(f"saved: {out}  ({time.time()-t0:.0f}s)")
