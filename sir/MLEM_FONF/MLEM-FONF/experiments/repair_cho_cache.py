"""Verify and repair a CHO ROI cache.

Checks each (method, dose, class) cache for duplicate cells (which happen if
two campaigns wrote the same cache concurrently) and for index/array length
mismatches; with --fix it rewrites deduplicated, canonically sorted arrays
and rebuilds the done-index.

  python experiments/repair_cho_cache.py results/cho_parallel_stripe0.25 [--fix]
"""
import argparse, json, pathlib, sys
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(a.cache)
    blocks = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    if not blocks:
        sys.exit(f"no lesion blocks under {root}")
    for blk in sorted(blocks):
        print(f"\n=== {blk.name} ===")
        idx = blk / "_done.json"
        done = set(map(tuple, json.load(open(idx)))) if idx.exists() else set()
        kept_cells = set()
        for f in sorted(blk.glob("*.npy")):
            if f.name.endswith("_keys.npy"):
                continue
            fk = f.with_name(f.stem + "_keys.npy")
            arr = np.load(f)
            if not fk.exists():
                print(f"  {f.name:<28} {len(arr):5d} ROIs | NO key file (cannot dedupe)")
                continue
            keys = np.load(fk)
            n_dup = len(keys) - len({tuple(k) for k in keys})
            status = "OK" if (n_dup == 0 and len(arr) == len(keys)) else "REPAIR"
            print(f"  {f.name:<28} {len(arr):5d} ROIs | {len(keys):5d} keys | "
                  f"duplicates {n_dup:4d} | {status}")
            if a.fix and (n_dup or len(arr) != len(keys)):
                m = min(len(arr), len(keys))
                arr, keys = arr[:m], keys[:m]
                seen, keep = set(), []
                for i, k in enumerate(map(tuple, keys)):
                    if k not in seen:
                        seen.add(k); keep.append(i)
                order = sorted(keep, key=lambda i: (keys[i][0], keys[i][1]))
                np.save(f, arr[order]); np.save(fk, keys[order])
                print(f"     -> repaired to {len(order)} unique ROIs")
            if a.fix:
                keys = np.load(fk)
                stem = f.stem.split("_")
                pres = int(stem[-1]); inc = float(stem[-2])
                meth = "_".join(stem[:-2])
                for si, k in keys:
                    kept_cells.add((meth, inc, int(si), int(k), pres))
        if a.fix:
            json.dump([list(c) for c in kept_cells], open(idx, "w"))
            print(f"  index rebuilt: {len(kept_cells)} cells")
