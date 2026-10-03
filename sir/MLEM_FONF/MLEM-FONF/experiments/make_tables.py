"""Build JBHI-grade summary tables from benchmark CSVs.

Per (dataset, geometry): for every incident level, mean +/- std of PSNR and
MSSIM per method, median runtime, and PAIRED Wilcoxon signed-rank tests of
A-FONF against every other method (pairs matched on slice x seed).

  python experiments/make_tables.py results/benchmark_mayo_parallel.csv
"""
import csv
import math
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.stats import wilcoxon

from mlem_fonf.version import banner


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def main(csv_path):
    banner("make_tables")
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print("empty csv"); return
    dataset, geometry = rows[0]["dataset"], rows[0]["geometry"]
    incs = sorted({float(r["incident"]) for r in rows})
    methods = sorted({r["method"] for r in rows})
    print(f"\n{dataset} / {geometry} — {len(rows)} rows, "
          f"{len(incs)} incident levels, {len(methods)} methods")
    out_rows = [["incident", "method", "PSNR_mean", "PSNR_std", "MSSIM_mean",
                 "MSSIM_std", "time_median_s", "n", "wilcoxon_p_seedpairs",
                 "sig"]]
    dup_warned = 0
    for inc in incs:
        # per-method DEDUPED per-cell values: vals[key] = (psnr, mssim, t)
        agg = defaultdict(dict)
        for r in rows:
            if float(r["incident"]) != inc:
                continue
            m = r["method"]
            try:
                p, s, t = (float(r["PSNR"]), float(r["MSSIM"]),
                           float(r["time_s"]))
            except ValueError:
                continue
            if not (math.isfinite(p) and math.isfinite(s)):
                continue
            tag = r["slice_tag"]
            if dataset == "lodopab" and tag.isdigit():
                tag = "f" + tag                     # canonicalize legacy tags
            key = (tag, r["slice_idx"], r["seed"])
            if key in agg[m]:
                dup_warned += 1
            agg[m][key] = (p, s, t)            # dedup: last wins, everywhere
        if dup_warned:
            print(f"[warn] {dup_warned} duplicate cell rows deduped "
                  f"(last occurrence used consistently)")
        ref = dict(agg.get("A-FONF", {}))
        def _safe_wilcoxon(diffs):
            try:
                if len(diffs) >= 10 and any(abs(d) > 0 for d in diffs):
                    return float(wilcoxon(diffs).pvalue)
            except Exception:
                pass
            return None

        order = sorted(agg, key=lambda m: -np.mean([v[0] for v in
                                                    agg[m].values()]))
        print(f"\n=== incident {inc:g} ===")
        print(f"{'method':<14} {'PSNR':>14} {'MSSIM':>15} {'t(med)':>8} "
              f"{'p(seed-pairs)':>14} {'p(slice-lvl)':>13}")
        for m in order:
            vals = agg[m]
            ps = np.array([v[0] for v in vals.values()])
            ss_ = np.array([v[1] for v in vals.values()])
            ts = np.array([v[2] for v in vals.values()])
            mp, sp = ps.mean(), ps.std()
            ms_, sdev = ss_.mean(), ss_.std()
            tm = float(np.median(ts))
            if m == "A-FONF" or not ref:
                p1 = p2 = None
                t1 = t2 = "—"
            else:
                common = [k for k in vals if k in ref]
                diffs = [ref[k][0] - vals[k][0] for k in common]
                p1 = _safe_wilcoxon(diffs)
                # slice-level: average over seeds within each slice first
                bysl = defaultdict(list)
                for k in common:
                    bysl[(k[0], k[1])].append(ref[k][0] - vals[k][0])
                sdiffs = [float(np.mean(v)) for v in bysl.values()]
                p2 = _safe_wilcoxon(sdiffs)
                t1 = f"{p1:.1e} {stars(p1)}" if p1 is not None else "n/a"
                t2 = f"{p2:.1e} {stars(p2)}" if p2 is not None else "n/a"
            print(f"{m:<14} {mp:7.2f}±{sp:5.2f} {ms_:8.3f}±{sdev:5.3f} "
                  f"{tm:8.2f} {t1:>14} {t2:>13}")
            out_rows.append([inc, m, f"{mp:.4f}", f"{sp:.4f}", f"{ms_:.4f}",
                             f"{sdev:.4f}", f"{tm:.3f}", len(ps),
                             p1 if p1 is not None else "",
                             stars(p1) if p1 is not None else ""])
    stem = pathlib.Path(csv_path).stem
    new = stem.replace("benchmark", "table")
    if new == stem:
        new = stem + "_table"                  # never overwrite the input
    outp = pathlib.Path(csv_path).with_name(new + ".csv")
    with open(outp, "w", newline="") as fh:
        csv.writer(fh).writerows(out_rows)
    print(f"\nsaved: {outp}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "results/benchmark_mayo_parallel.csv")
