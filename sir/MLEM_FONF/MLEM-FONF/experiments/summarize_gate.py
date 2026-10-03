"""Summarize the realistic-geometry gate: per-dose mean +/- 95% CI per method
and a paired Wilcoxon of A-FONF against each baseline (pairs matched on
slice x seed).

  python experiments/summarize_gate.py [results/benchmark_realistic_fan.csv]
"""
import csv, math, pathlib, sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.stats import wilcoxon
from mlem_fonf.version import banner


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def main(path):
    banner("summarize_gate")
    rows = list(csv.DictReader(open(path)))
    if not rows:
        sys.exit("empty csv")
    geo = f"{rows[0]['n']}^2, {rows[0]['bins']} bins, {rows[0]['views']} views"
    incs = sorted({float(r["incident"]) for r in rows})
    print(f"\nrealistic fan geometry: {geo} | {len(rows)} rows")
    for inc in incs:
        sel = [r for r in rows if float(r["incident"]) == inc]
        by = defaultdict(dict)
        for r in sel:
            try:
                p = float(r["PSNR"])
            except ValueError:
                continue
            if math.isfinite(p):
                by[r["method"]][(r["patient"], r["slice"], r["seed"])] = (p, float(r["MSSIM"]))
        if not by:
            continue
        ref = by.get("A-FONF", {})
        order = sorted(by, key=lambda m: -np.mean([v[0] for v in by[m].values()]))
        print(f"\n=== {inc:g} photons/ray ===")
        print(f"{'method':<22} {'PSNR (mean±95%CI)':>22} {'SSIM':>14} {'n':>4} "
              f"{'vs A-FONF':>14}")
        for m in order:
            ps = np.array([v[0] for v in by[m].values()])
            ss = np.array([v[1] for v in by[m].values()])
            half = 1.96 * ps.std(ddof=1) / math.sqrt(len(ps)) if len(ps) > 1 else 0.0
            if m == "A-FONF" or not ref:
                verdict = "—"
            else:
                common = [k for k in by[m] if k in ref]
                d = [ref[k][0] - by[m][k][0] for k in common]
                if len(d) >= 6 and any(abs(x) > 0 for x in d):
                    pv = wilcoxon(d).pvalue
                    verdict = f"{np.mean(d):+.2f} dB {stars(pv)}"
                else:
                    verdict = f"n={len(d)}"
            print(f"{m:<22} {ps.mean():10.2f} ± {half:<5.2f}{'':>5} "
                  f"{ss.mean():8.3f}{'':>5} {len(ps):4d} {verdict:>14}")
    # verdict line
    tv = [m for m in {r["method"] for r in rows} if m.startswith("MLEM+TV")]
    if tv and "A-FONF" in {r["method"] for r in rows}:
        gaps = []
        for inc in incs:
            a = [float(r["PSNR"]) for r in rows
                 if float(r["incident"]) == inc and r["method"] == "A-FONF"]
            b = [float(r["PSNR"]) for r in rows
                 if float(r["incident"]) == inc and r["method"].startswith("MLEM+TV")]
            if a and b:
                gaps.append(np.mean(a) - np.mean(b))
        print(f"\nGATE: A-FONF vs tuned TV across doses: "
              f"{', '.join(f'{g:+.2f}' for g in gaps)} dB")
        print("  -> " + ("PASS (advantage holds at realistic geometry)"
                         if min(gaps) > 0.3 else
                         "REVIEW (advantage does not hold; JBHI route)"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else str(pathlib.Path(__file__).resolve().parents[1]
                  / "results" / "benchmark_realistic_fan.csv"))
