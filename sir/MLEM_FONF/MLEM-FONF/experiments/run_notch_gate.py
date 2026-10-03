"""Gate G2a: robust-notch v2 on the full Mayo L506 (1mm, full-dose) test set.

Per slice: (i) clean detection count K (target 0 everywhere);
(ii) stripe-injection true-positive test at amplitudes 0.10 and 0.05
(target: detected within 3 px of the injected frequency).

Run on the data machine:
    python experiments/run_notch_gate.py --root /media/ant-pc/HDD2/datasets
"""
import argparse, csv, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from mlem_fonf import data as D
from mlem_fonf.fonf import select_notches_v2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--coverage", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=None, help="smoke-run on first N slices")
    args = ap.parse_args()
    from mlem_fonf.version import banner
    banner("run_notch_gate")
    yy, xx = np.mgrid[0:256, 0:256]
    tgt = np.array([0.11, 0.05]) * 2 * np.pi
    rows, t0 = [], time.time()
    nK0 = tp10 = tp05 = n = 0
    loc_hist = {}
    for pat, i, img in D.mayo_slices(args.root, "test", "full", every=args.every, limit=args.limit):
        det = select_notches_v2(img, kappa=args.kappa, coverage_min=args.coverage)
        K = len(det)
        for (a, b, _) in det:
            key = (round(a * 256 / (2 * np.pi)), round(b * 256 / (2 * np.pi)))
            loc_hist[key] = loc_hist.get(key, 0) + 1
        hits = {}
        for amp in (0.10, 0.05):
            st = img + amp * np.sin(2 * np.pi * (0.11 * xx + 0.05 * yy))
            d2 = select_notches_v2(st, kappa=args.kappa, coverage_min=args.coverage)
            ok = bool(d2) and min(np.hypot(a - tgt[0], b - tgt[1]) * 256 / (2 * np.pi)
                                  for a, b, _ in d2) <= 3.0
            hits[amp] = ok
        n += 1; nK0 += (K == 0); tp10 += hits[0.10]; tp05 += hits[0.05]
        rows.append([pat, i, K, int(hits[0.10]), int(hits[0.05])])
        if n % 25 == 0:
            print(f"  {n} slices | K=0 so far {nK0}/{n} | TP .10 {tp10}/{n} | TP .05 {tp05}/{n}")
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "notch_gate_L506.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["patient", "slice", "K_clean", "tp_amp10", "tp_amp05"])
        w.writerows(rows)
    print("=" * 60)
    print(f"GATE G2a SUMMARY ({n} slices, {time.time()-t0:.0f}s)")
    print(f"  clean K=0: {nK0}/{n}   ({100*nK0/n:.1f}%)")
    print(f"  stripe TP amp=0.10: {tp10}/{n}   amp=0.05: {tp05}/{n}")
    if loc_hist:
        top = sorted(loc_hist.items(), key=lambda kv: -kv[1])[:5]
        print("  detection locations (px, count):", top)
    print(f"  csv: {out}")
    print("  PASS" if (nK0 == n and tp10 == n) else "  REVIEW NEEDED — paste this summary back")
