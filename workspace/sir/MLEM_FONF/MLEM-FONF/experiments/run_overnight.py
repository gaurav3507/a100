"""Unattended overnight pass: ALL classical baselines, BOTH datasets.

Night-1 plan (fits ~8-9 h): Mayo (parallel) + LoDoPaB (parallel), 20 slices,
seeds {0,1}, supplement tier included, resume-safe (re-running after any
interruption skips completed cells). A third seed and the fan-geometry pass
are appended on later nights with the same command pattern.

    nohup python experiments/run_overnight.py --root /media/ant-pc/HDD2/datasets \
        > results/overnight1.log 2>&1 &
"""
import argparse, pathlib, subprocess, sys, time, csv
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--slices", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--datasets", nargs="+", default=["mayo", "lodopab"])
    ap.add_argument("--geometry", default="parallel")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE.parents[0] / "src"))
    from mlem_fonf.version import banner
    banner("run_overnight")
    t00 = time.time()
    for ds in a.datasets:
        print(f"\n===== JOB: {ds} / {a.geometry} | slices={a.slices} "
              f"seeds={a.seeds} =====", flush=True)
        cmd = [sys.executable, str(HERE / "run_benchmark.py"),
               "--dataset", ds, "--root", a.root, "--geometry", a.geometry,
               "--slices", str(a.slices), "--seeds", *map(str, a.seeds),
               "--with-supplement", "--append", "--resume"]
        rc = subprocess.call(cmd)
        print(f"===== JOB {ds} exit={rc} | elapsed total "
              f"{(time.time()-t00)/3600:.2f} h =====", flush=True)
    # morning summary: mean PSNR/MSSIM per dataset x incident x method
    print("\n================ MORNING SUMMARY ================")
    outdir = HERE.parents[0] / "results"
    lines = []
    for ds in a.datasets:
        p = outdir / f"benchmark_{ds}_{a.geometry}.csv"
        if not p.exists():
            continue
        acc = defaultdict(list)
        with open(p) as fh:
            rd = csv.DictReader(fh)
            for row in rd:
                acc[(row["dataset"], float(row["incident"]),
                     row["method"])].append((float(row["PSNR"]),
                                             float(row["MSSIM"])))
        for (d, inc, m), v in sorted(acc.items()):
            ps = sum(x[0] for x in v) / len(v)
            ss = sum(x[1] for x in v) / len(v)
            lines.append(f"{d:<8} inc={inc:>6.0f} {m:<14} "
                         f"PSNR {ps:6.2f}  MSSIM {ss:.3f}  (n={len(v)})")
    summary = "\n".join(lines)
    print(summary)
    (outdir / "overnight_summary.txt").write_text(summary + "\n")
    print(f"\ntotal wall time: {(time.time()-t00)/3600:.2f} h")
