"""Table-II-style comparison: all methods, mean over noise realizations."""
import argparse, csv, time, pathlib
import numpy as np
from common import setup, METHODS, N
from mlem_fonf import all_metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phantom", default="shepp_logan")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--realizations", type=int, default=10)
    args = ap.parse_args()
    out = pathlib.Path(__file__).resolve().parents[1] / "results"
    acc = {m: [] for m in METHODS}
    for r in range(args.realizations):
        A, ref, g = setup(args.phantom, seed=r)
        for name, fn in METHODS.items():
            t = time.time()
            met = all_metrics(ref, fn(g, A, args.iters))
            met["time_s"] = time.time() - t
            acc[name].append(met)
            print(f"[r{r}] {name:<13} PSNR {met['PSNR']:.2f}  SNR {met['SNR']:.2f}  ({met['time_s']:.0f}s)")
    keys = ["SNR", "MSE", "RMSE", "PSNR", "CP", "MSSIM", "time_s"]
    with open(out / f"comparison_{args.phantom}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method"] + [f"{k}_{s}" for k in keys for s in ("mean", "std")])
        print(f"\n{'method':<14}" + "".join(f"{k:>9}" for k in keys))
        for name, ms in acc.items():
            row = [name]
            line = f"{name:<14}"
            for k in keys:
                v = np.array([m[k] for m in ms]); row += [v.mean(), v.std()]
                line += f"{v.mean():9.4f}" if k in ("MSE", "RMSE") else f"{v.mean():9.3f}"
            w.writerow(row); print(line)
    print("saved:", out / f"comparison_{args.phantom}.csv")
