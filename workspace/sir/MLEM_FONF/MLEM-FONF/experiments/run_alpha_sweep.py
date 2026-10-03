"""Fig.-4a-style fractional-order ablation: PSNR vs alpha."""
import argparse, pathlib
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import setup, N
from mlem_fonf import mlem_fonf
from mlem_fonf.metrics import psnr

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--realizations", type=int, default=3)
    args = ap.parse_args()
    alphas = np.arange(0.2, 1.0, 0.1)
    means, stds = [], []
    for a in alphas:
        vals = []
        for r in range(args.realizations):
            A, ref, g = setup("shepp_logan", seed=r)
            vals.append(psnr(ref, mlem_fonf(g, A, N, args.iters, alpha=float(a), lam_dt=0.95)))
        means.append(np.mean(vals)); stds.append(np.std(vals))
        print(f"alpha={a:.1f}: PSNR {means[-1]:.2f} +/- {stds[-1]:.2f}")
    plt.figure(figsize=(4.2, 3))
    plt.errorbar(alphas, means, yerr=stds, marker="o", ms=4)
    plt.xlabel(r"fractional order $\alpha$"); plt.ylabel("PSNR (dB)")
    plt.title("Sensitivity to fractional order"); plt.tight_layout()
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "alpha_sweep.png"
    plt.savefig(out, dpi=160); print("saved:", out)
