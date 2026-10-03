"""Fig.-5-style convergence: PSNR/MSSIM vs iteration."""
import argparse, pathlib
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import setup, METHODS
from mlem_fonf.metrics import psnr, mssim

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phantom", default="shepp_logan")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--methods", nargs="+", default=["MLEM", "MLEM+TV", "PnP-ADMM(TV)", "MLEM+FONF"])
    args = ap.parse_args()
    A, ref, g = setup(args.phantom, seed=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    for name in args.methods:
        ps, ss, its = [], [], []
        def cb(k, f):
            if (k + 1) % args.every == 0:
                its.append(k + 1); ps.append(psnr(ref, f)); ss.append(mssim(ref, f))
        METHODS[name](g, A, args.iters, cb)
        a1.plot(its, ps, label=name); a2.plot(its, ss, label=name)
    a1.set_xlabel("iteration"); a1.set_ylabel("PSNR (dB)"); a1.legend(fontsize=7)
    a2.set_xlabel("iteration"); a2.set_ylabel("MSSIM")
    fig.suptitle(f"Convergence ({args.phantom})"); fig.tight_layout()
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / f"convergence_{args.phantom}.png"
    fig.savefig(out, dpi=160); print("saved:", out)
