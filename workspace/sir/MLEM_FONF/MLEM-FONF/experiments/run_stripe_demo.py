"""Fig.-6-style multi-notch proof of mechanism on a synthetic stripe."""
import pathlib
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import setup, N
from mlem_fonf import mlem_fonf, select_notches, h_fonf, apply_filter
from mlem_fonf.metrics import psnr

if __name__ == "__main__":
    A, ref, g = setup("shepp_logan", seed=0)
    clean = mlem_fonf(g, A, N, 400, alpha=0.5, lam_dt=0.95)          # clean recon
    yy, xx = np.mgrid[0:N, 0:N]
    stripe = clean + 0.12 * np.sin(2 * np.pi * (0.11 * xx + 0.05 * yy))
    notches = select_notches(stripe)
    print(f"detected K = {len(notches)} notch(es):",
          [f"({a:.2f},{b:.2f},s={c:.3f})" for a, b, c in notches])
    H = h_fonf(N, alpha=0.5, notches=notches)
    rec = apply_filter(stripe, H)
    p0, p1 = psnr(clean, stripe), psnr(clean, rec)
    print(f"PSNR: corrupted {p0:.1f} dB -> filtered {p1:.1f} dB  (gain {p1-p0:+.1f} dB)")
    F = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(stripe))))
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for a, im, t in zip(ax, (stripe, F, rec),
                        (f"stripe input ({p0:.1f} dB)", "spectrum + detections",
                         f"FONF output ({p1:.1f} dB)")):
        a.imshow(im, cmap="gray"); a.set_title(t, fontsize=8); a.axis("off")
    for (wx, wy, s) in notches:
        for sx, sy in ((wx, wy), (-wx, -wy)):
            ax[1].plot(N/2 + sx*N/(2*np.pi), N/2 + sy*N/(2*np.pi), "r+", ms=8)
    fig.tight_layout()
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "stripe_demo.png"
    fig.savefig(out, dpi=160); print("saved:", out)
