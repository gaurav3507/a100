"""Run the full benchmark on a user-supplied real CT slice (e.g., Mayo LDCT).

The Mayo / AAPM Low-Dose CT Grand Challenge data are distributed under a data
use agreement: do NOT commit patient images to a public repository; point this
script at your locally licensed copy instead.

Known behaviour on textured clinical anatomy (documented smoke test):
  * the optimal fractional order sits higher than on phantoms (use --alpha,
    cf. the paper's Fig. 4a monotonicity);
  * TV-family priors can score higher PSNR/MSSIM by over-smoothing while
    removing anatomy: inspect images, not only metrics;
  * the isotropic radial-background assumption of the notch-selection rule can
    false-trigger on strongly anisotropic spectra (spine/ribs); pass
    --no-notch to force K = 0 (the broadband regime of the paper).
"""
import argparse, pathlib, sys
import numpy as np
from PIL import Image
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from mlem_fonf import (build_system_matrix, poisson_sinogram, mlem, mlem_fonf,
                       mlem_spatial, pnp_admm_tv, all_metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to a grayscale slice (PNG/anything PIL reads)")
    ap.add_argument("--incident", type=float, default=600.0)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ref = np.asarray(Image.open(args.image).convert("L")).astype(float)
    if ref.shape != (256, 256):
        from skimage.transform import resize
        ref = resize(ref, (256, 256), anti_aliasing=True)
    ref /= max(ref.max(), 1e-12)
    A = build_system_matrix(256, 64, 90)
    g = poisson_sinogram(A, ref, args.incident, np.random.default_rng(args.seed))
    runs = [("MLEM", lambda: mlem(g, A, 256, args.iters)),
            ("MLEM+TV", lambda: mlem_spatial(g, A, 256, "tv", args.iters)),
            ("MLEM+AD", lambda: mlem_spatial(g, A, 256, "ad", args.iters)),
            ("PnP-ADMM(TV)", lambda: pnp_admm_tv(g, A, 256, args.iters)),
            (f"MLEM+FONF(a={args.alpha})",
             lambda: mlem_fonf(g, A, 256, args.iters, alpha=args.alpha, lam_dt=0.95))]
    print(f"{'method':<20} {'SNR':>7} {'RMSE':>8} {'PSNR':>7} {'CP':>7} {'MSSIM':>7}")
    for nm, fn in runs:
        m = all_metrics(ref, fn())
        print(f"{nm:<20} {m['SNR']:7.2f} {m['RMSE']:8.4f} {m['PSNR']:7.2f} "
              f"{m['CP']:7.3f} {m['MSSIM']:7.3f}")
