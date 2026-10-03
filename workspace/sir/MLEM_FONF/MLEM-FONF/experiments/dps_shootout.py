"""Pick the official diffusion checkpoint: run DPS from each candidate on a
few held-out slices and report PSNR/MSSIM + runtime. Winner is then renamed
to checkpoints/diffusion_3000.pt (the name the benchmark runner looks up).

  python experiments/dps_shootout.py --root /media/ant-pc/HDD2/datasets \
      --ckpts checkpoints/diffusion_tiny.pt checkpoints/diffusion_L_3000.pt \
              checkpoints/diffusion_XL_3000.pt --slices 3 --steps 200
"""
import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "training"))
import numpy as np
from mlem_fonf import build_system_matrix, poisson_sinogram, all_metrics
from mlem_fonf.version import banner

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--slices", type=int, default=3)
    ap.add_argument("--incident", type=float, default=3000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--nbins", type=int, default=64)
    ap.add_argument("--nang", type=int, default=90)
    ap.add_argument("--zeta", type=float, nargs="+", default=[1.0],
                    help="guidance scale(s); sweep on --split val, freeze, then test")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    a = ap.parse_args()
    banner("dps_shootout")
    import torch
    from eval_learned import dps_infer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = build_system_matrix(a.n, a.nbins, a.nang)
    if a.root:
        from mlem_fonf import data as D
        refs = [im for _, _, im in D.mayo_slices(a.root, a.split, "full",
                                                  every=4, limit=a.slices)]
    else:
        from mlem_fonf.phantoms import shepp_logan_mod
        refs = [shepp_logan_mod(a.n)] * a.slices
    print(f"split={a.split}  {'checkpoint':<28} {'zeta':>5} {'PSNR':>7} {'MSSIM':>7} {'s/slice':>8}")
    results = {}
    for ck in a.ckpts:
      for zeta in a.zeta:
        ps, ss, t0 = [], [], time.time()
        for i, ref in enumerate(refs):
            g, counts = poisson_sinogram(A, ref, a.incident,
                                         np.random.default_rng(i), return_counts=True)
            scale = (A @ ref.ravel()).max() / a.incident
            rec = dps_infer(ck, g, A, scale, n=a.n, steps=a.steps, seed=i,
                            zeta=zeta, device=dev)
            m = all_metrics(ref, np.clip(rec, 0, None))
            ps.append(m["PSNR"]); ss.append(m["MSSIM"])
        dt = (time.time() - t0) / len(refs)
        results[(ck, zeta)] = np.mean(ps)
        print(f"{'':<11}{pathlib.Path(ck).name:<28} {zeta:5.2f} {np.mean(ps):7.2f} {np.mean(ss):7.3f} {dt:8.1f}", flush=True)
    win = max(results, key=results.get)
    print(f"\nWINNER: {win[0]} @ zeta={win[1]}  -> cp it to checkpoints/diffusion_3000.pt; use this zeta in the grid")
