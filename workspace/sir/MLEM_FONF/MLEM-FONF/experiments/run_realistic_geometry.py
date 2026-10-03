"""TMI decision run: does the moderate-dose advantage hold in a realistic
clinical fan geometry (512^2, 736 bins, 576+ views)?

Uses the ASTRA GPU backend when available (required at this scale), falling
back to the certified sparse matrix only for small --n. Frozen A-FONF vs the
strongest classical baselines; identical protocol to the main campaign.

  python experiments/run_realistic_geometry.py --root /media/ant-pc/HDD2/datasets \
      --slices 5 --realizations 3
"""
import argparse, csv, os, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from mlem_fonf.version import banner


def astra_ops(n, n_bins, n_angles, sod, odd, gpu=True):
    """Return (forward, backward, shape) closures over an ASTRA fan operator."""
    import astra
    mag = (sod + odd) / sod
    det_width = mag * (n - 1) / (n_bins - 1)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    vol = astra.create_vol_geom(n, n)
    pg = astra.create_proj_geom("fanflat", det_width, n_bins, angles, sod, odd)
    pid = astra.create_projector("cuda" if gpu else "line_fanflat", pg, vol)
    W = astra.OpTomo(pid)

    def fwd(x):
        return np.asarray(W * x.ravel(), dtype=np.float64)

    def bwd(y):
        return np.asarray(W.T * y.ravel(), dtype=np.float64)

    return fwd, bwd, (n_bins * n_angles, n * n)


_FBP_SCALE = {}


def fbp_op(g, bwd, n, n_bins, n_angles, sod, odd, fwd=None):
    """Filtered backprojection through the operator adjoint (fan weighting).

    The global scale is calibrated ONCE on a synthetic phantom through the
    same operator and cached — never fitted to the test image (that would
    hand FBP oracle information)."""
    from mlem_fonf.fbp import _filter_sino
    mag = (sod + odd) / sod
    s_iso = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n_bins)
    w = sod / np.sqrt(sod ** 2 + (s_iso * mag) ** 2)
    def _bp(sino):
        gq = (sino.reshape(n_angles, n_bins) * w[None, :]).ravel()
        return bwd(_filter_sino(gq, n_bins, n_angles, True)).reshape(n, n)

    key = (n, n_bins, n_angles, float(sod), float(odd))
    if key not in _FBP_SCALE:
        from mlem_fonf.phantoms import shepp_logan_mod
        cal = shepp_logan_mod(n)
        r0 = _bp(np.asarray(fwd(cal.ravel())))
        den = float(np.sum(r0 * r0))
        _FBP_SCALE[key] = (float(np.sum(r0 * cal)) / den) if den > 0 else 1.0
    return np.maximum(_bp(g) * _FBP_SCALE[key], 0.0)


def mlem_op(g, fwd, bwd, n, n_iter, eps=1e-9):
    at1 = np.maximum(bwd(np.ones(g.size)), eps)
    f = np.ones(n * n)
    for _ in range(n_iter):
        f = f * (bwd(g / np.maximum(fwd(f), eps)) / at1)
        np.maximum(f, 0, out=f)
    return f.reshape(n, n)


def tv_op(g, fwd, bwd, n, n_iter, tv_weight=0.05, lam_dt=0.9, eps=1e-9):
    """MLEM+TV over operator closures — identical update to mlem_spatial's
    "tv" branch (TV in the relaxation slot), the strongest classical
    competitor at these doses."""
    from skimage.restoration import denoise_tv_chambolle
    at1 = np.maximum(bwd(np.ones(g.size)), eps)
    f = np.ones(n * n)
    for _ in range(n_iter):
        f_ml = f * (bwd(g / np.maximum(fwd(f), eps)) / at1)
        den = denoise_tv_chambolle(f_ml.reshape(n, n), weight=tv_weight)
        f = np.maximum((1 - lam_dt) * f_ml + lam_dt * den.ravel(), 0.0)
    return f.reshape(n, n)


def afonf_op(counts, scale, fwd, bwd, n, n_bins, n_iter, select_iter,
             grid=(0.2, 0.5, 0.8, 1.2, 1.6, 2.0), gamma=0.9, lam_dt=0.95):
    """Frozen Algorithm 1' over operator closures (Morozov + notches + damping)."""
    from mlem_fonf.fonf import h_fonf, apply_filter, select_notches_persistent
    from mlem_fonf.adaptive import poisson_deviance_per_ray
    g = counts * scale
    at1 = np.maximum(bwd(np.ones(g.size)), 1e-9)

    def run(alpha, iters, notches=(), gam=None, f0=None):
        """gam=None -> damped (gamma); gam=1.0 -> the undamped variant used by
        alpha_morozov's selection stage. f0 enables the warm start that
        pipeline.reconstruct uses for the persistence partner."""
        gam = gamma if gam is None else gam
        H = h_fonf(n, alpha=alpha, notches=notches)
        f = np.ones(n * n) if f0 is None else np.asarray(f0, float).ravel().copy()
        for _ in range(iters):
            # identical to algorithms.mlem_fonf_damped: multiplicative
            # correction with EXPONENT gamma (mirror-descent form), then the
            # spectral step in the relaxation slot
            ratio = bwd(g / np.maximum(fwd(f), 1e-9)) / at1
            f_ml = f * np.power(np.maximum(ratio, 1e-12), gam)
            f_hat = apply_filter(f_ml.reshape(n, n), H).ravel()
            f = np.maximum((1.0 - lam_dt) * f_ml + lam_dt * f_hat, 0.0)
        return f.reshape(n, n)

    devs, recs = {}, {}
    for a in grid:
        r = run(a, select_iter, gam=1.0)      # selection stage is UNDAMPED
        recs[a] = r
        devs[a] = poisson_deviance_per_ray(counts, fwd(r.ravel()) / scale)
    ok = [a for a in grid if devs[a] <= 1.0]
    a_sel = max(ok) if ok else min(devs, key=devs.get)
    f_sel = recs[a_sel]
    f_pair = run(a_sel, 20, f0=f_sel)     # warm start, as in pipeline.reconstruct
    notches = select_notches_persistent(f_sel, f_pair, r_max=n_bins // 2 - 2)
    return run(a_sel, n_iter, notches), a_sel, len(notches)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--bins", type=int, default=736)
    ap.add_argument("--views", type=int, default=576)
    ap.add_argument("--sod", type=float, default=1000.0)
    ap.add_argument("--odd", type=float, default=500.0)
    ap.add_argument("--incidents", type=float, nargs="+", default=None,
                    help="photons/ray; default = DOSE-MATCHED to the 64x90 "
                         "reference grid (same total photons), plus 2x and 8x")
    ap.add_argument("--slices", type=int, default=5)
    ap.add_argument("--realizations", type=int, default=3)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--select-iters", type=int, default=100)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--tv-weight", type=float, default=0.05,
                    help="TV strength for the baseline; tune it on the "
                         "validation patient for a fair comparison")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="subset of FBP MLEM MLEM+TV A-FONF")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--append", action="store_true",
                    help="append to the CSV instead of overwriting (per-dose "
                         "runs with different --tv-weight land in one file)")
    ap.add_argument("--out", default=None, help="CSV path override")
    a = ap.parse_args()
    banner("run_realistic_geometry")
    if a.incidents is None:
        # total photons = bins x views x incident; match the 64 x 90 x {60..3000}
        # reference so the comparison is a LOW-DOSE comparison, not a
        # high-flux one (736 x 576 carries 73x more rays than 64 x 90)
        base = 64 * 90
        f = base / (a.bins * a.views)
        a.incidents = [round(3000 * f), round(2 * 3000 * f), round(8 * 3000 * f)]
        print(f"dose-matched incidents (x{f:.4f} of the reference grid): "
              f"{a.incidents}", flush=True)
    from mlem_fonf import all_metrics, mlem_spatial, data as D
    from mlem_fonf.fbp import fbp
    try:
        fwd, bwd, shape = astra_ops(a.n, a.bins, a.views, a.sod, a.odd, gpu=not a.cpu)
    except ImportError:
        sys.exit("ASTRA not available: conda install -c astra-toolbox astra-toolbox "
                 "(needed for 512^2 geometry; the sparse backend cannot hold it)")
    # operator sanity before spending GPU hours: forward/backward shapes + finiteness
    _x = np.zeros(a.n * a.n); _x[a.n * a.n // 2 + a.n // 2] = 1.0
    _p = fwd(_x)
    assert _p.size == shape[0] and np.isfinite(_p).all() and _p.max() > 0, \
        "forward operator broken"
    _b = bwd(np.ones(shape[0]))
    assert _b.size == shape[1] and np.isfinite(_b).all() and _b.min() >= 0, \
        "backprojector broken"
    print(f"operator check: A delta -> {_p.max():.3g} peak, A^T 1 range "
          f"[{_b.min():.3g}, {_b.max():.3g}]", flush=True)
    print(f"geometry: {a.n}^2 volume, {a.bins} bins, {a.views} views, "
          f"sinogram {shape[0]:,} rays", flush=True)
    refs = [(p, i, im) for p, i, im in
            D.mayo_slices(a.root, a.split, "full", every=4, limit=a.slices, size=a.n)]
    assert refs, f"no slices found for split={a.split} under {a.root}"
    print(f"slices: {[(p, i) for p, i, _ in refs]}", flush=True)
    out = (pathlib.Path(a.out) if a.out else
           pathlib.Path(__file__).resolve().parents[1] / "results" / "benchmark_realistic_fan.csv")
    out.parent.mkdir(exist_ok=True)
    new_file = not (a.append and out.exists())
    fh_out = open(out, "w" if new_file else "a", newline="")
    wr = csv.writer(fh_out)
    if new_file:
        wr.writerow(["n", "bins", "views", "incident", "patient", "slice", "seed",
                     "method", "PSNR", "MSSIM", "time_s"])
    fh_out.flush()
    rows = []
    for inc in a.incidents:
        for pat, idx, ref in refs:
            proj = fwd(ref.ravel())
            scale = proj.max() / inc
            for seed in range(a.realizations):
                rng = np.random.default_rng(1000 + seed)
                counts = rng.poisson(np.maximum(proj / scale, 0)).astype(float)
                g = counts * scale
                _all = (
                    ("FBP", lambda: fbp_op(g, bwd, a.n, a.bins, a.views,
                                           a.sod, a.odd, fwd)),
                    ("MLEM", lambda: mlem_op(g, fwd, bwd, a.n, a.iters)),
                    ("MLEM+TV", lambda: tv_op(g, fwd, bwd, a.n, a.iters,
                                              tv_weight=a.tv_weight)),
                    ("A-FONF", lambda: afonf_op(counts, scale, fwd, bwd, a.n, a.bins,
                                                a.iters, a.select_iters)[0]),
                )
                for name, fn in [(k, v) for k, v in _all
                                 if (a.methods is None or k in a.methods)]:
                    t0 = time.time()
                    rec = fn()
                    m = all_metrics(ref, np.clip(np.asarray(rec), 0, None))
                    rows.append([a.n, a.bins, a.views, inc, pat, idx, seed,
                                 name if name != "MLEM+TV" else f"MLEM+TV(w={a.tv_weight:g})",
                                 m["PSNR"], m["MSSIM"], time.time() - t0])
                    wr.writerow(rows[-1]); fh_out.flush(); os.fsync(fh_out.fileno())
                    print(f"inc={inc:<6g} {pat}#{idx} r{seed} {name:<8} "
                          f"PSNR {m['PSNR']:6.2f} MSSIM {m['MSSIM']:.3f} "
                          f"{rows[-1][-1]:6.1f}s", flush=True)
    fh_out.close()
    print(f"saved: {out}  ({len(rows)} rows)")
