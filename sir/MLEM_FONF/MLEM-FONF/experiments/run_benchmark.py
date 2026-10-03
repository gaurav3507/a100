"""M3 benchmark grid: all baselines x incidents x seeds on one dataset.

Classical + frozen Algorithm 1' run everywhere; learned methods run when
their checkpoints exist in --ckpt-dir (skipped-with-note otherwise), so the
grid fills progressively as training completes.

Smoke:  python experiments/run_benchmark.py --dataset phantom --quick
Full:   python experiments/run_benchmark.py --dataset mayo \
            --root /media/ant-pc/HDD2/datasets --slices 20
"""
import argparse, csv, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "training"))
import numpy as np
from mlem_fonf import (build_system_matrix, build_fanbeam_matrix,
                       poisson_sinogram, mlem, mlem_spatial, pnp_admm_tv,
                       reconstruct, all_metrics)
from mlem_fonf.fbp import fbp
from mlem_fonf.version import banner


def _mlem_iters(geometry, inc, default):
    """Early-stopped MLEM protocol: iteration count frozen on the validation
    patient (results/mlem_iters.json); falls back to `default` if absent."""
    import json as _json
    p = pathlib.Path(__file__).resolve().parents[1] / "results" / "mlem_iters.json"
    if p.exists():
        try:
            return int(_json.load(open(p)).get(f"{geometry}:{int(inc)}", default))
        except Exception:
            return default
    return default

def get_slices(a):
    if a.dataset == "phantom":
        from mlem_fonf.phantoms import PHANTOMS
        names = ["shepp_logan", "ct_sim", "elliptical", "thorax_like"]
        return [(nm, 0, PHANTOMS[nm](256)) for nm in names[:a.slices]]
    from mlem_fonf import data as D
    if a.dataset == "mayo":
        return list(D.mayo_slices(a.root, "test", "full", every=4, limit=a.slices))
    if a.dataset == "lodopab":
        return [(f"f{fi}", si, im) for fi, si, im in
                D.lodopab_slices(a.root, "test", limit=a.slices)]
    raise ValueError(a.dataset)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="phantom",
                    choices=["phantom", "mayo", "lodopab"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--geometry", default="parallel", choices=["parallel", "fan"])
    ap.add_argument("--incidents", type=float, nargs="+",
                    default=[60, 300, 1500, 3000])
    ap.add_argument("--realizations", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--slices", type=int, default=8)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--with-dip", action="store_true")
    ap.add_argument("--with-supplement", action="store_true",
                    help="include supplement-tier methods (MLEM+FuzzyAD)")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="run only these methods (progressive fill)")
    ap.add_argument("--append", action="store_true",
                    help="append to existing CSV instead of overwriting")
    ap.add_argument("--dps-zeta", type=float, default=1.0,
                    help="DPS guidance scale (tuned on validation split)")
    ap.add_argument("--dps-steps", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    banner("run_benchmark")
    if a.quick:
        a.incidents, a.realizations, a.slices, a.iters = [3000], 1, 1, 300
        a.seeds = a.seeds or [0]
    A = (build_system_matrix(256, 64, 90) if a.geometry == "parallel"
         else build_fanbeam_matrix(256, 64, 90, sod=600, odd=600))
    ck = pathlib.Path(a.ckpt_dir)

    DOSE_AGNOSTIC = {"diffusion"}          # score model trained on clean slices

    def ckpt_for(method, inc):
        """Checkpoint lookup: exact per-dose file, then a generic file, then
        (for dose-agnostic models only) any available checkpoint of that
        method — the diffusion prior does not depend on the dose."""
        p = ck / f"{method}_{int(inc)}.pt"
        if p.exists():
            return p
        gen = ck / f"{method}.pt"
        if gen.exists():
            return gen
        if method in DOSE_AGNOSTIC:
            cands = sorted(ck.glob(f"{method}_*.pt"))
            if cands:
                print(f"[note] {method}: no checkpoint for inc={int(inc)}; "
                      f"using dose-agnostic {cands[0].name}")
                return cands[0]
        return None

    have = {m: any(p.name.startswith(m) for p in ck.glob("*.pt"))
            for m in ("redcnn", "lpd", "diffusion")}
    torch_ok = True
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_ok = False
    for m, h in have.items():
        if not h:
            print(f"[note] checkpoint missing -> skipping {m}")
    if not torch_ok:
        print("[note] torch unavailable -> all learned methods skipped")
    slices = get_slices(a)
    out = pathlib.Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    csvp = out / f"benchmark_{a.dataset}_{a.geometry}.csv"
    seeds = a.seeds if a.seeds is not None else list(range(a.realizations))
    done = set()
    if a.resume and csvp.exists():
        with open(csvp) as fh0:
            rd = csv.reader(fh0); next(rd, None)
            for row in rd:
                done.add((row[0], row[1], float(row[2]), row[3], int(row[4]),
                          int(row[5]), row[6]))
        print(f"[resume] {len(done)} completed rows found")
    new_file = not (csvp.exists() and (a.append or a.resume))
    fh = open(csvp, "w" if new_file else "a", newline="", buffering=1)
    w = csv.writer(fh)
    if new_file:
        w.writerow(["dataset", "geometry", "incident", "slice_tag",
                    "slice_idx", "seed", "method", "SNR", "MSE", "RMSE",
                    "PSNR", "CP", "MSSIM", "time_s"])
    n_done = n_skip = n_err = 0
    for inc in a.incidents:
        for sid, (tag, idx, ref) in enumerate(slices):
            for r in seeds:
                g, counts = poisson_sinogram(A, ref, inc,
                                             np.random.default_rng(100 * sid + r),
                                             return_counts=True)
                scale = (A @ ref.ravel()).max() / inc
                methods = {
                    "FBP": lambda: fbp(g, A, 256, 64, 90, a.geometry),
                    "MLEM": lambda: mlem(g, A, 256, _mlem_iters(a.geometry, inc, a.iters)),
                    "MLEM+TV": lambda: mlem_spatial(g, A, 256, "tv", a.iters),
                    "MLEM+AD": lambda: mlem_spatial(g, A, 256, "ad", a.iters),
                    "PnP-ADMM(TV)": lambda: pnp_admm_tv(g, A, 256, a.iters),
                    "A-FONF": lambda: reconstruct(counts, scale, A, 256,
                                                  n_bins=64, n_iter=a.iters)[0],
                }
                if a.with_supplement:
                    methods["MLEM+FuzzyAD"] = lambda: mlem_spatial(
                        g, A, 256, "fuzzyad", a.iters)
                if torch_ok:
                    import torch
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                    from eval_learned import (redcnn_infer, lpd_infer,
                                              dip_reconstruct, dps_infer)
                    cr, cl, cd = (ckpt_for("redcnn", inc), ckpt_for("lpd", inc),
                                  ckpt_for("diffusion", inc))
                    if cr:
                        methods["RED-CNN"] = lambda cr=cr: redcnn_infer(
                            cr, fbp(g, A, 256, 64, 90, a.geometry), dev)
                    if cl and a.geometry == "parallel":
                        methods["LPD"] = lambda cl=cl: lpd_infer(cl, g, A, dev)
                    if cd and a.geometry == "parallel":
                        methods["DPS"] = lambda cd=cd: dps_infer(
                            cd, g, A, scale, steps=a.dps_steps,
                            zeta=a.dps_zeta, seed=r, device=dev)
                    if a.with_dip:
                        methods["DIP"] = lambda: dip_reconstruct(
                            g, A, seed=r, device=dev)
                if a.methods:
                    methods = {k: v for k, v in methods.items()
                               if k in a.methods}
                    missing = [m for m in a.methods if m not in methods]
                    if missing:
                        raise SystemExit(
                            f"requested method(s) unavailable: {missing} — "
                            f"checkpoints in {ck}: "
                            f"{sorted(p.name for p in ck.glob('*.pt')) or 'none'}; "
                            "torch available: " + str(torch_ok))
                for name, fn in methods.items():
                    key = (a.dataset, a.geometry, float(inc), str(tag),
                           int(idx), int(r), name)
                    if key in done:
                        n_skip += 1
                        continue
                    t0 = time.time()
                    try:
                        rec = np.clip(np.asarray(fn(), dtype=float), 0, None)
                        met = all_metrics(ref, rec)
                    except Exception as e:  # noqa: BLE001
                        n_err += 1
                        print(f"[ERROR] inc={inc} {tag}#{idx} r{r} {name}: {e}",
                              flush=True)
                        continue
                    met["time_s"] = time.time() - t0
                    w.writerow([a.dataset, a.geometry, inc, tag, idx, r, name]
                               + [met[k] for k in ("SNR", "MSE", "RMSE",
                                                   "PSNR", "CP", "MSSIM",
                                                   "time_s")])
                    n_done += 1
                    print(f"inc={inc:>6} {tag}#{idx} r{r} {name:<13} "
                          f"PSNR {met['PSNR']:6.2f}  MSSIM {met['MSSIM']:.3f}  "
                          f"{met['time_s']:5.1f}s", flush=True)
    fh.close()
    print(f"saved: {csvp}  (+{n_done} rows, {n_skip} resumed-skips, "
          f"{n_err} errors)")
