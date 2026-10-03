"""Early-stopping protocol for the unregularized MLEM baseline.

Selects the iteration count that maximizes mean PSNR on the VALIDATION
patient (never test), per geometry x incident, and freezes it in
results/mlem_iters.json, which run_campaign / run_benchmark read.

  python experiments/select_mlem_iters.py --root /media/ant-pc/HDD2/datasets
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from mlem_fonf import build_system_matrix, build_fanbeam_matrix, poisson_sinogram, mlem
from mlem_fonf.metrics import psnr
from mlem_fonf.version import banner

GRID = (5, 10, 15, 20, 25, 35, 50, 75, 100, 150, 200, 400, 800)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", default="mayo")
    ap.add_argument("--slices", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--incidents", type=float, nargs="+", default=[60, 300, 1500, 3000])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    banner("select_mlem_iters")
    from mlem_fonf import data as D
    refs = [im for _, _, im in D.mayo_slices(a.root, "val", "full", every=4, limit=a.slices)]
    table = {}
    for geom, A in (("parallel", build_system_matrix(256, 64, 90)),
                    ("fan", build_fanbeam_matrix(256, 64, 90, sod=600, odd=600))):
        for inc in a.incidents:
            acc = {k: [] for k in GRID}
            for si, ref in enumerate(refs):
                for sd in range(a.seeds):
                    g = poisson_sinogram(A, ref, inc, np.random.default_rng(5000 + 10 * si + sd))
                    traj = {}
                    mlem(g, A, 256, max(GRID), callback=lambda k, f: traj.__setitem__(k + 1, psnr(ref, f)) if (k + 1) in GRID else None)
                    for k in GRID:
                        acc[k].append(traj[k])
            best = max(GRID, key=lambda k: np.mean(acc[k]))
            table[f"{geom}:{int(inc)}"] = int(best)
            print(f"{geom:<9} inc={int(inc):<5} best iters = {best:<4} "
                  f"(PSNR {np.mean(acc[best]):.2f} vs @800 {np.mean(acc[800]):.2f})", flush=True)
    out = pathlib.Path(a.out) if a.out else (pathlib.Path(__file__).resolve().parents[1] / "results" / "mlem_iters.json")
    out.parent.mkdir(exist_ok=True)
    json.dump(table, open(out, "w"), indent=1)
    print("saved:", out)
