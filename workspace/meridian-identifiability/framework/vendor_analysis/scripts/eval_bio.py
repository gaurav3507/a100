import os, sys, json
import numpy as np, torch
sys.path.insert(0, "/workspace/external/discrepancy_vae/src")
from inference import evaluate_single_leftout

RUN = "/workspace/meridian-identifiability/dvae_repro/results/bio/run1784607886"
DEV = "cuda:0"

def r2(pred, true):
    ss_res = ((true - pred)**2).sum()
    ss_tot = ((true - true.mean())**2).sum()
    return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

def per_interv_r2(gt_y, pred_y, c_y, gt_x, top_de=20):
    """Their protocol: per intervention, R2 between mean(generated) and mean(truth)."""
    keys = [tuple(np.nonzero(r)[0]) for r in c_y]
    uniq = sorted(set(keys))
    all_r2, de_r2 = [], []
    for k in uniq:
        m = np.array([kk == k for kk in keys])
        if m.sum() < 5: continue
        t = gt_y[m].mean(0); p = pred_y[m].mean(0); ctrl = gt_x[m].mean(0)
        all_r2.append(r2(p, t))
        de = np.argsort(-np.abs(t - ctrl))[:top_de]     # most changed vs control
        de_r2.append(r2(p[de], t[de]))
    return np.array(all_r2), np.array(de_r2), len(uniq)

out = {}
for tag in ["best", "last"]:
    path = f"{RUN}/{tag}_model.pt"
    if not os.path.exists(path): continue
    for temp in [1, 100]:
        m = torch.load(path, weights_only=False, map_location=DEV); m.eval()
        rmse, signerr, gt_y, pred_y, c_y, gt_x = evaluate_single_leftout(
            m, RUN, DEV, mode="CMVAE", temp=temp)
        a, d, n = per_interv_r2(gt_y, pred_y, c_y, gt_x)
        k = f"{tag}_temp{temp}"
        out[k] = dict(rmse=float(rmse), signerr=float(signerr),
                      n_interventions=int(n),
                      r2_all_mean=float(np.nanmean(a)), r2_all_median=float(np.nanmedian(a)),
                      r2_de_mean=float(np.nanmean(d)),  r2_de_median=float(np.nanmedian(d)),
                      n_cells=int(gt_y.shape[0]))
        print(f"\n=== {k} ===  cells={gt_y.shape[0]}  interventions={n}", flush=True)
        print(f"  RMSE        {rmse:.4f}      (paper: ~0.5-0.6)", flush=True)
        print(f"  R2 all      mean {np.nanmean(a):.4f}  median {np.nanmedian(a):.4f}   (paper: 0.99)", flush=True)
        print(f"  R2 DE-genes mean {np.nanmean(d):.4f}  median {np.nanmedian(d):.4f}   (paper: 0.95)", flush=True)

json.dump(out, open(f"{RUN}/eval_summary.json","w"), indent=2)
print("\nwrote eval_summary.json", flush=True)
