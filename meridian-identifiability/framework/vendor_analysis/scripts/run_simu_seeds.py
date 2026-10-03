import os, sys, json, random
import numpy as np, torch
from argparse import Namespace
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import train
from utils import get_simu_data

OUT = "/workspace/meridian-identifiability/dvae_repro/results/seeds"
os.makedirs(OUT, exist_ok=True)
ORDER = [0,1,2,3,4]
TRUE = np.diag(np.ones(4), 1)

def shd(G, th):
    return int(((np.abs(G) > th).astype(int) != TRUE.astype(int)).sum())

for seed in range(10):
    f = f"{OUT}/seed{seed}.json"
    if os.path.exists(f):
        print("SKIP", seed, flush=True); continue
    opts = Namespace(batch_size=256, mode='train', lr=5e-2, epochs=200,
                     grad_clip=False, mxAlpha=4, mxBeta=0.5, mxTemp=1,
                     lmbda=1e-1, MMD_sigma=1000, kernel_num=10,
                     matched_IO=False, latdim=5, seed=seed)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dl, dl2, dim, cdim, ptb, nl = get_simu_data(batch_size=opts.batch_size, mode='train')
    dl3, *_ = get_simu_data(batch_size=opts.batch_size, mode='test', perturb_targets=ptb)  # match notebook RNG
    opts.dim, opts.cdim = dim, cdim
    sd = f"{OUT}/run{seed}"; os.makedirs(sd, exist_ok=True)
    train(dl, opts, 'cuda:0', sd, log=False, simu=True, nonlinear=nl, order=ORDER)

    res = {"seed": seed}
    for tag, path in [("best", f"{sd}/best_model.pt"), ("final", f"{sd}/last_model.pt")]:
        if not os.path.exists(path): continue
        m = torch.load(path, weights_only=False); m.eval()
        G = torch.triu(m.G, diagonal=1).detach().cpu().numpy()
        res[tag] = {"max_abs": float(np.abs(G).max()),
                    "shd_0.01": shd(G, 0.01), "shd_0.05": shd(G, 0.05),
                    "G": np.round(G,4).tolist()}
    json.dump(res, open(f,"w"), indent=2)
    print(f"seed {seed}: " + " | ".join(
        f"{k} maxG={v['max_abs']:.4f} SHD01={v['shd_0.01']}"
        for k,v in res.items() if k!="seed"), flush=True)

print("\n=== SUMMARY ===", flush=True)
for s in range(10):
    f = f"{OUT}/seed{s}.json"
    if os.path.exists(f):
        r = json.load(open(f))
        for k in ("best","final"):
            if k in r: print(f"seed {s} {k}: SHD@0.01={r[k]['shd_0.01']}  maxG={r[k]['max_abs']:.4f}", flush=True)
