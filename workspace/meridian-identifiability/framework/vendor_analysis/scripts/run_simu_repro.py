import os, sys, json, random, time
import numpy as np, torch
from argparse import Namespace
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import train
from utils import get_simu_data

OUT = "/workspace/meridian-identifiability/dvae_repro/results"
os.makedirs(OUT, exist_ok=True)
ORDER = [0, 1, 2, 3, 4]          # true topological order, as in their notebook
SEED  = 12                        # their seed

opts = Namespace(batch_size=256, mode='train', lr=5e-2, epochs=200,
                 grad_clip=False, mxAlpha=4, mxBeta=0.5, mxTemp=1,
                 lmbda=1e-1, MMD_sigma=1000, kernel_num=10,
                 matched_IO=False, latdim=5, seed=SEED)

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
dl, dl2, dim, cdim, ptb, nonlinear = get_simu_data(batch_size=opts.batch_size, mode='train')
opts.dim, opts.cdim = dim, cdim
print(f"dim={dim} cdim={cdim} nonlinear={nonlinear} targets={ptb}", flush=True)

savedir = f"{OUT}/repro_seed{SEED}"
os.makedirs(savedir, exist_ok=True)
t0 = time.time()
train(dl, opts, 'cuda:0', savedir, log=False, simu=True,
      nonlinear=nonlinear, order=ORDER)
print(f"trained in {time.time()-t0:.0f}s", flush=True)

model = torch.load(f"{savedir}/best_model.pt", weights_only=False)
model.eval()
rec = []
for i in range(5):
    c = torch.from_numpy(np.eye(5)[i]).to('cuda:0').double().unsqueeze(0)
    bc, _ = model.c_encode(c, temp=1)
    rec.append(bc.argmax().item())
print("supplied order :", ORDER, flush=True)
print("recovered order:", rec, flush=True)
print("ROUND TRIP (recovered == supplied):", rec == ORDER, flush=True)

G = torch.triu(model.G, diagonal=1).detach().cpu().numpy()
true = np.diag(np.ones(4), 1)
for th in [0.01, 0.05, 0.1, 0.3]:
    shd = int((( np.abs(G) > th ).astype(int) != true.astype(int)).sum())
    print(f"threshold {th:<5} SHD={shd}", flush=True)
print("learned G:\n", np.round(G, 3), flush=True)

json.dump({"order": ORDER, "recovered": rec, "G": G.tolist()},
          open(f"{savedir}/summary.json", "w"), indent=2)
