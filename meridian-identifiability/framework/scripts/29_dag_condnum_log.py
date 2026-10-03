"""Experiment A (for the paper's completeness argument): log DAG numerical
stability during training of the FULL CMVAE_CB.

The ablation already shows removing the DAG changes nothing, so instability
is unlikely to be the culprit; this script records the condition-number
trajectory to state that definitively.

Reads G from the batch's forward() and computes stats every log_every optimizer
steps. Does NOT modify their dag() math. Does NOT save a model.

Output: results/model/dag_stability.csv
Columns: step, epoch, cond2, inv_norm_2, G_frobenius, loss_total

Usage:
    python scripts/29_dag_condnum_log.py --epochs 15 --log_every 50
"""
import sys, os, csv, argparse
import numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/external/discrepancy_vae/src")
sys.path.insert(0, "/workspace/meridian-identifiability/framework/model")
from train import loss_function
from cb_data import CBDataset
from cb_model import CMVAE_CB
import cb_init


def dag_stats(G):
    """Read-only DAG stats. Does NOT modify G or the model."""
    z_dim = G.shape[0]
    triu = torch.triu(G, diagonal=1)
    A = torch.eye(z_dim, dtype=G.dtype, device=G.device) - triu
    with torch.no_grad():
        try:
            cond2 = float(torch.linalg.cond(A).item())
        except Exception:
            cond2 = float("nan")
        try:
            inv_norm_2 = float(
                torch.linalg.norm(torch.linalg.inv(A), ord=2).item()
            )
        except Exception:
            inv_norm_2 = float("nan")
        g_frob = float(torch.linalg.norm(G).item())
    return cond2, inv_norm_2, g_frob


def main(a):
    dev = a.device
    print(f"[setup] device={dev} init={a.init} epochs={a.epochs} "
          f"log_every={a.log_every}", flush=True)

    ds_tr = CBDataset(a.h5ad, a.split, "train")
    dl = DataLoader(ds_tr, batch_size=a.batch, shuffle=True,
                    num_workers=0, drop_last=True)
    print(f"[setup] {len(ds_tr)} training cells, dim={ds_tr.dim} c_dim={ds_tr.c_dim}, "
          f"{len(dl)} batches/epoch", flush=True)

    model = CMVAE_CB(dim=ds_tr.dim, z_dim=a.zdim, c_dim=ds_tr.c_dim,
                     device=dev).double().to(dev)
    if a.init == "shift":
        print("[init] fitting decoder to shift basis", flush=True)
        W, _ = cb_init.shift_basis(a.npz, a.split, a.zdim)
        cb_init.init_decoder_fit(model, W, steps=800, lr=1e-2)
    else:
        print("[init] random", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    def ramp(E, start, mx, hold_flat_after=None):
        s = np.zeros(E)
        if E <= start + 1:
            s[:] = mx; s[:min(start, E)] = 0; return s
        end = hold_flat_after if hold_flat_after else E
        end = min(end, E)
        if end > start:
            s[start:end] = np.linspace(0, mx, end - start)
        s[end:] = mx
        return s
    aS = ramp(a.epochs, 5, a.mxAlpha, a.epochs // 2); aS[:5] = 0
    bS = ramp(a.epochs, 10, a.mxBeta); bS[:10] = 0
    tS = np.ones(a.epochs)
    if a.epochs > 5:
        tS[5:] = np.linspace(1, a.mxTemp, a.epochs - 5)

    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, "dag_stability.csv")
    tmp_path = csv_path + ".tmp"
    csv_f = open(tmp_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["step", "epoch", "cond2", "inv_norm_2",
                     "G_frobenius", "loss_total"])
    csv_f.flush()

    with torch.no_grad():
        model.eval()
        xb, yb, cb = next(iter(dl))
        xb, yb, cb = xb.to(dev), yb.to(dev), cb.to(dev)
        _, _, _, _, G0 = model(xb, cb, cb, num_interv=1, temp=float(tS[0]))
    c0, i0, g0 = dag_stats(G0.detach())
    writer.writerow([0, 0, c0, i0, g0, ""]); csv_f.flush()
    print(f"[step 0/pre] cond2={c0:.3e} inv_norm_2={i0:.3e} G_frob={g0:.3e}",
          flush=True)

    global_step = 0
    for ep in range(a.epochs):
        ds_tr.resample()
        model.train()
        for xb, yb, cb in dl:
            xb, yb, cb = xb.to(dev), yb.to(dev), cb.to(dev)
            opt.zero_grad()
            y_hat, x_recon, z_mu, z_var, G = model(xb, cb, cb, num_interv=1,
                                                    temp=float(tS[ep]))
            mmd_l, mse, kl, L1 = loss_function(y_hat, yb, x_recon, xb,
                                                z_mu, z_var, G,
                                                a.MMD_sigma, a.kernel_num, False)
            loss = aS[ep] * mmd_l + mse + bS[ep] * kl + a.lmbda * L1
            loss.backward(); opt.step()
            global_step += 1

            if global_step % a.log_every == 0:
                c, i, g = dag_stats(G.detach())
                writer.writerow([global_step, ep, c, i, g, float(loss.item())])
                csv_f.flush()
                print(f"[step {global_step:>5} ep {ep:>2}] "
                      f"cond2={c:.3e} inv_norm_2={i:.3e} "
                      f"G_frob={g:.3e} loss={float(loss.item()):.4f}",
                      flush=True)

    csv_f.close()
    os.rename(tmp_path, csv_path)
    print(f"\n[write] {csv_path}", flush=True)

    import numpy as _np
    rows = []
    with open(csv_path) as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            rows.append(row)
    if rows:
        conds = _np.array([float(x[2]) for x in rows if x[2] and x[2] != "nan"])
        invs = _np.array([float(x[3]) for x in rows if x[3] and x[3] != "nan"])
        gs = _np.array([float(x[4]) for x in rows if x[4] and x[4] != "nan"])
        print(f"[summary] rows={len(rows)}", flush=True)
        print(f"[summary] cond2      min={conds.min():.3e} "
              f"max={conds.max():.3e} final={conds[-1]:.3e}", flush=True)
        print(f"[summary] inv_norm_2 min={invs.min():.3e} "
              f"max={invs.max():.3e} final={invs[-1]:.3e}", flush=True)
        print(f"[summary] G_frob     min={gs.min():.3e} "
              f"max={gs.max():.3e} final={gs[-1]:.3e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", default="/workspace/external/discrepancy_vae/datasets/causalbench_k562.h5ad")
    p.add_argument("--npz", default="/workspace/meridian-identifiability/causalbench/data/dataset_k562.npz")
    p.add_argument("--split", default="/workspace/meridian-identifiability/framework/results/splits/k562_zeroshot_split.json")
    p.add_argument("--out", default="/workspace/meridian-identifiability/framework/results/model")
    p.add_argument("--init", choices=["shift", "random"], default="shift")
    p.add_argument("--zdim", type=int, default=15)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mxAlpha", type=float, default=10.0)
    p.add_argument("--mxBeta", type=float, default=2.0)
    p.add_argument("--mxTemp", type=float, default=5.0)
    p.add_argument("--MMD_sigma", type=float, default=1000.0)
    p.add_argument("--kernel_num", type=int, default=10)
    p.add_argument("--lmbda", type=float, default=1e-3)
    p.add_argument("--device", default="cuda:0")
    main(p.parse_args())
