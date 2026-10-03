"""Experiment A: DAG numerical-stability logging during training.

Question: does CMVAE's DAG layer become numerically unstable during training?
model.dag() computes (z + bc*csz) @ inverse(I - triu(G)); if I - triu(G) becomes
ill-conditioned this inverse amplifies noise. A static test with an untrained
encoder feeding the TRUE program target scored R2 = -2.15, worse than the readout
guess at -0.21, which should not happen if the pipeline were sound.

This script runs a short training run (default shift-init, 30 epochs) and, every
N gradient steps, logs from the batch's G:
    cond2(I - triu(G))          two-norm condition number
    ||inverse(I - triu(G))||_F
    ||triu(G)||_F
    ||G||_F
plus the loss components. It also runs the same zeroshot_eval as cb_train.py
every zs_every epochs so the DAG stats can be aligned with R2 trajectory.

Output: results/dag_stability/log_{init}.json (per-step DAG time series plus
per-epoch zeroshot R2). No model checkpoint is written.

Usage:
    python scripts/26_dag_stability.py                              # shift-init, 30 ep
    python scripts/26_dag_stability.py --init random --epochs 30    # random-init
"""
import sys, os, json, argparse
import numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/external/discrepancy_vae/src")
sys.path.insert(0, "/workspace/meridian-identifiability/framework/model")
from train import loss_function
from cb_data import CBDataset
from cb_model import CMVAE_CB
import cb_init


def r2(pred, true):
    return float(1 - ((true - pred) ** 2).sum() / ((true ** 2).sum() + 1e-12))


def dag_stats(G):
    """G: (z_dim, z_dim) learned adjacency. A = I - torch.triu(G, diagonal=1)."""
    z_dim = G.shape[0]
    triu = torch.triu(G, diagonal=1)
    A = torch.eye(z_dim, dtype=G.dtype, device=G.device) - triu
    with torch.no_grad():
        try:
            cond = float(torch.linalg.cond(A).item())
        except Exception:
            cond = float("nan")
        try:
            inv_norm = float(torch.linalg.norm(torch.linalg.inv(A)).item())
        except Exception:
            inv_norm = float("nan")
        triu_norm = float(torch.linalg.norm(triu).item())
        g_norm = float(torch.linalg.norm(G).item())
    return dict(cond=cond, inv_norm=inv_norm, triu_norm=triu_norm, G_norm=g_norm)


def zeroshot_eval(model, ds_tr, npz_path, split_json, device):
    d = np.load(npz_path, allow_pickle=True)
    X = d["expression_matrix"].astype(np.float64)
    iv = np.asarray(d["interventions"]).astype(str)
    genes = [str(v) for v in d["var_names"]]; gidx = {g: i for i, g in enumerate(genes)}
    keep = iv != "excluded"; X, iv = X[keep], iv[keep]
    ctrl_mu = X[iv == "non-targeting"].mean(0)
    held = json.load(open(split_json))["heldout_perturbations"]

    was_training = model.training
    model.eval(); model.bind(ds_tr.train_genes, ds_tr.genes)
    pg = model.program_gene_map()
    xc = torch.from_numpy(ds_tr.ctrl).double().to(device)
    with torch.no_grad():
        base = model.control_pred(xc).mean(0).cpu().numpy()
    rows = []
    for g in held:
        t = gidx[g]
        truth = (X[iv == g].mean(0) - ctrl_mu).copy(); truth[t] = 0.0
        with torch.no_grad():
            pred = model.predict(xc, g, seen=False, pg_map=pg).mean(0).cpu().numpy()
        mshift = (pred - base).copy(); mshift[t] = 0.0
        rows.append((g, r2(mshift, truth)))
    if was_training:
        model.train()
    r2_vals = np.array([x[1] for x in rows])
    return float(np.median(r2_vals)), float(np.mean(r2_vals))


def _to_float(t):
    return float(t.item()) if hasattr(t, "item") else float(t)


def main(a):
    dev = a.device
    print(f"[setup] device={dev} init={a.init} epochs={a.epochs} "
          f"log_every={a.log_every} zs_every={a.zs_every}", flush=True)

    ds_tr = CBDataset(a.h5ad, a.split, "train")
    dl = DataLoader(ds_tr, batch_size=a.batch, shuffle=True,
                    num_workers=0, drop_last=True)
    steps_per_epoch = len(dl)
    print(f"[setup] {len(ds_tr)} training cells, dim={ds_tr.dim} c_dim={ds_tr.c_dim}, "
          f"{steps_per_epoch} batches/epoch", flush=True)

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
    dag_log = []
    zs_log = []
    global_step = 0

    with torch.no_grad():
        model.eval()
        xb, yb, cb = next(iter(dl))
        xb, yb, cb = xb.to(dev), yb.to(dev), cb.to(dev)
        _, _, _, _, G0 = model(xb, cb, cb, num_interv=1, temp=float(tS[0]))
    entry = dict(step=0, epoch=0, phase="pretrain")
    entry.update(dag_stats(G0.detach()))
    entry["loss"] = None
    dag_log.append(entry)
    print(f"[step 0/pre] cond={entry['cond']:.3e} inv_norm={entry['inv_norm']:.3e} "
          f"triu_norm={entry['triu_norm']:.3e} G_norm={entry['G_norm']:.3e}",
          flush=True)

    zs_med0, zs_mean0 = zeroshot_eval(model, ds_tr, a.npz, a.split, dev)
    zs_log.append(dict(step=0, epoch=0, r2_median=zs_med0, r2_mean=zs_mean0))
    print(f"[step 0/pre] zeroshot R2 median {zs_med0:+.4f}  mean {zs_mean0:+.4f}",
          flush=True)

    for ep in range(a.epochs):
        ds_tr.resample()
        model.train()
        for xb, yb, cb in dl:
            xb, yb, cb = xb.to(dev), yb.to(dev), cb.to(dev)
            opt.zero_grad()
            y_hat, x_recon, z_mu, z_var, G = model(xb, cb, cb, num_interv=1,
                                                    temp=float(tS[ep]))
            mmd_l, mse, kl, L1 = loss_function(y_hat, yb, x_recon, xb, z_mu, z_var, G,
                                                a.MMD_sigma, a.kernel_num, False)
            loss = aS[ep] * mmd_l + mse + bS[ep] * kl + a.lmbda * L1
            loss.backward(); opt.step()
            global_step += 1

            if global_step % a.log_every == 0:
                entry = dict(step=global_step, epoch=ep, phase="train")
                entry.update(dag_stats(G.detach()))
                entry["loss"] = dict(
                    total=_to_float(loss), mmd=_to_float(mmd_l),
                    mse=_to_float(mse), kl=_to_float(kl), L1=_to_float(L1),
                )
                dag_log.append(entry)
                print(f"[step {global_step:>5} ep {ep:>2}] "
                      f"cond={entry['cond']:.3e} inv_norm={entry['inv_norm']:.3e} "
                      f"triu_norm={entry['triu_norm']:.3e} "
                      f"G_norm={entry['G_norm']:.3e} "
                      f"loss={entry['loss']['total']:.4f}", flush=True)

        if (ep + 1) % a.zs_every == 0 or ep == a.epochs - 1:
            zs_med, zs_mean = zeroshot_eval(model, ds_tr, a.npz, a.split, dev)
            zs_log.append(dict(step=global_step, epoch=ep,
                               r2_median=zs_med, r2_mean=zs_mean))
            print(f"[ep {ep:>2}] zeroshot R2 median {zs_med:+.4f} "
                  f"mean {zs_mean:+.4f} (ridge 0.226, gmean 0.129)", flush=True)

    cond_series = [e["cond"] for e in dag_log if e.get("cond") is not None
                   and not np.isnan(e["cond"])]
    inv_series = [e["inv_norm"] for e in dag_log if e.get("inv_norm") is not None
                  and not np.isnan(e["inv_norm"])]
    summary = dict(
        n_dag_entries=len(dag_log),
        cond_min=float(np.min(cond_series)) if cond_series else None,
        cond_max=float(np.max(cond_series)) if cond_series else None,
        cond_final=cond_series[-1] if cond_series else None,
        inv_norm_min=float(np.min(inv_series)) if inv_series else None,
        inv_norm_max=float(np.max(inv_series)) if inv_series else None,
        inv_norm_final=inv_series[-1] if inv_series else None,
        zs_median_pretrain=zs_log[0]["r2_median"] if zs_log else None,
        zs_median_best=max((e["r2_median"] for e in zs_log), default=None),
        zs_median_final=zs_log[-1]["r2_median"] if zs_log else None,
    )
    payload = dict(config=vars(a), summary=summary,
                   dag_log=dag_log, zeroshot_log=zs_log)
    out_json = os.path.join(a.out, f"log_{a.init}.json")
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.rename(tmp, out_json)
    print(f"\n[write] {out_json}", flush=True)
    print(f"[summary] cond min={summary['cond_min']} "
          f"max={summary['cond_max']} final={summary['cond_final']}", flush=True)
    print(f"[summary] inv_norm min={summary['inv_norm_min']} "
          f"max={summary['inv_norm_max']} final={summary['inv_norm_final']}",
          flush=True)
    print(f"[summary] zeroshot R2 med: pretrain={summary['zs_median_pretrain']} "
          f"best={summary['zs_median_best']} final={summary['zs_median_final']}",
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", default="/workspace/external/discrepancy_vae/datasets/causalbench_k562.h5ad")
    p.add_argument("--npz", default="/workspace/meridian-identifiability/causalbench/data/dataset_k562.npz")
    p.add_argument("--split", default="/workspace/meridian-identifiability/framework/results/splits/k562_zeroshot_split.json")
    p.add_argument("--out", default="/workspace/meridian-identifiability/framework/results/dag_stability")
    p.add_argument("--init", choices=["shift", "random"], default="shift")
    p.add_argument("--zdim", type=int, default=15)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--zs_every", type=int, default=5)
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
