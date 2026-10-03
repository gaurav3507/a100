"""Experiment B: no-causal-layer ablation on the same zero-shot split.

Question: does the causal DAG layer contribute to zero-shot prediction on
CausalBench? discrepancy_vae's own Table 3 already hints it might not
(MMD 0.324 with vs 0.348 without on single-node; 0.432 vs 0.427 on double-node,
better without). This is the same-shape test using our zero-shot metric.

Implementation: subclass CMVAE_CB and override the single method dag(). The
override replaces (z + bc*csz) @ inverse(I - triu(G)) with u = z + bc*csz.
Everything else is IDENTICAL to cb_train.py: same encoder, same decoder, same
c_encode, same forward (which now routes through the overridden dag), same
loss_function (including L1 on G), same optimizer, same ramps, same
CBDataset, same zeroshot_eval, same init options.

G remains a learned parameter so their loss_function signature does not change
and the training procedure stays a strict A/B against CMVAE_CB. G's L1 term
still pushes it toward zero; it just no longer influences predictions.

Usage:
    python scripts/27_no_causal_ablation.py --init shift  --epochs 100
    python scripts/27_no_causal_ablation.py --init random --epochs 100
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


class CMVAE_CB_NoDAG(CMVAE_CB):
    """DAG bypassed: u = z + bc1*csz1 + bc2*csz2 instead of the matrix inverse.
    G is kept as a learned parameter (still gets L1'd via loss_function) but
    is no longer used in any prediction path.
    """
    def dag(self, z, bc1, csz1, bc2, csz2, num_interv=1):
        u = z
        if num_interv >= 1 and bc1 is not None and csz1 is not None:
            u = u + bc1 * csz1.unsqueeze(-1)
        if num_interv >= 2 and bc2 is not None and csz2 is not None:
            u = u + bc2 * csz2.unsqueeze(-1)
        return u


def r2(pred, true):
    return float(1 - ((true - pred) ** 2).sum() / ((true ** 2).sum() + 1e-12))


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
    return np.array([x[1] for x in rows]), rows


def main(a):
    dev = a.device
    print(f"[setup] variant=no_causal_layer init={a.init} epochs={a.epochs} "
          f"device={dev}", flush=True)

    ds_tr = CBDataset(a.h5ad, a.split, "train")
    dl = DataLoader(ds_tr, batch_size=a.batch, shuffle=True,
                    num_workers=0, drop_last=True)

    model = CMVAE_CB_NoDAG(dim=ds_tr.dim, z_dim=a.zdim, c_dim=ds_tr.c_dim,
                            device=dev).double().to(dev)
    if a.init == "shift":
        print("[init] fitting decoder to shift basis (no-DAG variant)", flush=True)
        W, _ = cb_init.shift_basis(a.npz, a.split, a.zdim)
        cb_init.init_decoder_fit(model, W, steps=800, lr=1e-2)
    else:
        print("[init] random (no-DAG variant)", flush=True)

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
    best_zs, best_state = -1e9, None
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
        if ep % 10 == 0 or ep == a.epochs - 1:
            zs, _ = zeroshot_eval(model, ds_tr, a.npz, a.split, dev)
            print(f"ep {ep:3d}  zeroshot R2 median {np.median(zs):+.4f}  "
                  f"mean {np.mean(zs):+.4f}  (ridge 0.226, gmean 0.129, no-DAG)",
                  flush=True)
            if np.median(zs) > best_zs:
                best_zs = float(np.median(zs))
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

    torch.save(model, f"{a.out}/last_nodag_{a.init}.pt")
    if best_state:
        model.load_state_dict(best_state)
        torch.save(model, f"{a.out}/best_nodag_{a.init}.pt")
    zs, rows = zeroshot_eval(model, ds_tr, a.npz, a.split, dev)
    res = dict(
        variant="no_causal_layer",
        init=a.init, zdim=a.zdim,
        best_median_r2=float(best_zs),
        final_median_r2=float(np.median(zs)),
        final_mean_r2=float(np.mean(zs)),
        frac_beats_gmean=float(np.mean(zs > 0.129)),
        frac_beats_ridge=float(np.mean(zs > 0.226)),
        baselines=dict(global_mean=0.129, ridge=0.226),
        per_gene=[{"gene": g, "r2": v} for g, v in rows],
    )
    out_json = f"{a.out}/zeroshot_nodag_{a.init}.json"
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
    os.rename(tmp, out_json)
    print(f"\n[write] {out_json}", flush=True)
    print(f"=== no-DAG {a.init} init  (best over training) ===", flush=True)
    print(f"  median R2 {best_zs:+.4f}  vs ridge 0.226, gmean 0.129", flush=True)
    print(f"  beats gmean {res['frac_beats_gmean']:.0%}, "
          f"beats ridge {res['frac_beats_ridge']:.0%} of held-out genes",
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", default="/workspace/external/discrepancy_vae/datasets/causalbench_k562.h5ad")
    p.add_argument("--npz", default="/workspace/meridian-identifiability/causalbench/data/dataset_k562.npz")
    p.add_argument("--split", default="/workspace/meridian-identifiability/framework/results/splits/k562_zeroshot_split.json")
    p.add_argument("--out", default="/workspace/meridian-identifiability/framework/results/model")
    p.add_argument("--init", choices=["shift", "random"], default="shift")
    p.add_argument("--zdim", type=int, default=15)
    p.add_argument("--epochs", type=int, default=100)
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
