"""DGX scaffold: train a score/denoising diffusion model on Mayo abdomen
(clean full-dose slices), for DPS-style posterior sampling at eval time.

Run on the DGX (A100):
  # Option A (data lives on this machine):
  python training/train_diffusion_dgx.py --root <datasets-root>
  # Option B (recommended for DGX): train from a pre-exported npz
  #   [on ant-pc]  python training/export_train_npz.py --root <root> --out mayo_train.npz
  #   [scp npz to DGX, then]
  python training/train_diffusion_dgx.py --npz mayo_train.npz
Bring back: the checkpoint, this script's git hash, seeds, loss log, and
`conda env export > env_dgx.yml`.
"""
import argparse, math, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np

if __name__ == "__main__":
    import torch
    import torch.nn as nn  # noqa: F401
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from models import TinyUNet, BigUNet

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--npz", default=None,
                    help="pre-exported training slices (overrides --root)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--slices", type=int, default=1200)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--ch", type=int, default=64, help="UNet width")
    ap.add_argument("--arch", default="tiny", choices=["tiny", "big"],
                    help="score-model architecture (big = XL baseline)")
    ap.add_argument("--init-from", default=None,
                    help="warm-start net+EMA from a checkpoint (arch/ch must "
                         "match). NOTE: checkpoints store EMA weights — "
                         "meaningful only for runs with >~5k steps, where "
                         "EMA has converged to the trained weights.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=25,
                    help="periodic checkpoint interval (epochs); survives\n"
                         "container restarts (Jupyter-managed servers)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = f"checkpoints/diffusion_{int(3000)}.pt"  # trained dose-agnostic on clean slices
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.npz:
        X = np.load(a.npz)["slices"][:a.slices].astype(np.float32)
        print(f"loaded {len(X)} slices from {a.npz}")
    else:
        assert a.root, "--root or --npz required"
        from mlem_fonf import data as D      # root mode only (needs pydicom)
        X = np.stack([ref for _, _, ref in
                      D.mayo_slices(a.root, "train", "full", every=2,
                                    limit=a.slices)]).astype(np.float32)
    X = torch.tensor(X)[:, None] * 2 - 1                    # [-1, 1]
    # cosine schedule
    s = 0.008
    tt = torch.linspace(0, 1, a.T + 1)
    ab = torch.cos((tt + s) / (1 + s) * math.pi / 2) ** 2
    abar = (ab / ab[0]).to(dev)
    Net = BigUNet if a.arch == "big" else TinyUNet
    net = Net(ch=a.ch).to(dev)
    ema = Net(ch=a.ch).to(dev); ema.load_state_dict(net.state_dict())
    print(f"arch={a.arch} ch={a.ch} params="
          f"{sum(p.numel() for p in net.parameters())/1e6:.1f}M", flush=True)
    if a.init_from:
        pk = torch.load(a.init_from, map_location=dev)
        assert pk.get("arch", "tiny") == a.arch and int(pk.get("ch", 64)) == a.ch, \
            f"init-from mismatch: ckpt arch={pk.get('arch')} ch={pk.get('ch')}"
        net.load_state_dict(pk["state_dict"])
        ema.load_state_dict(pk["state_dict"])
        print(f"warm-started from {a.init_from} (epoch {pk.get('epoch','?')})",
              flush=True)
    opt = torch.optim.Adam(net.parameters(), 2e-4)
    try:
        _Scaler = torch.amp.GradScaler
    except AttributeError:                     # torch < 2.x
        from torch.cuda.amp import GradScaler as _Scaler
    scaler = _Scaler(enabled=(dev == "cuda"))
    nb = len(X) // a.batch
    assert nb > 0, f"batch {a.batch} > dataset {len(X)}"

    def _save_ckpt(epoch_done):
        """SINGLE save path for periodic + final (identical, DPS-loadable)."""
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": ema.state_dict(), "T": a.T, "seed": a.seed,
                    "abar": abar.cpu(), "ch": a.ch, "arch": a.arch,
                    "epoch": epoch_done}, a.out)
    for ep in range(a.epochs):
        perm = torch.randperm(len(X)); t0, tot = time.time(), 0.0
        for b in range(nb):
            x0 = X[perm[b * a.batch:(b + 1) * a.batch]].to(dev)
            t = torch.randint(1, a.T + 1, (x0.size(0),), device=dev)
            at = abar[t][:, None, None, None]
            eps = torch.randn_like(x0)
            xt = at.sqrt() * x0 + (1 - at).sqrt() * eps
            with torch.amp.autocast(device_type="cuda", enabled=(dev == "cuda")):
                loss = torch.nn.functional.mse_loss(
                    net(xt, t.float() / a.T), eps)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss ep{ep+1} b{b} — aborting "
                                   "BEFORE corrupting EMA/periodic checkpoints")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            with torch.no_grad():
                for p, q in zip(ema.parameters(), net.parameters()):
                    p.mul_(0.999).add_(q, alpha=0.001)
            tot += loss.item()
        print(f"epoch {ep+1}/{a.epochs}  loss {tot/nb:.5f}  {time.time()-t0:.0f}s",
              flush=True)
        if (ep + 1) % a.save_every == 0:
            _save_ckpt(ep + 1)
            print(f"  [periodic] saved @ epoch {ep+1} -> {a.out}", flush=True)
    _save_ckpt(a.epochs)
    print("saved:", a.out)
