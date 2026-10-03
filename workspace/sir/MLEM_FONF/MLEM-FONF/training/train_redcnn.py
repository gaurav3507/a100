"""Train RED-CNN on the locked Mayo split (published patch recipe).

Inputs: FBP reconstructions of simulated low-dose sinograms (incident level
--incident) | Targets: full-dose ground-truth slices. AMP on; seeds fixed;
state_dict checkpoints. Run on the A4000:
  python training/train_redcnn.py --root /media/ant-pc/HDD2/datasets --incident 3000
"""
import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from models import REDCNN
    from mlem_fonf import build_system_matrix, poisson_sinogram
    from mlem_fonf.fbp import fbp
    from mlem_fonf import data as D

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--incident", type=float, default=3000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--patch", type=int, default=55)
    ap.add_argument("--patches-per-slice", type=int, default=16)
    ap.add_argument("--slices", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = f"checkpoints/redcnn_{int(a.incident)}.pt"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = build_system_matrix(256, 64, 90)
    xs, ys = [], []
    rng = np.random.default_rng(a.seed)
    print("building training pairs (FBP low-dose -> GT)...")
    for i, (pat, idx, ref) in enumerate(D.mayo_slices(a.root, "train", "full",
                                                      every=4, limit=a.slices)):
        g = poisson_sinogram(A, ref, a.incident, np.random.default_rng(1000 + i))
        x = fbp(g, A, 256, 64, 90, "parallel")
        for _ in range(a.patches_per_slice):           # random patches/slice
            r0, c0 = rng.integers(0, 256 - a.patch, 2)
            xs.append(x[r0:r0 + a.patch, c0:c0 + a.patch])
            ys.append(ref[r0:r0 + a.patch, c0:c0 + a.patch])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{a.slices} slices")
    X = torch.tensor(np.stack(xs), dtype=torch.float32)[:, None]
    Y = torch.tensor(np.stack(ys), dtype=torch.float32)[:, None]
    dl = DataLoader(TensorDataset(X, Y), batch_size=a.batch, shuffle=True,
                    num_workers=2, pin_memory=True)
    net = REDCNN().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-4)
    try:
        _Scaler = torch.amp.GradScaler
    except AttributeError:                     # torch < 2.x
        from torch.cuda.amp import GradScaler as _Scaler
    scaler = _Scaler(enabled=(dev == "cuda"))
    prev_losses, skipped = [], 0
    for ep in range(a.epochs):
        t0, tot, gsum, nb2 = time.time(), 0.0, 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=(dev == "cuda")):
                loss = torch.nn.functional.mse_loss(net(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite LOSS ep{ep+1}: input range "
                                   f"[{xb.min():.3g},{xb.max():.3g}]")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            if not torch.isfinite(gn):
                skipped += 1
            gsum += float(gn) if torch.isfinite(gn) else 0.0
            nb2 += 1
            scaler.step(opt); scaler.update()
            tot += loss.item() * xb.size(0)
        with torch.no_grad():
            vout = net(X[:8].to(dev))
            vpsnr = -10 * torch.log10(
                torch.mean((vout - Y[:8].to(dev)) ** 2)).item()
        ep_loss = tot / len(X)
        print(f"epoch {ep+1}/{a.epochs}  mse {ep_loss:.6f}  gnorm {gsum/max(nb2,1):.3f}  "
              f"scale {scaler.get_scale():.0f}  skip {skipped}  "
              f"valPSNR {vpsnr:.2f}  {time.time()-t0:.0f}s", flush=True)
        prev_losses.append(round(ep_loss, 6))
        if len(prev_losses) >= 3 and len(set(prev_losses[-3:])) == 1:
            raise RuntimeError(
                f"FROZEN loss ({prev_losses[-1]}) for 3 epochs — "
                f"gnorm {gsum/max(nb2,1):.4f}, scaler scale "
                f"{scaler.get_scale():.1f}, skipped {skipped}. Aborting to "
                "save GPU time; paste this line back.")
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "incident": a.incident,
                "seed": a.seed}, a.out)
    print("saved:", a.out)
