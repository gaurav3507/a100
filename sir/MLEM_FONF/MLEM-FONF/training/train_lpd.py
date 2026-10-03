"""Train Learned Primal-Dual through the CERTIFIED torch-sparse projector.

  python training/train_lpd.py --root /media/ant-pc/HDD2/datasets --incident 3000
"""
import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

if __name__ == "__main__":
    import torch
    from models import LPD
    from mlem_fonf import build_system_matrix, poisson_sinogram
    from mlem_fonf.torch_ops import to_torch_projector
    from mlem_fonf import data as D

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--incident", type=float, default=3000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--slices", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = f"checkpoints/lpd_{int(a.incident)}.pt"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = build_system_matrix(256, 64, 90)
    # spectral-norm normalization (Adler-Oektem practice): keeps all network
    # inputs O(1); sigma is stored in the checkpoint for identical inference.
    v = np.random.default_rng(0).standard_normal(A.shape[1])
    for _ in range(25):
        v = A.T @ (A @ v)
        v /= np.linalg.norm(v)
    sigma = float(np.sqrt(np.linalg.norm(A.T @ (A @ v))))
    print(f"operator norm sigma = {sigma:.2f}")
    fwd0, bwd0, _ = to_torch_projector(A, device=dev)
    fwd = lambda x: fwd0(x) / sigma
    bwd = lambda y: bwd0(y) / sigma
    G, R = [], []
    print("simulating training sinograms...")
    for i, (pat, idx, ref) in enumerate(D.mayo_slices(a.root, "train", "full",
                                                      every=4, limit=a.slices)):
        g = poisson_sinogram(A, ref, a.incident, np.random.default_rng(2000 + i))
        G.append((g / sigma).astype(np.float32)); R.append(ref.astype(np.float32))
    G = torch.tensor(np.stack(G)); R = torch.tensor(np.stack(R))[:, None]
    net = LPD(fwd, bwd, 256, 64, 90).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-4)
    try:
        _Scaler = torch.amp.GradScaler
    except AttributeError:                     # torch < 2.x
        from torch.cuda.amp import GradScaler as _Scaler
    scaler = _Scaler(enabled=(dev == "cuda"))
    nb = len(G) // a.batch
    assert nb > 0, f"batch {a.batch} > dataset {len(G)}"
    for ep in range(a.epochs):
        perm = torch.randperm(len(G)); t0, tot = time.time(), 0.0
        for b in range(nb):
            idxs = perm[b * a.batch:(b + 1) * a.batch]
            gb, rb = G[idxs].to(dev), R[idxs].to(dev)
            with torch.amp.autocast(device_type="cuda", enabled=(dev == "cuda")):
                loss = torch.nn.functional.mse_loss(net(gb), rb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {ep+1} batch {b}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item() * len(idxs)
        print(f"epoch {ep+1}/{a.epochs}  mse {tot/(nb*a.batch):.6f}  {time.time()-t0:.0f}s")
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "incident": a.incident,
                "seed": a.seed, "sigma": sigma}, a.out)
    print("saved:", a.out)
