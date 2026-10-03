"""Inference helpers for learned baselines (used by run_benchmark).

All evaluation runs on the single A4000 machine (locked rule). DIP is
optimized per image here (no training). Diffusion DPS sampling loads a
DGX-trained score checkpoint when present.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np


def redcnn_infer(ckpt, x_img, device="cuda"):
    import torch
    from models import REDCNN
    net = REDCNN().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    net.eval()
    with torch.no_grad():
        y = net(torch.tensor(x_img, dtype=torch.float32,
                             device=device)[None, None])
    return y[0, 0].cpu().numpy()


def lpd_infer(ckpt, g, A, device="cuda"):
    import torch
    from models import LPD
    from mlem_fonf.torch_ops import to_torch_projector
    pack = torch.load(ckpt, map_location=device)
    sigma = float(pack.get("sigma", 1.0))
    fwd0, bwd0, _ = to_torch_projector(A, device=device)
    net = LPD(lambda x: fwd0(x) / sigma, lambda y: bwd0(y) / sigma,
              256, 64, 90).to(device)
    net.load_state_dict(pack["state_dict"])
    net.eval()
    with torch.no_grad():
        y = net(torch.tensor(np.asarray(g) / sigma, dtype=torch.float32,
                             device=device)[None]).clamp(min=0)
    return y[0, 0].cpu().numpy()


def dip_reconstruct(g, A, n=256, iters=4000, lr=1e-3, tv_w=1e-6, seed=0,
                    device="cuda"):
    """Per-image Deep Image Prior with data fidelity through the certified
    projector + light TV (Baguer et al. 2020 setting)."""
    import torch
    from models import DIPNet
    from mlem_fonf.torch_ops import to_torch_projector
    torch.manual_seed(seed)
    fwd, _, _ = to_torch_projector(A, device=device)
    gt = torch.tensor(g, dtype=torch.float32, device=device)[None]
    z = torch.randn(1, 1, n, n, device=device) * 0.1
    net = DIPNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr)
    for k in range(iters):
        f = net(z)
        Af = fwd(f.reshape(1, -1))
        loss = torch.nn.functional.mse_loss(Af, gt)
        if tv_w:
            loss = loss + tv_w * (f.diff(-1).abs().mean() + f.diff(-2).abs().mean())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        return net(z)[0, 0].cpu().numpy()


def dps_infer(ckpt, g, A, count_scale, n=256, steps=200, zeta=1.0, seed=0,
              device="cuda", net=None, abar=None, T=None, _legacy_step=False):
    """Diffusion posterior sampling (Chung et al., ICLR 2023) with likelihood
    guidance through the certified projector:
        x_{t-1} <- x'_{t-1} - zeta * grad_x ||y - A(x0_hat(x_t))||_2
    (gradient of the residual NORM, i.e. residual-scaled step). zeta is a
    tuned hyperparameter (select on the VALIDATION patient, never test).
    Model trained on [-1,1] slices; returns [0,1]."""
    import torch
    from models import TinyUNet, BigUNet
    from mlem_fonf.torch_ops import to_torch_projector
    torch.manual_seed(seed)
    if net is None:                                   # normal path: load ckpt
        pack = torch.load(ckpt, map_location=device)
        Net = BigUNet if pack.get("arch", "tiny") == "big" else TinyUNet
        net = Net(ch=int(pack.get("ch", 64))).to(device)
        net.load_state_dict(pack["state_dict"]); net.eval()
        abar = pack["abar"].to(device)
        T = pack["T"]
    else:                                             # injected oracle/test model
        abar = abar.to(device)
    fwd, _, _ = to_torch_projector(A, device=device)
    gt = torch.tensor(g, dtype=torch.float32, device=device)[None]
    ts = torch.linspace(T, 1, steps, device=device).long()
    x = torch.randn(1, 1, n, n, device=device)
    for i, t in enumerate(ts):
        at = abar[t]
        x = x.detach().requires_grad_(True)
        eps = net(x, t[None].float() / T)
        # clip_denoised (IDDPM/DPS standard): at t~T, abar~0 and 1/sqrt(abar)
        # amplifies eps-prediction error by ~1e7 — clamping x0 to the data
        # range bounds it. Without this, trained nets yield saturated noise.
        x0 = ((x - (1 - at).sqrt() * eps) / at.sqrt().clamp_min(1e-4)).clamp(-1, 1)
        f01 = ((x0 + 1) / 2).clamp(0, 1)
        resid = fwd(f01.reshape(1, -1)) - gt
        rnorm = resid.norm()                      # ||y - A x0_hat||_2
        grad = torch.autograd.grad(rnorm, x)[0]   # DPS: gradient of the NORM
        t_prev = ts[i + 1] if i + 1 < len(ts) else torch.tensor(0, device=device)
        ap_ = abar[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
        with torch.no_grad():
            x = ap_.sqrt() * x0 + (1 - ap_).sqrt() * eps
            if _legacy_step:                          # TEST HOOK: the v<=0.9.7 bug
                x = x - 0.3 * grad / grad.norm().clamp_min(1e-8)
            else:
                x = x - zeta * grad
    return ((x0.detach()[0, 0].cpu().numpy() + 1) / 2).clip(0, 1)
