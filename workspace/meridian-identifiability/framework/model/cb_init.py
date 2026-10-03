"""Case A: plant the intervention subspace in the NONLINEAR decoder by fitting, not
copying. Their decoder is decode(u) = leakyrelu(d2(leakyrelu(d1(u)))), so no single
weight maps programs->genes. We run a short least-squares fit on d1,d2 so that
decode(program_onehot_k) ~ shift-basis-direction k for each of the z_dim programs.
That starts the decoder in the intervention subspace (Finding 4) without pretending
the map is linear. Encoder is left at default init -- it cannot be set by pseudo-inverse
through a hidden layer.
"""
import json, numpy as np, torch

def shift_basis(npz_path, split_json, z_dim, ctrl_label="non-targeting", nmin=200):
    d = np.load(npz_path, allow_pickle=True)
    X = d["expression_matrix"].astype(np.float64)
    iv = np.asarray(d["interventions"]).astype(str)
    genes = [str(v) for v in d["var_names"]]; gidx = {g:i for i,g in enumerate(genes)}
    keep = iv != "excluded"; X, iv = X[keep], iv[keep]
    train = set(json.load(open(split_json))["train_perturbations"])
    mu = X[iv == ctrl_label].mean(0)
    M = []
    for g in train:
        r = np.where(iv == g)[0]
        if len(r) < nmin: continue
        v = X[r].mean(0) - mu; v[gidx[g]] = 0.0
        M.append(v)
    M = np.array(M)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:z_dim].T, genes                          # W: (dim, z_dim)

def init_decoder_fit(model, W, steps=800, lr=1e-2, verbose=True):
    """Fit d1,d2 so decode(e_k) ~ W[:,k]. e_k = one-hot program, W[:,k] = shift dir k."""
    dev, dt = model.d2.weight.device, model.d2.weight.dtype
    z = model.z_dim
    with torch.no_grad():
        base = torch.from_numpy(W.T).to(dt).to(dev)
        scale = (model.d2.weight.norm() / (base.norm() + 1e-8)).item()
    T = (torch.from_numpy(W.T).to(dt).to(dev) * scale).detach()   # (z, dim), no grad
    opt = torch.optim.Adam(list(model.d1.parameters()) + list(model.d2.parameters()), lr=lr)
    for i in range(steps):
        E = torch.eye(z, dtype=dt, device=dev)          # rebuild each step -> fresh graph
        opt.zero_grad()
        pred = model.decode(E)                          # (z, dim)
        loss = ((pred - T) ** 2).mean()
        loss.backward(); opt.step()
        if verbose and i % 200 == 0:
            print(f"    init-fit step {i}: mse {loss.item():.4f}", flush=True)
    if verbose:
        with torch.no_grad():
            r = torch.stack([torch.nn.functional.cosine_similarity(
                model.decode(E[k:k+1])[0], T[k], dim=0) for k in range(z)])
        print(f"    init-fit done: median program cosine {r.median().item():.3f}", flush=True)
    return model
