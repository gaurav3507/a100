import os, json
import numpy as np

D = "/workspace/meridian-identifiability/causalbench/data"
R = "/workspace/meridian-identifiability/causalbench/results/spectrum"
os.makedirs(R, exist_ok=True)
CTRL, SEED = "non-targeting", 0

def load(ds, filt):
    n = f"dataset_{ds}" + ("_filtered" if filt else "") + ".npz"
    d = np.load(os.path.join(D, n), allow_pickle=True)
    return d["expression_matrix"].astype(np.float64), np.asarray(d["interventions"]), \
           [str(v) for v in d["var_names"]]

def run(ds, filt, nmin, d):
    tag = f"{ds}_filt{int(filt)}_n{nmin}_d{d}"
    out = os.path.join(R, tag + ".json")
    if os.path.exists(out):
        print("SKIP", tag, flush=True); return
    rng = np.random.default_rng(SEED)
    X, iv, vn = load(ds, filt); gidx = {g: i for i, g in enumerate(vn)}
    ctrl = np.where(iv == CTRL)[0]
    mu = X[ctrl].mean(0)
    _, _, Vt = np.linalg.svd(X[ctrl] - mu, full_matrices=False)
    W = Vt[:d].T

    def proj(rows, drop):
        Wm = W.copy()
        if drop is not None: Wm[drop, :] = 0.0
        return (X[rows] - mu) @ Wm

    us, cs = np.unique(iv, return_counts=True)
    envs = [(g, np.where(iv == g)[0], gidx.get(g))
            for g, n in zip(us, cs) if g not in (CTRL, "excluded") and n >= nmin]

    cref = rng.permutation(ctrl); ref, rest = cref[:len(cref)//2], cref[len(cref)//2:]
    ref_mu = proj(ref, None).mean(0)

    M = np.array([proj(rng.choice(r, nmin, replace=False), t).mean(0) - ref_mu
                  for _, r, t in envs])
    k = len(M)
    N = np.array([proj(rng.choice(rest, nmin, replace=False), None).mean(0) - ref_mu
                  for _ in range(k)])

    s_sig = np.linalg.svd(M, compute_uv=False) / np.sqrt(len(M))
    s_noi = np.linalg.svd(N, compute_uv=False) / np.sqrt(len(N))
    n_cmp = min(len(s_sig), len(s_noi))
    ratio = s_sig[:n_cmp] / s_noi[:n_cmp]

    res = dict(tag=tag, n_envs=len(envs), n_null=k, d=d,
               sing_signal=s_sig.tolist(), sing_noise=s_noi.tolist(),
               sing_ratio=ratio.tolist(),
               n_dims_above_noise=int((ratio > 1.0).sum()),
               n_dims_above_2x=int((ratio > 2.0).sum()),
               participation_ratio=float(s_sig.sum()**2 / (s_sig**2).sum()),
               s2_over_s1=float(s_sig[1]/s_sig[0]))
    tmp = out + ".tmp"; json.dump(res, open(tmp, "w"), indent=2); os.rename(tmp, out)
    print(f"\n=== {tag} ===  n_envs={len(envs)}", flush=True)
    print(f"  dims above noise: {res['n_dims_above_noise']}/{d}   above 2x: {res['n_dims_above_2x']}/{d}", flush=True)
    print(f"  participation ratio: {res['participation_ratio']:.2f}   s2/s1: {res['s2_over_s1']:.3f}", flush=True)
    print(f"  sing_ratio[:10]: {np.round(ratio[:10],2).tolist()}", flush=True)

if __name__ == "__main__":
    for ds in ["k562", "rpe1"]:
        for filt in [False, True]:
            for d in [10, 20, 50]:
                try: run(ds, filt, 200, d)
                except Exception as e: print(f"FAIL {ds} {filt} {d}: {e}", flush=True)
    print("\nALL DONE", flush=True)
