"""Does the closed-form estimator hold on real CausalBench data?

Premise (verified on synthetic): if environment c differs from the reference on
exactly one latent coordinate, the observed mean shift lies along COLUMN c of the
mixing matrix. Stack those columns, pseudo-invert, done. No optimizer.

CausalBench is UNPAIRED (sequencing destroys the cell), so the paired form does not
apply and this reduces to the mean-shift estimator. That is fine: the screen showed
mean shift is the metric that carries signal here (2.0-5.6 vs a 1.0 null).

Four tests, in order of what would kill the plan first:
  0. GATE     control pseudo-environments -> split-half cosine must be ~0
  1. STABLE   real perturbations -> split-half cosine must be clearly > 0
  2. RANK1    is the per-environment deviation approximately 1-D?
  3. BASIS    does a shift-derived d=15 basis beat a control-PCA basis at
              reconstructing HELD-OUT perturbation shifts?

Test 3 is the one that justifies using this as initialization instead of PCA.
"""
import os, json
import numpy as np

D    = "/workspace/meridian-identifiability/causalbench/data"
OUT  = "/workspace/meridian-identifiability/framework/results/cf_estimator"
CTRL = "non-targeting"
NMIN = 200          # matched sample size, both halves get 100
DIMS = [5, 10, 15, 20, 30]
SEED = 0
os.makedirs(OUT, exist_ok=True)


def load(ds):
    d  = np.load(os.path.join(D, f"dataset_{ds}.npz"), allow_pickle=True)
    X  = d["expression_matrix"].astype(np.float64)
    iv = np.asarray(d["interventions"]).astype(str)
    vn = [str(v) for v in d["var_names"]]
    return X, iv, vn


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def run(ds):
    out_f = os.path.join(OUT, f"{ds}.json")
    if os.path.exists(out_f):
        print(f"SKIP {ds}"); return json.load(open(out_f))

    rng = np.random.default_rng(SEED)
    X, iv, vn = load(ds)
    gidx = {g: i for i, g in enumerate(vn)}
    half = NMIN // 2

    ctrl = np.where(iv == CTRL)[0]
    ctrl_mu = X[ctrl].mean(0)

    us, cs = np.unique(iv, return_counts=True)
    envs = [(g, np.where(iv == g)[0], gidx.get(g))
            for g, n in zip(us, cs) if g not in (CTRL, "excluded") and n >= NMIN]
    print(f"\n===== {ds} =====  {len(envs)} environments with >={NMIN} cells", flush=True)

    def shift(rows, drop):
        """mean(perturbed) - mean(control), with the targeted gene zeroed out."""
        v = X[rows].mean(0) - ctrl_mu
        if drop is not None:
            v = v.copy(); v[drop] = 0.0
        return v

    # ---- TEST 0: GATE -------------------------------------------------------
    cshuf = rng.permutation(ctrl)
    k = len(cshuf) // NMIN
    gate = []
    for i in range(k):
        blk = cshuf[i*NMIN:(i+1)*NMIN]
        gate.append(cos(shift(blk[:half], None), shift(blk[half:], None)))
    gate = np.array(gate)
    print(f"  [0] GATE   control split-half cosine: "
          f"median {np.median(gate):+.3f}  mean {gate.mean():+.3f}  (expect ~0)", flush=True)

    # ---- TEST 1: STABILITY --------------------------------------------------
    stab, rank1, norms = [], [], []
    for g, rows, drop in envs:
        sub = rng.choice(rows, NMIN, replace=False)
        a, b = shift(sub[:half], drop), shift(sub[half:], drop)
        stab.append(cos(a, b))
        norms.append(np.linalg.norm(shift(sub, drop)))

        # ---- TEST 2: is the deviation ~1-D? ---------------------------------
        Dv = X[sub] - ctrl_mu
        if drop is not None:
            Dv = Dv.copy(); Dv[:, drop] = 0.0
        m = Dv.mean(0)
        tot = (Dv**2).sum()
        rank1.append(float(NMIN * (m @ m) / tot) if tot > 0 else 0.0)

    stab, rank1, norms = np.array(stab), np.array(rank1), np.array(norms)
    print(f"  [1] STABLE real split-half cosine:    "
          f"median {np.median(stab):+.3f}  q10 {np.percentile(stab,10):+.3f}  "
          f"q90 {np.percentile(stab,90):+.3f}", flush=True)
    print(f"      fraction with cosine > 0.3: {(stab>0.3).mean():.2f}", flush=True)
    print(f"  [2] RANK1  mean-direction share of deviation energy: "
          f"median {np.median(rank1):.4f}", flush=True)

    # ---- TEST 3: BASIS COMPARISON -------------------------------------------
    # Only environments whose direction is estimable at all.
    ok = [e for e, s in zip(envs, stab) if s > 0.2]
    print(f"  [3] BASIS  using {len(ok)} environments with stable direction", flush=True)

    idx = rng.permutation(len(ok))
    ntr = int(0.8 * len(ok))
    tr = [ok[i] for i in idx[:ntr]]
    te = [ok[i] for i in idx[ntr:]]

    M_tr = np.array([shift(rng.choice(r, NMIN, replace=False), t) for _, r, t in tr])
    M_te = np.array([shift(rng.choice(r, NMIN, replace=False), t) for _, r, t in te])

    _, _, Vt_shift = np.linalg.svd(M_tr, full_matrices=False)          # shift basis
    Xc = X[ctrl] - ctrl_mu
    _, _, Vt_pca   = np.linalg.svd(Xc, full_matrices=False)            # control-PCA basis

    def recon_r2(M, V, d):
        W = V[:d].T
        P = (M @ W) @ W.T
        ss_res = ((M - P)**2).sum(); ss_tot = (M**2).sum()
        return float(1 - ss_res/ss_tot)

    basis = {}
    for d in DIMS:
        rs, rp = recon_r2(M_te, Vt_shift, d), recon_r2(M_te, Vt_pca, d)
        basis[d] = {"shift_basis_r2": rs, "control_pca_r2": rp, "gain": rs - rp}
        print(f"      d={d:>3}  shift-basis R2={rs:.4f}   control-PCA R2={rp:.4f}   "
              f"gain={rs-rp:+.4f}", flush=True)

    res = dict(dataset=ds, n_envs=len(envs), n_stable=len(ok), nmin=NMIN,
               gate_cos_median=float(np.median(gate)), gate_cos_mean=float(gate.mean()),
               stab_cos_median=float(np.median(stab)),
               stab_cos_q10=float(np.percentile(stab,10)),
               stab_cos_q90=float(np.percentile(stab,90)),
               frac_cos_gt_0p3=float((stab>0.3).mean()),
               rank1_median=float(np.median(rank1)),
               shift_norm_median=float(np.median(norms)),
               basis={str(k): v for k, v in basis.items()})
    tmp = out_f + ".tmp"
    json.dump(res, open(tmp, "w"), indent=2); os.rename(tmp, out_f)
    return res


if __name__ == "__main__":
    for ds in ["k562", "rpe1"]:
        try: run(ds)
        except Exception as e:
            print(f"FAIL {ds}: {type(e).__name__}: {e}", flush=True)
    print("\nALL DONE", flush=True)
