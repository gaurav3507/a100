"""Batch-nuisance null for CausalBench, matched to HCP's encode-based null.

HCP (mean_shift_v2.json, subject_pooled, d=10): split=estimation noise,
encode=+acquisition nuisance, task=+condition. Reports task/encode because
encode/split = 14.58 -- nuisance dominates and is CONFOUNDED with the sample.

03_screen.py has split (within) and task (pair) analogues but no middle term.
This adds it: same perturbation, DIFFERENT gem_group.

PARITY WITH 03_screen.py (verified against its source):
  - PCA basis from CONTROL cells only
  - environment subsampled to nmin FIRST, then split at half
  - EVERY distance uses `half` cells per side
  - pairs drop BOTH environments' target columns
  - one draw per unit, then median
Sanity check must reproduce mean_ratio_pairs = 4.270 before anything else counts.
Control cells have no targeted gene, so no column is dropped there -- an asymmetry
vs the perturbation terms, to be stated in Methods.
"""
import os, json
import numpy as np

DATA = "/workspace/meridian-identifiability/causalbench/data"
OUT  = "/workspace/meridian-identifiability/causalbench/results/screen"
D, NMIN, SEED, N_PAIRS, MINB = 10, 200, 0, 2000, 10
HALF = NMIN // 2
os.makedirs(OUT, exist_ok=True)

d   = np.load(f"{DATA}/dataset_k562.npz", allow_pickle=True)
X   = d["expression_matrix"].astype(np.float64)
iv  = np.asarray(d["interventions"]).astype(str)
vn  = [str(v) for v in d["var_names"]]; gidx = {g: i for i, g in enumerate(vn)}
gem = np.load(f"{DATA}/k562_gem_group.npy", allow_pickle=True).astype(str)
assert len(gem) == len(iv), "row count mismatch"
keep = iv != "excluded"; X, iv, gem = X[keep], iv[keep], gem[keep]

ctrl = np.where(iv == "non-targeting")[0]
mu = X[ctrl].mean(0)
_, _, Vt = np.linalg.svd(X[ctrl] - mu, full_matrices=False)
W = Vt[:D].T
print(f"cells {X.shape[0]}  control {len(ctrl)}  batches {len(set(gem))}", flush=True)

def proj(rows, drops):
    Wl = W.copy()
    for t in drops:
        if t is not None: Wl[t, :] = 0.0
    return ((X[rows] - mu) @ Wl).mean(0)

envs = [(g, np.where(iv == g)[0], gidx.get(g))
        for g in np.unique(iv) if g != "non-targeting" and (iv == g).sum() >= NMIN]
print(f"{len(envs)} environments >= {NMIN} cells", flush=True)

# ---------- SANITY: reproduce 03_screen.py mean_ratio_pairs ----------
rs = np.random.default_rng(SEED)
within, pair = [], []
for g, r, t in envs:
    s_ = rs.choice(r, NMIN, replace=False)
    within.append(np.linalg.norm(proj(s_[:HALF], {t}) - proj(s_[HALF:], {t})))
for _ in range(N_PAIRS):
    i, j = rs.choice(len(envs), 2, replace=False)
    (_, ra, da), (_, rb, db) = envs[i], envs[j]
    dr = {x for x in (da, db) if x is not None}
    pair.append(np.linalg.norm(proj(rs.choice(ra, HALF, replace=False), dr) -
                               proj(rs.choice(rb, HALF, replace=False), dr)))
sanity = float(np.median(pair) / np.median(within))
ok = abs(sanity - 4.270) <= 0.25
print(f"\nSANITY mean_ratio_pairs = {sanity:.3f}  (03_screen.py 4.270)  "
      f"{'PASS' if ok else '*** DIVERGES -- ignore everything below ***'}", flush=True)

# ---------- CONTROL batch null (well powered) ----------
rc = np.random.default_rng(SEED + 1)
cb = {b: ctrl[gem[ctrl] == b] for b in set(gem[ctrl])}
cb = {b: v for b, v in cb.items() if len(v) >= 2 * HALF}
keys = sorted(cb)
cwit = []
for b in keys:                                     # one per batch
    v = rc.permutation(cb[b])
    cwit.append(np.linalg.norm(proj(v[:HALF], set()) - proj(v[HALF:2*HALF], set())))
cbat = []
for a in range(len(keys)):                         # one per unordered batch pair
    for b in range(a + 1, len(keys)):
        cbat.append(np.linalg.norm(
            proj(rc.choice(cb[keys[a]], HALF, replace=False), set()) -
            proj(rc.choice(cb[keys[b]], HALF, replace=False), set())))
ctrl_ratio = float(np.median(cbat) / np.median(cwit))
print(f"CONTROL batch/within = {ctrl_ratio:.3f}   "
      f"({len(keys)} batches, {len(cbat)} pairs, {HALF} cells/side)", flush=True)

# ---------- PER-PERTURBATION batch null (underpowered) ----------
rp = np.random.default_rng(SEED + 2)
pe = []
for g, r, t in envs:
    bs = {b: r[gem[r] == b] for b in set(gem[r])}
    bs = {b: v for b, v in bs.items() if len(v) >= MINB}
    if len(bs) >= 2: pe.append((g, r, t, bs))
pb, pw = [], []
for g, r, t, bs in pe:
    k1, k2 = sorted(bs, key=lambda k: -len(bs[k]))[:2]
    n = min(MINB, len(bs[k1]), len(bs[k2]))
    pb.append(np.linalg.norm(proj(rp.choice(bs[k1], n, replace=False), {t}) -
                             proj(rp.choice(bs[k2], n, replace=False), {t})))
    v = rp.permutation(bs[k1])
    if len(v) >= 2 * n:
        pw.append(np.linalg.norm(proj(v[:n], {t}) - proj(v[n:2*n], {t})))
pert_ratio = float(np.median(pb) / np.median(pw)) if pw else float("nan")
print(f"PERTURB batch/within = {pert_ratio:.3f}   ({len(pe)} perts, {MINB} cells/side "
      f"-- UNDERPOWERED, both terms noise-dominated)", flush=True)

print(f"\n  CausalBench nuisance : {ctrl_ratio:.2f}x estimation noise (batch RANDOMIZED across perturbations)")
print(f"  HCP nuisance         : 14.58x estimation noise (acquisition CONFOUNDED with sample)")

json.dump(dict(d=D, nmin=NMIN, half=HALF, n_pairs=N_PAIRS, seed=SEED,
               sanity_mean_ratio_pairs=sanity, sanity_target=4.270, sanity_pass=bool(ok),
               control_batch_over_within=ctrl_ratio,
               n_control_batches=len(keys), n_control_pairs=len(cbat),
               perturbation_batch_over_within=pert_ratio,
               n_perturbations=len(pe), perturbation_cells_per_side=MINB,
               perturbation_underpowered=True,
               hcp_encode_over_split_subject_pooled=14.58,
               hcp_task_over_encode_subject_pooled=1.058),
          open(f"{OUT}/k562_batch_null.json", "w"), indent=2)
print(f"\nwrote {OUT}/k562_batch_null.json", flush=True)
