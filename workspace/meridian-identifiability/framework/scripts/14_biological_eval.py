"""Do the same baselines hold up against CURATED biological networks?

The statistical evaluator turned out to be an effect-reproducibility test: a trivial
differential-expression baseline scores 0.966 (K562) / 0.820 (RPE1). It also cannot
separate direct from indirect causation -- if A->B->C, perturbing A moves C, so A->C
counts as a true positive.

CORUM / StringDB / ChIP-Atlas are different: curated interaction records, not effect
measurements. Getting an edge right requires structural knowledge that effect detection
alone should not supply. This is the test that decides whether CausalBench poses a real
causal-discovery task or only an effect-detection task.

Their Evaluator returns ONLY true_positives, so precision is computed here directly
against the ground-truth edge set, at a MATCHED edge budget across all methods.

Baselines:
  random          null
  corr_top        association only
  empirical_shift differential expression -- won the statistical metric outright
  emp_shift_TC    transitive closure of the above. If this BEATS empirical_shift on the
                  statistical metric but LOSES here, the indirect-edge point is proven
                  with numbers rather than argument.
  hub             edges out of the most-perturbed genes
  program_shift   d=15 shift-basis clusters, dense DAG between them
  program_pca     same on control PCA -- isolates whether the shift basis matters
"""
import os, sys, json, itertools
import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, "/workspace/external/causalbench_repo")
from causalscbench.data_access.create_evaluation_datasets import CreateEvaluationDatasets

D       = "/workspace/meridian-identifiability/causalbench/data"
OUT     = "/workspace/meridian-identifiability/framework/results/bio_eval"
CTRL    = "non-targeting"
BUDGET  = 5000
D_LAT   = 15
NMIN    = 200
SEED    = 0
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)


def load(ds):
    d = np.load(os.path.join(D, f"dataset_{ds}.npz"), allow_pickle=True)
    X  = d["expression_matrix"].astype(np.float64)
    iv = np.asarray(d["interventions"]).astype(str)
    vn = [str(v) for v in d["var_names"]]
    keep = iv != "excluded"
    return X[keep], iv[keep], vn


def cap(edges, n):
    edges = list(dict.fromkeys(edges))
    if len(edges) > n:
        sel = rng.choice(len(edges), n, replace=False)
        edges = [edges[i] for i in sorted(sel)]
    return edges


def score(net, truth, directed=False):
    """Precision + recall against a curated edge set, computed here."""
    if not net: return dict(n_edges=0, tp=0, precision=None, recall=None)
    S = set(net) if directed else {(a, b) for a, b in net} | {(b, a) for a, b in net}
    T = set(truth) if directed else {(a, b) for a, b in truth} | {(b, a) for a, b in truth}
    hits = S & T
    tp = len(hits) if directed else len(hits) / 2
    n  = len(net)
    tt = len(T) if directed else len(T) / 2
    return dict(n_edges=n, tp=float(tp),
                precision=float(tp / n) if n else None,
                recall=float(tp / tt) if tt else None)


def build_networks(ds):
    X, iv, genes = load(ds)
    gset, gidx = set(genes), {g: i for i, g in enumerate(genes)}
    ctrl_mu = X[iv == CTRL].mean(0)

    us, cs = np.unique(iv, return_counts=True)
    envs = [(g, np.where(iv == g)[0], gidx.get(g))
            for g, c in zip(us, cs) if g != CTRL and c >= NMIN]
    src = [g for g, _, _ in envs]
    print(f"  {len(envs)} environments >= {NMIN} cells", flush=True)

    def shift(rows, drop):
        v = X[rows].mean(0) - ctrl_mu
        if drop is not None:
            v = v.copy(); v[drop] = 0.0
        return v

    M = np.array([shift(r, t) for _, r, t in envs])
    _, _, Vt_shift = np.linalg.svd(M, full_matrices=False)
    _, _, Vt_pca   = np.linalg.svd(X[iv == CTRL] - ctrl_mu, full_matrices=False)

    lab_s = KMeans(D_LAT, n_init=10, random_state=SEED).fit_predict(Vt_shift[:D_LAT].T)
    lab_p = KMeans(D_LAT, n_init=10, random_state=SEED).fit_predict(Vt_pca[:D_LAT].T)

    def prog(lab):
        mem = {k: [genes[i] for i in np.where(lab == k)[0]] for k in range(D_LAT)}
        e = [(a, b) for i in range(D_LAT) for j in range(D_LAT) if i < j
             for a in mem[i] for b in mem[j] if a in set(src)]
        return cap(e, BUDGET)

    nets = {}
    nets["random"] = cap([(src[rng.integers(len(src))], genes[rng.integers(len(genes))])
                          for _ in range(BUDGET * 3)], BUDGET)

    C = np.corrcoef(X.T); np.fill_diagonal(C, 0.0)
    order = np.dstack(np.unravel_index(np.argsort(-np.abs(C), axis=None), C.shape))[0]
    ce = []
    for i, j in order:
        if genes[i] in set(src): ce.append((genes[i], genes[j]))
        if len(ce) >= BUDGET: break
    nets["corr_top"] = ce

    scored = []
    for g, rows, drop in envs:
        v = np.abs(shift(rows, drop))
        for j in np.argsort(-v)[:60]:
            if genes[j] != g: scored.append((v[j], g, genes[j]))
    scored.sort(reverse=True)
    emp = [(a, b) for _, a, b in scored[:BUDGET]]
    nets["empirical_shift"] = emp

    # transitive closure of the empirical shift graph
    adj = {}
    for a, b in emp: adj.setdefault(a, set()).add(b)
    tc = set(emp)
    for a in list(adj):
        for b in list(adj[a]):
            for c in adj.get(b, ()):
                if c != a: tc.add((a, c))
    nets["emp_shift_TC"] = cap(sorted(tc), BUDGET)

    deg = sorted(src, key=lambda g: -int((iv == g).sum()))
    nets["hub"] = cap([(s, t) for s in deg[:20] for t in genes if t != s], BUDGET)

    nets["program_shift"] = prog(lab_s)
    nets["program_pca"]   = prog(lab_p)

    nets = {k: [(a, b) for a, b in v if a in gset and b in gset and a != b]
            for k, v in nets.items()}
    return nets, genes


def run(ds):
    out_f = os.path.join(OUT, f"{ds}.json")
    if os.path.exists(out_f):
        print(f"SKIP {ds}"); return
    print(f"\n===== {ds} =====", flush=True)

    nets, genes = build_networks(ds)

    print("  loading curated networks...", flush=True)
    ced = CreateEvaluationDatasets(D, f"weissmann_{ds}")
    gs = set(genes)
    truths = {}
    # CORUM's host serves a broken cert chain and now redirects to HTML (2026).
    # It is protein-complex co-membership, not regulation, so it was the weakest
    # ground truth here regardless.
    loaders = [m for m in dir(ced) if m.startswith("_load") and "corum" not in m]
    print(f"  loaders: {loaders}", flush=True)
    for m in loaders:
        try:
            t = getattr(ced, m)()
        except Exception as e:
            print(f"    {m}: FAILED ({type(e).__name__})", flush=True)
            continue
        parts = list(t) if isinstance(t, tuple) else [t]
        for k, sub in enumerate(parts):
            try:
                edges = {(a, b) for a, b in sub if a in gs and b in gs and a != b}
            except Exception:
                continue
            if edges:
                nm = m.replace("_load_", "") + (f"_{k}" if len(parts) > 1 else "")
                truths[nm] = edges
                print(f"    {nm}: {len(edges)} edges", flush=True)
    print(f"  curated sets usable: {sorted(truths)}", flush=True)

    res = {}
    for tname, truth in truths.items():
        print(f"\n  --- {tname}  ({len(truth)} ground-truth edges) ---", flush=True)
        print(f"  {'method':<18}{'edges':>7}{'TP':>8}{'precision':>11}{'recall':>10}", flush=True)
        res[tname] = {}
        for mname, net in nets.items():
            s = score(net, truth)
            res[tname][mname] = s
            p = f"{s['precision']:.4f}" if s['precision'] is not None else "-"
            r = f"{s['recall']:.4f}"    if s['recall']    is not None else "-"
            print(f"  {mname:<18}{s['n_edges']:>7}{s['tp']:>8.0f}{p:>11}{r:>10}", flush=True)

    json.dump({"dataset": ds, "budget": BUDGET, "d_latent": D_LAT, "results": res},
              open(out_f + ".tmp", "w"), indent=2)
    os.rename(out_f + ".tmp", out_f)


if __name__ == "__main__":
    for ds in ["k562", "rpe1"]:
        try: run(ds)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"FAIL {ds}: {e}", flush=True)
    print("\nALL DONE", flush=True)
