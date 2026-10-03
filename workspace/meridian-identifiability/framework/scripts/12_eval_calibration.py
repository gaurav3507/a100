"""What do trivial baselines score on CausalBench's statistical evaluator?

Their BIOLOGICAL evaluator returns only true_positives -- no precision. So predicting
more edges monotonically improves the score, and the number is uninterpretable on its
own. Their STATISTICAL evaluator is the real one: for each predicted edge A->B it takes
HELD-OUT cells where A was actually perturbed, compares B's expression against control
via Mann-Whitney, and returns true AND false positives. Precision is computable.

Before training any model we establish what the following score, ALL AT A MATCHED EDGE
BUDGET (otherwise edge count alone drives the result):

  empty         no edges                     -- floor
  random        random gene pairs            -- the real null
  corr_top      highest |correlation| pairs  -- no causality, just association
  hub           edges out of the most-perturbing genes
  program_shift genes clustered on the d=15 SHIFT basis, dense DAG between clusters
  program_pca   same but clustered on control PCA -- isolates whether the SHIFT
                basis matters, or just any clustering

program_shift is a preview of what the framework will output, built from the basis we
already have with NO training. If the trained model cannot beat it, the training is
not doing the work.
"""
import os, json, sys
import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, "/workspace/external/causalbench_repo")
from causalscbench.evaluation import statistical_evaluation

D      = "/workspace/meridian-identifiability/causalbench/data"
OUT    = "/workspace/meridian-identifiability/framework/results/eval_calibration"
CTRL   = "non-targeting"
BUDGET = 5000     # every method predicts exactly this many edges
D_LAT  = 15
NMIN   = 200
SEED   = 0
MAXPATH = 1       # direct edges only -- the strict test
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
    """Trim or report; every method must predict exactly n edges."""
    edges = list(dict.fromkeys(edges))
    if len(edges) > n:
        idx = rng.choice(len(edges), n, replace=False)
        edges = [edges[i] for i in sorted(idx)]
    return edges


def clusters_to_edges(labels, genes, d, budget):
    """Dense DAG over clusters -> gene edges. Cluster i -> j for all i<j."""
    members = {k: [genes[i] for i in np.where(labels == k)[0]] for k in range(d)}
    edges = []
    for i in range(d):
        for j in range(d):
            if i >= j: continue
            for a in members[i]:
                for b in members[j]:
                    edges.append((a, b))
    return cap(edges, budget), {k: len(v) for k, v in members.items()}


def run(ds):
    out_f = os.path.join(OUT, f"{ds}.json")
    if os.path.exists(out_f):
        print(f"SKIP {ds}"); return

    X, iv, genes = load(ds)
    print(f"\n===== {ds} =====  X={X.shape}  genes={len(genes)}", flush=True)

    # ---- held-out split: evaluator must never see training cells -------------
    n = X.shape[0]
    perm = rng.permutation(n)
    te = perm[: n // 5]
    tr = perm[n // 5:]
    Xte, ivte = X[te], iv[te]
    Xtr, ivtr = X[tr], iv[tr]

    ev = statistical_evaluation.Evaluator(Xte, list(ivte), genes)
    perturbed_te = set(ivte) - {CTRL}
    print(f"  test cells {len(te)}  source genes usable as edge parents: {len(perturbed_te)}",
          flush=True)

    # every predicted edge's SOURCE must have been perturbed in the test split,
    # otherwise the evaluator has no interventional cells to test against
    valid_src = [g for g in genes if g in perturbed_te]
    gset = set(genes)

    # ---- build the shift basis on TRAIN only --------------------------------
    ctrl_tr = np.where(ivtr == CTRL)[0]
    ctrl_mu = Xtr[ctrl_tr].mean(0)
    gidx = {g: i for i, g in enumerate(genes)}
    us, cs = np.unique(ivtr, return_counts=True)
    envs = [(g, np.where(ivtr == g)[0], gidx.get(g))
            for g, c in zip(us, cs) if g != CTRL and c >= NMIN]

    M = []
    for g, rows, drop in envs:
        v = Xtr[rows].mean(0) - ctrl_mu
        if drop is not None:
            v = v.copy(); v[drop] = 0.0
        M.append(v)
    M = np.array(M)
    _, _, Vt_shift = np.linalg.svd(M, full_matrices=False)
    _, _, Vt_pca   = np.linalg.svd(Xtr[ctrl_tr] - ctrl_mu, full_matrices=False)
    print(f"  shift basis from {len(envs)} train environments", flush=True)

    # gene loadings in each basis -> cluster genes into D_LAT programs
    lab_shift = KMeans(D_LAT, n_init=10, random_state=SEED).fit_predict(Vt_shift[:D_LAT].T)
    lab_pca   = KMeans(D_LAT, n_init=10, random_state=SEED).fit_predict(Vt_pca[:D_LAT].T)

    # ---- candidate networks -------------------------------------------------
    nets = {}
    nets["empty"] = []

    nets["random"] = cap([(rng.choice(valid_src), genes[rng.integers(len(genes))])
                          for _ in range(BUDGET * 3)], BUDGET)

    C = np.corrcoef(Xtr.T)
    np.fill_diagonal(C, 0.0)
    flat = np.dstack(np.unravel_index(np.argsort(-np.abs(C), axis=None), C.shape))[0]
    corr_edges = []
    for i, j in flat:
        if genes[i] in perturbed_te:
            corr_edges.append((genes[i], genes[j]))
        if len(corr_edges) >= BUDGET: break
    nets["corr_top"] = corr_edges

    deg = sorted(valid_src, key=lambda g: -int((ivtr == g).sum()))
    hub = [(s, t) for s in deg[:20] for t in genes if t != s]
    nets["hub"] = cap(hub, BUDGET)

    e_s, sz_s = clusters_to_edges(lab_shift, genes, D_LAT, BUDGET)
    e_p, sz_p = clusters_to_edges(lab_pca,   genes, D_LAT, BUDGET)
    nets["program_shift"] = [(a, b) for a, b in e_s if a in perturbed_te]
    nets["program_pca"]   = [(a, b) for a, b in e_p if a in perturbed_te]
    print(f"  shift cluster sizes: {sorted(sz_s.values(), reverse=True)}", flush=True)
    print(f"  pca   cluster sizes: {sorted(sz_p.values(), reverse=True)}", flush=True)

    # ---- evaluate -----------------------------------------------------------
    res = {}
    print(f"\n  {'method':<16}{'edges':>7}{'TP':>8}{'FP':>8}{'precision':>11}", flush=True)
    for name, net in nets.items():
        net = [(a, b) for a, b in net if a in perturbed_te and b in gset and a != b]
        if not net:
            print(f"  {name:<16}{0:>7}{'-':>8}{'-':>8}{'-':>11}", flush=True)
            res[name] = {"n_edges": 0}; continue
        out = ev.evaluate_network(net, max_path_length=MAXPATH)

        def flat(d, pre=""):
            f = {}
            for k, v in d.items():
                key = f"{pre}{k}"
                if isinstance(v, dict): f.update(flat(v, key + "."))
                else:
                    try: f[key] = float(v)
                    except (TypeError, ValueError): pass
            return f
        F = flat(out)
        if name == "random":
            print(f"    [raw keys] {sorted(F.keys())}", flush=True)

        tp_k = [k for k in F if "true_positive" in k.lower()]
        fp_k = [k for k in F if "false_positive" in k.lower()]
        res[name] = {"n_edges": len(net), "raw": F}
        line = f"  {name:<16}{len(net):>7}"
        for tk in sorted(tp_k):
            base = tk.replace("true_positive", "false_positive")
            tp = F[tk]; fp = F.get(base, np.nan)
            prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            res[name][f"precision[{tk}]"] = None if prec != prec else float(prec)
            line += f"  {tk.split('.')[-1] if '.' in tk else tk}={tp:.0f}/{fp:.0f} p={prec:.4f}"
        print(line, flush=True)

    json.dump({"dataset": ds, "budget": BUDGET, "d_latent": D_LAT,
               "max_path_length": MAXPATH, "results": res},
              open(out_f + ".tmp", "w"), indent=2)
    os.rename(out_f + ".tmp", out_f)


if __name__ == "__main__":
    for ds in ["k562", "rpe1"]:
        try: run(ds)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"FAIL {ds}: {e}", flush=True)
    print("\nALL DONE", flush=True)
