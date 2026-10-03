import numpy as np
from causalscbench.data_access.create_dataset import CreateDataset

D = "/workspace/meridian-identifiability/causalbench/data"

for filt in [False, True]:
    print(f"\n===== filter={filt} =====", flush=True)
    k, r = CreateDataset(D, filter=filt).load()
    for path in [k, r]:
        d = np.load(path, allow_pickle=True)
        X = d["expression_matrix"]; iv = d["interventions"]; vn = d["var_names"]
        u, c = np.unique(iv, return_counts=True)
        keep = u != "excluded"
        print(f"{path}", flush=True)
        print(f"  X: {X.shape}  genes(cols): {len(vn)}", flush=True)
        print(f"  environments (excl 'excluded'): {keep.sum()}", flush=True)
        print(f"  non-targeting cells: {int(c[u=='non-targeting'][0]) if 'non-targeting' in u else 0}", flush=True)
        print(f"  cells labelled 'excluded': {int(c[u=='excluded'][0]) if 'excluded' in u else 0}", flush=True)
        n = c[keep & (u != "non-targeting")]
        print(f"  per-env cells p10/p50/p90: {np.percentile(n,10):.0f}/{np.percentile(n,50):.0f}/{np.percentile(n,90):.0f}", flush=True)
        for t in [100, 200]:
            print(f"  envs >={t}: {(n>=t).sum()}", flush=True)
print("\nALL DONE", flush=True)
