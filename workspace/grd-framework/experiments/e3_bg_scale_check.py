import numpy as np, anndata as ad
from data_paths import perturbseq_path
A=ad.read_h5ad(perturbseq_path('causalbench_k562.h5ad'))
X=A.X.toarray().astype(np.float64) if hasattr(A.X,"toarray") else np.asarray(A.X,np.float64)
g=A.obs['guide_ids'].astype(str).values
Xc=X[g=='']
del A,X
D=10
mu=Xc.mean(0)
U,S,Vt=np.linalg.svd(Xc-mu,full_matrices=False)
Bp=Vt[:D].T
Yc=(Xc-mu)@Bp
print("real K562 control, projected to D=10:")
print("  n cells:", len(Yc))
print("  per-dim variance:", np.round(Yc.var(0),3))
print("  total background variance:", round(float(Yc.var(0).sum()),3))
