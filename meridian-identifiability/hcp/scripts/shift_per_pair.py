"""Per-task-pair shift, ranked. Pooling 21 pairs hides whether a few contrasts
carry real signal. Sample-size matched: both null and signal use half-length fits."""
import glob, itertools, os, numpy as np
TS="/workspace/hcp/ts"; TASKS=["WM","GAMBLING","MOTOR","LANGUAGE","SOCIAL","RELATIONAL","EMOTION"]
D=20; RIDGE=1e-2; ENC=os.environ.get("ENC","LR")
def load(s,t): 
    p=f"{TS}/{s}_{t}_{ENC}.npy"; return np.load(p) if os.path.exists(p) else None
def zn(x): return (x-x.mean(0))/(x.std(0)+1e-8)
def var(z):
    X,Y=z[:-1],z[1:]; G=X.T@X+RIDGE*len(X)*np.eye(z.shape[1])
    return np.linalg.solve(G,X.T@Y).T
subs=sorted({os.path.basename(f).split("_")[0] for f in glob.glob(f"{TS}/*.npy")})
subs=[s for s in subs if all(load(s,t) is not None for t in TASKS)]
pool=np.concatenate([zn(load(s,t)) for s in subs for t in TASKS],0); pool-=pool.mean(0)
W=np.linalg.svd(pool,full_matrices=False)[2][:D].T
H={}
for s in subs:
    for t in TASKS:
        z=zn(load(s,t))@W; h=len(z)//2
        H[(s,t)]=(var(z[:h]),var(z[h:]))
within=np.concatenate([np.abs(H[(s,t)][0]-H[(s,t)][1]).ravel() for s in subs for t in TASKS])
thr=np.percentile(within,95); wmed=np.median(within)
print(f"{len(subs)} subjects, d={D}, enc={ENC}, null median {wmed:.4f}\n")
rows=[]
for t1,t2 in itertools.combinations(TASKS,2):
    b=np.concatenate([np.abs(H[(s,t1)][k]-H[(s,t2)][k]).ravel() for s in subs for k in (0,1)])
    rows.append((np.median(b)/wmed, float((b>thr).mean()), t1, t2))
rows.sort(reverse=True)
print(f"{'pair':<26} {'ratio':>6} {'shift frac':>11}")
for r,f,t1,t2 in rows: print(f"{t1+' vs '+t2:<26} {r:>6.2f} {f:>11.3f}")
print(f"\nratio = between-task / within-task median coefficient difference.")
print(f"~1.0 = no more than noise. >2 = a genuinely strong contrast worth using as an environment.")
