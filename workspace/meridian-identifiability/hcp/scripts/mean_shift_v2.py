"""Does cognitive task shift the latent mean beyond acquisition-run noise?

Three distances, all on 88-frame half-run means, matched sample size:
  split   - same task, same run, two halves            (estimation noise only)
  encode  - same task, DIFFERENT run (LR vs RL)        (+ run-level offset/drift)
  task    - DIFFERENT task, same run                   (+ condition effect)

'encode' is the load-bearing null. Raw BOLD carries arbitrary per-run baseline and
gain, so task-vs-task differences include run nuisance. LR-vs-RL of the SAME task
contains that nuisance and no condition effect. If task/encode ~ 1, the apparent
condition effect is acquisition, not cognition.

Two scalings: raw (nuisance intact) and subject-pooled z-score (per-region stats
pooled over that subject's runs, so between-task offsets survive but subject scale
is removed). Per-run z-scoring is NOT used: it centers away the quantity measured.
"""
import glob, itertools, json, os
import numpy as np

TS="/workspace/meridian-identifiability/hcp/ts"
OUT="/workspace/meridian-identifiability/hcp/results"
TASKS=["WM","GAMBLING","MOTOR","LANGUAGE","SOCIAL","RELATIONAL","EMOTION"]
NF, H = 176, 88
DIMS=[10,20,30,50]

def ld(s,t,e):
    p=os.path.join(TS,f"{s}_{t}_{e}.npy")
    return np.load(p).astype(float)[:NF] if os.path.exists(p) else None

def main():
    subs=sorted({os.path.basename(f).split("_")[0] for f in glob.glob(f"{TS}/*.npy")})
    subs=[s for s in subs if all(ld(s,t,e) is not None for t in TASKS for e in ("LR","RL"))]
    print(f"{len(subs)} subjects with all 7 tasks x both encodings", flush=True)

    res={}
    for scaling in ["raw","subject_pooled"]:
        runs={}
        for s in subs:
            allr=np.concatenate([ld(s,t,e) for t in TASKS for e in ("LR","RL")],0)
            mu,sd=allr.mean(0),allr.std(0)+1e-8
            for t in TASKS:
                for e in ("LR","RL"):
                    x=ld(s,t,e)
                    runs[(s,t,e)]=(x-mu)/sd if scaling=="subject_pooled" else x

        pool=np.concatenate([runs[k] for k in runs],0); pool=pool-pool.mean(0)
        _,_,Vt=np.linalg.svd(pool,full_matrices=False)

        res[scaling]={}
        for d in DIMS:
            W=Vt[:d].T
            m={k:(v@W) for k,v in runs.items()}
            A={k:v[:H].mean(0) for k,v in m.items()}
            B={k:v[H:NF].mean(0) for k,v in m.items()}

            split =[np.linalg.norm(A[(s,t,e)]-B[(s,t,e)])
                    for s in subs for t in TASKS for e in ("LR","RL")]
            encode=[np.linalg.norm(A[(s,t,"LR")]-A[(s,t,"RL")]) for s in subs for t in TASKS]
            task  =[np.linalg.norm(A[(s,t1,e)]-A[(s,t2,e)])
                    for s in subs for e in ("LR","RL")
                    for t1,t2 in itertools.combinations(TASKS,2)]

            sp,en,tk=map(np.median,(split,encode,task))
            # spectrum: 7 group task means vs LR-RL null of same dimension
            M=np.array([np.mean([A[(s,t,"LR")] for s in subs],0) for t in TASKS]); M-=M.mean(0)
            N=np.array([np.mean([A[(s,t,"LR")]-A[(s,t,"RL")] for s in subs],0) for t in TASKS])
            N=(N-N.mean(0))/np.sqrt(2)
            ss=np.linalg.svd(M,compute_uv=False); sn=np.linalg.svd(N,compute_uv=False)
            k=min((sn>1e-12).sum(),len(ss)); sr=ss[:k]/sn[:k]

            res[scaling][d]=dict(split=float(sp),encode=float(en),task=float(tk),
                task_over_encode=float(tk/en), task_over_split=float(tk/sp),
                encode_over_split=float(en/sp), sing_ratio=sr.tolist(),
                dims_above_2x=int((sr>2).sum()), max_dims=int(k))
            print(f"  [{scaling:14s}] d={d:3d}  task/encode={tk/en:5.2f}  "
                  f"task/split={tk/sp:5.2f}  encode/split={en/sp:5.2f}  "
                  f"dims>2x {int((sr>2).sum())}/{k}  sr={np.round(sr,2).tolist()}", flush=True)

    p=os.path.join(OUT,"mean_shift_v2.json"); tmp=p+".tmp"
    json.dump(dict(n_subjects=len(subs),nframes=NF,tasks=TASKS,results=res),open(tmp,"w"),indent=2)
    os.rename(tmp,p); print(f"\nwrote {p}",flush=True)

if __name__=="__main__":
    main()
