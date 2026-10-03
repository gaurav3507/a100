"""Mean-shift analogue of shift_fraction.py, matched to the CausalBench screen.

Coefficient shift turned out not to discriminate working from non-working domains
(CausalBench scored 1.07-1.22, HCP 1.06-1.31). Mean shift did (CausalBench 2.0-5.6
against a ~1.0 null). This measures HCP on that second ruler.

Two numbers:
  mean_ratio  - between-task mean distance / within-task split-half mean distance.
                Directly comparable to CausalBench's mean_ratio_pairs.
  spectrum    - how many independent directions the 7 tasks span. Rank <= 6 by
                construction (7 environments, centered). CausalBench got ~15 from
                385 environments.

Matched sample size throughout: every task truncated to NFRAMES, split 88/88.
Frames are autocorrelated so a half-mean is noisier than 88 independent samples,
but the within-task null carries the identical autocorrelation, so the RATIO is valid.
"""
import glob, itertools, json, os
import numpy as np

TS   = "/workspace/meridian-identifiability/hcp/ts"
OUT  = "/workspace/meridian-identifiability/hcp/results"
TASKS = ["WM", "GAMBLING", "MOTOR", "LANGUAGE", "SOCIAL", "RELATIONAL", "EMOTION"]
NFRAMES = 176          # EMOTION is the shortest run; match everything to it
DIMS = [10, 20, 30, 50]

def load(s, t, e):
    p = os.path.join(TS, f"{s}_{t}_{e}.npy")
    return np.load(p) if os.path.exists(p) else None

def znorm(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-8)

def run(enc):
    subs = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(f"{TS}/*.npy")})
    subs = [s for s in subs if all(load(s, t, enc) is not None for t in TASKS)]
    print(f"\n===== enc {enc}: {len(subs)} subjects =====", flush=True)

    pool = np.concatenate([znorm(load(s, t, enc))[:NFRAMES] for s in subs for t in TASKS], 0)
    pool -= pool.mean(0)
    _, _, Vt = np.linalg.svd(pool, full_matrices=False)

    out = {}
    for d in DIMS:
        W = Vt[:d].T
        h = NFRAMES // 2
        A = np.zeros((len(subs), len(TASKS), d))   # first half
        B = np.zeros((len(subs), len(TASKS), d))   # second half
        for i, s in enumerate(subs):
            for j, t in enumerate(TASKS):
                z = (znorm(load(s, t, enc))[:NFRAMES]) @ W
                A[i, j] = z[:h].mean(0)
                B[i, j] = z[h:NFRAMES].mean(0)

        within  = [np.linalg.norm(A[i, j] - B[i, j])
                   for i in range(len(subs)) for j in range(len(TASKS))]
        between = [np.linalg.norm(A[i, j1] - A[i, j2])
                   for i in range(len(subs)) for j1, j2 in itertools.combinations(range(len(TASKS)), 2)]
        between += [np.linalg.norm(B[i, j1] - B[i, j2])
                    for i in range(len(subs)) for j1, j2 in itertools.combinations(range(len(TASKS)), 2)]
        ratio = float(np.median(between) / (np.median(within) + 1e-12))

        # spectrum: 7 group-level task directions vs a within-task noise null.
        # null rows are (a-b) so their noise is sqrt(2) x a half-mean's -> rescale.
        M = A.mean(0); M = M - M.mean(0)
        N = (A - B).mean(0) / np.sqrt(2); N = N - N.mean(0)
        s_sig = np.linalg.svd(M, compute_uv=False)
        s_noi = np.linalg.svd(N, compute_uv=False)
        k = min((s_noi > 1e-12).sum(), len(s_sig))
        sr = (s_sig[:k] / s_noi[:k])

        out[d] = dict(mean_ratio=ratio,
                      within_median=float(np.median(within)),
                      between_median=float(np.median(between)),
                      sing_ratio=sr.tolist(),
                      n_dims_above_noise=int((sr > 1).sum()),
                      n_dims_above_2x=int((sr > 2).sum()),
                      max_possible_dims=int(k),
                      participation_ratio=float(s_sig.sum()**2 / (s_sig**2).sum()),
                      s2_over_s1=float(s_sig[1]/s_sig[0]))
        print(f"  d={d:3d}  mean_ratio={ratio:5.2f}   "
              f"dims>2x: {out[d]['n_dims_above_2x']}/{k}   "
              f"PR={out[d]['participation_ratio']:.2f}   "
              f"sing_ratio={np.round(sr,2).tolist()}", flush=True)

    p = os.path.join(OUT, f"mean_shift_{enc}.json")
    tmp = p + ".tmp"
    json.dump(dict(enc=enc, n_subjects=len(subs), nframes=NFRAMES,
                   tasks=TASKS, results=out), open(tmp, "w"), indent=2)
    os.rename(tmp, p)
    print(f"  wrote {p}", flush=True)

if __name__ == "__main__":
    for e in ["LR", "RL"]:
        run(e)
    print("\nALL DONE", flush=True)
