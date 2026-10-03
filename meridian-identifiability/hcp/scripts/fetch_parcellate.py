"""Download HCP task dtseries -> parcellate to Schaefer-200 -> save -> delete raw.
Resumable: skips subject/task/run already saved. Atomic writes. Disk stays flat.
Usage:  python fetch_parcellate.py <n_subjects> [start_index]
"""
import os, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parcellate import load_atlas, parcellate

BUCKET = "s3://hcp-openaccess/HCP_1200"
RAW = "/workspace/hcp/raw"
TS  = "/workspace/hcp/ts"
SUBJ_FILE = "/workspace/hcp/subjects.txt"
TASKS = ["WM", "GAMBLING", "MOTOR", "LANGUAGE", "SOCIAL", "RELATIONAL", "EMOTION"]
ENCS  = ["LR", "RL"]

def log(m): print(m, flush=True)

def s3_cp(key, dest):
    r = subprocess.run(["aws", "s3", "cp", key, dest, "--request-payer", "requester",
                        "--only-show-errors"], capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip()

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    os.makedirs(RAW, exist_ok=True); os.makedirs(TS, exist_ok=True)
    subjects = [l.strip() for l in open(SUBJ_FILE) if l.strip()][start:start + n]
    atlas = load_atlas()
    log(f"{len(subjects)} subjects, {len(TASKS)} tasks x {len(ENCS)} encodings")
    t_all = time.time()
    for si, subj in enumerate(subjects):
        for task in TASKS:
            for enc in ENCS:
                run = f"tfMRI_{task}_{enc}"
                out = os.path.join(TS, f"{subj}_{task}_{enc}.npy")
                if os.path.exists(out):
                    continue
                fn  = f"{run}_Atlas_MSMAll.dtseries.nii"
                key = f"{BUCKET}/{subj}/MNINonLinear/Results/{run}/{fn}"
                local = os.path.join(RAW, f"{subj}_{fn}")
                t0 = time.time()
                ok, err = s3_cp(key, local)
                if not ok:
                    log(f"  [miss] {subj} {run}: {err.splitlines()[-1][:80] if err else 'download failed'}")
                    continue
                try:
                    ts, counts = parcellate(local, atlas)
                    tmp = out + ".tmp.npy"
                    np.save(tmp, ts); os.replace(tmp, out)
                    lag1 = float(np.mean([np.corrcoef(ts[:-1,i], ts[1:,i])[0,1]
                                          for i in range(ts.shape[1])]))
                    log(f"  [ok  ] {subj} {run}: T={ts.shape[0]} lag1={lag1:.3f} "
                        f"({time.time()-t0:.0f}s)")
                except Exception as e:
                    log(f"  [FAIL] {subj} {run}: parcellate error {e}")
                finally:
                    if os.path.exists(local):
                        os.remove(local)
        done = len([f for f in os.listdir(TS) if f.startswith(subj)])
        log(f"[subj {si+1}/{len(subjects)}] {subj}: {done}/14 runs  "
            f"(elapsed {(time.time()-t_all)/60:.1f} min)")
    log(f"DONE in {(time.time()-t_all)/60:.1f} min")

if __name__ == "__main__":
    main()
