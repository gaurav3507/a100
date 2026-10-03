#!/usr/bin/env python3
"""
CAFE: EDTalk feature extraction for the MEAD FPS25 tree.

Why not prep_video.py directly (verified against the released source):
  - its MEAD glob `*/{angle}/*/*/*.mp4` is commented out; the active glob is the
    flat CREMA-D one `data_root/*.mp4`, so on MEAD it finds 0 videos;
  - its loop unpacks the path into exactly 5 parts (CREMA-D layout), which
    raises on the 8-part MEAD path;
  - it prints and silently skips any per-video exception.

This driver reuses C-MET's EXACT components and settings, unmodified:
  sys.path 'src'; src.EDTalk.networks.generator.Generator(size=256,
  style_dim=512, lip_dim=20, pose_dim=6, exp_dim=10, channel_multiplier=1);
  weights torch.load('pretrained_weights/EDTalk.pt')['gen']; gen.eval();
  src.util.vid_preprocessing; gen.compute_alpha_D on the (T,C,H,W) clip
  ('full' mode, with their 'batch' mode of 100 frames as the OOM fallback).
Outputs are written exactly where C-MET expects them, next to each clip:
  <num>_ED_exp.npy, <num>_ED_pose.npy, <num>_ED_lip.npy.

Added safety: all three arrays are validated (2-D, first dim == frame count,
finite, trailing dims consistent across clips) before anything is written;
writes go to temp files then atomic rename; clips with all three outputs are
skipped (resume); every failure is recorded with a reason in
edtalk_failures.txt instead of being swallowed.

Run from the C-MET repo root (C_MET env):
  python ../extract_edtalk_mead.py --fps25_root dataset/MEAD/FPS25 --limit 3
  python ../extract_edtalk_mead.py --fps25_root dataset/MEAD/FPS25
  python ../extract_edtalk_mead.py --selftest       # mocks, no GPU/torch
"""

import argparse, glob, os, sys, time

SUFFIXES = ("_ED_exp.npy", "_ED_pose.npy", "_ED_lip.npy")
TMP = ".tmp.npy"


def discover(fps25_root, angle="front"):
    pat = os.path.join(fps25_root, "*", angle, "*", "*", "*.mp4")
    return sorted(v for v in glob.glob(pat) if ".tmp" not in os.path.basename(v))

def out_paths(video):
    stem = video[:-4]
    return [stem + s for s in SUFFIXES]

def all_exist(video):
    return all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in out_paths(video))


# ---- real components (lazy: only imported for real runs) -------------------

class RealBackend:
    def __init__(self):
        sys.path.insert(0, os.getcwd()); sys.path.append("src")
        import torch
        import numpy as np
        from src.EDTalk.networks.generator import Generator as EDTalk_Generator
        from src.util import vid_preprocessing
        self.torch, self.np, self.pre = torch, np, vid_preprocessing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gen = EDTalk_Generator(size=256, style_dim=512, lip_dim=20, pose_dim=6,
                               exp_dim=10, channel_multiplier=1).to(self.device)
        w = torch.load("pretrained_weights/EDTalk.pt",
                       map_location=lambda s, l: s)["gen"]
        gen.load_state_dict(w)
        gen.eval()
        self.gen = gen

    def load(self, path):
        vid, fps = self.pre(path)          # (1, T, C, H, W)
        return vid, fps, int(vid.shape[1])

    def compute(self, vid, mode):
        torch, np = self.torch, self.np
        vid = vid.to(self.device)
        with torch.no_grad():
            if mode == "full":
                v = vid.view(-1, vid.size(2), vid.size(3), vid.size(4))
                e, p, l = self.gen.compute_alpha_D(v)
                out = [e.cpu().numpy(), p.cpu().numpy(), l.cpu().numpy()]
            else:                           # their 'batch' mode, 100 frames
                parts = [[], [], []]
                for i in range(0, vid.shape[1], 100):
                    vb = vid[:, i:i + 100].reshape(-1, vid.size(2), vid.size(3), vid.size(4))
                    for k, t in enumerate(self.gen.compute_alpha_D(vb)):
                        parts[k].append(t.cpu().numpy())
                out = [np.concatenate(x, axis=0) for x in parts]
        del vid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    def save(self, path, arr):
        self.np.save(path, arr)

    def is_oom(self, exc):
        return "out of memory" in str(exc).lower()


# ---- per-clip + batch logic (unit-tested with a mock backend) -------------

def _valid(arrs, T, ref_dims, np_isfinite):
    for a in arrs:
        if getattr(a, "ndim", None) != 2:
            return f"bad_ndim_{getattr(a, 'ndim', None)}"
        if a.shape[0] != T:
            return f"frame_mismatch_{a.shape[0]}_vs_{T}"
        if not np_isfinite(a):
            return "non_finite"
    dims = tuple(a.shape[1] for a in arrs)
    if ref_dims[0] is not None and dims != ref_dims[0]:
        return f"dims_{dims}_vs_{ref_dims[0]}"
    return None

def process_one(video, backend, ref_dims, np_isfinite):
    outs = out_paths(video)
    tmps = [p + TMP for p in outs]
    for t in tmps:
        if os.path.exists(t):
            os.remove(t)
    vid, fps, T = backend.load(video)
    if T <= 0:
        return False, "zero_frames"
    try:
        arrs = backend.compute(vid, "full")
    except Exception as e:
        if not backend.is_oom(e):
            raise
        arrs = backend.compute(vid, "batch")
    bad = _valid(arrs, T, ref_dims, np_isfinite)
    if bad:
        return False, bad
    if ref_dims[0] is None:
        ref_dims[0] = tuple(a.shape[1] for a in arrs)
    try:
        for t, a in zip(tmps, arrs):
            backend.save(t, a)
        for t, o in zip(tmps, outs):
            os.replace(t, o)
    finally:
        for t in tmps:
            if os.path.exists(t):
                os.remove(t)
    return True, f"ok T={T} fps={fps}"

def run(fps25_root, backend, np_isfinite, limit=None, log=print, shard=0, nshards=1):
    videos = discover(fps25_root)
    # partition on the FULL sorted list (stable), so shards are fixed and disjoint
    # no matter how many clips are already done or which shards run concurrently
    mine = [v for k, v in enumerate(videos) if k % nshards == shard]
    pending = [v for v in mine if not all_exist(v)]
    already = len(mine) - len(pending)
    if limit is not None:
        pending = pending[:limit]
    ref_dims = [None]
    ok_n, failures = 0, []
    for i, v in enumerate(pending, 1):
        try:
            ok, reason = process_one(v, backend, ref_dims, np_isfinite)
        except Exception as e:
            ok, reason = False, f"exception_{type(e).__name__}: {e}"
        if ok:
            ok_n += 1
        else:
            failures.append((os.path.relpath(v, fps25_root), reason))
        if i % 50 == 0 or i == len(pending):
            log(f"[edtalk s{shard}/{nshards}] {i}/{len(pending)}  ok={ok_n} fail={len(failures)}")
    meta = os.path.dirname(fps25_root.rstrip("/")) or "."
    fname = "edtalk_failures.txt" if nshards == 1 else f"edtalk_failures_s{shard}of{nshards}.txt"
    fail_path = os.path.join(meta, fname)
    if failures:
        with open(fail_path, "w") as f:
            for rel, r in failures:
                f.write(f"{rel}\t{r}\n")
    return {"videos": len(videos), "shard": f"{shard}/{nshards}", "in_shard": len(mine),
            "already_done": already, "processed": len(pending), "ok": ok_n,
            "failed": len(failures), "feature_dims": ref_dims[0],
            "failures_file": fail_path if failures else None}


# ---- self-test ------------------------------------------------------------

class _Arr:
    """Minimal ndarray stand-in so the self-test needs no numpy/torch."""
    def __init__(self, T, d, finite=True):
        self.shape, self.ndim, self.finite = (T, d), 2, finite

class MockBackend:
    def __init__(self, frames, fail_load=(), oom_full=(), wrong_T=(), wrong_dim=(), nan=(),
                 crash_save=()):
        self.frames, self.fail_load, self.oom_full = frames, set(fail_load), set(oom_full)
        self.wrong_T, self.wrong_dim, self.nan = set(wrong_T), set(wrong_dim), set(nan)
        self.crash_save, self.modes, self.calls = set(crash_save), [], []
    def load(self, path):
        self.calls.append(path)
        if path in self.fail_load:
            raise IOError("unreadable video")
        return path, 25.0, self.frames
    def compute(self, vid, mode):
        self.modes.append((vid, mode))
        if vid in self.oom_full and mode == "full":
            raise RuntimeError("CUDA out of memory. Tried to allocate")
        T = self.frames + (1 if vid in self.wrong_T else 0)
        d = 11 if vid in self.wrong_dim else 10
        fin = vid not in self.nan
        return [_Arr(T, d, fin), _Arr(T, 6, fin), _Arr(T, 20, fin)]
    def save(self, path, arr):
        if any(path.startswith(c[:-4]) for c in self.crash_save) and "_ED_pose" in path:
            raise OSError("disk error mid-save")
        with open(path, "w") as f:
            f.write(f"{arr.shape}")
    def is_oom(self, e):
        return "out of memory" in str(e).lower()

def selftest():
    import tempfile
    fin = lambda a: a.finite
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "dataset", "MEAD", "FPS25")
        vids = []
        for ident in ("M003", "W009"):
            for emo in ("angry", "neutral"):
                for n in (1, 2, 3):
                    v = os.path.join(root, ident, "front", emo, "level_1", f"{n:03d}.mp4")
                    os.makedirs(os.path.dirname(v), exist_ok=True)
                    open(v, "w").close()
                    vids.append(v)
        open(os.path.join(root, "M003/front/angry/level_1/009.mp4.tmpcrop.mp4"), "w").close()

        # T1 discovery: nested MEAD tree, temp files excluded
        assert discover(root) == sorted(vids), len(discover(root))

        bad_load, oom, wT, wD, nan, crash = vids[1], vids[2], vids[3], vids[4], vids[5], vids[6]
        b = MockBackend(82, fail_load=[bad_load], oom_full=[oom], wrong_T=[wT],
                        wrong_dim=[wD], nan=[nan], crash_save=[crash])
        s = run(root, b, fin, log=lambda *_: None)
        failed_set = {bad_load, wT, wD, nan, crash}
        # T2 counts: 12 clips, 5 fail for distinct reasons, 7 succeed (incl OOM fallback)
        assert s["processed"] == 12 and s["ok"] == 7 and s["failed"] == 5, s
        assert s["already_done"] == 0, s
        # T3 every success has all 3 outputs; every failure has NONE (no partial files)
        for v in vids:
            exist = [os.path.isfile(p) for p in out_paths(v)]
            if v in failed_set:
                assert not any(exist), (v, exist)
            else:
                assert all(exist), (v, exist)
        assert os.path.basename(out_paths(vids[0])[0]) == "001_ED_exp.npy"
        assert not glob.glob(os.path.join(root, "**", "*" + TMP), recursive=True)
        # T4 OOM in full mode falls back to their batch mode and succeeds
        assert (oom, "batch") in b.modes and all(os.path.isfile(p) for p in out_paths(oom))
        # T5 failure reasons are specific, not swallowed
        reasons = dict(l.rstrip("\n").split("\t") for l in open(s["failures_file"]))
        rel = lambda v: os.path.relpath(v, root)
        assert reasons[rel(bad_load)].startswith("exception_OSError"), reasons
        assert reasons[rel(wT)] == "frame_mismatch_83_vs_82"
        assert reasons[rel(wD)].startswith("dims_(11, 6, 20)"), reasons[rel(wD)]
        assert reasons[rel(nan)] == "non_finite"
        assert reasons[rel(crash)].startswith("exception_OSError")
        # T6 feature dims recorded from real output (not hardcoded)
        assert s["feature_dims"] == (10, 6, 20), s["feature_dims"]
        # T7 resume: finished clips skipped, exactly the 5 failures retried
        b2 = MockBackend(82)
        s2 = run(root, b2, fin, log=lambda *_: None)
        assert s2["processed"] == 5 and s2["ok"] == 5 and s2["already_done"] == 7, s2
        assert sorted(b2.calls) == sorted(failed_set), b2.calls
        # T8 limit
        for v in vids:
            for p in out_paths(v):
                os.remove(p)
        s3 = run(root, MockBackend(82), fin, limit=3, log=lambda *_: None)
        assert s3["processed"] == 3 and s3["ok"] == 3

        # T9 shards: fixed, disjoint, and together cover every clip exactly once
        for v in vids:
            for p in out_paths(v):
                if os.path.exists(p):
                    os.remove(p)
        N = 4
        parts = [[v for k, v in enumerate(discover(root)) if k % N == i] for i in range(N)]
        flat = [v for p in parts for v in p]
        assert sorted(flat) == sorted(vids) and len(set(flat)) == len(flat)

        # T10 all N shards run CONCURRENTLY (threads): every clip computed exactly once
        import threading
        calls, lk = [], threading.Lock()
        class CountBackend(MockBackend):
            def load(self, path):
                with lk:
                    calls.append(path)
                return super().load(path)
        res = [None] * N
        def go(i):
            res[i] = run(root, CountBackend(82), fin, log=lambda *_: None, shard=i, nshards=N)
        th = [threading.Thread(target=go, args=(i,)) for i in range(N)]
        [t.start() for t in th]; [t.join() for t in th]
        assert sorted(calls) == sorted(vids) and len(calls) == len(set(calls)), len(calls)
        assert sum(r["ok"] for r in res) == len(vids) and all(r["failed"] == 0 for r in res)
        assert all(all_exist(v) for v in vids)

        # T11 shard membership does not depend on done-state (re-run: all skipped)
        res2 = [run(root, CountBackend(82), fin, log=lambda *_: None, shard=i, nshards=N)
                for i in range(N)]
        assert all(r["processed"] == 0 and r["already_done"] == r["in_shard"] for r in res2)
        assert sum(r["in_shard"] for r in res2) == len(vids)
    print("selftest OK: nested MEAD discovery (tmp excluded), C-MET output names beside clip, "
          "OOM full->batch fallback, frame/dim/NaN validation, specific failure reasons, "
          "atomic writes (no partial files on crash), dims recorded, resume, --limit, "
          "shards disjoint+complete, concurrent shards each clip once, shard membership stable")
    return True


def main():
    ap = argparse.ArgumentParser(description="EDTalk features for MEAD (C-MET components)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fps25_root")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="0/1", help="I/N: process clips whose sorted index %% N == I")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.fps25_root:
        ap.print_help(); sys.exit(1)
    try:
        shard, nshards = (int(x) for x in args.shard.split("/"))
        assert nshards >= 1 and 0 <= shard < nshards
    except Exception:
        print(f"bad --shard {args.shard!r}; use I/N with 0 <= I < N"); sys.exit(2)
    if not os.path.isfile("pretrained_weights/EDTalk.pt") or not os.path.isdir("src"):
        print("run from the C-MET repo root (needs src/ and pretrained_weights/EDTalk.pt)")
        sys.exit(2)
    t0 = time.time()
    backend = RealBackend()
    import numpy as np
    s = run(args.fps25_root, backend, lambda a: bool(np.isfinite(a).all()), limit=args.limit,
            shard=shard, nshards=nshards)
    print(f"[edtalk] summary: {s}")
    print(f"[done] {time.time() - t0:.1f}s")
    sys.exit(0 if s["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
