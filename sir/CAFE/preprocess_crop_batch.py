#!/usr/bin/env python3
"""
CAFE preprocess: crop + 25fps over the MEAD FPS25 tree, then EDTalk features.

C-MET requires every video cropped to 256x256 and resampled to 25 fps, using
data_preprocess/crop_video.py, before prep_video.py extracts EDTalk features.
Raw MEAD front videos are 30 fps and full-frame, so this batch:

  1. finds every front .mp4 under the FPS25 tree,
  2. runs crop_video.py on each (face-crop + 25fps, their exact method),
     replacing the raw video in place,
  3. after all crops, runs prep_video.py to extract EDTalk features.

Safety / resume (built for "a crash must not waste hours"):
  - a ledger (crop_done.txt) records every completed video; a re-run skips them.
  - each crop writes to a temp file first, then atomically replaces the original,
    so a crash never leaves a half-written video in the tree.
  - a stale temp from a previous crash is removed before retrying that video.
  - --limit N processes only the first N pending videos, so you can validate on
    10-20 clips before committing to the full ~1100-video, multi-hour run.

crop_video.py is treated as a black box through its verified CLI (--inp, --outp,
--cpu); this file never imports its internals.

Usage (from the C-MET repo, C_MET env active):
  # dry run: list what would be processed
  python preprocess_crop_batch.py --fps25_root dataset/MEAD/FPS25 --cmet_repo . --plan
  # test on 10 videos first
  python preprocess_crop_batch.py --fps25_root dataset/MEAD/FPS25 --cmet_repo . --limit 10
  # full run (in tmux)
  python preprocess_crop_batch.py --fps25_root dataset/MEAD/FPS25 --cmet_repo .
  # skip cropping, only extract features (if crops already done)
  python preprocess_crop_batch.py --fps25_root dataset/MEAD/FPS25 --cmet_repo . --features-only
  # self-test (no GPU, no videos, no ffmpeg)
  python preprocess_crop_batch.py --selftest
"""

import argparse, glob, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

TMP_SUFFIX = ".tmpcrop.mp4"
_LEDGER_LOCK = threading.Lock()


# ---- discovery / ledger -----------------------------------------------------

def discover_videos(fps25_root, angle="front"):
    """All <id>/<angle>/<emotion>/<level>/<num>.mp4 under the tree, sorted."""
    pat = os.path.join(fps25_root, "*", angle, "*", "*", "*.mp4")
    return sorted(v for v in glob.glob(pat) if not v.endswith(TMP_SUFFIX))

def ledger_path(fps25_root):
    return os.path.join(os.path.dirname(fps25_root.rstrip("/")) or ".", "crop_done.txt")

def load_ledger(path):
    if not os.path.isfile(path):
        return set()
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())

def append_ledger(path, rel):
    with open(path, "a") as f:
        f.write(rel + "\n")


# ---- the real single-video crop (black-box CLI) ---------------------------

def crop_one_real(cmet_repo, video_abs, tmp_abs, cpu=False):
    """Run crop_video.py on one video. Returns True if a cropped file was made.
       crop_video.py skips if outp exists, so we always start from a clean tmp."""
    if os.path.exists(tmp_abs):
        os.remove(tmp_abs)
    cmd = ["python", "data_preprocess/crop_video.py",
           "--inp", video_abs, "--outp", tmp_abs]
    if cpu:
        cmd.append("--cpu")
    proc = subprocess.run(cmd, cwd=cmet_repo, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True)
    ok = os.path.isfile(tmp_abs) and os.path.getsize(tmp_abs) > 0
    return ok, proc.stdout


# ---- batch orchestration (unit-tested) ------------------------------------

def run_batch(fps25_root, cmet_repo, crop_fn=crop_one_real, is_done_fn=None,
              limit=None, cpu=False, angle="front", workers=1, log=print):
    """Crop every pending video. crop_fn and is_done_fn are injectable for tests.
       workers>1 runs that many crops concurrently (each crop_video.py is a
       separate process; the GPU face model is light so many share one GPU).
       Returns a summary dict."""
    videos = discover_videos(fps25_root, angle)
    ledger = ledger_path(fps25_root)
    done = load_ledger(ledger)

    pending = []
    for v in videos:
        rel = os.path.relpath(v, fps25_root)
        if rel in done:
            continue
        if is_done_fn is not None and is_done_fn(v):
            append_ledger(ledger, rel)   # self-heal: already cropped, record it
            continue
        pending.append((v, rel))

    if limit is not None:
        pending = pending[:limit]

    total = len(pending)
    counters = {"ok": 0, "fail": 0, "n": 0}
    failures = []

    def _process(item):
        v, rel = item
        tmp = v + TMP_SUFFIX
        ok, _out = crop_fn(cmet_repo, v, tmp, cpu=cpu)
        with _LEDGER_LOCK:
            counters["n"] += 1
            i = counters["n"]
            if ok:
                os.replace(tmp, v)              # atomic in-place replace
                append_ledger(ledger, rel)
                counters["ok"] += 1
            else:
                if os.path.exists(tmp):
                    os.remove(tmp)
                counters["fail"] += 1
                failures.append(rel)
            if i % 10 == 0 or i == total:
                log(f"[crop] {i}/{total}  ok={counters['ok']} fail={counters['fail']}")

    if workers <= 1:
        for item in pending:
            _process(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_process, pending))

    if failures:
        fp = os.path.join(os.path.dirname(ledger), "crop_failures.txt")
        with open(fp, "w") as f:
            f.write("\n".join(failures) + "\n")

    return {"discovered": len(videos), "processed": total,
            "cropped": counters["ok"], "failed": counters["fail"],
            "workers": workers,
            "already_done": len(videos) - len(pending) if limit is None else None,
            "failures_file": (os.path.join(os.path.dirname(ledger), "crop_failures.txt")
                              if failures else None)}


def extract_features(cmet_repo, fps25_root, angle="front", process_num=4, log=print):
    """Run prep_video.py to extract EDTalk exp/pose/lip features."""
    cmd = ["python", "prep_video.py", "--data_root", fps25_root,
           "--angle", angle, "--type", "EDTalk", "--process_num", str(process_num)]
    log(f"[features] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cmet_repo)
    return proc.returncode == 0


# ---- self-test (logic only; mock cropper, no GPU/ffmpeg/videos) ------------

def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "dataset", "MEAD", "FPS25")
        # build synthetic tree: 2 ids x 2 emotions x 1 level x 3 sentences = 12 videos
        made = []
        for ident in ["M003", "W009"]:
            for emo in ["angry", "neutral"]:
                for n in (1, 2, 3):
                    v = os.path.join(root, ident, "front", emo, "level_1", f"{n:03d}.mp4")
                    os.makedirs(os.path.dirname(v), exist_ok=True)
                    with open(v, "wb") as f:
                        f.write(b"RAW30FPS")   # pretend raw video
                    made.append(v)

        # T1: discovery finds all 12, sorted, no tmp files
        vids = discover_videos(root)
        assert len(vids) == 12, len(vids)

        # mock cropper: creates tmp (success) for all EXCEPT one "no-face" clip
        noface = os.path.join(root, "W009", "front", "angry", "level_1", "002.mp4")
        def mock_crop(repo, v, tmp, cpu=False):
            if v == noface:
                return False, "no face detected"        # simulate failure
            with open(tmp, "wb") as f:
                f.write(b"CROPPED256x256_25FPS")         # simulate cropped output
            return True, "ok"

        # T2: full batch -> 11 cropped, 1 failed, ledger + failures written
        s = run_batch(root, td, crop_fn=mock_crop, log=lambda *_: None)
        assert s["cropped"] == 11 and s["failed"] == 1, s
        # cropped videos were replaced in place (content changed)
        assert open(made[0], "rb").read() == b"CROPPED256x256_25FPS"
        # failed video untouched (still raw) and no leftover tmp
        assert open(noface, "rb").read() == b"RAW30FPS"
        assert not os.path.exists(noface + TMP_SUFFIX)
        assert os.path.isfile(s["failures_file"])
        led = load_ledger(ledger_path(root))
        assert len(led) == 11, len(led)   # only successes recorded

        # T3: resume -> everything in ledger skipped; only the 1 failure retried
        calls = []
        def mock_crop2(repo, v, tmp, cpu=False):
            calls.append(v)
            with open(tmp, "wb") as f:
                f.write(b"CROPPED")
            return True, "ok"
        s2 = run_batch(root, td, crop_fn=mock_crop2, log=lambda *_: None)
        assert s2["processed"] == 1 and s2["cropped"] == 1, s2
        assert calls == [noface], calls   # only the previously-failed one retried
        assert len(load_ledger(ledger_path(root))) == 12   # now all done

        # T4: limit caps pending count
        # fresh tree for a clean limit test
        root2 = os.path.join(td, "t2", "FPS25")
        for n in range(1, 6):
            v = os.path.join(root2, "M003", "front", "happy", "level_1", f"{n:03d}.mp4")
            os.makedirs(os.path.dirname(v), exist_ok=True); open(v, "wb").close()
        s3 = run_batch(root2, td, crop_fn=mock_crop2, limit=2, log=lambda *_: None)
        assert s3["processed"] == 2, s3
        assert len(load_ledger(ledger_path(root2))) == 2

        # T5: is_done_fn self-heals ledger (already-cropped detected without ledger)
        root3 = os.path.join(td, "t3", "FPS25")
        for n in (1, 2):
            v = os.path.join(root3, "M003", "front", "sad", "level_1", f"{n:03d}.mp4")
            os.makedirs(os.path.dirname(v), exist_ok=True); open(v, "wb").close()
        already = os.path.join(root3, "M003", "front", "sad", "level_1", "001.mp4")
        s4 = run_batch(root3, td, crop_fn=mock_crop2,
                       is_done_fn=lambda v: v == already, log=lambda *_: None)
        assert s4["processed"] == 1, s4   # 001 self-healed, only 002 processed
        assert os.path.relpath(already, root3) in load_ledger(ledger_path(root3))

        # T6: stale tmp from a prior crash is cleaned before retry
        root4 = os.path.join(td, "t4", "FPS25")
        v = os.path.join(root4, "M003", "front", "fear", "level_1", "001.mp4")
        os.makedirs(os.path.dirname(v), exist_ok=True); open(v, "wb").close()
        open(v + TMP_SUFFIX, "wb").close()   # leftover stale tmp
        def mock_crop3(repo, vv, tmp, cpu=False):
            assert not (os.path.exists(tmp) and os.path.getsize(tmp) > 0), "stale tmp not cleared"
            with open(tmp, "wb") as f:
                f.write(b"C")
            return True, "ok"
        s5 = run_batch(root4, td, crop_fn=mock_crop3, log=lambda *_: None)
        assert s5["cropped"] == 1, s5

        # T7: parallel workers -- all processed exactly once, ledger thread-safe,
        # no double-processing, failures still isolated
        root5 = os.path.join(td, "t5par", "FPS25")
        made5 = []
        for ident in ["M003", "W009", "M030", "W015"]:
            for n in range(1, 6):
                v = os.path.join(root5, ident, "front", "happy", "level_1", f"{n:03d}.mp4")
                os.makedirs(os.path.dirname(v), exist_ok=True)
                with open(v, "wb") as f:
                    f.write(b"RAW")
                made5.append(v)
        # one designated failure
        pfail = made5[7]
        seen = []
        seen_lock = threading.Lock()
        def par_crop(repo, v, tmp, cpu=False):
            with seen_lock:
                seen.append(v)          # record every call to detect duplicates
            time.sleep(0.001)
            if v == pfail:
                return False, "no face"
            with open(tmp, "wb") as f:
                f.write(b"C")
            return True, "ok"
        s6 = run_batch(root5, td, crop_fn=par_crop, workers=8, log=lambda *_: None)
        assert s6["processed"] == 20 and s6["cropped"] == 19 and s6["failed"] == 1, s6
        assert len(seen) == 20 and len(set(seen)) == 20, "a video was processed twice"
        led5 = load_ledger(ledger_path(root5))
        assert len(led5) == 19, (len(led5))          # only successes, no dup lines
        assert os.path.relpath(pfail, root5) not in led5

        # T8: resume after parallel run -> only the 1 failure retried, concurrently safe
        seen.clear()
        def par_crop2(repo, v, tmp, cpu=False):
            with seen_lock:
                seen.append(v)
            with open(tmp, "wb") as f:
                f.write(b"C")
            return True, "ok"
        s7 = run_batch(root5, td, crop_fn=par_crop2, workers=8, log=lambda *_: None)
        assert s7["processed"] == 1 and seen == [pfail], (s7, seen)
        assert len(load_ledger(ledger_path(root5))) == 20

    print("selftest OK: discovery, batch crop (in-place replace + failure isolation), "
          "ledger resume (retries only failures), --limit, is_done self-heal, stale-tmp cleanup, "
          "parallel workers (each video once, thread-safe ledger), parallel resume")
    return True


# ---- cli ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="CAFE crop+25fps+features batch")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fps25_root", help="path to dataset/MEAD/FPS25")
    ap.add_argument("--cmet_repo", default=".", help="C-MET repo root (cwd for scripts)")
    ap.add_argument("--angle", default="front")
    ap.add_argument("--limit", type=int, default=None, help="process only first N pending videos")
    ap.add_argument("--workers", type=int, default=1, help="parallel crop workers (GPU is light; 6-8 is safe)")
    ap.add_argument("--cpu", action="store_true", help="pass --cpu to crop_video.py")
    ap.add_argument("--process_num", type=int, default=4, help="prep_video.py workers")
    ap.add_argument("--plan", action="store_true", help="list pending videos and exit")
    ap.add_argument("--features-only", action="store_true", help="skip cropping, only run prep_video.py")
    ap.add_argument("--no-features", action="store_true", help="crop only, skip prep_video.py")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.fps25_root:
        ap.print_help(); sys.exit(1)

    if args.plan:
        vids = discover_videos(args.fps25_root, args.angle)
        done = load_ledger(ledger_path(args.fps25_root))
        pending = [v for v in vids if os.path.relpath(v, args.fps25_root) not in done]
        print(f"discovered: {len(vids)}  done(ledger): {len(done)}  pending: {len(pending)}")
        for v in pending[:20]:
            print("  ", os.path.relpath(v, args.fps25_root))
        if len(pending) > 20:
            print(f"   ... and {len(pending)-20} more")
        sys.exit(0)

    t0 = time.time()
    if not args.features_only:
        summary = run_batch(args.fps25_root, args.cmet_repo, limit=args.limit,
                            cpu=args.cpu, angle=args.angle, workers=args.workers)
        print(f"[crop] summary: {summary}")
        if summary["failed"]:
            print(f"[crop] {summary['failed']} videos failed (no face / crop error); "
                  f"see {summary['failures_file']}. Review before features.")

    if not args.no_features and not (args.limit and not args.features_only):
        # only extract features on a full crop pass, not a --limit test run
        ok = extract_features(args.cmet_repo, args.fps25_root, args.angle, args.process_num)
        print(f"[features] {'OK' if ok else 'FAILED'}")
        sys.exit(0 if ok else 1)
    elif args.limit:
        print("[features] skipped (--limit test run; run without --limit for features)")
    print(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
