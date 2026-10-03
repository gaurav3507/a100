#!/usr/bin/env python3
"""
CAFE fps + wav fix. Runs AFTER crop, BEFORE prep_video.py features.

Why this exists (verified, not assumed):
  1. crop_video.py builds `ffmpeg ... <outp> -r 25 -y`. Options after the
     output file are trailing options that ffmpeg ignores ("Trailing option(s)
     found in the command: may be ignored"), so cropped clips stay ~30 fps.
     The C-MET README requires every video to be 25 fps.
  2. The FPS25 tree has no <stem>.wav files, which generation
     (--audio_driving_path) and SyncNet evaluation need.

For every video recorded in the crop ledger (crop_done.txt):
  a. confirm it is cropped (256x256); otherwise skip and report,
  b. if fps != 25: re-encode to 25 fps with the fps filter (audio re-encoded
     AAC) into a temp file, verify 25 fps, then atomically replace,
  c. extract the clip's own audio track to <stem>.wav (16 kHz, mono,
     pcm_s16le, the format of C-MET's demo wav) via temp + atomic rename.
     Taking audio from the cropped clip keeps it aligned with the trimmed video.

Safety: fps_done.txt ledger (resume skips finished clips), temp + atomic
replace (a crash never leaves a half-written file), per-clip failure isolation
with reasons in fps_failures.txt, --limit for small trial runs, --workers for
parallelism. Only ledger-listed (already cropped) clips are ever touched.

Usage (from the C-MET repo, C_MET env active):
  python ../fps_wav_fix.py --fps25_root dataset/MEAD/FPS25 --limit 5
  python ../fps_wav_fix.py --fps25_root dataset/MEAD/FPS25 --workers 16
  python ../fps_wav_fix.py --selftest     # real ffmpeg on synthetic clips
"""

import argparse, json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

TARGET_FPS = 25.0
WAV_SR = 16000
TMPV = ".tmpfps.mp4"
TMPW = ".tmpwav.wav"
_LOCK = threading.Lock()


# ---- probing --------------------------------------------------------------

def _rate(s):
    try:
        n, d = s.split("/")
        return float(n) / float(d) if float(d) else 0.0
    except Exception:
        return 0.0

def probe(path):
    """Return dict with video w,h,fps,nb_frames and audio sr,channels (or None)."""
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "stream=codec_type,width,height,avg_frame_rate,nb_frames,sample_rate,channels",
           "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        return None
    info = {"w": None, "h": None, "fps": 0.0, "nb_frames": 0, "sr": None, "ch": None}
    for s in json.loads(out.stdout or "{}").get("streams", []):
        if s.get("codec_type") == "video":
            info["w"], info["h"] = s.get("width"), s.get("height")
            info["fps"] = _rate(s.get("avg_frame_rate", "0/0"))
            try:
                info["nb_frames"] = int(s.get("nb_frames", 0))
            except (TypeError, ValueError):
                info["nb_frames"] = 0
        elif s.get("codec_type") == "audio":
            info["sr"] = int(s.get("sample_rate", 0) or 0)
            info["ch"] = s.get("channels")
    return info

def is_25(fps):
    return abs(fps - TARGET_FPS) < 0.01

def wav_ok(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    i = probe(path)
    return bool(i) and i["sr"] == WAV_SR and i["ch"] == 1


# ---- the real per-clip fix ------------------------------------------------

def _ff(args):
    return subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args,
                          capture_output=True, text=True).returncode == 0

def fix_one(fps25_root, rel, size=256):
    v = os.path.join(fps25_root, rel)
    if not os.path.isfile(v):
        return False, "missing_video"
    info = probe(v)
    if not info:
        return False, "probe_failed"
    if (info["w"], info["h"]) != (size, size):
        return False, f"not_cropped_{info['w']}x{info['h']}"

    # (b) fps -> 25
    if not is_25(info["fps"]):
        tmp = v + TMPV
        if os.path.exists(tmp):
            os.remove(tmp)
        ok = _ff(["-i", v, "-vf", f"fps={int(TARGET_FPS)}", "-c:v", "libx264",
                  "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
                  "-c:a", "aac", "-b:a", "128k", tmp])
        t = probe(tmp) if ok and os.path.isfile(tmp) else None
        if not t or not is_25(t["fps"]) or t["nb_frames"] <= 0:
            if os.path.exists(tmp):
                os.remove(tmp)
            return False, "fps_convert_failed"
        os.replace(tmp, v)

    # (c) wav from the clip's own audio track
    stem = v[:-4] if v.endswith(".mp4") else v
    wav, tmpw = stem + ".wav", stem + TMPW
    if not wav_ok(wav):
        if os.path.exists(tmpw):
            os.remove(tmpw)
        ok = _ff(["-i", v, "-vn", "-ac", "1", "-ar", str(WAV_SR),
                  "-c:a", "pcm_s16le", "-f", "wav", tmpw])
        if not ok or not wav_ok(tmpw):
            if os.path.exists(tmpw):
                os.remove(tmpw)
            return False, "wav_extract_failed_or_no_audio"
        os.replace(tmpw, wav)
    return True, "ok"


# ---- ledgers + orchestration ----------------------------------------------

def _meta_dir(fps25_root):
    return os.path.dirname(fps25_root.rstrip("/")) or "."

def _read(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

def run_fix(fps25_root, fix_fn=fix_one, workers=1, limit=None, log=print):
    meta = _meta_dir(fps25_root)
    cropped = _read(os.path.join(meta, "crop_done.txt"))
    done_path = os.path.join(meta, "fps_done.txt")
    done = set(_read(done_path))
    pending = [r for r in dict.fromkeys(cropped) if r not in done]
    if limit is not None:
        pending = pending[:limit]

    total = len(pending)
    c = {"ok": 0, "fail": 0, "n": 0}
    failures = []

    def _proc(rel):
        ok, reason = fix_fn(fps25_root, rel)
        with _LOCK:
            c["n"] += 1
            if ok:
                with open(done_path, "a") as f:
                    f.write(rel + "\n")
                c["ok"] += 1
            else:
                c["fail"] += 1
                failures.append((rel, reason))
            if c["n"] % 25 == 0 or c["n"] == total:
                log(f"[fps] {c['n']}/{total}  ok={c['ok']} fail={c['fail']}")

    if workers <= 1:
        for r in pending:
            _proc(r)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_proc, pending))

    fail_path = os.path.join(meta, "fps_failures.txt")
    if failures:
        with open(fail_path, "w") as f:
            for rel, reason in failures:
                f.write(f"{rel}\t{reason}\n")
    return {"cropped_in_ledger": len(set(cropped)), "processed": total,
            "fixed": c["ok"], "failed": c["fail"],
            "failures_file": fail_path if failures else None}


# ---- self-test: real ffmpeg on synthetic clips + mock orchestration --------

def _mk(path, size, fps, audio=True, secs=2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    args = ["-f", "lavfi", "-i", f"testsrc=size={size}x{size}:rate={fps}"]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    args += ["-t", str(secs), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    args += (["-c:a", "aac"] if audio else ["-an"])
    assert _ff(args + [path]), f"could not synthesize {path}"

def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "dataset", "MEAD", "FPS25")
        meta = os.path.dirname(root)
        R = lambda *p: os.path.join(*p)
        a = "M003/front/angry/level_1/001.mp4"      # cropped 30fps + audio -> fix
        b = "M003/front/happy/level_1/002.mp4"      # cropped 25fps + audio -> wav only
        cc = "W009/front/sad/level_1/003.mp4"       # cropped 30fps, NO audio -> fail
        d = "W009/front/fear/level_1/004.mp4"       # uncropped 512 -> skip/fail
        e = "M030/front/neutral/level_1/005.mp4"    # raw, NOT in crop ledger -> untouched
        _mk(R(root, a), 256, 30)
        _mk(R(root, b), 256, 25)
        _mk(R(root, cc), 256, 30, audio=False)
        _mk(R(root, d), 512, 30)
        _mk(R(root, e), 256, 30)
        with open(R(meta, "crop_done.txt"), "w") as f:
            f.write("\n".join([a, b, cc, d]) + "\n")
        e_before = open(R(root, e), "rb").read()

        s = run_fix(root, workers=4, log=lambda *_: None)
        # T1: counts
        assert s["processed"] == 4 and s["fixed"] == 2 and s["failed"] == 2, s
        # T2: 30fps clip now exactly 25 fps, still 256x256, frames ~ 2s*25
        ia = probe(R(root, a))
        assert is_25(ia["fps"]) and (ia["w"], ia["h"]) == (256, 256), ia
        assert 45 <= ia["nb_frames"] <= 55, ia["nb_frames"]
        # T3: wav extracted 16k mono for both fixed clips, no temp leftovers
        for rel in (a, b):
            assert wav_ok(R(root, rel[:-4] + ".wav")), rel
            assert not os.path.exists(R(root, rel + TMPV))
            assert not os.path.exists(R(root, rel[:-4] + TMPW))
        # T4: 25fps clip untouched on fps (still 25) and wav created
        assert is_25(probe(R(root, b))["fps"])
        # T5: failures carry reasons; no-audio + uncropped isolated
        fails = dict(l.split("\t") for l in _read(s["failures_file"]))
        assert fails[cc] == "wav_extract_failed_or_no_audio", fails
        assert fails[d].startswith("not_cropped_512x512"), fails
        # T6: clip not in crop ledger is never touched
        assert open(R(root, e), "rb").read() == e_before
        assert not os.path.exists(R(root, e[:-4] + ".wav"))
        # T7: resume -> fixed clips skipped, only failures retried
        calls = []
        def mock(rootp, rel):
            calls.append(rel); return False, "still_bad"
        s2 = run_fix(root, fix_fn=mock, log=lambda *_: None)
        assert sorted(calls) == sorted([cc, d]), calls
        # T8: idempotent real re-run on an already-fixed clip is a no-op success
        with open(R(meta, "fps_done.txt"), "w") as f:
            f.write("")                       # force reprocess of everything
        before = os.path.getmtime(R(root, a))
        ok, reason = fix_one(root, a)
        assert ok and reason == "ok"
        assert os.path.getmtime(R(root, a)) == before, "25fps clip was re-encoded"
        # T9: parallel mock -> each clip exactly once, ledger has no duplicates
        with open(R(meta, "crop_done.txt"), "w") as f:
            f.write("\n".join(f"X/front/e/level_1/{i:03d}.mp4" for i in range(40)) + "\n")
        open(R(meta, "fps_done.txt"), "w").close()
        seen, sl = [], threading.Lock()
        def pm(rootp, rel):
            with sl: seen.append(rel)
            time.sleep(0.001); return True, "ok"
        s3 = run_fix(root, fix_fn=pm, workers=8, log=lambda *_: None)
        assert s3["fixed"] == 40 and len(set(seen)) == 40 == len(seen)
        assert len(_read(R(meta, "fps_done.txt"))) == 40
    print("selftest OK: real 30->25fps convert (256x256 kept, frame count), wav 16k mono, "
          "25fps clip wav-only, no-audio + uncropped isolated with reasons, "
          "non-ledger clip untouched, resume retries only failures, idempotent, parallel once-only")
    return True


def main():
    ap = argparse.ArgumentParser(description="CAFE fps 30->25 + wav extraction")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fps25_root")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.fps25_root:
        ap.print_help(); sys.exit(1)
    t0 = time.time()
    s = run_fix(args.fps25_root, workers=args.workers, limit=args.limit)
    print(f"[fps] summary: {s}")
    print(f"[done] {time.time()-t0:.1f}s")
    sys.exit(0 if s["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
