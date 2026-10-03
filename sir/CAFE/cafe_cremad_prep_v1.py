#!/usr/bin/env python3
"""CAFE: prepare the C-MET CREMA-D test clips with the Phase 1 MEAD chain (v1).

Run from the C-MET repository root with the C_MET environment active:

    python3 ../cafe_cremad_prep_v1.py --limit 5                 # pilot: 5 clips through every stage
    python3 ../cafe_cremad_prep_v1.py --workers 16              # all clips (start with setsid nohup)
    python3 ../cafe_cremad_prep_v1.py --stage audit             # read-only check of every output

Scope. The clips named in dataset/CREMA_D/test.csv (the 523 neutral sources and the 1546
emotional ground truths, 2069 unique clips). They are needed by the Phase 6 cross-dataset
evaluation whatever is decided about a CREMA-D Gate 2 reproduction; the public C-MET code has no
CREMA-D branch for the emotion direction, so no generation is attempted here.

Stages (each resumable and idempotent; later stages only take clips the earlier ones finished):
  extract : the archive crema-d-mirror-main.tar.gz next to this script, SHA-256 checked against
            the value recorded when it was downloaded, is read as a stream and only the needed
            VideoFlash/<clip>.flv members are written to dataset/CREMA_D/raw_flv/ (temp file, then
            atomic rename). A needed clip absent from the archive, or present twice, stops the run.
  crop    : the Phase 1 preprocess_crop_batch.crop_one_real (C-MET's data_preprocess/crop_video.py
            through its CLI) turns raw_flv/<clip>.flv into FPS25/<clip>.mp4; the output must be
            256x256 before it is accepted; ledger dataset/CREMA_D/crop_done.txt.
  fps     : the Phase 1 fps_wav_fix.run_fix, unchanged: 25 fps re-encode (crop_video.py's -r 25
            is ignored by ffmpeg) and a 16 kHz mono PCM wav from the cropped clip; ledger fps_done.txt.
  edtalk  : the Phase 1 extract_edtalk_mead.run with C-MET's own EDTalk components, file discovery
            replaced by the flat CREMA-D list; <clip>_ED_exp/_pose/_lip.npy next to each clip.
  e2v     : the Phase 1 extract_e2v_mead.run (emotion2vec+large, local weights), discovery replaced;
            FPS25/emotion2vec+large_features/<clip>.npy, the same "next to the clip" convention as
            MEAD (the public C-MET code defines no CREMA-D location).
  audit   : every needed clip: mp4 256x256 at 25 fps, wav 16 kHz mono, the three EDTalk arrays
            with one row per frame, and a finite 1024-d emotion2vec vector.
The four Phase 1 scripts are imported from the folder of this script, never copied or edited.
Nothing is deleted. Exit code: 0 clean, 1 failures, 2 bad usage or another run active.
"""

import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import os
import re
import socket
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

SCRIPT = "cafe_cremad_prep_v1"
ARCHIVE = "crema-d-mirror-main.tar.gz"
ARCHIVE_SHA256 = "8b5d78525aa95f0acbc1ee284c5ce7168687299b24cb6d8c04e3ae7644f8b824"  # recorded on ant-pc and DGX
ROOT = "dataset/CREMA_D"
FPS25 = ROOT + "/FPS25"
RAW = ROOT + "/raw_flv"
TEST_CSV = ROOT + "/test.csv"
MODEL_DIR = "/workspace/sir/CAFE/models/emotion2vec_plus_large"
PHASE1 = {"pcb": "preprocess_crop_batch.py", "fwf": "fps_wav_fix.py", "em": "extract_edtalk_mead.py",
          "ee": "extract_e2v_mead.py"}
NAME_RE = re.compile(r"^\d{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}$")
_MODS = {}
_LOCK = threading.Lock()


class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, status, stage, name, detail, **extra):
        row = {"status": status, "stage": stage, "name": name, "detail": detail}
        row.update(extra)
        self.rows.append(row)
        print("[%-7s] %-7s %-24s %s" % (status, stage, name, detail), flush=True)

    def failed(self):
        return [r["stage"] + ":" + r["name"] for r in self.rows if r["status"] in ("FAIL", "MISSING")]


def module(key, here):
    """Import one Phase 1 script once, from the folder of this script."""
    if key not in _MODS:
        path = os.path.join(here, PHASE1[key])
        spec = importlib.util.spec_from_file_location("phase1_" + key, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _MODS[key] = m
    return _MODS[key]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_lines(path):
    if not os.path.isfile(path):
        return []
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def needed_names(repo):
    rows = list(csv.DictReader(open(os.path.join(repo, TEST_CSV), newline="")))
    names = set()
    for r in rows:
        for col in ("source_video_path", "gt_video_path"):
            names.add(os.path.basename(r[col].replace("\\", "/"))[:-4])
    bad = sorted(n for n in names if not NAME_RE.match(n))
    return rows, sorted(names), bad


# ---------------------------------------------------------------- stages

def stage_extract(rep, repo, here, names, expected_sha):
    raw = os.path.join(repo, RAW)
    manifest = os.path.join(raw, "extract_manifest.json")
    if os.path.isfile(manifest):
        try:
            m = json.load(open(manifest))
            if (m.get("archive_sha256") == expected_sha and
                    all(os.path.isfile(os.path.join(raw, n + ".flv")) and
                        os.path.getsize(os.path.join(raw, n + ".flv")) == m["sizes"].get(n) for n in names)):
                rep.add("OK", "extract", "raw clips", "%d/%d already extracted from the verified archive"
                        % (len(names), len(names)))
                return True
        except Exception:
            pass
    arc = os.path.join(here, ARCHIVE)
    if not os.path.isfile(arc):
        rep.add("MISSING", "extract", "archive", "%s not found next to this script" % ARCHIVE)
        return False
    t0 = time.time()
    digest = sha256_file(arc)
    if digest != expected_sha:
        rep.add("FAIL", "extract", "archive", "SHA-256 %s differs from the recorded %s" % (digest[:12], expected_sha[:12]))
        return False
    os.makedirs(raw, exist_ok=True)
    need, sizes, dup = set(names), {}, []
    with tarfile.open(arc, "r|gz") as t:
        for mem in t:
            if not mem.isfile() or "/VideoFlash/" not in "/" + mem.name or not mem.name.endswith(".flv"):
                continue
            stem = os.path.basename(mem.name)[:-4]
            if stem not in need:
                continue
            if stem in sizes:
                dup.append(stem)
                continue
            dst = os.path.join(raw, stem + ".flv")
            tmp = dst + ".tmp"
            src = t.extractfile(mem)
            with open(tmp, "wb") as out:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
            got = os.path.getsize(tmp)
            if got != mem.size:
                os.remove(tmp)
                rep.add("FAIL", "extract", stem, "short write (%d of %d bytes)" % (got, mem.size))
                return False
            os.replace(tmp, dst)
            sizes[stem] = mem.size
    missing = sorted(need - set(sizes))
    if dup or missing:
        rep.add("FAIL", "extract", "archive content", "%d needed clips missing from the archive (%s), %d duplicated (%s)"
                % (len(missing), ", ".join(missing[:4]), len(dup), ", ".join(dup[:4])))
        return False
    small = sorted(n for n, s in sizes.items() if s < 1000)
    if small:
        rep.add("FAIL", "extract", "archive content", "%d clips under 1 KB (Git LFS stubs?): %s" % (len(small), ", ".join(small[:4])))
        return False
    with open(manifest + ".tmp", "w") as fh:
        json.dump({"archive": ARCHIVE, "archive_sha256": digest, "sizes": sizes}, fh)
    os.replace(manifest + ".tmp", manifest)
    rep.add("OK", "extract", "raw clips", "%d clips (%.2f GB) from %s, SHA-256 %s verified, in %.0f s"
            % (len(sizes), sum(sizes.values()) / 1e9, ARCHIVE, digest[:12], time.time() - t0))
    return True


def stage_crop(rep, repo, here, names, workers, limit, crop_fn=None):
    pcb, fwf = module("pcb", here), module("fwf", here)
    crop_fn = crop_fn or pcb.crop_one_real
    fps25, raw = os.path.join(repo, FPS25), os.path.join(repo, RAW)
    os.makedirs(fps25, exist_ok=True)
    ledger = os.path.join(repo, ROOT, "crop_done.txt")
    done = set(read_lines(ledger))
    pending = []
    for n in names:
        rel = n + ".mp4"
        if rel in done:
            continue
        final = os.path.join(fps25, rel)
        info = fwf.probe(final) if os.path.isfile(final) else None
        if info and (info["w"], info["h"]) == (256, 256):
            with open(ledger, "a") as fh:     # self-heal: cropped earlier, ledger line lost
                fh.write(rel + "\n")
            continue
        pending.append(n)
    if limit is not None:
        pending = pending[:limit]
    fails = []
    t0 = time.time()

    def one(n):
        inp = os.path.abspath(os.path.join(raw, n + ".flv"))
        final = os.path.abspath(os.path.join(fps25, n + ".mp4"))
        tmp = final + pcb.TMP_SUFFIX
        try:
            ok, out = crop_fn(os.path.abspath(repo), inp, tmp)
            info = fwf.probe(tmp) if ok and os.path.isfile(tmp) else None
            if not info or (info["w"], info["h"]) != (256, 256):
                why = "not_256x256_%sx%s" % (info["w"], info["h"]) if info else "no_output"
                tail = [ln for ln in (out or "").strip().splitlines() if ln.strip()][-1:] or [""]
                raise RuntimeError("%s %s" % (why, tail[0][:120]))
            os.replace(tmp, final)
            with _LOCK:
                with open(ledger, "a") as fh:
                    fh.write(n + ".mp4\n")
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            with _LOCK:
                fails.append((n, "%s: %s" % (type(e).__name__, str(e)[:160])))

    if workers <= 1:
        for n in pending:
            one(n)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, pending))
    if fails:
        with open(os.path.join(repo, ROOT, "crop_failures.txt"), "w") as fh:
            fh.write("".join("%s\t%s\n" % f for f in fails))
    status = "OK" if not fails else "FAIL"
    rep.add(status, "crop", "crop_video.py", "%d processed, %d failed, %d in ledger, %.0f s%s"
            % (len(pending), len(fails), len(set(read_lines(ledger))), time.time() - t0,
               "; first: %s %s" % fails[0] if fails else ""))
    return not fails


def stage_fps(rep, repo, here, workers, limit):
    fwf = module("fwf", here)
    t0 = time.time()
    s = fwf.run_fix(os.path.join(repo, FPS25), workers=workers, limit=limit, log=lambda *_: None)
    status = "OK" if s["failed"] == 0 else "FAIL"
    rep.add(status, "fps", "fps_wav_fix.run_fix", "%d processed, %d fixed, %d failed, %.0f s%s"
            % (s["processed"], s["fixed"], s["failed"], time.time() - t0,
               "; see " + os.path.relpath(s["failures_file"], repo) if s.get("failures_file") else ""))
    return s["failed"] == 0


def _ready(repo, names):
    fps_done = set(read_lines(os.path.join(repo, ROOT, "fps_done.txt")))
    return [n for n in names if n + ".mp4" in fps_done]


def stage_edtalk(rep, repo, here, names, limit, backend_factory=None):
    em = module("em", here)
    import numpy as np
    ready = _ready(repo, names)
    fps25 = os.path.join(repo, FPS25)
    em.discover = lambda root, angle="front": sorted(os.path.join(root, n + ".mp4") for n in ready)
    backend = (backend_factory or em.RealBackend)()
    t0 = time.time()
    s = em.run(fps25, backend, lambda a: bool(np.isfinite(a).all()), limit=limit, log=lambda *_: None)
    status = "OK" if s["failed"] == 0 else "FAIL"
    rep.add(status, "edtalk", "extract_edtalk_mead.run", "%d clips ready, %d processed, %d failed, dims %s, %.0f s"
            % (len(ready), s["processed"], s["failed"], s["feature_dims"], time.time() - t0))
    return s["failed"] == 0


def stage_e2v(rep, repo, here, names, limit, model_dir, backend_factory=None):
    ee = module("ee", here)
    ready = [n for n in _ready(repo, names) if os.path.isfile(os.path.join(repo, FPS25, n + ".wav"))]
    ee.discover_wavs = lambda root, angle="front": sorted(os.path.join(root, n + ".wav") for n in ready)
    if backend_factory is None:
        if not os.path.isfile(os.path.join(model_dir, "model.pt")):
            rep.add("MISSING", "e2v", "emotion2vec model", "%s has no model.pt" % model_dir)
            return False
        backend = ee.RealBackend(model_dir)
    else:
        backend = backend_factory()
    t0 = time.time()
    try:
        s = ee.run(os.path.join(repo, FPS25), backend, limit=limit, log=lambda *_: None)
    finally:
        tmpd = getattr(backend, "tmp", None)
        if tmpd and os.path.isdir(tmpd):
            import shutil
            shutil.rmtree(tmpd, ignore_errors=True)
    status = "OK" if s["failed"] == 0 else "FAIL"
    rep.add(status, "e2v", "extract_e2v_mead.run", "%d wavs ready, %d processed, %d failed, %.0f s"
            % (len(ready), s["processed"], s["failed"], time.time() - t0))
    return s["failed"] == 0


def stage_audit(rep, repo, here, names):
    fwf = module("fwf", here)
    import numpy as np
    fps25 = os.path.join(repo, FPS25)
    counts = {k: 0 for k in ("mp4", "wav", "edtalk", "e2v")}
    bad = {k: [] for k in counts}
    for n in names:
        base = os.path.join(fps25, n)
        info = fwf.probe(base + ".mp4") if os.path.isfile(base + ".mp4") else None
        if info and (info["w"], info["h"]) == (256, 256) and fwf.is_25(info["fps"]) and info["nb_frames"] > 0:
            counts["mp4"] += 1
        else:
            bad["mp4"].append(n)
            info = None
        if fwf.wav_ok(base + ".wav"):
            counts["wav"] += 1
        else:
            bad["wav"].append(n)
        try:
            arrs = [np.load(base + s, mmap_mode="r") for s in ("_ED_exp.npy", "_ED_pose.npy", "_ED_lip.npy")]
            rows_ok = info is not None and all(a.ndim == 2 and a.shape[0] == info["nb_frames"] for a in arrs)
            if rows_ok and all(np.isfinite(np.asarray(a)).all() for a in arrs):
                counts["edtalk"] += 1
            else:
                bad["edtalk"].append(n)
        except Exception:
            bad["edtalk"].append(n)
        try:
            v = np.load(os.path.join(fps25, "emotion2vec+large_features", n + ".npy"))
            if v.shape == (1024,) and np.isfinite(v).all():
                counts["e2v"] += 1
            else:
                bad["e2v"].append(n)
        except Exception:
            bad["e2v"].append(n)
    for k, label in (("mp4", "mp4 256x256 at 25 fps"), ("wav", "wav 16 kHz mono"),
                     ("edtalk", "EDTalk exp/pose/lip, rows = frames"), ("e2v", "emotion2vec 1024-d finite")):
        rep.add("OK" if not bad[k] else "FAIL", "audit", k, "%d/%d %s%s" % (counts[k], len(names), label,
                "; missing or bad: " + ", ".join(bad[k][:4]) + (" (+%d)" % (len(bad[k]) - 4) if len(bad[k]) > 4 else "")
                if bad[k] else ""))
    return not any(bad.values())


# ---------------------------------------------------------------- main

def other_run_active(here):
    for lock, tag in (("fullrun.lock", b"cafe_eval_prep"), ("fvd.lock", b"cafe_fvd_setup"), ("aitv.lock", b"cafe_aitv"),
                      ("cremad.lock", b"cafe_cremad_prep")):
        try:
            pid = int(open(os.path.join(here, "reports", lock)).read().split()[0])
            if pid != os.getpid() and tag in open("/proc/%d/cmdline" % pid, "rb").read():
                return "%s (pid %d)" % (tag.decode(), pid)
        except Exception:
            pass
    return None


def run(args, repo, here, rep, hooks=None):
    hooks = hooks or {}
    rows, names, bad = needed_names(repo)
    if bad:
        rep.add("FAIL", "list", "test.csv", "unexpected clip names: %s" % ", ".join(bad[:4]))
        return False
    rep.add("OK", "list", "test.csv", "%d rows, %d unique clips needed" % (len(rows), len(names)))
    sel = names[:args.limit] if args.limit else names
    stages = ["extract", "crop", "fps", "edtalk", "e2v", "audit"] if args.stage == "all" else [args.stage]
    ok = True
    for st in stages:
        if st == "extract":
            ok = stage_extract(rep, repo, here, names, args.expected_sha256) and ok
            if not ok:
                break
        elif st == "crop":
            ok = stage_crop(rep, repo, here, sel, args.workers, None, crop_fn=hooks.get("crop")) and ok
        elif st == "fps":
            ok = stage_fps(rep, repo, here, args.workers, None) and ok
        elif st == "edtalk":
            ok = stage_edtalk(rep, repo, here, sel, None, backend_factory=hooks.get("edtalk")) and ok
        elif st == "e2v":
            ok = stage_e2v(rep, repo, here, sel, None, args.model_dir, backend_factory=hooks.get("e2v")) and ok
        elif st == "audit":
            ok = stage_audit(rep, repo, here, sel) and ok
    return ok


def main(argv=None, hooks=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="all", choices=["all", "extract", "crop", "fps", "edtalk", "e2v", "audit"])
    ap.add_argument("--limit", type=int, default=None, help="first N needed clips (sorted by name) only")
    ap.add_argument("--workers", type=int, default=8, help="parallel crops and fps fixes")
    ap.add_argument("--model_dir", default=MODEL_DIR)
    ap.add_argument("--expected_sha256", default=ARCHIVE_SHA256, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    repo, here = os.getcwd(), os.path.dirname(os.path.abspath(__file__))
    if not (os.path.isfile(os.path.join(repo, "inference.py")) and os.path.isfile(os.path.join(repo, TEST_CSV))):
        print("Run this from the C-MET repository root (cd /workspace/sir/CAFE/C-MET).")
        return 2
    missing = [f for f in PHASE1.values() if not os.path.isfile(os.path.join(here, f))]
    if missing:
        print("Phase 1 scripts missing next to this script: %s" % ", ".join(missing))
        return 2
    busy = other_run_active(here)
    if busy:
        print("Another run is active (%s); wait for it to finish." % busy)
        return 2
    tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(here, "reports"), exist_ok=True)
    lock = os.path.join(here, "reports", "cremad.lock")
    with open(lock, "w") as fh:
        fh.write("%d %s\n" % (os.getpid(), tag))
    print("%s | %s | host %s | stage %s | limit %s | workers %d" % (SCRIPT, tag, socket.gethostname(), args.stage,
                                                                   args.limit, args.workers), flush=True)
    rep = Report()
    try:
        ok = run(args, repo, here, rep, hooks)
    finally:
        try:
            if int(open(lock).read().split()[0]) == os.getpid():
                os.remove(lock)
        except Exception:
            pass
    path = os.path.join(here, "reports", "cremad_prep_%s.json" % tag)
    with open(path, "w") as fh:
        json.dump({"script": SCRIPT, "stamp": tag, "args": vars(args), "rows": rep.rows}, fh, indent=1)
    print("FAILURES: " + (", ".join(rep.failed()) or "none"))
    print("REPORT: " + path)
    return 0 if ok and not rep.failed() else 1


if __name__ == "__main__":
    sys.exit(main())
