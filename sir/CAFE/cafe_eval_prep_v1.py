#!/usr/bin/env python3
"""CAFE Gate 2, step 2: evaluation CSV build and subset dry run (v1).

Run from the C-MET repository root, with the C_MET environment active:

    python3 ../cafe_eval_prep_v1.py               # validate, build CSVs, run the dry run
    python3 ../cafe_eval_prep_v1.py --check-only  # validate and build CSVs only

What it does
  1. Reads evaluation/runs/mead_cmet_repro/ours.csv (the generation output), resolves every
     path against the repository root and validates it: required columns; gt_emotion equal to
     the Emotion-FAN label strings (check_quantitative_all.py compares raw strings); every
     generated file named after its own row (ID_emotion_level_NNN.mp4), so rows and videos
     cannot be misaligned; no duplicates; coverage of dataset/MEAD/test.csv. The CSVs it writes
     hold only the six standard columns plus orig_row, in the standard order: generated_path is
     column 5, which vide2frame_custom.py, frame2face_custom.py, all_pipeline.py and
     all_syncnet.py read, and no "gt" column exists for the last two to prefer.
  2. Source audio: all_pipeline.py swaps the extension of source_audio_path for .wav and
     run_pipeline.py copies that file unchanged as the track audio. An existing .wav is used as
     is and its format is reported (16 kHz mono PCM s16 is what run_pipeline.py extracts on its
     own). A missing .wav is extracted from source_audio_path in exactly that format into
     evaluation/runs/source_wav/, never next to the dataset files.
  3. Writes the validated full-run CSV (absolute paths) to evaluation/runs/staging/ours.csv and
     a stratified subset (one row per emotion x intensity, identities rotated) to
     evaluation/runs/mead_ours/ours_dryrun.csv.
  4. Dry run: runs the README commands on the subset from evaluation/ with stdin closed
     (run_pipeline.py drops into pdb when ffmpeg fails; vide2frame_custom.py's ffmpeg prompts
     before overwriting), logs each command, then checks every output row by row: frames,
     face crops, FID, predicted_emotion, SyncNet face tracks and Sync_conf.

Nothing is deleted. Outputs of an earlier dry run are moved to
evaluation/runs/archive/dryrun_<stamp>/ before a new one starts.
Exit code: 0 clean, 1 blockers or incomplete dry run, 2 bad usage.
"""

import argparse
import datetime
import glob
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import wave

SCRIPT = "cafe_eval_prep_v1"
EVAL = "evaluation"
GEN_CSV = "evaluation/runs/mead_cmet_repro/ours.csv"
TEST_CSV = "dataset/MEAD/test.csv"
STAGING_CSV = "evaluation/runs/staging/ours.csv"
RUN_DIR = "evaluation/runs/mead_ours"
DRY_NAME = "ours_dryrun"
WAV_DIR = "evaluation/runs/source_wav"
ARCHIVE_DIR = "evaluation/runs/archive"
COLUMNS = ["source_video_path", "gt_video_path", "gt_emotion", "intensity", "generated_path",
           "source_audio_path"]
PATH_COLUMNS = ["source_video_path", "gt_video_path", "generated_path", "source_audio_path"]
LABELS = ["happy", "angry", "disgusted", "fear", "sad", "contempt", "surprised"]  # emotion-fan.py
LEVELS = ["level_1", "level_2", "level_3"]
SAMPLE_COUNT = 16  # frame2face_custom.py sample_count
GT_RE = re.compile(r"/([^/]+)/front/([^/]+)/(level_\d+)/(\d+)\.mp4$")


class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, status, group, name, detail, **extra):
        row = {"status": status, "group": group, "name": name, "detail": detail}
        row.update(extra)
        self.rows.append(row)
        print("[%-7s] %-6s %-30s %s" % (status, group, name, detail), flush=True)

    def blockers(self):
        return [r["name"] for r in self.rows if r["status"] in ("FAIL", "MISSING")]


def stamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def resolve(repo, p):
    p = str(p).strip()
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(repo, p))


def examples(items, k=3):
    items = list(items)
    more = "" if len(items) <= k else " (+%d more)" % (len(items) - k)
    return ", ".join(str(x) for x in items[:k]) + more


def expected_gen_name(gt_path):
    m = GT_RE.search(gt_path.replace("\\", "/"))
    if not m:
        return None
    ident, emo, level, num = m.groups()
    return "%s_%s_%s_%s.mp4" % (ident, emo, level, num)


def identity_of(gt_path):
    m = GT_RE.search(gt_path.replace("\\", "/"))
    return m.group(1) if m else "?"


def wav_format(path):
    try:
        w = wave.open(path, "rb")
        try:
            return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        finally:
            w.close()
    except Exception as e:
        return "unreadable as PCM wav (%s: %s)" % (type(e).__name__, str(e)[:60])


# ---------------------------------------------------------------- validation and CSV build

def load_generation_csv(rep, repo):
    import pandas as pd
    path = os.path.join(repo, GEN_CSV)
    if not os.path.isfile(path):
        rep.add("MISSING", "csv", "generation CSV", GEN_CSV + " not found")
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        rep.add("FAIL", "csv", "generation CSV", "missing columns: " + ", ".join(missing))
        return None
    rep.add("OK", "csv", "generation CSV", "%d rows; columns %s" % (len(df), ", ".join(df.columns)))
    return df


def validate_rows(rep, repo, df):
    """Return the DataFrame with absolute paths, or None when a blocker makes it unusable."""
    import pandas as pd
    ok = True
    labels = sorted(set(df["gt_emotion"]))
    bad = [x for x in labels if x not in LABELS]
    if bad:
        rep.add("FAIL", "csv", "gt_emotion labels", "not Emotion-FAN label strings: %s (allowed: %s)"
                % (examples(bad), ", ".join(LABELS)))
        ok = False
    else:
        counts = df["gt_emotion"].value_counts()
        rep.add("OK", "csv", "gt_emotion labels", ", ".join("%s %d" % (k, counts[k]) for k in LABELS
                                                          if k in counts.index))
    badlev = sorted(set(df["intensity"]) - set(LEVELS))
    if badlev:
        rep.add("FAIL", "csv", "intensity labels", "unexpected: " + examples(badlev))
        ok = False

    out = df.copy()
    out["orig_row"] = [str(i) for i in range(len(df))]
    for c in PATH_COLUMNS:
        out[c] = [resolve(repo, p) for p in df[c]]
    for c, required in (("generated_path", True), ("gt_video_path", True), ("source_video_path", False)):
        gone = [p for p in out[c] if not os.path.isfile(p)]
        if gone:
            rep.add("FAIL" if required else "WARN", "files", c,
                    "%d/%d missing, e.g. %s" % (len(gone), len(out), examples(gone, 2)))
            ok = ok and not required
        else:
            rep.add("OK", "files", c, "%d/%d present" % (len(out), len(out)))

    mism = []
    for i, (gt, gen) in enumerate(zip(out["gt_video_path"], out["generated_path"])):
        want = expected_gen_name(gt)
        if want is None or os.path.basename(gen) != want:
            mism.append("row %d: %s vs %s" % (i, os.path.basename(gen), want))
    if mism:
        rep.add("FAIL", "csv", "row/video alignment", "%d generated file names do not match their "
                "row's gt_video_path: %s" % (len(mism), examples(mism, 2)))
        ok = False
    else:
        rep.add("OK", "csv", "row/video alignment", "every generated file is named after its row's "
                "gt_video_path")
    lab = [i for i, (gt, e, lv) in enumerate(zip(out["gt_video_path"], out["gt_emotion"], out["intensity"]))
           if GT_RE.search(gt.replace("\\", "/")) is None
           or GT_RE.search(gt.replace("\\", "/")).group(2) != e
           or GT_RE.search(gt.replace("\\", "/")).group(3) != lv]
    if lab:
        rep.add("FAIL", "csv", "label/path agreement", "%d rows whose gt_emotion or intensity differ "
                "from gt_video_path, e.g. rows %s" % (len(lab), examples(lab)))
        ok = False
    dup_pair = out.duplicated(subset=["source_video_path", "gt_video_path"]).sum()
    dup_gen = out.duplicated(subset=["generated_path"]).sum()
    if dup_pair or dup_gen:
        rep.add("FAIL", "csv", "duplicates", "%d duplicate (source, gt) pairs, %d duplicate generated "
                "files" % (dup_pair, dup_gen))
        ok = False
    else:
        rep.add("OK", "csv", "duplicates", "none")

    tpath = os.path.join(repo, TEST_CSV)
    if os.path.isfile(tpath):
        t = pd.read_csv(tpath, dtype=str, keep_default_na=False)
        tkeys = set(zip((resolve(repo, p) for p in t["source_video_path"]),
                        (resolve(repo, p) for p in t["gt_video_path"])))
        okeys = set(zip(out["source_video_path"], out["gt_video_path"]))
        absent = sorted(os.path.relpath(g, repo) for _, g in tkeys - okeys)
        extra = sorted(os.path.relpath(g, repo) for _, g in okeys - tkeys)
        if extra:
            rep.add("FAIL", "csv", "test.csv coverage", "%d rows not in %s, e.g. %s"
                    % (len(extra), TEST_CSV, examples(extra, 2)))
            ok = False
        elif absent:
            rep.add("WARN", "csv", "test.csv coverage", "%d of %d test rows have no generated video "
                    "(report as a deviation): %s" % (len(absent), len(tkeys), examples(absent, 4)),
                    absent=absent)
        else:
            rep.add("OK", "csv", "test.csv coverage", "all %d test rows present" % len(tkeys))
    else:
        rep.add("WARN", "csv", "test.csv coverage", TEST_CSV + " not found, coverage not checked")
    return out if ok else None


def prepare_audio(rep, repo, out, check_only):
    """Point source_audio_path at a .wav that all_pipeline.py will find; extract missing ones."""
    ff = shutil.which("ffmpeg")
    wavs, fmt, extracted, failed, planned = [], {}, 0, [], set()
    for src in out["source_audio_path"]:
        w = os.path.splitext(src)[0] + ".wav"  # exactly what all_pipeline.py looks for
        if os.path.isfile(w):
            wavs.append(w)
            continue
        rel = os.path.relpath(os.path.splitext(src)[0], repo)
        if rel.startswith(".."):
            rel = "external/" + re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(src)[0].strip("/"))
        dst = os.path.join(repo, WAV_DIR, rel + ".wav")
        wavs.append(dst)
        if os.path.isfile(dst):
            continue
        if not os.path.isfile(src):
            failed.append("source audio missing: " + os.path.relpath(src, repo))
            continue
        if check_only:
            planned.add(dst)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        part = dst + ".part.wav"
        res = subprocess.run([ff or "ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", "1",
                              "-ar", "16000", "-acodec", "pcm_s16le", part],
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if res.returncode == 0 and os.path.isfile(part) and os.path.getsize(part) > 44:
            os.replace(part, dst)
            extracted += 1
        else:
            if os.path.exists(part):
                os.remove(part)
            failed.append("%s: ffmpeg rc=%d" % (os.path.relpath(src, repo), res.returncode))
    out["source_audio_path"] = wavs
    uniq = sorted(set(w for w in wavs if os.path.isfile(w)))
    for w in uniq:
        f = wav_format(w)
        key = "%d Hz, %d ch, %d-bit" % (f[0], f[1], 8 * f[2]) if isinstance(f, tuple) else f
        fmt.setdefault(key, []).append(w)
    short = [os.path.relpath(w, repo) for w in uniq if isinstance(wav_format(w), tuple)
             and wav_format(w)[3] < 16000 * 0.5]
    detail = "%d rows, %d unique wav files; %s" % (len(wavs), len(set(wavs)),
                                                  "; ".join("%s: %d" % (k, len(v)) for k, v in fmt.items()))
    if extracted:
        detail += "; extracted %d missing .wav (16 kHz mono PCM s16) into %s" % (extracted, WAV_DIR)
    if planned:
        detail += "; %d .wav to extract into %s on a full run" % (len(planned), WAV_DIR)
    if failed:
        rep.add("FAIL", "audio", "source audio (.wav)", "%s; %d problems: %s" % (detail, len(failed),
                                                                                examples(failed, 2)))
        return False
    odd = [k for k in fmt if k != "16000 Hz, 1 ch, 16-bit"]
    if odd or short:
        rep.add("WARN", "audio", "source audio (.wav)", detail + ("; formats other than 16 kHz mono PCM "
                "s16 are copied unchanged by run_pipeline.py" if odd else "") +
                ("; very short files: " + examples(short, 2) if short else ""))
    else:
        rep.add("OK", "audio", "source audio (.wav)", detail)
    return True


def pick_subset(out):
    """One row per (emotion, intensity) cell, identities rotated, first matching row in CSV order."""
    ids = sorted(set(identity_of(g) for g in out["gt_video_path"]))
    picks, k = [], 0
    for emo in LABELS:
        for lev in LEVELS:
            cell = [i for i in range(len(out)) if out["gt_emotion"].iloc[i] == emo and out["intensity"].iloc[i] == lev]
            if not cell:
                continue
            want = ids[k % len(ids)] if ids else "?"
            k += 1
            same = [i for i in cell if identity_of(out["gt_video_path"].iloc[i]) == want]
            picks.append((same or cell)[0])
    return picks


def write_csv(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    frame.to_csv(part, index=False)
    os.replace(part, path)


# ---------------------------------------------------------------- dry run

def dry_outputs(repo):
    ev = os.path.join(repo, EVAL)
    return [os.path.join(ev, "runs/mead_ours", DRY_NAME + ".csv"),
            os.path.join(ev, "runs/mead_ours/frames", DRY_NAME),
            os.path.join(ev, "runs/mead_ours/frames", DRY_NAME + "_GT"),
            os.path.join(ev, "runs/mead_ours/faces", DRY_NAME),
            os.path.join(ev, "syncnet_python/workspace", DRY_NAME)]


def archive_previous(rep, repo, tag):
    moved = []
    for p in dry_outputs(repo):
        if os.path.lexists(p):
            dst = os.path.join(repo, ARCHIVE_DIR, "dryrun_" + tag, os.path.relpath(p, os.path.join(repo, EVAL)))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(p, dst)
            moved.append(os.path.relpath(p, repo))
    if moved:
        rep.add("INFO", "dry", "previous dry run", "moved to %s/dryrun_%s: %s" % (ARCHIVE_DIR, tag,
                                                                               examples(moved, 5)))


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def stop_group(proc, grace=10.0):
    """SIGTERM the step's whole process group, then SIGKILL whatever is left after `grace` s."""
    pgid = proc.pid
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        end = time.time() + wait
        while time.time() < end:
            proc.poll()  # reap the leader, or its zombie keeps the group "alive"
            if not group_alive(pgid):
                return
            time.sleep(0.2)


def run_step(rep, repo, logdir, n, name, args, timeout):
    """Run one README command in its own process group, so that a timeout also stops the
    driver's own children (multiprocessing workers, ffmpeg, face_align_cuda.py), which would
    otherwise keep running and write into the next run's outputs."""
    log = os.path.join(logdir, "%d_%s.log" % (n, name))
    t0 = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    leftovers = False
    with open(log, "wb") as fh:
        fh.write(("$ cd %s && %s\n" % (EVAL, " ".join(args))).encode())
        fh.flush()
        proc = subprocess.Popen(args, cwd=os.path.join(repo, EVAL), stdin=subprocess.DEVNULL, stdout=fh,
                                stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_group(proc)
            proc.wait()
            rc = "timeout"
        except KeyboardInterrupt:
            stop_group(proc, grace=3.0)
            raise
        if group_alive(proc.pid):  # children still running after the driver itself exited
            leftovers = True
            stop_group(proc)
    dt = time.time() - t0
    status = "OK" if rc == 0 and not leftovers else "FAIL"
    note = "; stopped leftover child processes" if leftovers else ""
    rep.add(status, "dry", "%d %s" % (n, name), "rc=%s in %.0f s%s; log %s" % (rc, dt, note, os.path.relpath(log, repo)),
            seconds=round(dt, 1), log=log)
    return rc == 0


def tail(path, pattern, k=3):
    try:
        lines = open(path, errors="replace").read().splitlines()
    except Exception:
        return []
    return [ln.strip()[:140] for ln in lines if re.search(pattern, ln)][-k:]


def verify_outputs(rep, repo, n_rows, logdir):
    """Row-by-row check of everything the dry run should have produced."""
    import pandas as pd
    ev = os.path.join(repo, EVAL)
    df = pd.read_csv(os.path.join(ev, "runs/mead_ours", DRY_NAME + ".csv"))
    ws = os.path.join(ev, "syncnet_python/workspace", DRY_NAME)
    rows, bad = [], {"frames": [], "faces": [], "FID": [], "emotion": [], "tracks": [], "Sync_conf": []}
    for i in range(n_rows):
        d = "%07d" % i
        fg = len(glob.glob(os.path.join(ev, "runs/mead_ours/frames", DRY_NAME, d, "*.jpg")))
        ft = len(glob.glob(os.path.join(ev, "runs/mead_ours/frames", DRY_NAME + "_GT", d, "*.jpg")))
        fc = len(glob.glob(os.path.join(ev, "runs/mead_ours/faces", DRY_NAME, d, "*.jpg")))
        ref = os.path.splitext(os.path.basename(str(df.at[i, "generated_path"])))[0]
        conf_file = os.path.join(ws, "confidences", ref + ".txt")
        tracks = 0
        if os.path.isfile(conf_file):
            tracks = len([ln for ln in open(conf_file) if ln.strip()])
        fid = df.at[i, "FID"] if "FID" in df.columns else None
        pred = df.at[i, "predicted_emotion"] if "predicted_emotion" in df.columns else None
        sc = df.at[i, "Sync_conf"] if "Sync_conf" in df.columns else None
        pred = None if (pred is None or pd.isna(pred) or str(pred).strip() == "") else str(pred).strip()
        fid = None if (fid is None or pd.isna(fid)) else float(fid)
        sc = None if (sc is None or pd.isna(sc)) else float(sc)
        if fg == 0 or ft == 0:
            bad["frames"].append(i)
        if fc == 0:
            bad["faces"].append(i)
        if fid is None:
            bad["FID"].append(i)
        if pred is None:
            bad["emotion"].append(i)
        if tracks == 0:
            bad["tracks"].append(i)
        if sc is None:
            bad["Sync_conf"].append(i)
        rows.append({"row": i, "video": ref, "frames_gen": fg, "frames_gt": ft, "crops": fc, "FID": fid,
                     "gt_emotion": df.at[i, "gt_emotion"], "predicted": pred, "tracks": tracks, "Sync_conf": sc})

    print("\n  row video                           frames gen/gt crops      FID  gt -> predicted        tracks Sync_conf")
    for r in rows:
        print("  %3d %-31s %6d/%-6d %5d %8s  %-9s -> %-10s %6d %9s" % (
            r["row"], r["video"][:31], r["frames_gen"], r["frames_gt"], r["crops"],
            "-" if r["FID"] is None else "%.2f" % r["FID"], r["gt_emotion"], r["predicted"] or "-",
            r["tracks"], "-" if r["Sync_conf"] is None else "%.3f" % r["Sync_conf"]))
    print("", flush=True)

    def cover(name, key, what, warn_only=False, hint=""):
        miss = bad[key]
        if miss:
            rep.add("WARN" if warn_only else "FAIL", "check", name, "%d/%d rows %s: rows %s%s"
                    % (len(miss), n_rows, what, examples(miss, 6), hint))
        else:
            rep.add("OK", "check", name, "%d/%d rows" % (n_rows, n_rows))

    cover("frames (gen and GT)", "frames", "without frames")
    few = [r["row"] for r in rows if 0 < r["crops"] < SAMPLE_COUNT]
    cover("face crops", "faces", "without any face crop")
    if few:
        rep.add("WARN", "check", "face crops per video", "%d rows with fewer than %d crops: rows %s"
                % (len(few), SAMPLE_COUNT, examples(few, 6)))
    cover("FID column", "FID", "without FID")
    cover("predicted_emotion column", "emotion", "without a prediction")
    cover("SyncNet face tracks", "tracks", "without a face track",
          hint="; see %s" % os.path.relpath(os.path.join(logdir, "5_all_pipeline.log"), repo))
    cover("Sync_conf column", "Sync_conf", "without Sync_conf")
    multi = [r["row"] for r in rows if r["tracks"] > 1]
    if multi:
        rep.add("WARN", "check", "tracks per video", "%d rows with more than one face track (Sync_conf "
                "is their mean): rows %s" % (len(multi), examples(multi, 6)))
    fails = tail(os.path.join(logdir, "5_all_pipeline.log"), r"->\s*returncode=|->\s*error=|audio_errors: \[.+\]|video_errors: \[.+\]")
    if fails:
        rep.add("WARN", "check", "all_pipeline.py messages", " | ".join(fails))

    fids = [r["FID"] for r in rows if r["FID"] is not None]
    scs = [r["Sync_conf"] for r in rows if r["Sync_conf"] is not None]
    preds = [r for r in rows if r["predicted"] is not None]
    acc = sum(1 for r in preds if r["predicted"] == r["gt_emotion"])
    summary = "n=%d; FID mean %s (%d rows); Acc_emo %s (%d/%d); Sync_conf mean %s (%d rows)" % (
        n_rows, "%.3f" % (sum(fids) / len(fids)) if fids else "-", len(fids),
        "%.2f%%" % (100.0 * acc / len(preds)) if preds else "-", acc, len(preds),
        "%.4f" % (sum(scs) / len(scs)) if scs else "-", len(scs))
    rep.add("INFO", "check", "subset metrics", summary + "; pipeline sanity only, not comparable to Table 1")
    return rows


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-only", action="store_true", help="validate and build CSVs, no dry run")
    ap.add_argument("--step-timeout", type=int, default=3600, help="seconds per dry-run command")
    args = ap.parse_args()

    repo = os.getcwd()
    here = os.path.dirname(os.path.abspath(__file__))
    tag = stamp()
    if not (os.path.isfile(os.path.join(repo, "inference.py")) and
            os.path.isdir(os.path.join(repo, EVAL, "syncnet_python"))):
        print("Run this from the C-MET repository root (cd /workspace/sir/CAFE/C-MET).")
        return 2
    try:
        import pandas  # noqa: F401
    except Exception as e:
        print("pandas is not importable in this interpreter (%s); activate C_MET first." % e)
        return 2
    print("%s | %s | host %s | repo %s | mode %s" % (SCRIPT, tag, socket.gethostname(), repo,
                                                    "check-only" if args.check_only else "check + dry run"),
          flush=True)
    rep = Report()

    df = load_generation_csv(rep, repo)
    out = validate_rows(rep, repo, df) if df is not None else None
    audio_ok = prepare_audio(rep, repo, out, args.check_only) if out is not None else False

    run_dir = os.path.join(repo, RUN_DIR)
    others = sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "*.csv"))
                    if os.path.basename(p) != DRY_NAME + ".csv")
    if others:
        rep.add("FAIL", "dry", "runs/mead_ours is clean", "other CSVs present (%s); vide2frame_custom.py "
                "and frame2face_custom.py process every CSV there, move them out first" % examples(others))
    else:
        rep.add("OK", "dry", "runs/mead_ours is clean", "no CSV other than %s.csv" % DRY_NAME)

    subset = []
    if out is not None and audio_ok:
        full = out[COLUMNS + ["orig_row"]]
        if not args.check_only:
            write_csv(full, os.path.join(repo, STAGING_CSV))
            rep.add("OK", "csv", "full-run CSV", "%d rows with absolute paths -> %s (used in the full run, "
                    "not now)" % (len(full), STAGING_CSV))
        subset = pick_subset(out)
        cells = sorted(set((out["gt_emotion"].iloc[i], out["intensity"].iloc[i]) for i in subset))
        ids = sorted(set(identity_of(out["gt_video_path"].iloc[i]) for i in subset))
        rep.add("OK", "csv", "dry-run subset", "%d rows covering %d emotion x intensity cells, identities %s"
                % (len(subset), len(cells), ", ".join(ids)))

    block = rep.blockers()
    if args.check_only or block:
        if block and not args.check_only:
            rep.add("INFO", "dry", "dry run", "not started because of the blockers above")
        return finish(rep, here, tag, 1 if block else 0, [])

    # ---- dry run
    archive_previous(rep, repo, tag)
    sub = out[COLUMNS + ["orig_row"]].iloc[subset].reset_index(drop=True)
    dry_csv = os.path.join(run_dir, DRY_NAME + ".csv")
    write_csv(sub, dry_csv)
    rep.add("OK", "dry", "subset CSV", "%d rows -> %s" % (len(sub), os.path.relpath(dry_csv, repo)))
    logdir = os.path.join(here, "reports", "dryrun_" + tag)
    os.makedirs(logdir, exist_ok=True)
    py = sys.executable
    csv_rel = "runs/mead_ours/%s.csv" % DRY_NAME
    steps = [
        ("vide2frame", [py, "vide2frame_custom.py"]),
        ("frame2face", [py, "frame2face_custom.py"]),
        ("fid", [py, "pytorch-fid/custom.py", "runs/mead_ours/frames/" + DRY_NAME,
                 "runs/mead_ours/frames/%s_GT" % DRY_NAME, "--csv_path", csv_rel]),
        ("emotion_fan", [py, "Emotion-FAN/emotion-fan.py", "--csv_file", csv_rel, "--checkpoint",
                         "Emotion-FAN/checkpoints/Emotion-FAN_MEAD.pth", "--num_frames", "16"]),
        ("all_pipeline", [py, "syncnet_python/all_pipeline.py", "--csv_path", csv_rel]),
        ("all_syncnet", [py, "syncnet_python/all_syncnet.py", "--csv_path", csv_rel]),
        ("check_quantitative", [py, "check_quantitative_all.py", "--csv_path", csv_rel]),
    ]
    t0 = time.time()
    for n, (name, cmd) in enumerate(steps, 1):
        print("... step %d/%d %s" % (n, len(steps), name), flush=True)
        run_step(rep, repo, logdir, n, name, cmd, args.step_timeout)
    rows = verify_outputs(rep, repo, len(sub), logdir)
    q = tail(os.path.join(logdir, "7_check_quantitative.log"), r"^(FID|fvd|Sync_conf|Accemo):", 4)
    q = [ln.split(":")[0] + ": missing" if "(missing" in ln else ln for ln in q]
    if q:
        rep.add("INFO", "check", "check_quantitative_all.py", " | ".join(q))
    rep.add("INFO", "dry", "wall time", "%.1f min for %d videos" % ((time.time() - t0) / 60.0, len(sub)))
    return finish(rep, here, tag, 1 if rep.blockers() else 0, rows)


def finish(rep, here, tag, code, rows):
    counts = {}
    for r in rep.rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("SUMMARY " + " ".join("%s=%d" % (k.lower(), counts.get(k, 0))
                                for k in ("OK", "WARN", "FAIL", "MISSING", "INFO")))
    print("BLOCKERS: " + (", ".join(rep.blockers()) or "none"))
    os.makedirs(os.path.join(here, "reports"), exist_ok=True)
    path = os.path.join(here, "reports", "eval_prep_%s.json" % tag)
    n = 2
    while os.path.exists(path):  # never overwrite an earlier report
        path = os.path.join(here, "reports", "eval_prep_%s-%d.json" % (tag, n))
        n += 1
    with open(path, "w") as fh:
        json.dump({"script": SCRIPT, "stamp": tag, "rows": rep.rows, "dryrun_rows": rows,
                   "blockers": rep.blockers()}, fh, indent=1, default=str)
    print("REPORT: " + path)
    return code


if __name__ == "__main__":
    sys.exit(main())
