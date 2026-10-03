#!/usr/bin/env python3
"""CAFE Gate 2: AITV (average inference time per video) with the Gate 2 generation code (v2).

Run from the C-MET repository root with the C_MET environment active, for example in the
background (about 12 s per row end to end on the DGX, so about 4 h for all 1141 rows):

    python3 ../cafe_aitv_v2.py                # all rows of evaluation/runs/mead_cmet_repro/ours.csv
    python3 ../cafe_aitv_v2.py --limit 21     # first 21 rows only

Method. C-MET's evaluation/README.md (section 7) defines AITV as the mean over test rows of a
wall-clock window that starts before the source image and driving signals are preprocessed and
stops after the last frame is generated, excluding the ffmpeg mux and the save, with no warm-up
and no torch.cuda.synchronize(). The script it names, inference_dataset_ref.py, is not in the
public repository (no commit ever contained it), so the README is the only specification.

This harness times the SAME generation code that produced the 1141 Gate 2 videos:
overnight_gate2_v4.py (next to this script) is imported, its RealGen is built once (models
loaded outside any timing), and RealGen.generate() runs unmodified. Two of its own calls are
wrapped only to read the clock:
  - img_preprocessing (self.ip), the first call inside the README window  -> window start;
  - save_video (self.sv), the first call after all frames exist          -> window end.
Everything between them (image, audio and video preprocessing, lip and expression features,
the connector, every generated frame) is inside the window; the per-(emotion, level) direction
lookup before it, and the save and mux after it, are outside, as in the README. At window end
the clock is read first (the README value), then CUDA is synchronized and read again, so the
report also shows the time including pending GPU work. No warm-up row is run.

Outputs go to evaluation/runs/aitv_<stamp>/ (never into the Gate 2 folder). Each new video is
also compared byte for byte with its Gate 2 counterpart (informational). Per-row results:
reports/aitv_<stamp>/rows.csv; the 1-minute load average is recorded per row because the DGX
is shared. AITV is hardware dependent (the paper does not state its timing GPU), so it is
reported next to the paper value for information, not as a pass or fail.
Safety for a long unattended run: before importing overnight_gate2_v4.py its source is parsed
(not executed) and refused if any top-level statement could run work on import (anything except
imports, assignments, definitions, the docstring and an if __name__ == "__main__" block). Every
row is appended to rows.csv (and every error to errors.txt) as soon as it finishes, so an
interruption loses nothing already timed; SIGTERM or Ctrl+C stops after the current row and
still prints the summary of the rows done.
Exit code: 0 done, 1 errors or interrupted, 2 bad usage or another run active.
"""

import argparse
import ast
import csv
import datetime
import hashlib
import importlib.util
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time

SCRIPT = "cafe_aitv_v2"
GEN_SCRIPT = "overnight_gate2_v4.py"
GEN_SCRIPT_SIZE = 42382          # the version that generated Gate 2 (handoff, script inventory)
OURS = "evaluation/runs/mead_cmet_repro/ours.csv"
PAPER_AITV = 2.643


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def other_run_active(here):
    for lock, tag in (("fullrun.lock", b"cafe_eval_prep"), ("fvd.lock", b"cafe_fvd_setup"), ("aitv.lock", b"cafe_aitv")):
        try:
            pid = int(open(os.path.join(here, "reports", lock)).read().split()[0])
            if pid != os.getpid() and tag in open("/proc/%d/cmdline" % pid, "rb").read():
                return "%s (pid %d)" % (tag.decode(), pid)
        except Exception:
            pass
    return None


def import_is_safe(path):
    """True if importing the module can only define things (checked on the parsed source)."""
    tree = ast.parse(open(path).read(), filename=path)
    bad = []
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.FunctionDef,
                          ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
            continue
        if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                and n.test.left.id == "__name__" and len(n.test.comparators) == 1
                and isinstance(n.test.comparators[0], ast.Constant) and n.test.comparators[0].value == "__main__"):
            continue
        if isinstance(n, ast.Try) and all(isinstance(b, (ast.Import, ast.ImportFrom)) for b in n.body):
            continue
        bad.append("line %d: %s" % (n.lineno, type(n).__name__))
    return (not bad), bad


def gpu_snapshot():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except Exception as e:
        return "nvidia-smi failed: %s" % e


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None, help="first N rows of ours.csv only")
    args = ap.parse_args()
    repo, here = os.getcwd(), os.path.dirname(os.path.abspath(__file__))
    tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not (os.path.isfile(os.path.join(repo, "inference.py")) and os.path.isfile(os.path.join(repo, OURS))):
        print("Run this from the C-MET repository root (cd /workspace/sir/CAFE/C-MET); %s must exist." % OURS)
        return 2
    gen_path = os.path.join(here, GEN_SCRIPT)
    if not os.path.isfile(gen_path):
        print("%s not found next to this script." % GEN_SCRIPT)
        return 2
    safe, bad = import_is_safe(gen_path)
    if not safe:
        print("Refusing to import %s: top-level statements that would run on import: %s"
              % (GEN_SCRIPT, "; ".join(bad[:5])))
        return 2
    busy = other_run_active(here)
    if busy:
        print("Another run is active (%s); AITV would be distorted. Wait for it to finish." % busy)
        return 2
    os.makedirs(os.path.join(here, "reports"), exist_ok=True)
    lock = os.path.join(here, "reports", "aitv.lock")
    with open(lock, "w") as fh:
        fh.write("%d %s\n" % (os.getpid(), tag))
    try:
        return run(args, repo, here, tag, gen_path)
    finally:
        try:
            if int(open(lock).read().split()[0]) == os.getpid():
                os.remove(lock)
        except Exception:
            pass


def run(args, repo, here, tag, gen_path):
    size, digest = os.path.getsize(gen_path), sha256_file(gen_path)
    print("%s | %s | host %s | %s %d B sha256 %s%s" % (SCRIPT, tag, socket.gethostname(), GEN_SCRIPT, size,
                                                     digest[:12], "" if size == GEN_SCRIPT_SIZE else
                                                     "  (WARNING: size differs from the Gate 2 version, %d B)"
                                                     % GEN_SCRIPT_SIZE), flush=True)
    rows = list(csv.DictReader(open(os.path.join(repo, OURS), newline="")))
    if args.limit:
        rows = rows[:args.limit]
    outdir = os.path.join(repo, "evaluation", "runs", "aitv_" + tag)
    gdir, work = os.path.join(outdir, "gen"), os.path.join(outdir, "work")
    os.makedirs(gdir)
    os.makedirs(work)
    logdir = os.path.join(here, "reports", "aitv_" + tag)
    os.makedirs(logdir)
    gpu_before = gpu_snapshot()
    print("GPU before loading models: %s | load average %s" % (gpu_before, os.getloadavg()), flush=True)

    spec = importlib.util.spec_from_file_location("overnight_gate2_v4", gen_path)
    og = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(og)
    t_load = time.perf_counter()
    gen = og.RealGen(print)
    t_load = time.perf_counter() - t_load
    print("models loaded in %.1f s (outside every timed window)" % t_load, flush=True)

    marks = {}
    orig_ip, orig_sv = gen.ip, gen.sv
    torch = getattr(gen, "torch", None)
    cuda = bool(torch is not None and torch.cuda.is_available())

    def ip_timed(*a, **k):
        marks["start"] = time.perf_counter()
        return orig_ip(*a, **k)

    def sv_timed(vid, *a, **k):
        marks["end"] = time.perf_counter()          # README value: no synchronize
        if cuda:
            torch.cuda.synchronize()
        marks["end_sync"] = time.perf_counter()
        marks["frames"] = int(vid.shape[2]) if hasattr(vid, "shape") and len(vid.shape) > 2 else -1
        return orig_sv(vid, *a, **k)

    gen.ip, gen.sv = ip_timed, sv_timed
    results, errors = [], []
    cols = ["row", "video", "aitv_window_s", "window_synced_s", "pre_window_s", "save_mux_s", "row_total_s",
            "frames", "load1", "identical_to_gate2"]
    rows_fh = open(os.path.join(logdir, "rows.csv"), "w", newline="")
    wr = csv.DictWriter(rows_fh, fieldnames=cols)
    wr.writeheader()
    rows_fh.flush()
    err_fh = open(os.path.join(logdir, "errors.txt"), "w")
    stop = {"flag": False}

    def on_signal(signum, frame):
        stop["flag"] = True
        print("signal %d: stopping after the current row" % signum, flush=True)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    t_all = time.perf_counter()
    for i, r in enumerate(rows):
        if stop["flag"]:
            break
        name = og.out_name(r)
        out = os.path.join(gdir, name)
        marks.clear()
        t_row = time.perf_counter()
        try:
            img = og.source_frame(r["source_video_path"], os.path.join(outdir, "src_frames"))
            t_gen = time.perf_counter()
            gen.generate(img, r["source_video_path"], r["source_video_path"][:-4] + ".wav",
                         r["gt_emotion"], r["intensity"], out, work)
            t_done = time.perf_counter()
            good, why = og.probe_ok(out)
            if not good:
                raise RuntimeError("invalid output: " + why)
            if "start" not in marks or "end" not in marks:
                raise RuntimeError("timing marks missing (generate() call order changed?)")
            ref = os.path.join(repo, r["generated_path"])
            same = os.path.isfile(ref) and sha256_file(ref) == sha256_file(out)
            rec = {"row": i, "video": name, "aitv_window_s": marks["end"] - marks["start"],
                   "window_synced_s": marks["end_sync"] - marks["start"],
                   "pre_window_s": marks["start"] - t_gen, "save_mux_s": t_done - marks["end_sync"],
                   "row_total_s": t_done - t_row, "frames": marks.get("frames", -1),
                   "load1": os.getloadavg()[0], "identical_to_gate2": same}
            results.append(rec)
            wr.writerow(rec)
            rows_fh.flush()
        except Exception as e:
            errors.append((i, name, "%s: %s" % (type(e).__name__, str(e)[:200])))
            err_fh.write("%d\t%s\t%s\n" % errors[-1])
            err_fh.flush()
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            w = [x["aitv_window_s"] for x in results]
            print("[aitv] %d/%d ok=%d err=%d | window mean %.3f s | %.1f s/row end to end | load %.0f"
                  % (i + 1, len(rows), len(results), len(errors), statistics.mean(w) if w else float("nan"),
                     (time.perf_counter() - t_all) / (i + 1), os.getloadavg()[0]), flush=True)

    rows_fh.close()
    err_fh.close()
    interrupted = stop["flag"] and len(results) + len(errors) < len(rows)
    if interrupted:
        print("INTERRUPTED after %d of %d rows; the summary below covers the rows done" % (len(results) + len(errors), len(rows)))
    summary = {"script": SCRIPT, "stamp": tag, "generation_script": GEN_SCRIPT, "generation_script_size": size,
               "generation_script_sha256": digest, "rows_requested": len(rows), "rows_timed": len(results),
               "errors": errors, "model_load_s": t_load, "gpu_before": gpu_before, "interrupted": interrupted}
    print("")
    if results:
        w = [x["aitv_window_s"] for x in results]
        ws = [x["window_synced_s"] for x in results]
        tot = [x["row_total_s"] for x in results]
        same = sum(1 for x in results if x["identical_to_gate2"])
        loads = [x["load1"] for x in results]
        summary.update({"aitv_readme_s": statistics.mean(w), "aitv_median_s": statistics.median(w),
                        "aitv_first_row_s": w[0], "aitv_without_first_s": statistics.mean(w[1:]) if len(w) > 1 else None,
                        "window_synced_mean_s": statistics.mean(ws), "row_total_mean_s": statistics.mean(tot),
                        "identical_to_gate2": same, "load1_min": min(loads), "load1_max": max(loads)})
        print("AITV (README method, n=%d)         %.4f s   (paper %.3f s on unstated hardware; information only)"
              % (len(w), summary["aitv_readme_s"], PAPER_AITV))
        print("  median %.4f s | first row (cold, included as in the README) %.4f s | mean without it %s"
              % (summary["aitv_median_s"], w[0], "%.4f s" % summary["aitv_without_first_s"]
                 if summary["aitv_without_first_s"] is not None else "-"))
        print("  same window with torch.cuda.synchronize(): mean %.4f s" % summary["window_synced_mean_s"])
        print("  end to end per row (source frame, direction, window, save, mux): mean %.2f s"
              % summary["row_total_mean_s"])
        print("  outputs byte-identical to Gate 2: %d/%d | 1-min load average %.0f to %.0f"
              % (same, len(results), min(loads), max(loads)))
    print("errors: %d%s" % (len(errors), "" if not errors else " (first: %s)" % (errors[0],)))
    print("GPU at end: %s" % gpu_snapshot())
    with open(os.path.join(logdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print("RESULT: %s (videos in %s)" % (logdir, os.path.relpath(outdir, repo)))
    return 1 if errors or not results or interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
