#!/usr/bin/env python3
"""CAFE Gate 2, FVD: environment setup, I3D installation, smoke test and full run (v3).

Run from the C-MET repository root with the C_MET environment active:

    python3 ../cafe_fvd_setup_v3.py          # set up and verify (safe to repeat)
    python3 ../cafe_fvd_setup_v3.py --run    # FVD on evaluation/runs/mead_ours/ours.csv

Setup (idempotent):
  1. I3D archive: the newest i3d-kinetics*.tar.gz next to this script (Kaggle,
     deepmind/i3d-kinetics, TensorFlow1, variation 400, version 1). Size and SHA-256 are
     recorded. Members are checked (no absolute or parent paths, no links, a TF-Hub module at
     the top level or inside one top-level folder) and extracted atomically into
     <script dir>/tfhub_cache/<sha1 of the handle>. tensorflow_hub 0.15.0 maps the handle used
     by frechet_video_distance.py, https://tfhub.dev/deepmind/i3d-kinetics-400/1, to exactly
     that folder and uses it without any download when it is not empty. fvd.py therefore runs
     unchanged and offline.
  2. A separate virtual environment <script dir>/envs/fvd, created from this interpreter
     (Python 3.9), with pinned packages. The C_MET environment is not touched. Wheels are first
     downloaded into <script dir>/envs/wheelhouse in up to 5 attempts (each limited in time; a
     stalled connection is dropped and the next attempt skips every file already complete), then
     installed offline from that folder. A rerun of the setup reuses the wheelhouse.
  3. Probes inside that environment, with network access blocked by a dead proxy: package
     versions, GPU visibility, resolution of the handle to the cache folder, and a graph with
     the module applied the way frechet_video_distance.py applies it (tensor
     RGB/inception_i3d/Mean:0, 400 values per video, finite).
  4. Smoke test: fvd.py itself on a two-row CSV under reports/: row 1 is the first pair of
     ours.csv, row 2 is that generated video against itself, which must give about 0.

Run (--run): requires a passed setup for the same archive and pins, no other CSV in
runs/mead_ours (fvd.py processes every CSV there), and no cafe_eval_prep run active. Backs up
ours.csv, runs `python fvd.py runs/mead_ours` from evaluation/ in the FVD environment (own
process group, stdin closed), then checks: every row has a finite fvd, every other column is
unchanged against the backup, and reports the mean next to C-MET Table 1 (FVD 329.862,
10 percent tolerance) before running check_quantitative_all.py with this interpreter.

Nothing is deleted. Exit code: 0 clean, 1 failure, 2 bad usage or another run active.
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time

SCRIPT = "cafe_fvd_setup_v3"
HANDLE = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
HANDLE_DIR = "092225fb776e28d6d64ac605ab6be03f18dd2027"  # sha1(HANDLE), checked at start
ARCHIVE_GLOB = "i3d-kinetics*.tar.gz"
TF_SPEC = os.environ.get("CAFE_FVD_TF_SPEC", "tensorflow[and-cuda]==2.15.1")  # override: testing only
# tensorflow-hub 0.15.0 is the last release with the TF1 hub.Module API that frechet_video_distance.py
# calls (0.16 removed it, and also pulls tf-keras). Its resolver maps an https handle to
# TFHUB_CACHE_DIR/sha1(handle) and skips the download when that folder is not empty.
PINS = [TF_SPEC, "tensorflow-probability==0.23.0", "tensorflow-gan==2.1.0", "tensorflow-hub==0.15.0",
        "numpy==1.26.4", "pandas==2.3.3", "opencv-python-headless==4.11.0.86", "tqdm==4.70.1", "six==1.17.0"]
COLUMNS = ["source_video_path", "gt_video_path", "gt_emotion", "intensity", "generated_path",
           "source_audio_path"]
RUN_CSV = "evaluation/runs/mead_ours/ours.csv"
PAPER_FVD, TOLERANCE = 329.862, 0.10
DL_ATTEMPTS = 5
DL_TIMEOUT = int(os.environ.get("CAFE_FVD_DL_TIMEOUT", "1800"))  # seconds per download attempt
DEAD_PROXY = {"HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
              "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9", "NO_PROXY": "", "no_proxy": ""}

PROBE = r'''
import json, os, sys
out = {}
try:
    import tensorflow as tf, tensorflow_hub as hub, tensorflow_gan as tfgan, tensorflow_probability as tfp
    import cv2, numpy, pandas
    out["versions"] = {"tensorflow": tf.__version__, "tensorflow_hub": hub.__version__,
                       "tensorflow_gan": getattr(tfgan, "__version__", "?"), "tensorflow_probability": tfp.__version__,
                       "cv2": cv2.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__}
    out["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    out["resolved"] = hub.resolve(sys.argv[1])
    tf1 = tf.compat.v1
    tf1.disable_v2_behavior()
    with tf1.Graph().as_default():
        x = tf1.placeholder(tf1.float32, [16, 15, 224, 224, 3])
        m = hub.Module(sys.argv[1], name="probe_module")
        m(x)
        t = tf1.get_default_graph().get_tensor_by_name("probe_module_apply_default/RGB/inception_i3d/Mean:0")
        cfg = tf1.ConfigProto()
        cfg.gpu_options.allow_growth = True
        with tf1.Session(config=cfg) as s:
            s.run(tf1.global_variables_initializer())
            s.run(tf1.tables_initializer())
            v = s.run(t, {x: numpy.zeros([16, 15, 224, 224, 3], "float32")})
    out["embedding_shape"] = list(v.shape)
    out["embedding_finite"] = bool(numpy.isfinite(v).all())
    out["n_variables"] = len(m.variables)
except Exception as e:
    out["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
print("PROBE_JSON " + json.dumps(out))
'''


class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, status, group, name, detail, **extra):
        row = {"status": status, "group": group, "name": name, "detail": detail}
        row.update(extra)
        self.rows.append(row)
        print("[%-7s] %-6s %-26s %s" % (status, group, name, detail), flush=True)

    def blockers(self):
        return [r["name"] for r in self.rows if r["status"] in ("FAIL", "MISSING")]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def err(e):
    return "%s: %s" % (type(e).__name__, str(e).strip().splitlines()[0][:160] if str(e).strip() else "")


def fvd_env(here):
    env = dict(os.environ, TFHUB_CACHE_DIR=os.path.join(here, "tfhub_cache"), PYTHONUNBUFFERED="1",
               TF_CPP_MIN_LOG_LEVEL="1", PYTHONNOUSERSITE="1")
    for k in ("PYTHONPATH", "PYTHONHOME"):  # never let C_MET or user packages leak into the FVD env
        env.pop(k, None)
    env.update(DEAD_PROXY)  # any download attempt fails loudly instead of fetching something else
    return env


# ---------------------------------------------------------------- setup steps

def install_module(rep, here):
    cands = sorted(glob.glob(os.path.join(here, ARCHIVE_GLOB)), key=os.path.getmtime, reverse=True)
    if not cands:
        rep.add("MISSING", "i3d", "I3D archive", "no %s next to this script" % ARCHIVE_GLOB)
        return None
    arc = cands[0]
    digest = sha256_file(arc)
    info = {"archive": os.path.basename(arc), "size": os.path.getsize(arc), "sha256": digest}
    dest = os.path.join(here, "tfhub_cache", HANDLE_DIR)
    marker = os.path.join(dest, ".cafe_source.json")
    if os.path.isfile(marker):
        try:
            if json.load(open(marker)).get("sha256") == digest:
                rep.add("OK", "i3d", "I3D module", "in place from %s (%d B, sha256 %s)" % (info["archive"], info["size"],
                                                                                     digest[:12]), **info)
                return info
        except Exception:
            pass
    try:
        with tarfile.open(arc, "r:gz") as tf_:
            members = tf_.getmembers()
            bad = [m.name for m in members if m.name.startswith("/") or ".." in m.name.split("/")
                   or m.issym() or m.islnk() or not (m.isfile() or m.isdir())]
            if bad:
                rep.add("FAIL", "i3d", "I3D archive", "unsafe members: %s" % ", ".join(bad[:3]))
                return None
            names = [m.name[2:] if m.name.startswith("./") else m.name for m in members]
            roots = set(n.split("/")[0] for n in names if n and n != ".")
            prefix = ""
            if "saved_model.pb" not in names and len(roots) == 1:
                prefix = list(roots)[0] + "/"
            need = [prefix + "saved_model.pb", prefix + "tfhub_module.pb"]
            if not all(n in names for n in need) or not any(n.startswith(prefix + "variables/") for n in names):
                rep.add("FAIL", "i3d", "I3D archive", "not a TF-Hub TF1 module (needs saved_model.pb, tfhub_module.pb "
                        "and variables/); top level: %s" % ", ".join(sorted(roots)[:6]))
                return None
            tmp = os.path.join(here, "tfhub_cache", HANDLE_DIR + ".tmp_%d" % os.getpid())
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            for m, n in zip(members, names):
                if not n or n == "." or not n.startswith(prefix):
                    continue
                rel = n[len(prefix):]
                if not rel:
                    continue
                target = os.path.join(tmp, rel)
                if m.isdir():
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with tf_.extractfile(m) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out, 1 << 20)
    except Exception as e:
        rep.add("FAIL", "i3d", "I3D archive", "cannot read %s: %s" % (info["archive"], err(e)))
        return None
    with open(os.path.join(tmp, ".cafe_source.json"), "w") as fh:
        json.dump(info, fh)
    if os.path.exists(dest):
        kept = dest + ".replaced_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.replace(dest, kept)
    os.replace(tmp, dest)
    n_files = sum(len(f) for _, _, f in os.walk(dest)) - 1
    rep.add("OK", "i3d", "I3D module", "extracted %s (%d B, sha256 %s) to tfhub_cache/%s (%d files)"
            % (info["archive"], info["size"], digest[:12], HANDLE_DIR, n_files), **info)
    return info


def ensure_venv(rep, here, logdir):
    venv = os.path.join(here, "envs", "fvd")
    py = os.path.join(venv, "bin", "python")
    marker = os.path.join(venv, ".cafe_pins.json")
    if os.path.isfile(py) and os.path.isfile(marker):
        try:
            if json.load(open(marker)) == PINS:
                rep.add("OK", "env", "FVD environment", "in place (%s)" % os.path.relpath(venv, here))
                return py
        except Exception:
            pass
    log = os.path.join(logdir, "pip.log")
    t0 = time.time()
    with open(log, "ab") as fh:
        if not os.path.isfile(py):
            r = subprocess.run([sys.executable, "-m", "venv", venv], stdout=fh, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, timeout=600)
            if r.returncode != 0 or not os.path.isfile(py):
                rep.add("FAIL", "env", "FVD environment", "python -m venv failed; see %s" % os.path.relpath(log, here))
                return None
        # pip 24.2 resolves from PyPI metadata files instead of downloading whole wheels
        subprocess.run([py, "-m", "pip", "install", "--upgrade", "pip==24.2"], stdout=fh,
                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=900)
        wheelhouse = os.path.join(here, "envs", "wheelhouse")
        os.makedirs(wheelhouse, exist_ok=True)
        done = False
        for attempt in range(1, DL_ATTEMPTS + 1):
            n_before = len(os.listdir(wheelhouse))
            t1 = time.time()
            fh.write(("\n=== download attempt %d/%d (%d files already in the wheelhouse)\n"
                      % (attempt, DL_ATTEMPTS, n_before)).encode())
            fh.flush()
            try:
                r = subprocess.run([py, "-m", "pip", "download", "--dest", wheelhouse, "--timeout", "60",
                                    "--retries", "10", "--progress-bar", "off"] + PINS, stdout=fh,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=DL_TIMEOUT)
                outcome = "rc=%d" % r.returncode
                done = r.returncode == 0
            except subprocess.TimeoutExpired:
                outcome = "stopped after %d s" % DL_TIMEOUT
            n_after = len(os.listdir(wheelhouse))
            size = sum(os.path.getsize(os.path.join(wheelhouse, f)) for f in os.listdir(wheelhouse))
            rep.add("OK" if done else "INFO", "env", "download attempt %d" % attempt,
                    "%s in %.0f s; wheelhouse %d files (%+d), %.2f GB" % (outcome, time.time() - t1, n_after,
                                                                          n_after - n_before, size / 1e9))
            if done:
                break
        if not done:
            rep.add("FAIL", "env", "FVD environment", "download incomplete after %d attempts; rerun the same command, "
                    "finished files are kept in envs/wheelhouse; see %s" % (DL_ATTEMPTS, os.path.relpath(log, here)))
            return None
        r = subprocess.run([py, "-m", "pip", "install", "--no-index", "--find-links", wheelhouse] + PINS, stdout=fh,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=3600)
    if r.returncode != 0:
        tailtxt = open(log, errors="replace").read().strip().splitlines()[-1:] or [""]
        rep.add("FAIL", "env", "FVD environment", "pip install failed (%s); see %s" % (tailtxt[0][:120],
                                                                                     os.path.relpath(log, here)))
        return None
    with open(marker, "w") as fh:
        json.dump(PINS, fh)
    rep.add("OK", "env", "FVD environment", "created %s with %s in %.0f s" % (os.path.relpath(venv, here),
                                                                          ", ".join(PINS), time.time() - t0))
    return py


def run_probe(rep, here, py, logdir):
    path = os.path.join(logdir, "probe.py")
    with open(path, "w") as fh:
        fh.write(PROBE)
    res = subprocess.run([py, path, HANDLE], env=fvd_env(here), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, timeout=1800)
    text = res.stdout.decode("utf-8", "replace")
    open(os.path.join(logdir, "probe.log"), "w").write(text)
    lines = [ln for ln in text.splitlines() if ln.startswith("PROBE_JSON ")]
    if not lines:
        rep.add("FAIL", "probe", "FVD environment probe", "no result (rc=%d); see probe.log" % res.returncode)
        return None
    out = json.loads(lines[-1][len("PROBE_JSON "):])
    if "versions" in out:
        rep.add("INFO", "probe", "versions", ", ".join("%s %s" % kv for kv in out["versions"].items()))
    if "error" in out:
        rep.add("FAIL", "probe", "FVD environment probe", out["error"])
        return None
    expect = os.path.join(here, "tfhub_cache", HANDLE_DIR)
    if os.path.realpath(out.get("resolved", "")) == os.path.realpath(expect):
        rep.add("OK", "probe", "offline handle resolution", "%s -> tfhub_cache/%s with network blocked" % (HANDLE, HANDLE_DIR))
    else:
        rep.add("FAIL", "probe", "offline handle resolution", "resolved to %s, expected %s" % (out.get("resolved"), expect))
    if out.get("embedding_shape") == [16, 400] and out.get("embedding_finite"):
        rep.add("OK", "probe", "I3D embedding", "RGB/inception_i3d/Mean:0 gives [16, 400], finite (%d variables)"
                % out.get("n_variables", -1))
    else:
        rep.add("FAIL", "probe", "I3D embedding", "shape %s, finite %s" % (out.get("embedding_shape"), out.get("embedding_finite")))
    if out.get("gpus"):
        rep.add("OK", "probe", "TensorFlow GPU", ", ".join(out["gpus"]))
    else:
        rep.add("WARN", "probe", "TensorFlow GPU", "no GPU visible to TensorFlow; FVD would run on CPU (slower, same method)")
    return out


def run_fvd(py, here, repo, target_dir, log, timeout):
    """python fvd.py <target_dir> from evaluation/, own process group, stdin closed."""
    with open(log, "ab") as fh:
        proc = subprocess.Popen([py, "fvd.py", target_dir], cwd=os.path.join(repo, "evaluation"), env=fvd_env(here),
                                stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as e:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                end = time.time() + 10
                while time.time() < end and proc.poll() is None:
                    time.sleep(0.2)
            proc.wait()
            if isinstance(e, KeyboardInterrupt):
                raise
            return "timeout"


def smoke_test(rep, here, repo, py, logdir):
    import pandas as pd
    src = os.path.join(repo, RUN_CSV)
    if not os.path.isfile(src):
        rep.add("MISSING", "smoke", "fvd.py smoke test", "%s not found" % RUN_CSV)
        return False
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    if any(c not in df.columns for c in COLUMNS) or df.columns[4] != "generated_path" or len(df) == 0:
        rep.add("FAIL", "smoke", "fvd.py smoke test", "ours.csv lacks the standard columns in order")
        return False
    a = df.iloc[0][COLUMNS].to_dict()
    b = dict(a, gt_video_path=a["generated_path"])
    sdir = os.path.join(logdir, "smoke")
    os.makedirs(sdir)
    pd.DataFrame([a, b], columns=COLUMNS).to_csv(os.path.join(sdir, "smoke.csv"), index=False)
    t0 = time.time()
    rc = run_fvd(py, here, repo, sdir, os.path.join(logdir, "smoke_fvd.log"), 3600)
    try:
        out = pd.read_csv(os.path.join(sdir, "smoke.csv"))
        v = [float(x) for x in out["fvd"]]
    except Exception as e:
        rep.add("FAIL", "smoke", "fvd.py smoke test", "rc=%s, no fvd values (%s); see smoke_fvd.log" % (rc, err(e)))
        return False
    ok = rc == 0 and all(x == x for x in v) and v[0] > 0 and abs(v[1]) < 1e-3
    rep.add("OK" if ok else "FAIL", "smoke", "fvd.py smoke test", "rc=%s in %.0f s; real pair %.4f, video vs itself %.2e "
            "(must be about 0)" % (rc, time.time() - t0, v[0], v[1]))
    return ok


# ---------------------------------------------------------------- full run

def prep_running(here):
    try:
        pid = int(open(os.path.join(here, "reports", "fullrun.lock")).read().split()[0])
        return pid if b"cafe_eval_prep" in open("/proc/%d/cmdline" % pid, "rb").read() else None
    except Exception:
        return None


def take_lock(here, tag):
    path = os.path.join(here, "reports", "fvd.lock")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        pid = int(open(path).read().split()[0])
        if b"cafe_fvd_setup" in open("/proc/%d/cmdline" % pid, "rb").read():
            return None, "another FVD run is active (pid %d)" % pid
    except Exception:
        pass
    if os.path.exists(path):
        os.remove(path)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, ("%d %s\n" % (os.getpid(), tag)).encode())
    os.close(fd)
    return path, None


def full_run(rep, here, repo, tag, timeout):
    import numpy as np
    import pandas as pd
    state_path = os.path.join(here, "reports", "fvd_setup_state.json")
    try:
        state = json.load(open(state_path))
    except Exception:
        state = {}
    arcs = sorted(glob.glob(os.path.join(here, ARCHIVE_GLOB)), key=os.path.getmtime, reverse=True)
    if not (state.get("passed") and state.get("pins") == PINS and arcs
            and state.get("archive_sha256") == sha256_file(arcs[0])):
        rep.add("FAIL", "run", "setup passed", "run the setup (without --run) first; it must pass for this archive and pins")
        return
    rep.add("OK", "run", "setup passed", "on %s for %s" % (state.get("stamp"), state.get("archive")))
    run_dir = os.path.join(repo, "evaluation/runs/mead_ours")
    others = sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "*.csv")) if os.path.basename(p) != "ours.csv")
    if others:
        rep.add("FAIL", "run", "runs/mead_ours is clean", "fvd.py processes every CSV there; move out: " + ", ".join(others))
        return
    csvp = os.path.join(repo, RUN_CSV)
    before = pd.read_csv(csvp)
    backup = os.path.join(here, "reports", "ours_before_fvd_%s.csv" % tag)
    shutil.copy2(csvp, backup)
    done_before = int(pd.to_numeric(before["fvd"], errors="coerce").notna().sum()) if "fvd" in before.columns else 0
    rep.add("OK", "run", "backup", "%s (%d rows; %d already have fvd, fvd.py skips those)"
            % (os.path.relpath(backup, here), len(before), done_before))
    logdir = os.path.join(here, "reports", "fvd_run_" + tag)
    os.makedirs(logdir, exist_ok=True)
    py = os.path.join(here, "envs", "fvd", "bin", "python")
    print("... fvd.py runs/mead_ours (%s); log %s" % (datetime.datetime.now().strftime("%H:%M:%S"),
                                                     os.path.relpath(os.path.join(logdir, "fvd.log"), here)), flush=True)
    t0 = time.time()
    rc = run_fvd(py, here, repo, "runs/mead_ours", os.path.join(logdir, "fvd.log"), timeout)
    rep.add("OK" if rc == 0 else "FAIL", "run", "fvd.py", "rc=%s in %.1f min" % (rc, (time.time() - t0) / 60.0))
    after = pd.read_csv(csvp)
    if len(after) != len(before) or "fvd" not in after.columns:
        rep.add("FAIL", "check", "ours.csv after fvd.py", "rows %d -> %d, fvd column %s" % (len(before), len(after),
                "present" if "fvd" in after.columns else "missing"))
        return
    changed = []
    for c in before.columns:
        if c == "fvd":
            continue
        x, y = before[c], after[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            same = np.allclose(x.to_numpy(float), y.to_numpy(float), rtol=0, atol=0, equal_nan=True)
        else:
            same = x.fillna("<NA>").astype(str).equals(y.fillna("<NA>").astype(str))
        if not same:
            changed.append(c)
    if changed:
        rep.add("FAIL", "check", "other columns unchanged", "changed by fvd.py: " + ", ".join(changed))
    else:
        rep.add("OK", "check", "other columns unchanged", "all %d columns identical to the backup" % (len(before.columns) - ("fvd" in before.columns)))
    v = pd.to_numeric(after["fvd"], errors="coerce")
    fin = v[np.isfinite(v)]
    miss = [int(i) for i in v.index[~np.isfinite(v.fillna(np.nan))]]
    if miss:
        rep.add("FAIL", "check", "fvd column", "%d/%d rows without a finite fvd: rows %s" % (len(miss), len(v), miss[:8]))
    else:
        rep.add("OK", "check", "fvd column", "%d/%d rows" % (len(v), len(v)))
    neg = int((fin < 0).sum())
    if neg:
        rep.add("WARN", "check", "fvd sign", "%d rows below 0 (min %.3e)" % (neg, fin.min()))
    if len(fin):
        m = float(fin.mean())
        dev = (m - PAPER_FVD) / PAPER_FVD
        cov = "n=%d/%d" % (len(fin), len(v)) + ("" if len(fin) == len(v) else ", INCOMPLETE coverage")
        rep.add("INFO", "gate2", "FVD", "%.4f vs paper %.4f: %+.1f%%, %s the 10%% tolerance (%s)"
                % (m, PAPER_FVD, 100 * dev, "within" if abs(dev) <= TOLERANCE else "OUTSIDE", cov))
    try:
        stats = [ln.strip() for ln in open(os.path.join(logdir, "fvd.log"), errors="replace").read().replace("\r", "\n")
                 .splitlines() if ln.strip().startswith("Total:")]
    except Exception:
        stats = []
    if stats and "Success: 0 " in stats[-1] + " " and len(fin):
        rep.add("INFO", "check", "fvd.py console counter", "fvd.py printed '%s': its counter only accepts Python "
                "floats and TensorFlow returns numpy float32, so it reports 0 although %d values were written; "
                "the CSV values above are what count" % (stats[-1], len(fin)))
    q = subprocess.run([sys.executable, "check_quantitative_all.py", "--csv_path", "runs/mead_ours/ours.csv"],
                       cwd=os.path.join(repo, "evaluation"), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=600)
    lines = [ln.strip() for ln in q.stdout.decode("utf-8", "replace").splitlines()
             if ln.split(":")[0] in ("FID", "fvd", "Sync_conf", "Accemo")]
    lines = [ln.split(":")[0] + ": missing" if "(missing" in ln else ln for ln in lines]
    rep.add("INFO", "check", "check_quantitative_all.py", " | ".join(lines) or "no output (rc=%d)" % q.returncode)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true", help="run fvd.py on evaluation/runs/mead_ours/ours.csv")
    ap.add_argument("--timeout", type=int, default=86400, help="seconds for fvd.py in --run")
    args = ap.parse_args()
    repo, here = os.getcwd(), os.path.dirname(os.path.abspath(__file__))
    tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not (os.path.isfile(os.path.join(repo, "inference.py")) and os.path.isfile(os.path.join(repo, "evaluation/fvd.py"))):
        print("Run this from the C-MET repository root (cd /workspace/sir/CAFE/C-MET).")
        return 2
    if hashlib.sha1(HANDLE.encode("utf8")).hexdigest() != HANDLE_DIR:
        print("internal error: handle hash mismatch")
        return 2
    if sys.version_info[:2] != (3, 9):
        print("Run this with the C_MET interpreter (Python 3.9); this is Python %d.%d. Activate C_MET first."
              % sys.version_info[:2])
        return 2
    if prep_running(here):
        print("A cafe_eval_prep run is active; wait for it to finish.")
        return 2
    print("%s | %s | host %s | repo %s | mode %s" % (SCRIPT, tag, socket.gethostname(), repo, "run" if args.run else "setup"),
          flush=True)
    rep = Report()

    def on_term(signum, frame):
        raise KeyboardInterrupt("signal %d" % signum)
    signal.signal(signal.SIGTERM, on_term)
    lock = None
    try:
        lock, e = take_lock(here, tag)  # one setup or run at a time (both write envs/, tfhub_cache/, ours.csv)
        if e:
            print(e)
            return 2
        if args.run:
            full_run(rep, here, repo, tag, args.timeout)
        else:
            logdir = os.path.join(here, "reports", "fvd_setup_" + tag)
            os.makedirs(logdir, exist_ok=True)
            state_path = os.path.join(here, "reports", "fvd_setup_state.json")
            if os.path.exists(state_path):  # a setup that dies halfway must not leave an old "passed" behind
                with open(state_path, "w") as fh:
                    json.dump({"passed": False, "stamp": tag, "note": "setup in progress or interrupted"}, fh)
            info = install_module(rep, here)
            py = ensure_venv(rep, here, logdir) if info else None
            probe = run_probe(rep, here, py, logdir) if py else None
            ok = bool(probe) and not rep.blockers() and smoke_test(rep, here, repo, py, logdir)
            state = {"passed": bool(ok and not rep.blockers()), "stamp": tag, "pins": PINS,
                     "archive": info and info["archive"], "archive_sha256": info and info["sha256"],
                     "versions": probe and probe.get("versions"), "gpus": probe and probe.get("gpus")}
            with open(state_path, "w") as fh:
                json.dump(state, fh, indent=1)
    except KeyboardInterrupt as e:
        print("STOPPED (%s): any running child was stopped; for --run, ours.csv keeps the rows saved so far and the "
              "backup is in reports/" % e)
        return 130
    except subprocess.TimeoutExpired as e:
        print("FAIL: %s timed out after %s s (network or PyPI stall); rerun the same command" % (e.cmd[:4], e.timeout))
        return 1
    finally:
        if lock and os.path.exists(lock):
            try:
                if int(open(lock).read().split()[0]) == os.getpid():
                    os.remove(lock)
            except Exception:
                pass
    counts = {}
    for r in rep.rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("SUMMARY " + " ".join("%s=%d" % (k.lower(), counts.get(k, 0)) for k in ("OK", "WARN", "FAIL", "MISSING", "INFO")))
    print("BLOCKERS: " + (", ".join(rep.blockers()) or "none"))
    path = os.path.join(here, "reports", "fvd_%s_%s.json" % ("run" if args.run else "setup", tag))
    with open(path, "w") as fh:
        json.dump({"script": SCRIPT, "stamp": tag, "rows": rep.rows}, fh, indent=1, default=str)
    print("REPORT: " + path)
    return 1 if rep.blockers() else 0


if __name__ == "__main__":
    sys.exit(main())
