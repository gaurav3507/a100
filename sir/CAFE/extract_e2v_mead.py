#!/usr/bin/env python3
"""
CAFE: per-clip emotion2vec+large features for the MEAD FPS25 tree.

These are the features C-MET's evaluation protocol reads
(src/dataset_emo12.py, init_e2v_MEAD_paths):
  {FPS25}/{ID}/front/{emotion}/{level}/emotion2vec+large_features/{NNN}.npy

Same model and call as C-MET's extract_e2v+L.py:
  funasr AutoModel(emotion2vec_plus_large).generate(wav, output_dir=...,
  granularity="utterance")  -> {NNN}.npy (1024-d utterance embedding)
Differences, all deliberate:
  - model loads from a LOCAL folder with HF_HUB_OFFLINE=1 (the original
    downloads from ModelScope, which is unreliable on this network); the HF
    model card states the weights are identical to the ModelScope release;
  - no data_root string trap (the original silently does nothing unless the
    argument is exactly './dataset/MEAD/FPS25');
  - validation (shape, finite), atomic writes, resume, failure log with reasons.

Fidelity mode compares our features against the authors' released subset in
audios/MEAD/{emotion}/emotion2vec+large_features/{ID}_{level}_{NNN}.npy and
reports cosine statistics over every overlapping clip.

Run from the C-MET repo root (C_MET env):
  python ../extract_e2v_mead.py --fps25_root dataset/MEAD/FPS25 \
      --model_dir /workspace/sir/CAFE/models/emotion2vec_plus_large --limit 5
  python ../extract_e2v_mead.py --fps25_root dataset/MEAD/FPS25 \
      --model_dir /workspace/sir/CAFE/models/emotion2vec_plus_large
  python ../extract_e2v_mead.py --fps25_root dataset/MEAD/FPS25 --fidelity audios/MEAD
  python ../extract_e2v_mead.py --selftest
"""

import argparse, glob, os, shutil, sys, tempfile, time

FEAT_DIR = "emotion2vec+large_features"
DIM = 1024


def discover_wavs(fps25_root, angle="front"):
    pat = os.path.join(fps25_root, "*", angle, "*", "*", "*.wav")
    return sorted(w for w in glob.glob(pat) if ".tmp" not in os.path.basename(w))

def out_path(wav):
    d, name = os.path.split(wav)
    return os.path.join(d, FEAT_DIR, name[:-4] + ".npy")


class RealBackend:
    def __init__(self, model_dir):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from funasr import AutoModel
        import numpy as np
        self.np = np
        self.m = AutoModel(model=model_dir, disable_update=True)
        self.tmp = tempfile.mkdtemp(prefix="e2v_")

    def extract(self, wav):
        """Run the C-MET call into a private temp dir; return the saved array."""
        for f in glob.glob(os.path.join(self.tmp, "*")):
            os.remove(f)
        self.m.generate(wav, output_dir=self.tmp, granularity="utterance")
        files = glob.glob(os.path.join(self.tmp, "*.npy"))
        if len(files) != 1:
            raise RuntimeError(f"expected 1 npy from funasr, got {len(files)}")
        return self.np.load(files[0])

    def save(self, path, arr):
        self.np.save(path, arr)

    def check(self, arr):
        if getattr(arr, "shape", None) != (DIM,):
            return f"bad_shape_{getattr(arr, 'shape', None)}"
        if not bool(self.np.isfinite(arr).all()):
            return "non_finite"
        return None


def is_done(wav, backend=None):
    p = out_path(wav)
    if not os.path.isfile(p) or os.path.getsize(p) == 0:
        return False
    return True

def process_one(wav, backend):
    arr = backend.extract(wav)
    bad = backend.check(arr)
    if bad:
        return False, bad
    dst = out_path(wav)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp.npy"
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        backend.save(tmp, arr)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return True, "ok"

def run(fps25_root, backend, limit=None, log=print):
    wavs = discover_wavs(fps25_root)
    pending = [w for w in wavs if not is_done(w, backend)]
    already = len(wavs) - len(pending)
    if limit is not None:
        pending = pending[:limit]
    ok_n, failures = 0, []
    for i, w in enumerate(pending, 1):
        try:
            ok, reason = process_one(w, backend)
        except Exception as e:
            ok, reason = False, f"exception_{type(e).__name__}: {e}"
        if ok:
            ok_n += 1
        else:
            failures.append((os.path.relpath(w, fps25_root), reason))
        if i % 100 == 0 or i == len(pending):
            log(f"[e2v] {i}/{len(pending)}  ok={ok_n} fail={len(failures)}")
    meta = os.path.dirname(fps25_root.rstrip("/")) or "."
    fail_path = os.path.join(meta, "e2v_failures.txt")
    if failures:
        with open(fail_path, "w") as f:
            for rel, r in failures:
                f.write(f"{rel}\t{r}\n")
    return {"wavs": len(wavs), "already_done": already, "processed": len(pending),
            "ok": ok_n, "failed": len(failures),
            "failures_file": fail_path if failures else None}


def fidelity(fps25_root, audios_root, np):
    """Cosine between our features and the authors' released subset."""
    pat = os.path.join(audios_root, "*", FEAT_DIR, "*.npy")
    rows, missing = [], []
    for a_path in sorted(glob.glob(pat)):
        emotion = a_path.split(os.sep)[-3]
        stem = os.path.basename(a_path)[:-4]            # {ID}_{level}_{NNN}
        parts = stem.split("_")
        if len(parts) != 4:
            continue
        ident, level, num = parts[0], f"{parts[1]}_{parts[2]}", parts[3]
        ours = os.path.join(fps25_root, ident, "front", emotion, level, FEAT_DIR, num + ".npy")
        if not os.path.isfile(ours):
            missing.append(stem + "/" + emotion)
            continue
        a = np.load(a_path).astype("float64").reshape(-1)
        b = np.load(ours).astype("float64").reshape(-1)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        rows.append((cos, f"{emotion}/{stem}"))
    if not rows:
        return {"compared": 0, "no_ours_for": len(missing)}
    c = sorted(r[0] for r in rows)
    n = len(c)
    return {"compared": n, "no_ours_for": len(missing),
            "min": round(c[0], 4), "median": round(c[n // 2], 4),
            "mean": round(sum(c) / n, 4), "below_0.98": sum(1 for x in c if x < 0.98),
            "worst3": [(round(x, 4), s) for x, s in sorted(rows)[:3]]}


# ---- self-test (mock backend; no funasr/torch) ---------------------------

class _A:
    def __init__(self, shape, finite=True):
        self.shape, self.finite = shape, finite

class MockBackend:
    def __init__(self, bad_shape=(), nan=(), raise_on=()):
        self.bad_shape, self.nan, self.raise_on = set(bad_shape), set(nan), set(raise_on)
        self.calls = []
    def extract(self, wav):
        self.calls.append(wav)
        if wav in self.raise_on:
            raise RuntimeError("expected 1 npy from funasr, got 0")
        return _A((768,) if wav in self.bad_shape else (DIM,), wav not in self.nan)
    def save(self, path, arr):
        with open(path, "w") as f:
            f.write(str(arr.shape))
    def check(self, arr):
        if arr.shape != (DIM,):
            return f"bad_shape_{arr.shape}"
        return None if arr.finite else "non_finite"

def selftest():
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "dataset", "MEAD", "FPS25")
        wavs = []
        for ident in ("M003", "W009"):
            for emo in ("angry", "neutral"):
                for n in (1, 2, 3):
                    w = os.path.join(root, ident, "front", emo, "level_1", f"{n:03d}.wav")
                    os.makedirs(os.path.dirname(w), exist_ok=True)
                    open(w, "w").close()
                    wavs.append(w)
        open(os.path.join(root, "M003/front/angry/level_1/004.tmpwav.wav"), "w").close()
        # T1 discovery excludes temp wavs
        assert discover_wavs(root) == sorted(wavs)
        # T2 output path is exactly C-MET's layout
        assert out_path(wavs[0]).endswith("M003/front/angry/level_1/" + FEAT_DIR + "/001.npy")
        bad, nan, boom = wavs[1], wavs[2], wavs[3]
        b = MockBackend(bad_shape=[bad], nan=[nan], raise_on=[boom])
        s = run(root, b, log=lambda *_: None)
        assert s["processed"] == 12 and s["ok"] == 9 and s["failed"] == 3, s
        # T3 failures produce no output; successes do; no temp files anywhere
        for w in wavs:
            assert os.path.isfile(out_path(w)) == (w not in (bad, nan, boom)), w
        assert not glob.glob(os.path.join(root, "**", "*.tmp.npy"), recursive=True)
        reasons = dict(l.rstrip("\n").split("\t") for l in open(s["failures_file"]))
        rel = lambda w: os.path.relpath(w, root)
        assert reasons[rel(bad)] == "bad_shape_(768,)"
        assert reasons[rel(nan)] == "non_finite"
        assert reasons[rel(boom)].startswith("exception_RuntimeError")
        # T4 resume retries exactly the failures
        b2 = MockBackend()
        s2 = run(root, b2, log=lambda *_: None)
        assert sorted(b2.calls) == sorted([bad, nan, boom]) and s2["already_done"] == 9, s2
        # T5 limit
        for w in wavs:
            os.remove(out_path(w))
        s3 = run(root, MockBackend(), limit=4, log=lambda *_: None)
        assert s3["processed"] == 4 and s3["ok"] == 4
    # T6 fidelity mapping + stats with real numpy if available
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        with tempfile.TemporaryDirectory() as td:
            fps = os.path.join(td, "FPS25"); aud = os.path.join(td, "audios")
            rng = np.random.default_rng(0)
            v1, v2 = rng.normal(size=DIM), rng.normal(size=DIM)
            def put(p, arr):
                os.makedirs(os.path.dirname(p), exist_ok=True); np.save(p, arr)
            put(os.path.join(aud, "angry", FEAT_DIR, "M003_level_3_005.npy"), v1)
            put(os.path.join(fps, "M003", "front", "angry", "level_3", FEAT_DIR, "005.npy"), v1 * 2.0)
            put(os.path.join(aud, "sad", FEAT_DIR, "W009_level_1_012.npy"), v2)
            put(os.path.join(fps, "W009", "front", "sad", "level_1", FEAT_DIR, "012.npy"), -v2)
            put(os.path.join(aud, "happy", FEAT_DIR, "W015_level_2_007.npy"), v1)
            r = fidelity(fps, aud, np)
            assert r["compared"] == 2 and r["no_ours_for"] == 1, r
            assert r["min"] == -1.0 and r["below_0.98"] == 1, r
            assert r["worst3"][0][1] == "sad/W009_level_1_012", r
    print("selftest OK: temp-excluded discovery, C-MET output layout, shape/NaN validation, "
          "specific failure reasons, atomic writes, resume retries failures, --limit, "
          "fidelity mapping (ID_level_NNN -> tree) + cosine stats")
    return True


def main():
    ap = argparse.ArgumentParser(description="emotion2vec+large features for MEAD")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fps25_root")
    ap.add_argument("--model_dir")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fidelity", metavar="AUDIOS_MEAD", help="compare with authors' audios/MEAD")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.fps25_root:
        ap.print_help(); sys.exit(1)
    if args.fidelity:
        import numpy as np
        print("[fidelity]", fidelity(args.fps25_root, args.fidelity, np))
        return
    if not args.model_dir or not os.path.isfile(os.path.join(args.model_dir, "model.pt")):
        print("--model_dir must contain model.pt"); sys.exit(2)
    t0 = time.time()
    backend = RealBackend(args.model_dir)
    try:
        s = run(args.fps25_root, backend, limit=args.limit)
    finally:
        shutil.rmtree(backend.tmp, ignore_errors=True)
    print(f"[e2v] summary: {s}")
    print(f"[done] {time.time() - t0:.1f}s")
    sys.exit(0 if s["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
