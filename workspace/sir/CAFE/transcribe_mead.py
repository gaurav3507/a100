#!/usr/bin/env python3
"""
CAFE: transcribe every MEAD FPS25 wav with Whisper (for sentence-number alignment).

Why: emotion2vec comparison against the authors' released features showed a
cyclic numbering shift in some level_3 folders of the official MEAD release
(e.g. disgusted/W015 authors 008 -> ours 005). Sentence identity is best
recovered from WHAT is said, so every clip (all 2641, including the
emotion-specific sentences 004-020) is transcribed.

Settings: whisper small.en loaded from the local cache (download is blocked on
this network; the cached file is SHA256-verified by whisper itself),
language='en', temperature=0 (deterministic), fp16 on GPU.

Output: <meta>/transcripts_all.csv with columns relpath,transcript where
relpath = ID/front/emotion/level/NNN.wav. Rows are appended and flushed one by
one, so a re-run resumes; a torn last line from a crash is repaired and that
clip is transcribed again. Failures go to <meta>/transcribe_failures.txt.

Run from the C-MET repo root (C_MET env):
  python ../transcribe_mead.py --fps25_root dataset/MEAD/FPS25 --limit 5
  python ../transcribe_mead.py --fps25_root dataset/MEAD/FPS25
  python ../transcribe_mead.py --selftest
"""

import argparse, csv, glob, os, sys, time

COLS = ["relpath", "transcript"]


def discover(fps25_root, angle="front"):
    pat = os.path.join(fps25_root, "*", angle, "*", "*", "*.wav")
    return sorted(w for w in glob.glob(pat) if ".tmp" not in os.path.basename(w))

def meta_dir(fps25_root):
    return os.path.dirname(fps25_root.rstrip("/")) or "."

def load_done(csv_path):
    """Return {relpath: transcript} from complete rows; ignore torn/malformed rows."""
    done = {}
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0 and row == COLS:
                continue
            if len(row) == 2 and row[0].endswith(".wav"):
                done[row[0]] = row[1]
    return done

def open_for_append(csv_path):
    """Open CSV for appending; write header if new; repair a torn last line."""
    new = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
    if not new:
        with open(csv_path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
        if last != b"\n":
            # drop the torn partial last line so it cannot glue onto the next row
            with open(csv_path, "rb") as f:
                data = f.read()
            cut = data.rfind(b"\n")
            with open(csv_path, "wb") as f:
                f.write(data[:cut + 1] if cut >= 0 else b"")
            new = os.path.getsize(csv_path) == 0
    fh = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(COLS); fh.flush()
    return fh, w


class RealBackend:
    def __init__(self, model_name="small.en"):
        import whisper, torch
        self.fp16 = torch.cuda.is_available()
        self.m = whisper.load_model(model_name)
    def transcribe(self, wav):
        r = self.m.transcribe(wav, language="en", temperature=0.0, fp16=self.fp16)
        return r["text"].strip()


def run(fps25_root, backend, limit=None, log=print):
    meta = meta_dir(fps25_root)
    csv_path = os.path.join(meta, "transcripts_all.csv")
    wavs = discover(fps25_root)
    done = load_done(csv_path)
    pending = [w for w in wavs if os.path.relpath(w, fps25_root) not in done]
    already = len(wavs) - len(pending)
    if limit is not None:
        pending = pending[:limit]
    fh, writer = open_for_append(csv_path)
    ok_n, failures = 0, []
    try:
        for i, w in enumerate(pending, 1):
            rel = os.path.relpath(w, fps25_root)
            try:
                text = (backend.transcribe(w) or "").strip()
                if not text:
                    raise ValueError("empty transcript")
                writer.writerow([rel, text]); fh.flush(); os.fsync(fh.fileno())
                ok_n += 1
            except Exception as e:
                failures.append((rel, f"{type(e).__name__}: {e}"))
            if i % 100 == 0 or i == len(pending):
                log(f"[asr] {i}/{len(pending)}  ok={ok_n} fail={len(failures)}")
    finally:
        fh.close()
    fail_path = os.path.join(meta, "transcribe_failures.txt")
    if failures:
        with open(fail_path, "w") as f:
            for rel, r in failures:
                f.write(f"{rel}\t{r}\n")
    return {"wavs": len(wavs), "already_done": already, "processed": len(pending),
            "ok": ok_n, "failed": len(failures), "csv": csv_path,
            "failures_file": fail_path if failures else None}


# ---- self-test -------------------------------------------------------------

class Mock:
    def __init__(self, fail=(), empty=()):
        self.fail, self.empty, self.calls = set(fail), set(empty), []
    def transcribe(self, wav):
        self.calls.append(wav)
        if wav in self.fail:
            raise RuntimeError("decode error")
        if wav in self.empty:
            return "   "
        # include a comma and a quote to exercise CSV quoting
        return f'Sentence, "{os.path.basename(wav)}"'

def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "dataset", "MEAD", "FPS25")
        wavs = []
        for ident in ("M003", "W015"):
            for emo in ("angry", "neutral"):
                for n in (1, 2, 3):
                    w = os.path.join(root, ident, "front", emo, "level_1", f"{n:03d}.wav")
                    os.makedirs(os.path.dirname(w), exist_ok=True); open(w, "w").close()
                    wavs.append(w)
        open(os.path.join(root, "M003/front/angry/level_1/009.tmpwav.wav"), "w").close()
        assert discover(root) == sorted(wavs)                                   # T1
        s = run(root, Mock(fail=[wavs[1]], empty=[wavs[2]]), log=lambda *_: None)
        assert s["processed"] == 12 and s["ok"] == 10 and s["failed"] == 2, s     # T2
        done = load_done(s["csv"])
        rel = lambda w: os.path.relpath(w, root)
        assert done[rel(wavs[0])] == 'Sentence, "001.wav"'                       # T3 quoting
        assert rel(wavs[1]) not in done and rel(wavs[2]) not in done
        reasons = dict(l.rstrip("\n").split("\t") for l in open(s["failures_file"]))
        assert reasons[rel(wavs[1])].startswith("RuntimeError")
        assert reasons[rel(wavs[2])].startswith("ValueError: empty")
        m2 = Mock()                                                               # T4 resume
        s2 = run(root, m2, log=lambda *_: None)
        assert sorted(m2.calls) == sorted([wavs[1], wavs[2]]) and s2["already_done"] == 10, s2
        # T5 torn last line: simulate crash mid-write, clip is redone, file stays valid
        with open(s["csv"], "a", encoding="utf-8") as f:
            f.write(rel(wavs[5]).replace(".wav", "") )    # partial, no newline
        # also remove wavs[5]'s real row so it counts as pending
        rows = [r for r in csv.reader(open(s["csv"], newline="", encoding="utf-8"))]
        with open(s["csv"], "w", newline="", encoding="utf-8") as f:
            cw = csv.writer(f)
            for r in rows[:-1]:
                if not (len(r) == 2 and r[0] == rel(wavs[5])):
                    cw.writerow(r)
            f.write(rel(wavs[5])[:10])                   # torn tail
        m3 = Mock()
        s3 = run(root, m3, log=lambda *_: None)
        assert m3.calls == [wavs[5]], m3.calls
        allrows = list(csv.reader(open(s["csv"], newline="", encoding="utf-8")))
        assert allrows[0] == COLS and all(len(r) == 2 for r in allrows), allrows[-3:]
        assert len(load_done(s["csv"])) == 12
        # T6 limit
        os.remove(s["csv"])
        s4 = run(root, Mock(), limit=3, log=lambda *_: None)
        assert s4["processed"] == 3 and len(load_done(s4["csv"])) == 3
    print("selftest OK: discovery (tmp excluded), CSV quoting, empty/exception failures with "
          "reasons, resume, torn-last-line repair, --limit")
    return True


def main():
    ap = argparse.ArgumentParser(description="Whisper transcripts for all MEAD FPS25 wavs")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fps25_root")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.fps25_root:
        ap.print_help(); sys.exit(1)
    t0 = time.time()
    s = run(args.fps25_root, RealBackend(args.model), limit=args.limit)
    print(f"[asr] summary: {s}")
    print(f"[done] {time.time() - t0:.1f}s")
    sys.exit(0 if s["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
