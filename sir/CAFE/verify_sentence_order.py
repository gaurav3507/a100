#!/usr/bin/env python3
"""
CAFE Phase 1 - MEAD sentence-order verification and remap (Gate 1 core).

Why this exists
---------------
C-MET pairs a NEUTRAL source utterance with an EMOTIONAL target of the SAME
sentence. Two facts from the official C-MET repo make this fragile:

  1. Common sentences use files 001-003 in every emotion (including neutral),
     paired by identical number.
  2. Generic sentences use files 021-030 in each emotion but 031-040 in
     neutral, so generic sentence k pairs emotion file (020+k) with neutral
     file (030+k), NOT the same number.
  3. The official MEAD Part0 release's actual generic file order does not
     always match the appendix order C-MET assumes, so pairing by the
     intended slot can pair DIFFERENT sentences across emotions. The C-MET
     authors fixed this by listening and renumbering.

This tool reconstructs the correct pairing from transcripts (ground truth of
what is actually spoken), independent of the file numbers. It does NOT
fabricate sentence text. Transcripts come from ASR (whisper, which is already
in C-MET's requirements) or from a transcripts file you supply. A canonical
appendix list is OPTIONAL and only used to align our canonical indices to
C-MET's absolute numbering; correct pairing does not require it.

Subcommands
-----------
  transcribe   run whisper ASR over the MEAD tree -> transcripts.csv
               (runs on the GPU node; not exercised by --selftest)
  verify       transcripts.csv (+ optional canonical.csv) -> remap.json + report
  --selftest   CPU-only self-check of the clustering/pairing/detection logic

remap.json is the contract the manifest builder consumes: it maps every used
MEAD file to a per-identity canonical_sentence_id, so pairing is done by
sentence identity, never by raw file number.
"""

import argparse, csv, difflib, glob, json, os, re, sys
from collections import defaultdict

USED_EMOTIONS = ["angry", "contempt", "disgusted", "fear",
                 "happy", "neutral", "sad", "surprised"]

# ---------- text handling ----------

def normalize(text):
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

# ---------- path parsing / sentence classing ----------

def parse_relpath(relpath):
    """
    Expect <ID>/front/<emotion>/<level_n>/<num>.<ext> (mp4 or wav).
    Returns dict or None if it does not match the used layout.
    """
    parts = relpath.replace("\\", "/").split("/")
    if len(parts) < 5 or parts[1] != "front":
        return None
    ident, _, emotion, level, fname = parts[-5], parts[-4], parts[-3], parts[-2], parts[-1]
    stem, _, ext = fname.rpartition(".")
    if not stem.isdigit():
        return None
    return {"id": ident, "emotion": emotion, "level": level,
            "num": int(stem), "ext": ext.lower(), "relpath": relpath}

def classify(emotion, num):
    """
    Map a used file to (sentence_class, intended_slot) per C-MET numbering.
      common : num 1..3            -> ('common', num)
      generic: emotion 21..30      -> ('generic', num-20)
               neutral 31..40      -> ('generic', num-30)
    Returns None for files C-MET does not use (e.g., 004-020).
    """
    if 1 <= num <= 3:
        return ("common", num)
    if emotion == "neutral":
        if 31 <= num <= 40:
            return ("generic", num - 30)
    else:
        if 21 <= num <= 30:
            return ("generic", num - 20)
    return None

# ---------- transcripts I/O ----------

def load_transcripts(path):
    """CSV with columns: relpath, transcript. Returns {relpath: transcript}."""
    out = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        need = {"relpath", "transcript"}
        if not need.issubset({c.strip() for c in r.fieldnames or []}):
            raise ValueError(f"transcripts csv must have columns {need}, got {r.fieldnames}")
        for row in r:
            out[row["relpath"].strip()] = row["transcript"]
    return out

def load_canonical(path):
    """
    Optional CSV: sentence_class(common|generic), canonical_index(int), sentence_text.
    You paste these from the MEAD paper appendix. The tool validates shape only
    (3 common, 10 generic) and never invents text.
    """
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({"sentence_class": row["sentence_class"].strip(),
                         "canonical_index": int(row["canonical_index"]),
                         "sentence_text": normalize(row["sentence_text"])})
    n_common = sum(1 for x in rows if x["sentence_class"] == "common")
    n_generic = sum(1 for x in rows if x["sentence_class"] == "generic")
    problems = []
    if n_common != 3:
        problems.append(f"expected 3 common sentences, got {n_common}")
    if n_generic != 10:
        problems.append(f"expected 10 generic sentences, got {n_generic}")
    if problems:
        raise ValueError("canonical list malformed: " + "; ".join(problems))
    return rows

# ---------- core: cluster by sentence, build remap ----------

def cluster_by_sentence(files, transcripts, threshold):
    """
    Greedy clustering of files that share a spoken sentence, within one
    (id, sentence_class). files: list of parsed dicts. Returns list of clusters;
    each cluster is a list of parsed dicts.
    """
    clusters = []  # each: {"rep": normalized_text, "members": [parsed,...]}
    for fp in files:
        txt = normalize(transcripts.get(fp["relpath"], ""))
        placed = False
        best_i, best_s = -1, 0.0
        for i, c in enumerate(clusters):
            s = similarity(txt, c["rep"])
            if s > best_s:
                best_s, best_i = s, i
        if best_i >= 0 and best_s >= threshold:
            clusters[best_i]["members"].append(fp)
            placed = True
        if not placed:
            clusters.append({"rep": txt, "members": [fp]})
    return clusters

def assign_canonical(cluster_rep, sentence_class, canonical, threshold):
    """Match a cluster's representative text to the canonical appendix list."""
    if not canonical:
        return None, None
    best_idx, best_s, best_txt = None, 0.0, None
    for row in canonical:
        if row["sentence_class"] != sentence_class:
            continue
        s = similarity(cluster_rep, row["sentence_text"])
        if s > best_s:
            best_s, best_idx, best_txt = s, row["canonical_index"], row["sentence_text"]
    if best_s >= threshold:
        return best_idx, best_s
    return None, best_s

def build_remap(parsed_files, transcripts, canonical, threshold):
    """
    Returns (entries, inconsistencies, low_conf, cluster_issues).
    entries: {relpath: {id,emotion,intensity,num,sentence_class,intended_slot,
                        canonical_sentence_id, provisional(bool), score}}
    """
    # group by (id, sentence_class)
    by_group = defaultdict(list)
    for fp in parsed_files:
        cls = classify(fp["emotion"], fp["num"])
        if cls is None:
            continue
        fp["sentence_class"], fp["intended_slot"] = cls
        by_group[(fp["id"], fp["sentence_class"])].append(fp)

    entries = {}
    inconsistencies = []
    low_conf = []
    cluster_issues = []

    for (ident, cls), files in sorted(by_group.items()):
        clusters = cluster_by_sentence(files, transcripts, threshold)
        # order clusters deterministically for provisional numbering:
        # by the median intended_slot of their members
        def med_slot(c):
            slots = sorted(m["intended_slot"] for m in c["members"])
            return slots[len(slots) // 2]
        clusters.sort(key=med_slot)

        for prov_i, c in enumerate(clusters, start=1):
            canon_idx, canon_score = assign_canonical(c["rep"], cls, canonical, threshold)
            provisional = canon_idx is None
            csid = f"{cls}:{canon_idx}" if canon_idx is not None else f"{cls}:prov{prov_i}"

            # cluster sanity: how many distinct neutral files? should be exactly 1
            neutral_members = [m for m in c["members"] if m["emotion"] == "neutral"]
            if len(neutral_members) == 0:
                cluster_issues.append({"id": ident, "sentence_class": cls,
                    "cluster": csid, "issue": "no neutral source in cluster"})
            elif len({m["num"] for m in neutral_members}) > 1:
                cluster_issues.append({"id": ident, "sentence_class": cls,
                    "cluster": csid, "issue": "multiple distinct neutral files",
                    "nums": sorted({m["num"] for m in neutral_members})})

            # detect intended-slot inconsistency: within this real sentence
            # cluster, do the non-neutral members share one intended slot?
            emo_slots = defaultdict(set)
            for m in c["members"]:
                emo_slots[m["intended_slot"]].add(m["emotion"])
            if cls == "generic" and len(emo_slots) > 1:
                inconsistencies.append({
                    "id": ident, "cluster": csid,
                    "intended_slots_seen": sorted(emo_slots.keys()),
                    "note": "same sentence appears under different intended slots; "
                            "naive slot pairing would mispair"})

            for m in c["members"]:
                score = similarity(normalize(transcripts.get(m["relpath"], "")), c["rep"])
                if score < threshold:
                    low_conf.append({"relpath": m["relpath"], "score": round(score, 3)})
                entries[m["relpath"]] = {
                    "id": ident, "emotion": m["emotion"], "intensity": m["level"],
                    "num": m["num"], "sentence_class": cls,
                    "intended_slot": m["intended_slot"],
                    "canonical_sentence_id": csid,
                    "provisional": provisional,
                    "match_score": round(score, 3),
                }
    return entries, inconsistencies, low_conf, cluster_issues

# ---------- subcommands ----------

def cmd_verify(args):
    transcripts = load_transcripts(args.transcripts)
    canonical = load_canonical(args.canonical) if args.canonical else None

    parsed = []
    for relpath in transcripts:
        p = parse_relpath(relpath)
        if p:
            parsed.append(p)
    if not parsed:
        print("[verify] no files matched the expected <ID>/front/<emo>/<level>/<num> layout")
        sys.exit(2)

    entries, inconsist, low_conf, issues = build_remap(
        parsed, transcripts, canonical, args.threshold)

    remap = {
        "schema": "cafe.phase1.mead_remap/1",
        "canonical_source": args.canonical or "provisional (no appendix supplied)",
        "threshold": args.threshold,
        "n_files": len(entries),
        "entries": entries,
        "inconsistencies": inconsist,
        "low_confidence": low_conf,
        "cluster_issues": issues,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(remap, f, indent=2)

    prov = sum(1 for e in entries.values() if e["provisional"])
    print(f"[verify] files mapped         : {len(entries)}")
    print(f"[verify] provisional (no appx): {prov}")
    print(f"[verify] slot inconsistencies : {len(inconsist)}  (naive pairing would mispair these)")
    print(f"[verify] low-confidence files : {len(low_conf)}  (below threshold {args.threshold})")
    print(f"[verify] cluster issues       : {len(issues)}")
    print(f"[verify] wrote {args.out}")
    if inconsist:
        print("[verify] ACTION: official numbering differs from appendix; "
              "the remap fixes pairing, but review inconsistencies before Gate 1.")
    # non-zero exit if anything needs a human before proceeding
    sys.exit(1 if (low_conf or issues) else 0)

def cmd_transcribe(args):
    """Runs on the GPU node. Kept minimal; not covered by --selftest."""
    try:
        import whisper  # openai-whisper, already in C-MET requirements
    except Exception as e:
        print("[transcribe] whisper not importable in this env:", repr(e))
        print("[transcribe] run this subcommand on the GPU node with C-MET's env.")
        sys.exit(3)
    model = whisper.load_model(args.model)
    wavs = []
    for emo in USED_EMOTIONS:
        wavs += glob.glob(os.path.join(args.data_root, "*", "front", emo, "*", "*.wav"))
    wavs.sort()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["relpath", "transcript"])
        for i, wav in enumerate(wavs, 1):
            rel = os.path.relpath(wav, args.data_root)
            p = parse_relpath(rel.replace(".wav", ".mp4"))
            if p is None or classify(p["emotion"], p["num"]) is None:
                continue  # skip files C-MET does not use
            text = model.transcribe(wav, language="en")["text"]
            w.writerow([rel, text])
            if i % 200 == 0:
                print(f"[transcribe] {i}/{len(wavs)}")
    print(f"[transcribe] wrote {args.out}")

# ---------- self-test ----------

def selftest():
    import tempfile
    # Canonical: 3 common + 10 generic. Use pairwise-distinct sentences so
    # transcript clustering can separate them (real MEAD sentences are distinct;
    # near-identical strings would legitimately fail to cluster).
    common_txt = {
        1: "the quick brown fox jumps over the lazy dog",
        2: "she sells sea shells by the sea shore",
        3: "a stitch in time saves nine in the morning",
    }
    generic_txt = {
        1: "please call stella and ask her to bring these things",
        2: "we went to the market to buy fresh vegetables today",
        3: "the children played happily in the sunny green park",
        4: "my older brother drives a red car to his work",
        5: "the ocean waves crashed against the rocky northern shore",
        6: "she wrote a very long letter to her old friend",
        7: "the tall mountain peak was covered in white snow",
        8: "they cooked a warm dinner together in the small kitchen",
        9: "a gentle breeze moved slowly through the tall pine trees",
        10: "the old wooden clock on the wall stopped ticking",
    }

    transcripts = {}

    def add(ident, emo, level, num, text):
        rel = f"{ident}/front/{emo}/{level}/{num:03d}.mp4"
        transcripts[rel] = text

    ident = "M003"
    emos = [e for e in USED_EMOTIONS if e != "neutral"]
    # common: same number across emotions and neutral
    for k in (1, 2, 3):
        add(ident, "neutral", "level_1", k, common_txt[k])
        for e in emos:
            add(ident, e, "level_1", k, common_txt[k])
    # generic: neutral 31..40 = sentence k; most emotions 21..30 = sentence k
    for k in range(1, 11):
        add(ident, "neutral", "level_1", 30 + k, generic_txt[k])
        for e in emos:
            if e == "happy":
                # INJECT the bug: happy's numbering is shifted by one
                shifted = k % 10 + 1
                add(ident, "happy", "level_1", 20 + k, generic_txt[shifted])
            else:
                add(ident, e, "level_1", 20 + k, generic_txt[k])

    parsed = [parse_relpath(r) for r in transcripts]
    assert all(parsed), "parse failed on a synthetic path"

    # canonical file
    with tempfile.TemporaryDirectory() as td:
        cpath = os.path.join(td, "canon.csv")
        with open(cpath, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["sentence_class", "canonical_index", "sentence_text"])
            for k, t in common_txt.items(): w.writerow(["common", k, t])
            for k, t in generic_txt.items(): w.writerow(["generic", k, t])
        canonical = load_canonical(cpath)

        entries, inconsist, low_conf, issues = build_remap(
            parsed, transcripts, canonical, threshold=0.82)

        # 1) every used file mapped
        assert len(entries) == len(transcripts), (len(entries), len(transcripts))
        # 2) correct pairing recovered: neutral 031 and each non-happy emotion 021
        #    share canonical_sentence_id generic:1
        nid = entries[f"{ident}/front/neutral/level_1/031.mp4"]["canonical_sentence_id"]
        aid = entries[f"{ident}/front/angry/level_1/021.mp4"]["canonical_sentence_id"]
        assert nid == aid == "generic:1", (nid, aid)
        # 3) the injected happy shift is caught as an inconsistency
        assert len(inconsist) >= 1, "failed to detect injected numbering bug"
        # 4) happy 021 actually speaks sentence 2, so it must land in generic:2
        hid = entries[f"{ident}/front/happy/level_1/021.mp4"]["canonical_sentence_id"]
        assert hid == "generic:2", hid
        # 5) canonical (non-provisional) since appendix supplied
        assert entries[f"{ident}/front/angry/level_1/021.mp4"]["provisional"] is False
        # 6) common paired by identical number
        cn = entries[f"{ident}/front/neutral/level_1/001.mp4"]["canonical_sentence_id"]
        cs = entries[f"{ident}/front/sad/level_1/001.mp4"]["canonical_sentence_id"]
        assert cn == cs == "common:1", (cn, cs)

        # 7) provisional path (no appendix) still pairs correctly
        e2, inc2, _, _ = build_remap(parsed, transcripts, None, threshold=0.82)
        n2 = e2[f"{ident}/front/neutral/level_1/031.mp4"]["canonical_sentence_id"]
        a2 = e2[f"{ident}/front/angry/level_1/021.mp4"]["canonical_sentence_id"]
        assert n2 == a2 and n2.startswith("generic:prov"), (n2, a2)
        assert all(e["provisional"] for e in e2.values() if e["sentence_class"] == "generic")

    print("selftest OK: parsing, transcript clustering, correct pairing, "
          "bug detection, canonical + provisional modes all pass")
    return True

# ---------- cli ----------

def main():
    ap = argparse.ArgumentParser(description="MEAD sentence-order verify/remap")
    ap.add_argument("--selftest", action="store_true", help="CPU self-check and exit")
    sub = ap.add_subparsers(dest="cmd")

    v = sub.add_parser("verify", help="build remap.json from transcripts")
    v.add_argument("--transcripts", required=True, help="csv: relpath,transcript")
    v.add_argument("--canonical", default=None, help="optional appendix csv")
    v.add_argument("--threshold", type=float, default=0.82)
    v.add_argument("--out", default="./dataset/MEAD/mead_sentence_remap.json")
    v.set_defaults(func=cmd_verify)

    t = sub.add_parser("transcribe", help="whisper ASR over MEAD (GPU node)")
    t.add_argument("--data_root", required=True, help="./dataset/MEAD/FPS25")
    t.add_argument("--model", default="base.en")
    t.add_argument("--out", default="./dataset/MEAD/transcripts.csv")
    t.set_defaults(func=cmd_transcribe)

    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not getattr(args, "cmd", None):
        ap.print_help()
        sys.exit(1)
    args.func(args)

if __name__ == "__main__":
    main()
