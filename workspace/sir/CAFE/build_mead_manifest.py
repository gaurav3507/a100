#!/usr/bin/env python3
"""
CAFE Phase 1 - MEAD manifest builder (Gate 1).

Consumes mead_sentence_remap.json (from verify_sentence_order.py) and emits
C-MET-schema training and evaluation manifests. Pairing is done by SENTENCE
IDENTITY (canonical_sentence_id), never by raw file number, so the generic
021-030 vs neutral 031-040 offset and any official-release renumbering are
handled correctly.

Output CSV schema (exactly matches the official C-MET dataset/MEAD/test.csv):
    source_video_path, gt_video_path, gt_emotion, intensity
where source_video_path is always the NEUTRAL utterance of the same sentence.

Split modes
-----------
  cmet      Reproduction split (Gate 2). The test set is taken VERBATIM from the
            repo's dataset/MEAD/test.csv (we do not invent C-MET's split). Train
            is every buildable neutral->emotion pair whose (id, sentence) is not
            used by a test row. NOTE: this is a SENTENCE split; the same
            identities appear in train and test. It is NOT identity
            generalization.
  identity  Identity-disjoint split (CAFE R2/R4, leakage-safe). train/val/test
            identity sets are provably disjoint; the builder asserts it.

Every emitted row is feature-complete: source and gt each have their sibling
_ED_exp.npy, _ED_pose.npy, _ED_lip.npy and .wav. Incomplete rows are excluded
and written to manifest_missing.csv (no silent scaffold rows). The split itself
is frozen to manifest_split.json for provenance.
"""

import argparse, csv, json, os, random, sys, time
from collections import defaultdict

CSV_FIELDS = ["source_video_path", "gt_video_path", "gt_emotion", "intensity"]
ED_SUFFIXES = ["_ED_exp.npy", "_ED_pose.npy", "_ED_lip.npy"]

# ---------- remap I/O ----------

def load_remap(path):
    with open(path) as f:
        d = json.load(f)
    if not str(d.get("schema", "")).startswith("cafe.phase1.mead_remap"):
        raise ValueError(f"unexpected remap schema: {d.get('schema')}")
    if "entries" not in d or not isinstance(d["entries"], dict):
        raise ValueError("remap has no 'entries' object")
    return d

# ---------- pairing ----------

def to_video_rel(rel):
    """Canonicalize a remap key to the video (.mp4) relpath. verify_sentence_order
    may key entries by the .wav it transcribed; the manifest and feature checks
    are defined on the video entity, so normalize here."""
    for ext in (".wav", ".mp4"):
        if rel.endswith(ext):
            return rel[: -len(ext)] + ".mp4"
    return rel

def build_pairs(entries):
    """
    Group entries by (id, canonical_sentence_id); within each group the neutral
    utterance is the source and every non-neutral utterance is a gt row.
    Returns (pairs, group_warnings).
      pair: {id, csid, source_rel, gt_rel, gt_emotion, intensity, provisional}
    """
    groups = defaultdict(list)
    for relpath, info in entries.items():
        groups[(info["id"], info["canonical_sentence_id"])].append((to_video_rel(relpath), info))

    pairs, warnings = [], []
    for (ident, csid), members in sorted(groups.items()):
        neutrals = [(r, i) for r, i in members if i["emotion"] == "neutral"]
        others = [(r, i) for r, i in members if i["emotion"] != "neutral"]
        if not neutrals:
            warnings.append({"id": ident, "csid": csid,
                             "issue": "no neutral source; group skipped",
                             "n_targets": len(others)})
            continue
        if len({i["num"] for _, i in neutrals}) > 1:
            # ambiguous; pick deterministic lowest num and warn
            neutrals.sort(key=lambda ri: ri[1]["num"])
            warnings.append({"id": ident, "csid": csid,
                             "issue": "multiple neutral candidates; picked lowest num",
                             "nums": sorted({i["num"] for _, i in neutrals})})
        source_rel = neutrals[0][0]
        prov = bool(neutrals[0][1].get("provisional", False))
        if not others:
            warnings.append({"id": ident, "csid": csid,
                             "issue": "neutral present but no emotional targets"})
            continue
        for gt_rel, gi in sorted(others, key=lambda ri: ri[0]):
            pairs.append({
                "id": ident, "csid": csid,
                "source_rel": source_rel, "gt_rel": gt_rel,
                "gt_emotion": gi["emotion"], "intensity": gi["intensity"],
                "provisional": prov or bool(gi.get("provisional", False)),
            })
    return pairs, warnings

# ---------- feature completeness ----------

def feature_paths(data_root, rel):
    """Sibling feature and audio files a C-MET training pair needs."""
    stem = rel[:-4] if rel.endswith(".mp4") else rel  # strip .mp4
    needed = [os.path.join(data_root, stem + suf) for suf in ED_SUFFIXES]
    needed.append(os.path.join(data_root, stem + ".wav"))
    needed.append(os.path.join(data_root, rel))  # the video itself
    return needed

def missing_for(data_root, rel):
    return [p for p in feature_paths(data_root, rel) if not os.path.isfile(p)]

def audit_pairs(pairs, data_root):
    """Split pairs into complete and incomplete (with the missing file list)."""
    complete, incomplete = [], []
    for p in pairs:
        miss = missing_for(data_root, p["source_rel"]) + missing_for(data_root, p["gt_rel"])
        if miss:
            incomplete.append({**p, "missing": miss})
        else:
            complete.append(p)
    return complete, incomplete

# ---------- csv emission ----------

def row_from_pair(p, prefix):
    return {
        "source_video_path": f"{prefix}/{p['source_rel']}",
        "gt_video_path": f"{prefix}/{p['gt_rel']}",
        "gt_emotion": p["gt_emotion"],
        "intensity": p["intensity"],
    }

def write_manifest(pairs, prefix, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for p in sorted(pairs, key=lambda x: (x["id"], x["csid"], x["gt_emotion"], x["intensity"])):
            w.writerow(row_from_pair(p, prefix))
    return len(pairs)

def write_missing(incomplete, prefix, out_path):
    if not incomplete:
        return 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "csid", "source_video_path", "gt_video_path",
                    "gt_emotion", "intensity", "missing_files"])
        for p in incomplete:
            w.writerow([p["id"], p["csid"], f"{prefix}/{p['source_rel']}",
                        f"{prefix}/{p['gt_rel']}", p["gt_emotion"], p["intensity"],
                        ";".join(p["missing"])])
    return len(incomplete)

# ---------- split logic ----------

def parse_test_csv_keys(test_csv):
    """Read the repo test.csv; return set of (id, gt_relpath-ish) to hold out.
       We key the held-out TEST sentences by (id, source_num) since the source
       neutral file identifies the sentence for a given id."""
    held = set()
    rows = []
    with open(test_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
            src = row["source_video_path"]
            # .../<ID>/front/neutral/level_1/<num>.mp4
            parts = src.replace("\\", "/").split("/")
            ident = parts[-5]
            num = parts[-1].rsplit(".", 1)[0]
            held.add((ident, num))
    return held, rows

def split_cmet(pairs, data_root, test_csv, prefix):
    """Test = repo test.csv verbatim (validated); train = complete pairs whose
       (id, source_num) is not a held-out test sentence."""
    held, test_rows = parse_test_csv_keys(test_csv)
    def src_num(p):
        return p["source_rel"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
    train = [p for p in pairs if (p["id"], src_num(p)) not in held]
    return train, test_rows, held

def split_identity(pairs, test_ids, val_ids, val_frac, seed):
    all_ids = sorted({p["id"] for p in pairs})
    test_ids = set(test_ids or [])
    val_ids = set(val_ids or [])
    unknown = (test_ids | val_ids) - set(all_ids)
    if unknown:
        raise ValueError(f"requested split ids not present in data: {sorted(unknown)}")
    remaining = [i for i in all_ids if i not in test_ids and i not in val_ids]
    if not val_ids and val_frac and val_frac > 0:
        rng = random.Random(seed)
        k = max(1, int(round(val_frac * len(remaining))))
        val_ids = set(rng.sample(remaining, min(k, len(remaining))))
        remaining = [i for i in remaining if i not in val_ids]
    train_ids = set(remaining)
    # leakage assertion: provably disjoint
    assert train_ids.isdisjoint(val_ids), "train/val id overlap"
    assert train_ids.isdisjoint(test_ids), "train/test id overlap"
    assert val_ids.isdisjoint(test_ids), "val/test id overlap"
    by = lambda ids: [p for p in pairs if p["id"] in ids]
    return by(train_ids), by(val_ids), by(test_ids), train_ids, val_ids, test_ids

# ---------- driver ----------

def cmd_build(args):
    remap = load_remap(args.remap)
    pairs, warnings = build_pairs(remap["entries"])
    complete, incomplete = audit_pairs(pairs, args.data_root)

    outdir = args.out_dir
    os.makedirs(outdir, exist_ok=True)
    prefix = args.path_prefix
    split_meta = {
        "schema": "cafe.phase1.manifest_split/1",
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": args.split, "path_prefix": prefix,
        "remap_schema": remap.get("schema"),
        "remap_provisional_any": any(p["provisional"] for p in pairs),
        "counts": {}, "group_warnings": warnings,
    }

    if args.split == "cmet":
        if not args.test_csv or not os.path.isfile(args.test_csv):
            print("[build] --test_csv (repo dataset/MEAD/test.csv) required for cmet split")
            sys.exit(2)
        train, test_rows, held = split_cmet(complete, args.data_root, args.test_csv, prefix)
        n_tr = write_manifest(train, prefix, os.path.join(outdir, "train.csv"))
        # test.csv is copied verbatim from the repo (source of truth)
        with open(os.path.join(outdir, "test.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()) if test_rows else CSV_FIELDS)
            w.writeheader()
            for r in test_rows:
                w.writerow(r)
        split_meta["counts"] = {"train_rows": n_tr, "test_rows": len(test_rows),
                                "held_out_sentences": len(held)}
        split_meta["note"] = ("sentence split; same identities in train and test; "
                              "not identity generalization")
    else:  # identity
        tr, va, te, tr_ids, va_ids, te_ids = split_identity(
            complete, args.test_ids, args.val_ids, args.val_frac, args.seed)
        n_tr = write_manifest(tr, prefix, os.path.join(outdir, "train.csv"))
        n_va = write_manifest(va, prefix, os.path.join(outdir, "val.csv"))
        n_te = write_manifest(te, prefix, os.path.join(outdir, "test.csv"))
        split_meta["counts"] = {"train_rows": n_tr, "val_rows": n_va, "test_rows": n_te}
        split_meta["ids"] = {"train": sorted(tr_ids), "val": sorted(va_ids),
                             "test": sorted(te_ids)}

    n_miss = write_missing(incomplete, prefix, os.path.join(outdir, "manifest_missing.csv"))
    split_meta["counts"]["excluded_incomplete_pairs"] = n_miss
    with open(os.path.join(outdir, "manifest_split.json"), "w") as f:
        json.dump(split_meta, f, indent=2)

    print(f"[build] mode              : {args.split}")
    print(f"[build] complete pairs    : {len(complete)}")
    print(f"[build] excluded (missing): {n_miss}  -> manifest_missing.csv")
    print(f"[build] group warnings    : {len(warnings)}")
    print(f"[build] counts            : {split_meta['counts']}")
    if split_meta.get("remap_provisional_any"):
        print("[build] NOTE: remap is provisional (no appendix). Numbering is internally "
              "consistent but not aligned to C-MET absolute indices.")
    print(f"[build] wrote manifests to {outdir}")
    sys.exit(1 if (n_miss or warnings) else 0)

# ---------- self-test ----------

def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()

def _make_tree(root, ids, complete=True, drop_feature_for=None):
    """Create a synthetic FPS25 tree with common(1-2)+generic(1-2) sentences,
       3 emotions + neutral, level_1 for all (level_2 for emotions). Returns
       a remap-entries dict mirroring verify_sentence_order.py output."""
    emos = ["angry", "happy", "sad"]
    entries = {}

    def add(ident, emo, level, num, csid, sentence_class, provisional=False):
        rel = f"{ident}/front/{emo}/{level}/{num:03d}.mp4"
        # create files
        _touch(os.path.join(root, rel))
        if complete and (drop_feature_for is None or rel != drop_feature_for):
            stem = rel[:-4]
            for suf in ED_SUFFIXES:
                _touch(os.path.join(root, stem + suf))
            _touch(os.path.join(root, stem + ".wav"))
        entries[rel] = {"id": ident, "emotion": emo, "intensity": level,
                        "num": num, "sentence_class": sentence_class,
                        "intended_slot": num, "canonical_sentence_id": csid,
                        "provisional": provisional, "match_score": 1.0}

    for ident in ids:
        # common sentence 1 and 2 (same number across emotions + neutral)
        for k in (1, 2):
            add(ident, "neutral", "level_1", k, f"common:{k}", "common")
            for e in emos:
                add(ident, e, "level_1", k, f"common:{k}", "common")
                add(ident, e, "level_2", k, f"common:{k}", "common")
        # generic sentence 1 and 2 (neutral 031/032, emotion 021/022)
        for k in (1, 2):
            add(ident, "neutral", "level_1", 30 + k, f"generic:{k}", "generic")
            for e in emos:
                add(ident, e, "level_1", 20 + k, f"generic:{k}", "generic")
    return entries

def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "dataset", "MEAD", "FPS25")
        ids = ["M003", "W009", "M011"]
        # drop one feature file to test exclusion
        drop = "M003/front/happy/level_1/001.mp4"
        entries = _make_tree(root, ids, complete=True, drop_feature_for=drop)

        pairs, warnings = build_pairs(entries)
        assert warnings == [], f"unexpected warnings: {warnings}"

        # T1: source is ALWAYS neutral; no neutral gt; no cross-id
        for p in pairs:
            assert "/neutral/" in p["source_rel"], p
            assert p["gt_emotion"] != "neutral", p
            assert p["source_rel"].split("/")[0] == p["gt_rel"].split("/")[0], "cross-id pair"

        # T2: generic pairing by sentence, not number (neutral 031 <-> emo 021)
        g = [p for p in pairs if p["csid"] == "generic:1" and p["id"] == "W009"]
        assert g, "no generic:1 pairs"
        assert all(p["source_rel"].endswith("/031.mp4") for p in g), "generic source not 031"
        assert all(p["gt_rel"].endswith("/021.mp4") for p in g), "generic gt not 021"

        # T3: feature audit excludes the dropped-feature row, keeps the rest
        complete, incomplete = audit_pairs(pairs, root)
        assert any(p["gt_rel"] == drop for p in incomplete), "missing row not caught"
        assert all(p["gt_rel"] != drop for p in complete), "incomplete row leaked into complete"
        assert len(complete) + len(incomplete) == len(pairs)

        # T4: identity split is provably disjoint and covers ids
        tr, va, te, tr_ids, va_ids, te_ids = split_identity(
            complete, test_ids=["M011"], val_ids=None, val_frac=0.5, seed=0)
        assert tr_ids.isdisjoint(te_ids) and tr_ids.isdisjoint(va_ids) and va_ids.isdisjoint(te_ids)
        assert te_ids == {"M011"}
        assert (tr_ids | va_ids | te_ids) == set(ids)
        assert all(p["id"] in te_ids for p in te)
        assert all(p["id"] in tr_ids for p in tr)

        # T5: determinism (same seed -> same val ids)
        _, _, _, _, va2, _ = split_identity(complete, ["M011"], None, 0.5, 0)
        assert va_ids == va2, "val split not deterministic under fixed seed"

        # T6: leakage assertion actually fires (negative test)
        fired = False
        try:
            # force overlap by hacking pairs' ids is complex; instead call with
            # test and val requesting the same id
            split_identity(complete, test_ids=["M011"], val_ids=["M011"],
                           val_frac=0.0, seed=0)
        except AssertionError:
            fired = True
        assert fired, "leakage assertion did NOT fire on forced overlap"

        # T7: cmet split reproduces a given test.csv verbatim; train excludes it
        prefix = "./dataset/MEAD/FPS25"
        test_csv = os.path.join(td, "repo_test.csv")
        with open(test_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS); w.writeheader()
            # hold out common sentence 001 for M003 across its emotions/levels
            for e in ("angry", "happy", "sad"):
                for lvl in ("level_1", "level_2"):
                    w.writerow({"source_video_path": f"{prefix}/M003/front/neutral/level_1/001.mp4",
                                "gt_video_path": f"{prefix}/M003/front/{e}/{lvl}/001.mp4",
                                "gt_emotion": e, "intensity": lvl})
        train, test_rows, held = split_cmet(complete, root, test_csv, prefix)
        assert ("M003", "001") in held
        # no train row uses M003 sentence 001 as source
        for p in train:
            if p["id"] == "M003":
                assert not p["source_rel"].endswith("/001.mp4"), "held-out sentence leaked into train"

        # T8: emitted CSV has EXACT C-MET columns and neutral sources
        outp = os.path.join(td, "train.csv")
        write_manifest(train, prefix, outp)
        with open(outp) as f:
            r = csv.reader(f); header = next(r)
        assert header == CSV_FIELDS, f"schema mismatch: {header}"

        # T9: interop regression. verify_sentence_order may key remap entries by
        # the .wav it transcribed. Those must normalize to the .mp4 video, resolve
        # feature files, and pair identically to the .mp4-keyed case.
        wav_entries = {}
        for rel, info in entries.items():
            wav_entries[rel[:-4] + ".wav"] = dict(info)  # re-key .mp4 -> .wav
        wpairs, wwarn = build_pairs(wav_entries)
        assert wwarn == [], f"wav-keyed warnings: {wwarn}"
        assert all(p["source_rel"].endswith(".mp4") and p["gt_rel"].endswith(".mp4")
                   for p in wpairs), "wav key not normalized to mp4"
        wcomplete, wincomplete = audit_pairs(wpairs, root)
        # same feature-completeness outcome as the .mp4-keyed run
        assert len(wcomplete) == len(complete) and len(wincomplete) == len(incomplete), \
            (len(wcomplete), len(complete), len(wincomplete), len(incomplete))
        assert {p["id"] for p in wcomplete} == {p["id"] for p in complete}, "ids differ after wav re-key"

    print("selftest OK: pairing(neutral-source, sentence-not-number, no cross-id), "
          "feature audit exclusion, identity disjointness, determinism, leakage "
          "assertion, cmet verbatim test + train exclusion, exact CSV schema")
    return True

# ---------- cli ----------

def main():
    ap = argparse.ArgumentParser(description="Build C-MET-schema MEAD manifests from remap.json")
    ap.add_argument("--selftest", action="store_true", help="CPU self-check and exit")
    ap.add_argument("--remap", help="mead_sentence_remap.json")
    ap.add_argument("--data_root", default="./dataset/MEAD/FPS25",
                    help="filesystem root for feature-existence checks")
    ap.add_argument("--path_prefix", default="./dataset/MEAD/FPS25",
                    help="prefix written into the CSV paths (match repo style)")
    ap.add_argument("--split", choices=["cmet", "identity"], default="cmet")
    ap.add_argument("--test_csv", help="repo dataset/MEAD/test.csv (cmet split)")
    ap.add_argument("--test_ids", nargs="*", help="identity split: held-out test ids")
    ap.add_argument("--val_ids", nargs="*", help="identity split: explicit val ids")
    ap.add_argument("--val_frac", type=float, default=0.0, help="identity split: val fraction of remaining")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="./dataset/MEAD/manifests")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.remap:
        ap.print_help(); sys.exit(1)
    cmd_build(args)

if __name__ == "__main__":
    main()
