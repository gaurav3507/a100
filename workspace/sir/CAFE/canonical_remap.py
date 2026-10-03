#!/usr/bin/env python3
"""
CAFE: renumber MEAD clips to the AUTHORS' numbering via a canonical sentence table.

Finding that motivates this (measured): whole (ID, emotion) folders of the
official MEAD release are offset from the numbering C-MET's authors used
(e.g. disgusted/M003: authors 008 = ours 005 at ALL levels; even 001-003 there
are not the common sentences). Level_1 is therefore NOT a valid reference.

Canonical table per emotion e (numbers 1..30, authors' numbering):
  1-3   common sentences = neutral 001-003 (neutral numbering verified by e2v)
  21-30 generic sentences = neutral 031-040 (C-MET pairs emo k with neutral k+10)
  4-20  and any other number: authors' level_3 e2v anchors (our clip whose
        emotion2vec feature matches the authors' file with cosine >= 0.99), plus
        "trusted folders" (level_3 folders with >= 3 anchors that ALL agree our
        number == authors' number).
Each (ID, emotion, level) folder is then matched to the table by transcript
similarity with an optimal one-to-one assignment.

Validations:
  A  leave-one-speaker-out: rebuild the table without speaker X, map X's
     folders, check X's anchors (non-circular);
  B  common sentences: every clip assigned 1-3 must have that number as its
     individual best match;
  C  test pairs: for every row of the authors' test.csv, does the GT clip carry
     the same sentence as the neutral source? reported before vs after remap;
  D  structure: coverage of the table, bijection per folder, weak matches.

Output <meta>/canonical_remap.json, directly usable by overnight_gate2.py
(--remap_json). Analysis only: this script never renames files.

Run from the C-MET repo root:
  python ../canonical_remap.py --fps25_root dataset/MEAD/FPS25 --audios audios/MEAD
  python ../canonical_remap.py --selftest
"""

import argparse, csv, glob, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from align_mead_levels import load_transcripts, sim, assign, classify, norm  # noqa: E402

E2V = "emotion2vec+large_features"
# calibrated on real data: unrelated English sentences reach max 0.60 (median 0.31);
# same sentence with our actual whisper errors scored 0.83-0.92
WEAK_SIM, WEAK_MARGIN, PAIR_OK = 0.65, 0.10, 0.60
TRUST_MIN_ANCHORS = 3


def medoid(texts):
    texts = [t for t in texts if t]
    if not texts:
        return None
    return max(texts, key=lambda t: sum(sim(t, u) for u in texts))


# ------------------------------------------------------------- anchors -----

def find_anchors(fps25_root, audios_root, np):
    """[(ID, emo, level, our_num, authors_num, cos)] for cos >= 0.99."""
    L = lambda p: np.load(p).astype("float64").ravel()
    cs = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    out, uncertain = [], 0
    for a in sorted(glob.glob(os.path.join(audios_root, "*", E2V, "*.npy"))):
        emo = a.split(os.sep)[-3]
        p = os.path.basename(a)[:-4].split("_")
        if len(p) == 4:
            ident, lvl, n = p[0], f"{p[1]}_{p[2]}", int(p[3])
        elif len(p) == 2 and emo == "neutral":
            ident, lvl, n = p[0], "level_1", int(p[1])
        else:
            continue
        cands = glob.glob(os.path.join(fps25_root, ident, "front", emo, lvl, E2V, "*.npy"))
        if not cands:
            continue
        x = L(a)
        c, m = max((cs(x, L(q)), int(os.path.basename(q)[:-4])) for q in cands)
        if c >= 0.99:
            out.append((ident, emo, lvl, m, n, c))
        else:
            uncertain += 1
    return out, uncertain


# ------------------------------------------------------ canonical table -----

def build_table(T, anchors, exclude_id=None):
    """{emo: {num: text}} in the authors' numbering, plus per-entry support."""
    ids = sorted({i for (i, _, _) in T} - {exclude_id})
    emos = sorted({e for (_, e, _) in T if e != "neutral"})
    neu = {c: medoid([T.get((i, "neutral", "level_1"), {}).get(c) for i in ids]) for c in range(1, 41)}
    table, support = {}, {}
    # trusted folders: all anchors agree identity, >= TRUST_MIN_ANCHORS
    byfolder = defaultdict(list)
    for (i, e, l, m, n, _) in anchors:
        if e != "neutral" and i != exclude_id:
            byfolder[(i, e, l)].append((m, n))
    trusted = {k for k, v in byfolder.items() if len(v) >= TRUST_MIN_ANCHORS and all(m == n for m, n in v)}
    for e in emos:
        cand = defaultdict(list)
        for c in (1, 2, 3):
            if neu.get(c):
                cand[c].append(("neutral", neu[c]))
        for c in range(21, 31):
            if neu.get(c + 10):
                cand[c].append(("neutral+10", neu[c + 10]))
        for (i, ee, l, m, n, _) in anchors:
            if ee == e and i != exclude_id:
                t = T.get((i, e, l), {}).get(m)
                if t:
                    cand[n].append((f"anchor:{i}/{l}/{m:03d}", t))
        for (i, ee, l) in trusted:
            if ee == e:
                for m, t in T.get((i, e, l), {}).items():
                    cand[m].append((f"trusted:{i}/{l}", t))
        table[e], support[e] = {}, {}
        for c, lst in cand.items():
            med = medoid([t for _, t in lst])
            if med:
                table[e][c] = med
                support[e][c] = {"n": len(lst), "sources": sorted({s.split(":")[0] for s, _ in lst}),
                                 "min_agreement": round(min(sim(t, med) for _, t in lst), 3)}
    return table, support, sorted("/".join(k) for k in trusted)


# -------------------------------------------------------------- mapping -----

def map_folders(T, table, only_id=None):
    groups, weak = {}, []
    for (i, e, l), clips in sorted(T.items()):
        if e == "neutral" or (only_id and i != only_id) or not clips:
            continue
        ref = table.get(e, {})
        if len(ref) < len(clips):
            groups[f"{i}/{e}/{l}"] = {"verdict": "unresolved_table_too_small", "n": len(clips), "map": {}}
            continue
        mp = assign(clips, ref)
        key = f"{i}/{e}/{l}"
        sims = [s for _, s, _ in mp.values()]
        groups[key] = {"verdict": classify(mp), "n": len(mp),
                       "mean_sim": round(sum(sims) / len(sims), 4), "min_sim": round(min(sims), 4),
                       "map": {f"{m:03d}": f"{r:03d}" for m, (r, _, _) in sorted(mp.items())}}
        for m, (r, s, mg) in mp.items():
            if s < WEAK_SIM or mg < WEAK_MARGIN:
                ranked = sorted(ref, key=lambda c: sim(clips[m], ref[c]), reverse=True)
                ru = next(c for c in ranked if c != r) if len(ranked) > 1 else None
                weak.append({"group": key, "ours": f"{m:03d}", "to": f"{r:03d}",
                             "sim": round(s, 3), "margin": round(mg, 3),
                             "assigned_is_best": ranked[0] == r,
                             "text": clips[m], "assigned_text": ref[r],
                             "runner_up": f"{ru:03d}" if ru else None,
                             "runner_up_text": ref.get(ru) if ru else None})
    return groups, weak

def canon_of(groups, i, e, l, m):
    if e == "neutral":
        return m
    g = groups.get(f"{i}/{e}/{l}")
    r = g["map"].get(f"{m:03d}") if g else None
    return int(r) if r else None


# ---------------------------------------------------------- validations -----

def validation_a_loso(T, anchors):
    agree, disagree, examples = 0, 0, []
    for x in sorted({i for (i, _, _) in T}):
        table, _, _ = build_table(T, anchors, exclude_id=x)
        groups, _ = map_folders(T, table, only_id=x)
        for (i, e, l, m, n, _) in anchors:
            if i != x or e == "neutral":
                continue
            c = canon_of(groups, i, e, l, m)
            if c == n:
                agree += 1
            else:
                disagree += 1
                examples.append(f"{e}/{i}/{l}: authors {n:03d} = ours {m:03d} -> mapped {c}")
    return {"agree": agree, "disagree": disagree, "examples": examples[:10]}

def validation_b_common(T, table, groups):
    mis, checked = [], 0
    for (i, e, l), clips in T.items():
        if e == "neutral":
            continue
        for m, t in clips.items():
            c = canon_of(groups, i, e, l, m)
            if c in (1, 2, 3):
                checked += 1
                best = max(table[e], key=lambda k: sim(t, table[e][k]))
                if best != c:
                    mis.append(f"{i}/{e}/{l}/{m:03d} assigned {c:03d} but best {best:03d}")
    return {"checked": checked, "misassigned": len(mis), "examples": mis[:5]}

def validation_c_pairs(test_csv, T, groups):
    """For every authors' test row: is the GT clip the same sentence as the neutral
       source? Relative test: among all 40 neutral sentences (speaker consensus), the
       GT transcript must be closest to the source's sentence. Robust to ASR noise."""
    ids = sorted({i for (i, _, _) in T})
    neu = {c: medoid([T.get((i, "neutral", "level_1"), {}).get(c) for i in ids]) for c in range(1, 41)}
    neu = {c: t for c, t in neu.items() if t}
    def txt(i, e, l, m):
        return T.get((i, e, l), {}).get(m)
    before = {"ok": 0, "bad": 0, "missing": 0}
    after = {"ok": 0, "bad": 0, "missing": 0}
    bad_after = []
    for r in csv.DictReader(open(test_csv, newline="")):
        s = r["source_video_path"].replace("\\", "/").split("/")
        g = r["gt_video_path"].replace("\\", "/").split("/")
        i, sn = s[-5], int(s[-1][:-4])
        e, l, k = g[-3], g[-2], int(g[-1][:-4])
        inv = {int(c): int(m) for m, c in (groups.get(f"{i}/{e}/{l}", {}).get("map") or {}).items()}
        gt0 = txt(i, e, l, k)
        gt1 = txt(i, e, l, inv[k]) if k in inv else None
        for bucket, gt in ((before, gt0), (after, gt1)):
            if not gt or sn not in neu:
                bucket["missing"] += 1
                continue
            best = max(neu, key=lambda c: sim(gt, neu[c]))
            if best == sn:
                bucket["ok"] += 1
            else:
                bucket["bad"] += 1
                if bucket is after:
                    bad_after.append(f"{i} {e}/{l}/{k:03d}: source neutral {sn:03d}, GT reads like neutral {best:03d}")
    return {"before": before, "after": after, "bad_after_examples": bad_after[:10]}


def run(fps25_root, audios_root, meta, np, log=print):
    T = load_transcripts(os.path.join(meta, "transcripts_all.csv"))
    anchors, uncertain = find_anchors(fps25_root, audios_root, np)
    table, support, trusted = build_table(T, anchors)
    groups, weak = map_folders(T, table)
    for (i, e, l), clips in T.items():                  # neutral folders: identity by design
        if e == "neutral":
            groups[f"{i}/{e}/{l}"] = {"verdict": "identity", "n": len(clips),
                                      "map": {f"{m:03d}": f"{m:03d}" for m in sorted(clips)}}
    coverage = {e: sorted(t) for e, t in table.items()}
    missing_nums = {e: [c for c in range(1, 31) if c not in t] for e, t in table.items()}
    va = validation_a_loso(T, anchors)
    vb = validation_b_common(T, table, groups)
    vc = validation_c_pairs(os.path.join(meta, "test.csv"), T, groups)
    verdicts = defaultdict(int)
    for g in groups.values():
        v = g["verdict"]
        verdicts["cyclic" if v.startswith("cyclic") else v] += 1
    rep = {"method": "canonical_table", "anchors": len(anchors), "anchors_uncertain": uncertain,
           "trusted_folders": trusted, "table_missing_numbers": missing_nums,
           "table_support": {e: {str(c): s for c, s in sup.items()} for e, sup in support.items()},
           "groups": groups, "weak_matches": weak, "summary": dict(verdicts),
           "validation_a_authors_e2v": va, "validation_b_common_sentences": vb,
           "validation_c_test_pairs": vc}
    out = os.path.join(meta, "canonical_remap.json")
    json.dump(rep, open(out, "w"), indent=1)
    log(f"[canon] anchors {len(anchors)} (uncertain {uncertain}) | trusted folders {len(trusted)}")
    log(f"[canon] table gaps: " + str({e: v for e, v in missing_nums.items() if v} or "none"))
    log(f"[canon] folders: {dict(verdicts)} | weak matches: {len(weak)}")
    nonid = sorted(k for k, g in groups.items() if g["verdict"] not in ("identity",))
    for k in nonid[:40]:
        log(f"   {k}: {groups[k]['verdict']} (min_sim {groups[k].get('min_sim')})")
    for w in weak:
        tag = "RENAMED-GROUP" if groups[w["group"]]["verdict"] != "identity" else "identity-group"
        log(f"   WEAK [{tag}] {w['group']}:{w['ours']} -> {w['to']} sim {w['sim']} margin {w['margin']} best={w['assigned_is_best']}")
        log(f"        heard   : {w['text']}")
        log(f"        assigned: {w['assigned_text']}")
        log(f"        runnerup: {w['runner_up']} {w['runner_up_text']}")
    log(f"[canon] A leave-one-speaker-out: {va}")
    log(f"[canon] B common sentences: {vb}")
    log(f"[canon] C test pairs: {vc}")
    log(f"[canon] wrote {out}")
    return rep


# ------------------------------------------------------------- self-test -----

def selftest():
    import tempfile, random, zlib
    import numpy as np
    # realistic vocabulary: 600 pseudo-words, 8-word sentences (distinct like TIMIT prompts);
    # the conservative weak-match gate on confusable text is tested in overnight_gate2 (T2)
    vr = random.Random(11)
    W = sorted({"".join(vr.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(vr.randint(4, 9)))
                for _ in range(700)})[:600]
    rs = lambda seed: " ".join(random.Random(seed).sample(W, 8))
    noise_rng = random.Random(5)
    noisy = lambda t: " ".join(w if noise_rng.random() > 0.05 else noise_rng.choice(W) for w in t.split())
    IDS, EMOS, LV = ["M003", "M030", "W009", "W015"], ["angry", "disgusted", "sad"], ["level_1", "level_2", "level_3"]
    NEU = {c: rs(1000 + c) for c in range(1, 41)}
    SPEC = {e: {c: rs(zlib.crc32(f"{e}{c}".encode())) for c in range(4, 21)} for e in EMOS}
    def canon_text(e, c):
        return NEU[c] if c <= 3 else (NEU[c + 10] if c >= 21 else SPEC[e][c])
    # truth: our file number -> authors canonical number, per folder
    shift3 = lambda m: (m + 2) % 30 + 1
    truth = {}
    for i in IDS:
        for e in EMOS:
            for l in LV:
                f = {m: m for m in range(1, 31)}
                if (i, e) in (("M003", "disgusted"), ("W015", "sad")):
                    f = {m: shift3(m) for m in range(1, 31)}               # whole folder shifted
                if (i, e, l) == ("M030", "angry", "level_2"):
                    perm = list(range(1, 31)); random.Random(3).shuffle(perm)
                    f = {m: perm[m - 1] for m in range(1, 31)}             # level-only permutation
                if (i, e, l) == ("W009", "sad", "level_2"):
                    f = {m: c for m, c in f.items() if m not in (26, 27, 28, 29)}   # missing clips
                truth[(i, e, l)] = f
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "dataset", "MEAD", "FPS25"); meta = os.path.dirname(root)
        aud = os.path.join(td, "audios", "MEAD"); os.makedirs(root)
        rows = []
        vec = lambda e, c: np.random.default_rng(zlib.crc32(f"{e}{c}".encode())).normal(size=24)
        for i in IDS:
            for c in range(1, 41):
                rows.append((f"{i}/front/neutral/level_1/{c:03d}.wav", noisy(NEU[c])))
                p = os.path.join(root, i, "front", "neutral", "level_1", E2V, f"{c:03d}.npy")
                os.makedirs(os.path.dirname(p), exist_ok=True); np.save(p, vec("neutral", c))
                if c % 4 == 0:
                    q = os.path.join(aud, "neutral", E2V, f"{i}_{c:03d}.npy")
                    os.makedirs(os.path.dirname(q), exist_ok=True); np.save(q, vec("neutral", c))
            for e in EMOS:
                for l in LV:
                    for m, c in truth[(i, e, l)].items():
                        rows.append((f"{i}/front/{e}/{l}/{m:03d}.wav", noisy(canon_text(e, c))))
                        p = os.path.join(root, i, "front", e, l, E2V, f"{m:03d}.npy")
                        os.makedirs(os.path.dirname(p), exist_ok=True); np.save(p, vec(e + i + l, c))
                        # authors' level_3 subset: every 3rd canonical number
                        if l == "level_3" and c % 3 == (IDS.index(i) % 3):
                            q = os.path.join(aud, e, E2V, f"{i}_{l}_{c:03d}.npy")
                            os.makedirs(os.path.dirname(q), exist_ok=True); np.save(q, vec(e + i + l, c) * 1.2)
        with open(os.path.join(meta, "transcripts_all.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["relpath", "transcript"]); w.writerows(rows)
        with open(os.path.join(meta, "test.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["source_video_path", "gt_video_path", "gt_emotion", "intensity"])
            for i in IDS:
                for e in EMOS:
                    for l in LV:
                        for k in list(range(1, 4)) + list(range(21, 31)):
                            sn = k if k <= 3 else k + 10
                            w.writerow([f"./dataset/MEAD/FPS25/{i}/front/neutral/level_1/{sn:03d}.mp4",
                                        f"./dataset/MEAD/FPS25/{i}/front/{e}/{l}/{k:03d}.mp4", e, l])
        rep = run(root, aud, meta, np, log=lambda *_: None)
        G = rep["groups"]
        # T1 table complete for every emotion
        assert all(not v for v in rep["table_missing_numbers"].values()), rep["table_missing_numbers"]
        # T2 every folder mapping equals the truth exactly (whole-folder shifts, level-only perm, missing)
        for (i, e, l), f in truth.items():
            got = {int(m): int(c) for m, c in G[f"{i}/{e}/{l}"]["map"].items()}
            assert got == f, (i, e, l)
        assert G["M003/disgusted/level_1"]["verdict"].startswith("cyclic"), G["M003/disgusted/level_1"]["verdict"]
        assert G["M003/angry/level_1"]["verdict"] == "identity"
        # T3 validation A (leave-one-speaker-out) perfect, B clean
        va, vb, vc = rep["validation_a_authors_e2v"], rep["validation_b_common_sentences"], rep["validation_c_test_pairs"]
        assert va["disagree"] == 0 and va["agree"] > 100, va
        assert vb["misassigned"] == 0 and vb["checked"] > 0, vb
        # T4 validation C: before remap there are content-mismatched pairs, after remap none
        assert vc["before"]["bad"] > 0, vc
        assert vc["after"]["bad"] == 0 and vc["after"]["ok"] > 0, vc
        # T5 output is accepted by overnight_gate2.decide_gate (rename set = non-identity groups)
        import overnight_gate2 as O
        for (i, e, l), f in truth.items():                          # create files decide_gate inspects
            d = os.path.join(root, i, "front", e, l)
            for m in f:
                open(os.path.join(d, f"{m:03d}.mp4"), "w").close()
        gate = O.decide_gate(rep, root, 100)
        assert gate["passed"], gate["reasons"]
        exp_nonid = sorted(k for k, g in G.items() if g["verdict"] != "identity")
        assert gate["rename_groups"] == exp_nonid and "M003/disgusted/level_1" in exp_nonid
        # T6 negative control: corrupt one anchor's transcript source -> LOSO must catch it
        T = load_transcripts(os.path.join(meta, "transcripts_all.csv"))
        anchors, _ = find_anchors(root, aud, np)
        bad = [(i, e, l, m, (n % 30) + 1, c) if (i, e) == ("M030", "angry") and l == "level_3" else (i, e, l, m, n, c)
               for (i, e, l, m, n, c) in anchors]
        va_bad = validation_a_loso(T, bad)
        assert va_bad["disagree"] > 0, va_bad
        # T7 negative control: a clip whose sentence is NOT in the table, inside a folder that
        # would be renamed, must be flagged weak and must block the gate
        rows2 = [(r, ("zzqx vvbn qqpl mmrt kkjh wwxc ffgd yyuo" if r == "M003/front/disgusted/level_2/010.wav" else t))
                 for r, t in rows]
        with open(os.path.join(meta, "transcripts_all.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["relpath", "transcript"]); w.writerows(rows2)
        rep2 = run(root, aud, meta, np, log=lambda *_: None)
        assert any(x["group"] == "M003/disgusted/level_2" and x["ours"] == "010" for x in rep2["weak_matches"]), rep2["weak_matches"]
        g2 = O.decide_gate(rep2, root, 100)
        assert not g2["passed"] and any("weak" in r for r in g2["reasons"]), g2
    print("selftest OK: complete canonical table from neutral+anchors+trusted folders, exact recovery of "
          "whole-folder shifts / level-only permutation / missing clips under ASR noise, "
          "leave-one-speaker-out clean, common sentences clean, test-pair content fixed (before bad>0, "
          "after bad=0), accepted by overnight gate, negative controls caught (corrupted anchor; "
          "out-of-table clip is flagged weak and blocks the gate)")
    return True


def main():
    ap = argparse.ArgumentParser(description="Canonical-table MEAD renumbering (analysis only)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fps25_root")
    ap.add_argument("--audios", default="audios/MEAD")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.fps25_root:
        ap.print_help(); sys.exit(1)
    import numpy as np
    meta = os.path.dirname(args.fps25_root.rstrip("/")) or "."
    run(args.fps25_root, args.audios, meta, np)


if __name__ == "__main__":
    main()
