#!/usr/bin/env python3
"""
CAFE: renumber MEAD clips to the AUTHORS' numbering (canonical_remap v2, algorithm v4).

What the real data showed (DGX, 2641 transcripts + 192 authors' e2v anchors):
  - neutral scripts are identical across the 4 speakers (all 40 numbers);
  - C-MET's own rule pairs emotion k with neutral k+10 for the generic sentences
    021-030, and the common sentences 001-003 equal neutral 001-003;
  - release errors can be the MAJORITY: in angry, 3 of 4 speakers are shifted by one
    from ~016 onward while M003 is correct (authors 016=ours 016, 025=ours 025);
    a plain majority vote therefore reproduces the release error (algorithm v3 failed);
  - gt 020 never matches neutral 030 in any folder: the authors' own test.csv pairs
    020 with a different sentence (inherent, documented, not a renumbering issue);
  - some clips hold a sentence outside their script (quarantined, never forced).

Reference (ground truth = authors' numbering):
  001-003  neutral 001-003
  021-030  neutral 031-040
  004-020  majority ONLY over TRUSTED folders: a folder is trusted when its 13
           common+generic clips each read as the expected neutral sentence of the
           same speaker (argmax over all 40 neutral sentences; at most one miss).
           Shifted folders fail this independent test and cast no vote.
Mapping: optimal assignment to the reference; quarantine (9NN) for unresolvable
clips; near-duplicate script pairs handled explicitly.

Validations: A authors' e2v anchors (all, and the 004-020 subset separately),
B every remapped common/generic clip reads as its speaker's expected neutral
sentence, C test pairs per gt number (before/after), LOSO (reference rebuilt
without speaker X maps X identically: contradictions must be 0).

Thresholds from measured real data: unrelated sentences <= 0.60 similarity
(median 0.31); same sentence with our whisper errors 0.83-0.92.

  python ../canonical_remap_v2.py --fps25_root dataset/MEAD/FPS25 --audios audios/MEAD
  python ../canonical_remap_v2.py --selftest
"""

import argparse, csv, glob, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from align_mead_levels import load_transcripts, sim, classify  # noqa: E402

E2V = "emotion2vec+large_features"
# NEARDUP_SIM: clip must clearly belong to the near-duplicate PAIR (unrelated max 0.60,
# same-sentence with real whisper errors observed down to 0.83); which member = assignment
SAME, WEAK_SIM, WEAK_MARGIN, NEARDUP, NEARDUP_SIM = 0.65, 0.65, 0.10, 0.80, 0.75
MAJ_FRAC = 0.5          # canonical text needs support > MAJ_FRAC of available instances
QBASE = 900


def medoid(texts):
    texts = [t for t in texts if t]
    return max(texts, key=lambda t: sum(sim(t, u) for u in texts)) if texts else None


# --------------------------------------------------------- majority table ----

def majority_text(texts):
    """Largest agreeing cluster (pairwise sim >= SAME to a pivot). Returns
       (canonical_text, support, available)."""
    texts = [t for t in texts if t]
    if not texts:
        return None, 0, 0
    best_pivot, best_members = None, []
    for t in texts:
        members = [u for u in texts if sim(t, u) >= SAME]
        if len(members) > len(best_members):
            best_pivot, best_members = t, members
    return medoid(best_members), len(best_members), len(texts)

COMMON = [1, 2, 3]
GENERIC = list(range(21, 31))
ANCHOR_POS = COMMON + GENERIC

def expected_neutral(c):
    return c if c <= 3 else c + 10

def neutral_by_id(T, ids):
    return {i: T.get((i, "neutral", "level_1"), {}) for i in ids}

def neutral_twins(T, ids):
    """Near-duplicate neutral sentence pairs, decided on the cross-speaker CONSENSUS text
       (a single speaker's noisy transcripts can hide the relation)."""
    cons = {c: medoid([T.get((i, "neutral", "level_1"), {}).get(c) for i in ids]) for c in range(1, 41)}
    ks = [c for c in cons if cons[c]]
    return {frozenset((a, b)) for x, a in enumerate(ks) for b in ks[x + 1:] if sim(cons[a], cons[b]) >= NEARDUP}

def reads_as(text, neu_i, n, twins=frozenset()):
    """True if `text` is closest to neutral sentence n among this speaker's 40, or to
       n's near-duplicate twin (e.g. '... will help ...' / '... would help ...'), which
       transcripts cannot reliably tell apart."""
    cands = {k: v for k, v in neu_i.items() if v}
    if not cands or n not in cands:
        return False
    best = max(cands, key=lambda k: sim(text, cands[k]))
    return best == n or frozenset((best, n)) in twins

def trusted_folders(T, ids):
    nb = neutral_by_id(T, ids)
    twins = neutral_twins(T, ids)
    trusted, score = set(), {}
    for (i, e, l), clips in T.items():
        if e == "neutral" or i not in ids:
            continue
        pos = [c for c in ANCHOR_POS if c in clips]
        ok = sum(reads_as(clips[c], nb[i], expected_neutral(c), twins) for c in pos)
        score[f"{i}/{e}/{l}"] = f"{ok}/{len(pos)}"
        if len(pos) >= 10 and ok >= len(pos) - 1:
            trusted.add((i, e, l))
    return trusted, score

def build_table(T, exclude_id=None):
    ids = sorted({i for (i, _, _) in T} - {exclude_id})
    emos = sorted({e for (_, e, _) in T if e != "neutral"})
    neu = {c: medoid([T.get((i, "neutral", "level_1"), {}).get(c) for i in ids]) for c in range(1, 41)}
    trusted, score = trusted_folders(T, ids)
    table, info = {}, {}
    for e in emos:
        table[e], info[e] = {}, {}
        for c in COMMON:
            if neu.get(c):
                table[e][c] = neu[c]; info[e][c] = {"source": "neutral", "support": None}
        for c in GENERIC:
            if neu.get(c + 10):
                table[e][c] = neu[c + 10]; info[e][c] = {"source": "neutral+10", "support": None}
        tf = sorted(k for k in trusted if k[1] == e)
        for c in range(4, 21):
            inst = [T[k].get(c) for k in tf]
            txt, sup, avail = majority_text(inst)
            info[e][c] = {"source": "trusted_majority", "support": sup, "available": avail,
                          "trusted_folders": len(tf)}
            if txt and (sup > MAJ_FRAC * avail) and (sup >= 2 or avail == 1):
                table[e][c] = txt
    return table, info, neu, trusted, score


def near_duplicates(table):
    out = []
    for e, t in table.items():
        ks = sorted(t)
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                if sim(t[ks[a]], t[ks[b]]) >= NEARDUP:
                    out.append((e, ks[a], ks[b]))
    return out


# -------------------------------------------------------------- mapping ------

def map_folder(clips, ref, dups):
    """Return {ours: (canon or None, sim, margin, status)} with quarantine for
       unresolvable clips. dups: set of frozenset({a,b}) near-duplicate pairs."""
    from scipy.optimize import linear_sum_assignment
    rows, cols = sorted(clips), sorted(ref)
    S = [[sim(clips[r], ref[c]) for c in cols] for r in rows]
    ri, ci = linear_sum_assignment([[1.0 - x for x in line] for line in S])
    out = {}
    for i, j in zip(ri, ci):
        others = [k for k in range(len(cols)) if k != j]
        ru = max(others, key=lambda k: S[i][k]) if others else None
        s_ = S[i][j]
        margin = s_ - (S[i][ru] if ru is not None else 0.0)       # >0 means assigned is the best
        if s_ >= WEAK_SIM and margin >= WEAK_MARGIN:
            status = "ok"
        elif s_ >= NEARDUP_SIM and ru is not None and frozenset({cols[j], cols[ru]}) in dups:
            status = "ok_near_duplicate"
        else:
            status = "quarantine"
        out[rows[i]] = (cols[j], round(s_, 3), round(margin, 3), status)
    for r in rows:                                    # clips left unassigned (more clips than table)
        if r not in out:
            out[r] = (None, 0.0, 0.0, "quarantine")
    return out

def finalize(res):
    """Turn assignment into a rename map; quarantined clips get 9NN names."""
    mp, q = {}, []
    used = {c for (c, _, _, st) in res.values() if st != "quarantine" and c is not None}
    nxt = QBASE + 1
    for m in sorted(res):
        c, s, mg, st = res[m]
        if st == "quarantine":
            if m > QBASE:                            # already quarantined earlier: keep its name
                mp[f"{m:03d}"] = f"{m:03d}"
            else:
                while nxt in used or nxt in res:
                    nxt += 1
                mp[f"{m:03d}"] = f"{nxt:03d}"; used.add(nxt); nxt += 1
            q.append(m)
        else:
            mp[f"{m:03d}"] = f"{c:03d}"
    return mp, q

def map_all(T, table, dups, only_id=None):
    groups, quarantined, flags = {}, [], []
    for (i, e, l), clips in sorted(T.items()):
        if e == "neutral" or not clips or (only_id and i != only_id):
            continue
        res = map_folder(clips, table.get(e, {}), dups)
        mp, q = finalize(res)
        key = f"{i}/{e}/{l}"
        verdict = "identity" if all(k == v for k, v in mp.items()) else (
            "quarantine_only" if all(k == v or int(v) > QBASE for k, v in mp.items()) else
            classify({int(k): (int(v), 0, 0) for k, v in mp.items() if int(v) <= 30 and int(k) <= 30}) )
        sims = [s for (_, s, _, st) in res.values() if st != "quarantine"]
        groups[key] = {"verdict": verdict, "n": len(mp), "map": mp,
                       "min_sim": round(min(sims), 3) if sims else None,
                       "quarantined": [f"{m:03d}" for m in q]}
        for m in q:
            c, s, mg, st = res[m]
            quarantined.append({"group": key, "ours": f"{m:03d}", "text": clips[m],
                                "closest": f"{c:03d}" if c else None, "sim": s})
        for m, (c, s, mg, st) in res.items():
            if st == "ok_near_duplicate":
                flags.append({"group": key, "ours": f"{m:03d}", "to": f"{c:03d}", "sim": s, "note": "near-duplicate script pair"})
    return groups, quarantined, flags

def canon_of(groups, i, e, l, m):
    if e == "neutral":
        return m
    g = groups.get(f"{i}/{e}/{l}")
    v = g["map"].get(f"{m:03d}") if g else None
    return int(v) if v and int(v) <= 30 else None


# ---------------------------------------------------------- validations -----

def find_anchors(fps25_root, audios_root, np):
    L = lambda p: np.load(p).astype("float64").ravel()
    cs = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    out = []
    for a in sorted(glob.glob(os.path.join(audios_root, "*", E2V, "*_level_*_*.npy"))):
        e = a.split(os.sep)[-3]
        p = os.path.basename(a)[:-4].split("_")
        if len(p) != 4:
            continue
        i, l, n = p[0], f"{p[1]}_{p[2]}", int(p[3])
        cands = glob.glob(os.path.join(fps25_root, i, "front", e, l, E2V, "*.npy"))
        if not cands:
            continue
        x = L(a)
        c, m = max((cs(x, L(q)), int(os.path.basename(q)[:-4])) for q in cands)
        if c >= 0.99:
            out.append((i, e, l, m, n))
    return out

def validation_a(anchors, groups):
    agree, bad, sa, sb = 0, [], 0, 0
    for (i, e, l, m, n) in anchors:
        c = canon_of(groups, i, e, l, m)
        spec = 4 <= n <= 20
        if c == n:
            agree += 1; sa += spec
        else:
            sb += spec
            bad.append(f"{e}/{i}/{l}: authors {n:03d} = ours {m:03d} -> mapped {c}")
    return {"agree": agree, "disagree": len(bad), "specific_004_020": {"agree": sa, "disagree": sb},
            "examples": bad[:12]}

def validation_b_folders(T, groups):
    """Every clip remapped to a common/generic number must read as its speaker's
       expected neutral sentence (argmax over that speaker's 40)."""
    ids = sorted({i for (i, _, _) in T})
    nb = neutral_by_id(T, ids)
    twins = neutral_twins(T, ids)
    checked, bad = 0, []
    for (i, e, l), clips in T.items():
        if e == "neutral":
            continue
        for m, t in clips.items():
            c = canon_of(groups, i, e, l, m)
            if c in ANCHOR_POS:
                checked += 1
                if not reads_as(t, nb[i], expected_neutral(c), twins):
                    bad.append(f"{i}/{e}/{l}/{m:03d} -> {c:03d} does not read as neutral {expected_neutral(c):03d}")
    return {"checked": checked, "misassigned": len(bad), "examples": bad[:10]}


def validation_c(test_csv, T, groups, neu):
    neu = {k: v for k, v in neu.items() if v}
    twins = neutral_twins(T, sorted({i for (i, _, _) in T}))
    per = defaultdict(lambda: {"before_ok": 0, "before_bad": 0, "after_ok": 0, "after_bad": 0, "missing_after": 0})
    tot = {"before": {"ok": 0, "bad": 0, "missing": 0}, "after": {"ok": 0, "bad": 0, "missing": 0}}
    for r in csv.DictReader(open(test_csv, newline="")):
        s = r["source_video_path"].replace("\\", "/").split("/")
        g = r["gt_video_path"].replace("\\", "/").split("/")
        i, sn = s[-5], int(s[-1][:-4])
        e, l, k = g[-3], g[-2], int(g[-1][:-4])
        inv = {int(v): int(m) for m, v in (groups.get(f"{i}/{e}/{l}", {}).get("map") or {}).items() if int(v) <= 30}
        folder = T.get((i, e, l), {})
        for tag, gt in (("before", folder.get(k)), ("after", folder.get(inv[k]) if k in inv else None)):
            if not gt or sn not in neu:
                tot[tag]["missing"] += 1
                if tag == "after":
                    per[k]["missing_after"] += 1
                continue
            best = max(neu, key=lambda c: sim(gt, neu[c]))
            ok = best == sn or frozenset((best, sn)) in twins
            tot[tag]["ok" if ok else "bad"] += 1
            per[k][f"{tag}_{'ok' if ok else 'bad'}"] += 1
    return {**tot, "by_gt_number": {f"{k:03d}": v for k, v in sorted(per.items())}}

def validation_loso(T, groups, dups_full):
    """Table from 3 speakers maps the 4th. contradiction = a DIFFERENT valid number
       (real danger); unresolved = the reduced table could not decide (data too thin)."""
    agree, contra, unres, bad = 0, 0, 0, []
    for x in sorted({i for (i, _, _) in T}):
        table, _, _, _, _ = build_table(T, exclude_id=x)
        g2, _, _ = map_all(T, table, set(frozenset(p[1:]) for p in near_duplicates(table)), only_id=x)
        for k, g in g2.items():
            for m, v in g["map"].items():
                full = groups[k]["map"][m]
                fq, lq = int(full) > QBASE, int(v) > QBASE
                if full == v or (fq and lq):
                    agree += 1
                elif lq or fq:
                    unres += 1
                else:
                    contra += 1
                    bad.append(f"{k}:{m} full->{full} loso->{v}")
    return {"agree": agree, "contradictions": contra, "unresolved": unres,
            "disagree": contra, "examples": bad[:10]}


def run(fps25_root, audios_root, meta, np, log=print):
    T = load_transcripts(os.path.join(meta, "transcripts_all.csv"))
    table, info, neu, trusted, tscore = build_table(T)
    dups_list = near_duplicates(table)
    dups = set(frozenset((a, b)) for (_, a, b) in dups_list)
    groups, quarantined, flags = map_all(T, table, dups)
    for (i, e, l), clips in T.items():
        if e == "neutral":
            groups[f"{i}/{e}/{l}"] = {"verdict": "identity", "n": len(clips),
                                      "map": {f"{m:03d}": f"{m:03d}" for m in sorted(clips)}, "quarantined": []}
    gaps = {e: [c for c in range(1, 31) if c not in t] for e, t in table.items()}
    anchors = find_anchors(fps25_root, audios_root, np) if audios_root else []
    va = validation_a(anchors, groups)
    vb = validation_b_folders(T, groups)
    vc = validation_c(os.path.join(meta, "test.csv"), T, groups, neu)
    vl = validation_loso(T, groups, dups)
    verdicts = defaultdict(int)
    for g in groups.values():
        verdicts["cyclic" if g["verdict"].startswith("cyclic") else g["verdict"]] += 1
    rep = {"method": "neutral_anchored_trusted_majority_v4", "table_gaps": gaps,
           "trusted_folders": sorted("/".join(k) for k in trusted), "trust_scores": tscore,
           "table_weak_support": {e: {str(c): v for c, v in d.items()
                                      if v.get("source") == "trusted_majority"
                                      and (v.get("available", 0) == 0 or v["support"] <= 0.75 * v["available"])}
                                  for e, d in info.items()},
           "near_duplicates": [f"{e} {a:03d}~{b:03d}" for (e, a, b) in dups_list],
           "groups": groups, "quarantined": quarantined, "near_duplicate_assignments": flags,
           "weak_matches": [], "summary": dict(verdicts), "anchors": len(anchors),
           "validation_a_authors_e2v": va, "validation_b_common_sentences": vb,
           "validation_c_test_pairs": vc, "validation_loso": vl,
           "table_missing_numbers": gaps}
    out = os.path.join(meta, "canonical_remap.json")
    json.dump(rep, open(out, "w"), indent=1)
    log(f"[canon4] trusted folders: {len(trusted)} of {len(tscore)} | per emotion: "
        + str({e: sum(1 for k in trusted if k[1] == e) for e in sorted(table)}))
    log(f"[canon4] table gaps: {({e: v for e, v in gaps.items() if v}) or 'none'}")
    log(f"[canon4] near-duplicate script pairs: {rep['near_duplicates'] or 'none'}")
    log(f"[canon4] folders: {dict(verdicts)} | quarantined clips: {len(quarantined)} | near-dup assignments: {len(flags)}")
    for k in sorted(k for k, g in groups.items() if g["verdict"] != "identity"):
        log(f"   {k}: {groups[k]['verdict']} min_sim {groups[k]['min_sim']} quarantined {groups[k]['quarantined']}")
    for q in quarantined:
        log(f"   QUARANTINE {q['group']}:{q['ours']} (closest {q['closest']} sim {q['sim']}): {q['text']}")
    log(f"[canon4] A authors e2v anchors: {va}")
    log(f"[canon4] B script vs neutral (independent): {vb}")
    log(f"[canon4] C test pairs: before {vc['before']} after {vc['after']}")
    log(f"[canon4] C by gt number: {vc['by_gt_number']}")
    log(f"[canon4] LOSO stability: {vl}")
    log(f"[canon4] wrote {out}")
    return rep


# ------------------------------------------------------------ self-test -----

def _scenario(td):
    """Synthetic MEAD mirroring the measured real failure modes. Returns (root, meta, aud, truth, OUT)."""
    import random, zlib
    import numpy as np
    vr = random.Random(11)
    W = sorted({"".join(vr.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(vr.randint(4, 9))) for _ in range(700)})[:600]
    rs = lambda seed: " ".join(random.Random(seed).sample(W, 8))
    nr = random.Random(5)
    noisy = lambda t: " ".join(w if nr.random() > 0.04 else nr.choice(W) for w in t.split())
    IDS, EMOS, LV = ["M003", "M030", "W009", "W015"], ["angry", "contempt", "happy"], ["level_1", "level_2", "level_3"]
    NEU = {c: rs(1000 + c) for c in range(1, 41)}
    NEU[34] = NEU[33].rsplit(" ", 1)[0] + " " + NEU[33].split()[-1][::-1]      # generic near-duplicate 023~024
    SPEC = {e: {c: rs(zlib.crc32(f"{e}{c}".encode())) for c in range(4, 21)} for e in EMOS}
    ctext = lambda e, c: NEU[c] if c <= 3 else (NEU[c + 10] if c >= 21 else SPEC[e][c])
    truth = {}
    for i in IDS:
        for e in EMOS:
            for l in LV:
                f = {m: m for m in range(1, 31)}
                if e == "angry" and i != "M003":           # MAJORITY shifted from 015 on, M003 correct
                    f.update({m: m + 1 for m in range(15, 30)}); f[30] = 15
                if e == "contempt" and i == "M003":        # generic region shifted in one speaker
                    f.update({m: m + 1 for m in range(20, 30)}); f[30] = 20
                if (i, e, l) == ("W009", "happy", "level_2"):
                    f[5], f[6] = 6, 5                      # local swap inside a trusted folder
                if (i, e, l) == ("W009", "contempt", "level_1"):
                    f = {m: c for m, c in f.items() if m not in (27, 28, 29, 30)}
                truth[(i, e, l)] = f
    OUT = ("W015", "contempt", "level_3", 24)
    root = os.path.join(td, "dataset", "MEAD", "FPS25"); meta = os.path.dirname(root)
    aud = os.path.join(td, "audios", "MEAD"); os.makedirs(root)
    rows = []
    vec = lambda k, c: np.random.default_rng(zlib.crc32(f"{k}{c}".encode())).normal(size=16)
    for i in IDS:
        for c in range(1, 41):
            rows.append((f"{i}/front/neutral/level_1/{c:03d}.wav", noisy(NEU[c])))
        for e in EMOS:
            for l in LV:
                for m, c in truth[(i, e, l)].items():
                    t = NEU[15] if (i, e, l, m) == OUT else ctext(e, c)
                    rows.append((f"{i}/front/{e}/{l}/{m:03d}.wav", noisy(t)))
                    p = os.path.join(root, i, "front", e, l, E2V, f"{m:03d}.npy")
                    os.makedirs(os.path.dirname(p), exist_ok=True); np.save(p, vec(e + i + l, c))
                    open(os.path.join(root, i, "front", e, l, f"{m:03d}.mp4"), "w").close()
                    if l == "level_3" and c % 4 == IDS.index(i) and (i, e, l, m) != OUT:
                        q = os.path.join(aud, e, E2V, f"{i}_{l}_{c:03d}.npy")
                        os.makedirs(os.path.dirname(q), exist_ok=True); np.save(q, vec(e + i + l, c) * 1.1)
    with open(os.path.join(meta, "transcripts_all.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["relpath", "transcript"]); w.writerows(rows)
    with open(os.path.join(meta, "test.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["source_video_path", "gt_video_path", "gt_emotion", "intensity"])
        for i in IDS:
            for e in EMOS:
                for l in LV:
                    for k in [1, 2, 3] + list(range(20, 31)):
                        sn = k if k <= 3 else k + 10
                        w.writerow([f"./dataset/MEAD/FPS25/{i}/front/neutral/level_1/{sn:03d}.mp4",
                                    f"./dataset/MEAD/FPS25/{i}/front/{e}/{l}/{k:03d}.mp4", e, l])
    return root, meta, aud, truth, OUT

def selftest():
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        root, meta, aud, truth, OUT = _scenario(td)
        rep = run(root, aud, meta, np, log=lambda *_: None)
        G = rep["groups"]
        tr = set(rep["trusted_folders"])
        # T1 trust: shifted folders cast no vote, the correct minority speaker is trusted
        assert all(f"{i}/angry/{l}" not in tr for i in ("M030", "W009", "W015") for l in ("level_1", "level_2", "level_3")), tr
        assert all(f"M003/angry/{l}" in tr for l in ("level_1", "level_2", "level_3")), tr
        assert "M003/contempt/level_1" not in tr and "W009/happy/level_2" in tr
        # T2 reference complete
        assert all(not v for v in rep["table_gaps"].values()), rep["table_gaps"]
        # T3 exact recovery of every folder (the out-of-script clip is quarantined instead)
        for (i, e, l), f in truth.items():
            got = {int(m): int(c) for m, c in G[f"{i}/{e}/{l}"]["map"].items()}
            exp = dict(f)
            if (i, e, l) == OUT[:3]:
                assert got[OUT[3]] > QBASE, got[OUT[3]]; got.pop(OUT[3]); exp.pop(OUT[3])
            diff = {k: (got.get(k), exp.get(k)) for k in exp if got.get(k) != exp.get(k)}
            assert not diff, (i, e, l, diff)
        # T4 the v3 failure case: the CORRECT minority folder is left untouched
        assert all(G[f"M003/angry/{l}"]["verdict"] == "identity" for l in ("level_1", "level_2", "level_3"))
        assert any(q["group"] == "W015/contempt/level_3" and q["ours"] == "024" for q in rep["quarantined"])
        va, vb, vc, vl = (rep[k] for k in ("validation_a_authors_e2v", "validation_b_common_sentences",
                                           "validation_c_test_pairs", "validation_loso"))
        assert va["disagree"] == 0 and va["specific_004_020"]["agree"] > 20, va          # T5
        assert vb["misassigned"] == 0 and vb["checked"] > 400, vb                        # T6
        bad = {k for k, v in vc["by_gt_number"].items() if v["after_bad"]}              # T7
        assert bad == {"020"} and vc["by_gt_number"]["020"]["after_bad"] == 36, (bad, vc["by_gt_number"].get("020"))
        assert vc["before"]["bad"] > 36
        assert vl["contradictions"] == 0, vl                                             # T8
        import overnight_gate2_v2 as O                                                   # T9
        gate = O.decide_gate(rep, root, 20)
        assert gate["passed"], gate["reasons"]
        # T10 regression: the previous algorithm (canonical_remap_v1, plain majority) FAILS here
        import canonical_remap_v1 as OLD
        old = OLD.run(root, aud, meta, np, log=lambda *_: None)
        assert any(old["groups"][f"M003/angry/{l}"]["verdict"] != "identity" for l in ("level_1", "level_2", "level_3")), \
            "test would not detect the plain-majority failure"
        mp, q = finalize({901: (None, 0.2, 0.0, "quarantine"), 1: (1, 0.9, 0.5, "ok")})    # T11
        assert mp == {"901": "901", "001": "001"} and q == [901]
    print("selftest OK: trust test excludes majority-shifted folders and keeps the correct minority, "
          "complete neutral-anchored reference, exact recovery (majority shift, generic shift, local swap, "
          "missing clips), out-of-script quarantine, anchors agree (incl. 004-020), per-clip neutral check, "
          "only the inherent gt-020 pairs mismatch, LOSO no contradictions, gate passes, "
          "previous plain-majority algorithm demonstrably fails the same data")
    return True


def main():
    ap = argparse.ArgumentParser(description="Majority canonical-table MEAD renumbering (analysis only)")
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
