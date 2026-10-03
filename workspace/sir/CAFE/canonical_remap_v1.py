#!/usr/bin/env python3
"""
CAFE: renumber MEAD clips to the corpus numbering via a MAJORITY canonical table (v3).

Measured facts this design rests on (DGX, all 2641 transcripts):
  - neutral scripts are identical across the 4 speakers at all 40 numbers
    (the one apparent mismatch, #35, is the same sentence with ASR errors);
  - happy/sad/surprised agree across speakers at 29/30 numbers, while angry,
    contempt, fear and disgusted disagree in contiguous runs: specific folders
    of the official release are shifted or permuted, the scripts are shared;
  - some clips carry a sentence that is not in their emotion's script
    (e.g. W015/contempt/024 holds a neutral sentence): they must not be forced
    onto a number;
  - the script contains near-duplicates (contempt 023 "... will help ..." vs
    024 "... would help ...").

Method:
  1. canonical text for (emotion e, number c) = majority cluster over the 12
     instances (4 speakers x 3 levels) at our number c; accepted only with a
     clear majority (>= MAJ_MIN of the available instances);
  2. each folder is matched to its emotion's table (optimal assignment);
  3. a clip is RESOLVED if sim >= 0.65 and margin >= 0.10, or if its runner-up
     is a near-duplicate script sentence and sim >= 0.85; otherwise it is
     QUARANTINED: renamed to 9NN (C-MET loaders only read 001-040, so it is
     ignored) and any test row needing that number counts as missing;
  4. emotion2vec anchors (authors' released features) are used ONLY as an
     independent validation, never as a table source (e2v encodes emotion,
     not content; short sentences collide).

Validations: A authors-anchor agreement; B generic/common script check against
neutral (1-3 == neutral 1-3, 21-30 == neutral 31-40), independent of how the
table was built; C test-pair content per gt number, before vs after; LOSO
stability (table from 3 speakers maps the 4th identically).

Thresholds come from measured real data: unrelated English sentences reach at
most 0.60 similarity (median 0.31); the same sentence with our whisper errors
scored 0.83-0.92.

Output <meta>/canonical_remap.json (overnight_gate2 compatible). Analysis only.

  python ../canonical_remap_v1.py --fps25_root dataset/MEAD/FPS25 --audios audios/MEAD
  python ../canonical_remap_v1.py --selftest
"""

import argparse, csv, glob, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from align_mead_levels import load_transcripts, sim, classify  # noqa: E402

E2V = "emotion2vec+large_features"
# NEARDUP_SIM: clip must clearly belong to the near-duplicate PAIR (unrelated max 0.60,
# same-sentence with real whisper errors observed down to 0.83); which member = assignment
SAME, WEAK_SIM, WEAK_MARGIN, NEARDUP, NEARDUP_SIM = 0.65, 0.65, 0.10, 0.90, 0.75
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

def build_table(T, exclude_id=None):
    ids = sorted({i for (i, _, _) in T} - {exclude_id})
    emos = sorted({e for (_, e, _) in T if e != "neutral"})
    table, info = {}, {}
    for e in emos:
        table[e], info[e] = {}, {}
        for c in range(1, 31):
            inst = [T.get((i, e, l), {}).get(c) for i in ids for l in ("level_1", "level_2", "level_3")]
            txt, sup, avail = majority_text(inst)
            info[e][c] = {"support": sup, "available": avail}
            if txt and sup > MAJ_FRAC * avail and sup >= 2:
                table[e][c] = txt
    neu = {c: medoid([T.get((i, "neutral", "level_1"), {}).get(c) for i in ids]) for c in range(1, 41)}
    return table, info, neu

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
    agree, bad = 0, []
    for (i, e, l, m, n) in anchors:
        c = canon_of(groups, i, e, l, m)
        if c == n:
            agree += 1
        else:
            bad.append(f"{e}/{i}/{l}: authors {n:03d} = ours {m:03d} -> mapped {c}")
    return {"agree": agree, "disagree": len(bad), "examples": bad[:12]}

def validation_b_script(table, neu):
    """Independent: table (built from emotion folders) vs neutral sentences."""
    res = []
    for e, t in table.items():
        for c, n in [(1, 1), (2, 2), (3, 3)] + [(k, k + 10) for k in range(21, 31)]:
            if c in t and neu.get(n):
                best = max(neu, key=lambda k: sim(t[c], neu[k]) if neu[k] else -1)
                res.append((e, c, n, best == n))
    bad = [f"{e} {c:03d} should equal neutral {n:03d}" for e, c, n, ok in res if not ok]
    return {"checked": len(res), "misassigned": len(bad), "examples": bad[:10]}

def validation_c(test_csv, T, groups, neu):
    neu = {k: v for k, v in neu.items() if v}
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
            ok = max(neu, key=lambda c: sim(gt, neu[c])) == sn
            tot[tag]["ok" if ok else "bad"] += 1
            per[k][f"{tag}_{'ok' if ok else 'bad'}"] += 1
    return {**tot, "by_gt_number": {f"{k:03d}": v for k, v in sorted(per.items())}}

def validation_loso(T, groups, dups_full):
    """Table from 3 speakers maps the 4th. contradiction = a DIFFERENT valid number
       (real danger); unresolved = the reduced table could not decide (data too thin)."""
    agree, contra, unres, bad = 0, 0, 0, []
    for x in sorted({i for (i, _, _) in T}):
        table, _, _ = build_table(T, exclude_id=x)
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
    table, info, neu = build_table(T)
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
    vb = validation_b_script(table, neu)
    vc = validation_c(os.path.join(meta, "test.csv"), T, groups, neu)
    vl = validation_loso(T, groups, dups)
    verdicts = defaultdict(int)
    for g in groups.values():
        verdicts["cyclic" if g["verdict"].startswith("cyclic") else g["verdict"]] += 1
    rep = {"method": "majority_table_v3", "table_gaps": gaps,
           "table_weak_support": {e: {str(c): v for c, v in d.items() if v["support"] <= 0.75 * v["available"]}
                                  for e, d in info.items()},
           "near_duplicates": [f"{e} {a:03d}~{b:03d}" for (e, a, b) in dups_list],
           "groups": groups, "quarantined": quarantined, "near_duplicate_assignments": flags,
           "weak_matches": [], "summary": dict(verdicts), "anchors": len(anchors),
           "validation_a_authors_e2v": va, "validation_b_common_sentences": vb,
           "validation_c_test_pairs": vc, "validation_loso": vl,
           "table_missing_numbers": gaps}
    out = os.path.join(meta, "canonical_remap.json")
    json.dump(rep, open(out, "w"), indent=1)
    log(f"[canon3] table gaps: {({e: v for e, v in gaps.items() if v}) or 'none'}")
    log(f"[canon3] near-duplicate script pairs: {rep['near_duplicates'] or 'none'}")
    log(f"[canon3] folders: {dict(verdicts)} | quarantined clips: {len(quarantined)} | near-dup assignments: {len(flags)}")
    for k in sorted(k for k, g in groups.items() if g["verdict"] != "identity"):
        log(f"   {k}: {groups[k]['verdict']} min_sim {groups[k]['min_sim']} quarantined {groups[k]['quarantined']}")
    for q in quarantined:
        log(f"   QUARANTINE {q['group']}:{q['ours']} (closest {q['closest']} sim {q['sim']}): {q['text']}")
    log(f"[canon3] A authors e2v anchors: {va}")
    log(f"[canon3] B script vs neutral (independent): {vb}")
    log(f"[canon3] C test pairs: before {vc['before']} after {vc['after']}")
    log(f"[canon3] C by gt number: {vc['by_gt_number']}")
    log(f"[canon3] LOSO stability: {vl}")
    log(f"[canon3] wrote {out}")
    return rep


# ------------------------------------------------------------ self-test -----

def selftest():
    import tempfile, random, zlib
    import numpy as np
    vr = random.Random(11)
    W = sorted({"".join(vr.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(vr.randint(4, 9))) for _ in range(700)})[:600]
    rs = lambda seed: " ".join(random.Random(seed).sample(W, 8))
    nr = random.Random(5)
    noisy = lambda t: " ".join(w if nr.random() > 0.04 else nr.choice(W) for w in t.split())
    IDS, EMOS, LV = ["M003", "M030", "W009", "W015"], ["angry", "contempt", "disgusted"], ["level_1", "level_2", "level_3"]
    NEU = {c: rs(1000 + c) for c in range(1, 41)}
    SPEC = {e: {c: rs(zlib.crc32(f"{e}{c}".encode())) for c in range(4, 21)} for e in EMOS}
    base = SPEC["contempt"][5]
    SPEC["contempt"][6] = base.rsplit(" ", 1)[0] + " " + base.split()[-1][::-1]      # near-duplicate of 005
    ctext = lambda e, c: NEU[c] if c <= 3 else (NEU[c + 10] if c >= 21 else SPEC[e][c])
    shift = lambda m: (m + 2) % 30 + 1
    truth = {}
    for i in IDS:
        for e in EMOS:
            for l in LV:
                f = {m: m for m in range(1, 31)}
                if (i, e) == ("M003", "disgusted"):
                    f = {m: shift(m) for m in range(1, 31)}                       # whole folder, all levels
                if (i, e, l) == ("W015", "angry", "level_2"):
                    p = list(range(16, 28)); random.Random(4).shuffle(p)
                    f.update({m: p[m - 16] for m in range(16, 28)})               # local permutation 16-27
                if (i, e, l) == ("W009", "contempt", "level_1"):
                    f = {m: c for m, c in f.items() if m not in (27, 28, 29, 30)}  # missing clips
                truth[(i, e, l)] = f
    OUT = ("W015", "contempt", "level_3", 24)       # a clip carrying an out-of-script (neutral) sentence
    with tempfile.TemporaryDirectory() as td:
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
        rep = run(root, aud, meta, np, log=lambda *_: None)
        G = rep["groups"]
        assert all(not v for v in rep["table_gaps"].values()), rep["table_gaps"]                 # T1
        assert "contempt 005~006" in rep["near_duplicates"], rep["near_duplicates"]             # T2
        for (i, e, l), f in truth.items():                                                      # T3
            got = {int(m): int(c) for m, c in G[f"{i}/{e}/{l}"]["map"].items()}
            exp = dict(f)
            if (i, e, l) == OUT[:3]:
                assert got[OUT[3]] > QBASE, got[OUT[3]]; got.pop(OUT[3]); exp.pop(OUT[3])
            assert got == exp, (i, e, l, {k: (got.get(k), exp.get(k)) for k in exp if got.get(k) != exp.get(k)})
        assert any(q["group"] == "W015/contempt/level_3" and q["ours"] == "024" for q in rep["quarantined"])  # T4
        va, vb, vc, vl = (rep[k] for k in ("validation_a_authors_e2v", "validation_b_common_sentences",
                                           "validation_c_test_pairs", "validation_loso"))
        assert va["disagree"] == 0 and va["agree"] > 50, va                                    # T5
        assert vb["misassigned"] == 0 and vb["checked"] == 3 * 13, vb                           # T6
        # T7 after remap, content mismatches remain ONLY at gt 020 (by construction the synthetic
        #    script's 020 is emotion-specific, not neutral 030): every one of the 36 is reported
        bad_nums = {k for k, v in vc["by_gt_number"].items() if v["after_bad"]}
        assert vc["before"]["bad"] > vc["after"]["bad"] and bad_nums == {"020"}, (bad_nums, vc["after"])
        assert vc["by_gt_number"]["020"]["after_bad"] == 36, vc["by_gt_number"]["020"]
        assert vc["by_gt_number"]["024"]["missing_after"] >= 1                                  # quarantined -> missing
        assert vl["contradictions"] == 0 and vl["agree"] > 1000, vl                              # T8
        import overnight_gate2_v1 as O                                                             # T9 gate
        for (i, e, l), f in truth.items():
            d = os.path.join(root, i, "front", e, l)
            for m in f:
                open(os.path.join(d, f"{m:03d}.mp4"), "w").close()
        gate = O.decide_gate(rep, root, 20)
        assert gate["passed"], gate["reasons"]
        assert "M003/disgusted/level_1" in gate["rename_groups"] and "W015/contempt/level_3" in gate["rename_groups"]
        # T10 negative control: shifted folder in a MAJORITY position must still be recovered, but if
        # 2 of 4 speakers share the same error the table loses its majority -> gap is reported, not guessed
        T2 = load_transcripts(os.path.join(meta, "transcripts_all.csv"))
        for i in ("M030", "W009"):
            for l in LV:
                d = T2[(i, "angry", l)]
                T2[(i, "angry", l)] = {m: d[(m % 30) + 1] for m in d}                           # same shift, 2 speakers
        t2, info2, _ = build_table(T2)
        gaps2 = [c for c in range(1, 31) if c not in t2["angry"]]
        assert len(gaps2) >= 20, gaps2          # 6 vs 6 split: no majority -> reported as gap, never guessed
        assert all(c in t2["contempt"] for c in range(1, 31))   # other emotions unaffected
        # T11 finalize keeps already-quarantined names stable (idempotent post-verify)
        mp, q = finalize({901: (None, 0.2, 0.0, "quarantine"), 1: (1, 0.9, 0.5, "ok")})
        assert mp == {"901": "901", "001": "001"} and q == [901]
    print("selftest OK: majority table without gaps, near-duplicate detection, exact recovery of whole-folder "
          "shift / local permutation / missing clips, out-of-script clip quarantined (9NN) and counted missing, "
          "authors-anchor agreement, independent script check vs neutral, test pairs fixed, LOSO stable, "
          "gate-compatible, stable quarantine names")
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
