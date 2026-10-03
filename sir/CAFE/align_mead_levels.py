#!/usr/bin/env python3
"""
CAFE: align MEAD sentence numbering across intensity levels using transcripts.

Problem (measured, not assumed): emotion2vec features of the authors' released
level_3 clips match OUR clip at a DIFFERENT number for several folders of the
official MEAD release (e.g. disgusted/W015 authors 008 -> ours 005, a cyclic
shift; surprised/M003 shows a non-cyclic permutation). C-MET's test.csv and
evaluation use the authors' numbering, so our tree must be renumbered first.

Method (analysis only, this script never renames files):
  1. per (ID, emotion): level_1 is the reference numbering; level_2 and level_3
     clips are matched to level_1 clips by transcript similarity using an
     optimal one-to-one assignment (Hungarian), which handles shifts, arbitrary
     permutations and missing clips;
  2. every match gets a similarity and a margin over the runner-up; weak
     matches are flagged;
  3. VALIDATION A: authors' level_3 e2v features give an independent
     ground truth (authors number N == our clip M when cosine >= 0.99); we
     report how often the transcript mapping sends M to N;
  4. VALIDATION B: common sentences 001-003 are identical across emotions, so
     after remapping their texts must agree across every emotion and level.

Output: <meta>/level_remap.json (full mapping + per-group verdicts) and a
printed report. Neutral (level_1 only) is reported for completeness.

Run from the C-MET repo root:
  python ../align_mead_levels.py --fps25_root dataset/MEAD/FPS25 --audios audios/MEAD
  python ../align_mead_levels.py --selftest
"""

import argparse, csv, difflib, glob, json, os, re, sys, zlib
from collections import defaultdict

LEVELS = ("level_1", "level_2", "level_3")
REF = "level_1"
SIM_WEAK, MARGIN_WEAK = 0.60, 0.10
FEAT_DIR = "emotion2vec+large_features"


def norm(t):
    t = re.sub(r"[^a-z0-9' ]+", " ", (t or "").lower())
    return re.sub(r"\s+", " ", t).strip()

def sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

def load_transcripts(csv_path):
    """{(ID, emotion, level): {num(int): normalized_text}} from relpath,transcript."""
    T = defaultdict(dict)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0 and row and row[0] == "relpath":
                continue
            if len(row) != 2:
                continue
            p = row[0].replace("\\", "/").split("/")
            if len(p) != 5 or p[1] != "front" or not p[4].endswith(".wav"):
                continue
            T[(p[0], p[2], p[3])][int(p[4][:-4])] = norm(row[1])
    return T

def assign(tgt, ref):
    """Optimal 1-1 match of tgt clips to ref clips by text similarity.
       Returns {tgt_num: (ref_num, sim, margin)}."""
    from scipy.optimize import linear_sum_assignment
    rows, cols = sorted(tgt), sorted(ref)
    S = [[sim(tgt[r], ref[c]) for c in cols] for r in rows]
    cost = [[1.0 - s for s in line] for line in S]
    ri, ci = linear_sum_assignment(cost)
    out = {}
    for i, j in zip(ri, ci):
        line = sorted(S[i], reverse=True)
        margin = line[0] - (line[1] if len(line) > 1 else 0.0)
        out[rows[i]] = (cols[j], S[i][j], margin)
    return out

def classify(mapping, n=30):
    pairs = [(m, r) for m, (r, _, _) in mapping.items()]
    if all(m == r for m, r in pairs):
        return "identity"
    shifts = {(r - m) % n for m, r in pairs}
    if len(shifts) == 1:
        k = shifts.pop()
        return f"cyclic_shift_{k if k <= n // 2 else k - n:+d}"
    return "permutation"

def analyze(T):
    groups, weak = {}, []
    ids_emos = sorted({(i, e) for (i, e, l) in T if e != "neutral"})
    for ident, emo in ids_emos:
        ref = T.get((ident, emo, REF))
        if not ref:
            continue
        for lvl in ("level_2", "level_3"):
            tgt = T.get((ident, emo, lvl))
            if not tgt:
                continue
            mp = assign(tgt, ref)
            key = f"{ident}/{emo}/{lvl}"
            sims = [s for _, s, _ in mp.values()]
            groups[key] = {
                "verdict": classify(mp),
                "n": len(mp),
                "mean_sim": round(sum(sims) / len(sims), 4),
                "min_sim": round(min(sims), 4),
                "map": {f"{m:03d}": f"{r:03d}" for m, (r, _, _) in sorted(mp.items())},
            }
            for m, (r, s, mg) in mp.items():
                if s < SIM_WEAK or mg < MARGIN_WEAK:
                    weak.append({"group": key, "ours": f"{m:03d}", "to": f"{r:03d}",
                                 "sim": round(s, 3), "margin": round(mg, 3)})
    return groups, weak

def canonical(groups, ident, emo, lvl, num):
    """Our file number -> reference (level_1) number."""
    if lvl == REF or emo == "neutral":
        return num
    g = groups.get(f"{ident}/{emo}/{lvl}")
    if not g:
        return num
    r = g["map"].get(f"{num:03d}")
    return int(r) if r else None

def validation_a(fps25_root, audios_root, groups, np):
    """Authors' e2v ground truth: authors N == our M (cos>=0.99) -> expect canonical(M)==N."""
    L = lambda p: np.load(p).astype("float64").ravel()
    cs = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    agree, disagree, uncertain = 0, [], 0
    for a in sorted(glob.glob(os.path.join(audios_root, "*", FEAT_DIR, "*_level_*_*.npy"))):
        emo = a.split(os.sep)[-3]
        p = os.path.basename(a)[:-4].split("_")
        if len(p) != 4:
            continue
        ident, lvl, n_auth = p[0], f"{p[1]}_{p[2]}", int(p[3])
        cands = glob.glob(os.path.join(fps25_root, ident, "front", emo, lvl, FEAT_DIR, "*.npy"))
        if not cands:
            continue
        x = L(a)
        best = max((cs(x, L(c)), int(os.path.basename(c)[:-4])) for c in cands)
        if best[0] < 0.99:
            uncertain += 1
            continue
        c = canonical(groups, ident, emo, lvl, best[1])
        if c == n_auth:
            agree += 1
        else:
            disagree.append(f"{emo}/{ident}/{lvl}: authors {n_auth:03d} = ours {best[1]:03d} "
                            f"-> transcript says {c}")
    return {"agree": agree, "disagree": len(disagree), "uncertain_e2v": uncertain,
            "disagree_examples": disagree[:10]}

def validation_b(T, groups):
    """Common sentences 001-003 are the same across emotions/levels per ID. After
       remapping, every clip assigned to canonical c must be closer to the consensus
       text of c than to the consensus of any other common sentence (relative test,
       robust to ASR noise; a wrong mapping lands closer to a different sentence)."""
    misassigned, checked, margins = [], 0, []
    for ident in sorted({i for (i, _, _) in T}):
        buckets = defaultdict(list)
        for (i, e, l), d in T.items():
            if i != ident:
                continue
            for num, txt in d.items():
                c = canonical(groups, i, e, l, num)
                if c in (1, 2, 3):
                    buckets[c].append((f"{e}/{l}/{num:03d}", txt))
        cons = {c: max(v, key=lambda t: sum(sim(t[1], u[1]) for u in v))[1]
                for c, v in buckets.items() if v}
        for c, items in buckets.items():
            others = [cons[o] for o in cons if o != c]
            for tag, txt in items:
                checked += 1
                own = sim(txt, cons[c])
                rival = max((sim(txt, o) for o in others), default=0.0)
                margins.append(own - rival)
                if own <= rival:
                    misassigned.append(f"{ident} {tag} assigned {c:03d} (own {own:.2f} <= rival {rival:.2f})")
    return {"checked": checked, "misassigned": len(misassigned),
            "min_margin": round(min(margins), 3) if margins else None,
            "examples": misassigned[:5]}


def run(fps25_root, audios_root, csv_path, out_path, np=None, log=print):
    T = load_transcripts(csv_path)
    groups, weak = analyze(T)
    rep = {"groups": groups, "weak_matches": weak,
           "summary": defaultdict(int)}
    for g in groups.values():
        rep["summary"][g["verdict"].split("_")[0] if g["verdict"].startswith("cyclic") else g["verdict"]] += 1
    rep["summary"] = dict(rep["summary"])
    rep["validation_b_common_sentences"] = validation_b(T, groups)
    if np is not None and audios_root:
        rep["validation_a_authors_e2v"] = validation_a(fps25_root, audios_root, groups, np)
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=1)
    nonid = {k: v["verdict"] for k, v in groups.items() if v["verdict"] != "identity"}
    log(f"[align] groups: {len(groups)} | verdicts: {rep['summary']} | weak matches: {len(weak)}")
    for k, v in sorted(nonid.items()):
        log(f"   {k}: {v} (mean_sim {groups[k]['mean_sim']}, min_sim {groups[k]['min_sim']})")
    log(f"[align] validation B (common sentences): {rep['validation_b_common_sentences']}")
    if "validation_a_authors_e2v" in rep:
        log(f"[align] validation A (authors e2v): {rep['validation_a_authors_e2v']}")
    log(f"[align] wrote {out_path}")
    return rep


# ---- self-test --------------------------------------------------------------

WORDS = ("apple river stone cloud forest candle window garden silver morning "
         "yellow bridge thunder paper ocean mirror pencil summer castle rabbit "
         "violin desert harbor meadow lantern glacier compass orchard velvet signal").split()

def _sentences(seed):
    import random
    rng = random.Random(seed)
    return {n: " ".join(rng.sample(WORDS, 7)) for n in range(1, 31)}

def _noisy(txt, rng, p):
    w = txt.split()
    return " ".join(x if rng.random() > p else rng.choice(WORDS) for x in w)

def selftest():
    import tempfile, random
    import numpy as np
    rng = random.Random(0)
    with tempfile.TemporaryDirectory() as td:
        fps = os.path.join(td, "dataset", "MEAD", "FPS25"); meta = os.path.dirname(fps)
        aud = os.path.join(td, "audios", "MEAD")
        os.makedirs(fps, exist_ok=True)
        rows = []
        common = {1: "kids are talking by the door", 2: "the birch canoe slid on the smooth planks",
                  3: "glue the sheet to the dark blue background"}
        truth = {}   # (ID,emo,lvl) -> {our_num: canonical}
        plans = {("M003", "disgusted", "level_3"): "shift-3", ("W015", "surprised", "level_3"): "perm",
                 ("M003", "disgusted", "level_2"): "missing4"}
        for ident in ("M003", "W015"):
            for k, emo in enumerate(("disgusted", "surprised")):
                S = _sentences(zlib.crc32(f"{ident}{emo}".encode()))
                S.update(common)
                for lvl in LEVELS:
                    plan = plans.get((ident, emo, lvl))
                    nums = list(range(1, 31))
                    if plan == "shift-3":
                        m = {o: ((o + 3 - 1) % 30) + 1 for o in nums}        # ours o holds canon o+3
                    elif plan == "perm":
                        perm = nums[:]; random.Random(7).shuffle(perm)
                        m = {o: perm[o - 1] for o in nums}
                    elif plan == "missing4":
                        m = {o: o for o in nums if o not in (5, 11, 17, 29)}
                    else:
                        m = {o: o for o in nums}
                    truth[(ident, emo, lvl)] = m
                    for o, c in m.items():
                        txt = S[c] if lvl == REF else _noisy(S[c], rng, 0.15)   # ASR noise
                        rows.append((f"{ident}/front/{emo}/{lvl}/{o:03d}.wav", txt))
        # one garbled transcript -> must be flagged weak
        rows = [(r, "uh" if r == "W015/front/disgusted/level_3/010.wav" else t) for r, t in rows]
        csvp = os.path.join(meta, "transcripts_all.csv")
        with open(csvp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["relpath", "transcript"]); w.writerows(rows)

        # authors' e2v ground truth for level_3: authors N == our o where truth[o] == N
        vec = {}
        for (ident, emo, lvl), m in truth.items():
            if lvl != "level_3":
                continue
            for o, c in m.items():
                v = np.random.default_rng(zlib.crc32(f"{ident}{emo}{c}".encode())).normal(size=64)
                p = os.path.join(fps, ident, "front", emo, lvl, FEAT_DIR, f"{o:03d}.npy")
                os.makedirs(os.path.dirname(p), exist_ok=True); np.save(p, v)
                q = os.path.join(aud, emo, FEAT_DIR, f"{ident}_{lvl}_{c:03d}.npy")
                os.makedirs(os.path.dirname(q), exist_ok=True); np.save(q, v * 1.3)

        out = os.path.join(meta, "level_remap.json")
        rep = run(fps, aud, csvp, out, np=np, log=lambda *_: None)
        G = rep["groups"]
        # T1 verdicts
        assert G["M003/disgusted/level_3"]["verdict"] == "cyclic_shift_+3", G["M003/disgusted/level_3"]["verdict"]
        assert G["W015/surprised/level_3"]["verdict"] == "permutation"
        assert G["M003/surprised/level_3"]["verdict"] == "identity"
        assert G["M003/disgusted/level_2"]["verdict"] == "identity" and G["M003/disgusted/level_2"]["n"] == 26
        # T2 exact recovery of every mapping (despite 15% ASR word noise)
        for (ident, emo, lvl), m in truth.items():
            if lvl == REF:
                continue
            got = {int(k): int(v) for k, v in G[f"{ident}/{emo}/{lvl}"]["map"].items()}
            if (ident, emo, lvl) == ("W015", "disgusted", "level_3"):
                got.pop(10, None); m = {k: v for k, v in m.items() if k != 10}
            assert got == m, (ident, emo, lvl)
        # T3 garbled transcript is flagged weak
        assert any(w["group"] == "W015/disgusted/level_3" and w["ours"] == "010" for w in rep["weak_matches"])
        # T4 validation A: all authors e2v pairs agree with transcript mapping (one garbled may miss)
        va = rep["validation_a_authors_e2v"]
        assert va["disagree"] <= 1 and va["agree"] >= 119, va
        # T5 validation B: common sentences agree after remap
        vb = rep["validation_b_common_sentences"]
        assert vb["misassigned"] == 0 and vb["checked"] >= 30 and vb["min_margin"] > 0, vb
        # T5b negative control for B: scramble level_3 map so common sentences land wrong
        scr = json.loads(json.dumps(G))
        g3 = scr["M003/surprised/level_3"]["map"]
        g3["001"], g3["004"] = "004", "001"
        vb_bad = validation_b(load_transcripts(csvp), scr)
        assert vb_bad["misassigned"] >= 1, vb_bad
        # T6 negative control: if the reference itself were wrong, validation A must fail loudly
        bad = json.loads(json.dumps(G))
        for k in bad:
            if k.endswith("level_3"):
                bad[k]["map"] = {kk: kk for kk in bad[k]["map"]}           # pretend identity
        va_bad = validation_a(fps, aud, bad, np)
        assert va_bad["disagree"] >= 50, va_bad
        # T7 JSON written and loadable
        assert json.load(open(out))["summary"]
    print("selftest OK: shift/permutation/missing recovered exactly under 15% ASR noise, verdict labels, "
          "weak-match flag, validation A agreement, validation B relative common-sentence test, "
          "negative controls for A and B fail loudly, JSON output")
    return True


def main():
    ap = argparse.ArgumentParser(description="Align MEAD levels by transcripts (analysis only)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fps25_root")
    ap.add_argument("--audios", default=None, help="authors' audios/MEAD for validation A")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.fps25_root:
        ap.print_help(); sys.exit(1)
    try:
        import scipy.optimize  # noqa
    except ImportError:
        print("scipy is required (optimal assignment)"); sys.exit(2)
    meta = os.path.dirname(args.fps25_root.rstrip("/")) or "."
    csvp = os.path.join(meta, "transcripts_all.csv")
    if not os.path.isfile(csvp):
        print(f"missing {csvp}; run transcribe_mead.py first"); sys.exit(2)
    import numpy as np
    run(args.fps25_root, args.audios, csvp, os.path.join(meta, "level_remap.json"), np=np)


if __name__ == "__main__":
    main()
