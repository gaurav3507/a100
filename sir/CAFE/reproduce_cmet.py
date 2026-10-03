#!/usr/bin/env python3
"""
CAFE Phase 2 - C-MET reproduction runner (Gate 2).

This does NOT reimplement C-MET's metrics. The official evaluation/ directory
already computes FID, FVD, Sync_conf, and Accemo by accumulating columns into a
per-video CSV. This runner does three things around that pipeline:

  plan   Emit the exact, ordered command sequence for the official pipeline,
         with the repo's real gotchas baked in (FID needs --csv_path or the
         aggregator sees nothing; SyncNet pairs the generated video with the
         SOURCE audio, not GT; Emotion-FAN inference uses --csv_file). Use this
         so the student runs reproduction correctly and identically each time.

  score  Read the final per-video CSV, aggregate the four required metrics
         exactly as evaluation/check_quantitative_all.py does, compare against
         the paper's Table 1 (paper_reference.json), attach the Phase 0
         provenance (commit + checkpoint hashes) if present, and print a Gate 2
         PASS / FAIL table with per-metric tolerance.

  --selftest  CPU-only checks of the aggregation, tolerance, gate verdict,
              provenance attach, and plan generation. No GPU or data needed.

Gate 2 passes only when all four required metrics are present and each is within
tolerance of the paper (not much worse; better than paper is fine but a huge
improvement is flagged as suspicious).
"""

import argparse, csv, json, os, sys, time

REQUIRED = ["FID", "fvd", "Sync_conf", "Accemo"]

# ---------- aggregation (mirrors evaluation/check_quantitative_all.py) ----------

def _mean(rows, col):
    vals = []
    for r in rows:
        v = r.get(col, "")
        if v is None or str(v).strip() == "":
            continue
        try:
            vals.append(float(v))
        except ValueError:
            continue
    return sum(vals) / len(vals) if vals else None

def _accemo(rows):
    pairs = [(r.get("gt_emotion"), r.get("predicted_emotion")) for r in rows]
    pairs = [(g, p) for g, p in pairs
             if g not in (None, "") and p not in (None, "")]
    if not pairs:
        return None
    correct = sum(1 for g, p in pairs if str(g) == str(p))
    return 100.0 * correct / len(pairs)

def summarize_csv(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "FID": _mean(rows, "FID"),
        "fvd": _mean(rows, "fvd"),
        "Sync_conf": _mean(rows, "Sync_conf"),
        "Accemo": _accemo(rows),
        "_n_rows": len(rows),
    }

# ---------- comparison / gate ----------

def worse_ratio(metric, reproduced, target, higher_is_better):
    """Relative amount by which `reproduced` is WORSE than `target` (signed).
       <= 0 means as good or better. Positive means worse."""
    if target == 0:
        return 0.0
    if higher_is_better.get(metric, False):
        return (target - reproduced) / abs(target)
    return (reproduced - target) / abs(target)

def grade(metric, reproduced, target, higher_is_better, tol, too_good=0.5):
    if reproduced is None:
        return "MISSING", None
    wr = worse_ratio(metric, reproduced, target, higher_is_better)
    if wr > tol:
        return "FAIL", wr
    if wr < -too_good:
        return "SUSPICIOUS", wr  # implausibly better than paper -> likely a bug
    return "PASS", wr

def load_provenance(prov_path):
    if not prov_path or not os.path.isfile(prov_path):
        return None
    try:
        with open(prov_path) as f:
            d = json.load(f)
        return {
            "git_commit": d.get("git", {}).get("commit"),
            "git_dirty": d.get("git", {}).get("dirty"),
            "checkpoints": {k: (v[0]["sha256"] if v and v[0].get("sha256") else None)
                            for k, v in d.get("checkpoints", {}).items()},
        }
    except Exception:
        return None

def cmd_score(args):
    if not os.path.isfile(args.csv_path):
        print(f"[score] csv not found: {args.csv_path}"); sys.exit(2)
    with open(args.reference) as f:
        ref = json.load(f)
    if args.dataset not in ref["targets"]:
        print(f"[score] dataset '{args.dataset}' not in reference; have {list(ref['targets'])}")
        sys.exit(2)
    targets = ref["targets"][args.dataset]
    hib = ref["higher_is_better"]

    summ = summarize_csv(args.csv_path)
    prov = load_provenance(args.provenance)

    report = {
        "schema": "cafe.phase2.gate2/1",
        "scored_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": args.dataset, "csv_path": args.csv_path,
        "tolerance": args.tol, "n_rows": summ["_n_rows"],
        "provenance": prov, "reference_source": ref.get("_source"),
        "metrics": {}, "verdict": None,
    }

    present = [m for m in REQUIRED if summ.get(m) is not None]
    all_pass = len(present) == len(REQUIRED)
    for m in REQUIRED:
        repro = summ.get(m)
        tgt = targets.get(m)
        status, wr = grade(m, repro, tgt, hib, args.tol)
        report["metrics"][m] = {
            "reproduced": repro, "paper": tgt,
            "worse_by_frac": None if wr is None else round(wr, 4),
            "status": status,
        }
        if status != "PASS":
            all_pass = False
    report["verdict"] = "GATE2_PASS" if all_pass else "GATE2_FAIL"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # human-readable table
    print(f"Gate 2 reproduction - {args.dataset}  (n={summ['_n_rows']} rows, tol={args.tol:.0%})")
    print("-" * 64)
    print(f"{'metric':<12}{'reproduced':>14}{'paper':>12}{'worse_by':>12}  status")
    for m in REQUIRED:
        r = report["metrics"][m]
        rep = "n/a" if r["reproduced"] is None else f"{r['reproduced']:.4f}"
        pap = "n/a" if r["paper"] is None else f"{r['paper']:.4f}"
        wb = "n/a" if r["worse_by_frac"] is None else f"{r['worse_by_frac']*100:+.1f}%"
        print(f"{m:<12}{rep:>14}{pap:>12}{wb:>12}  {r['status']}")
    print("-" * 64)
    if prov and prov.get("git_commit"):
        dirty = " (DIRTY tree)" if prov.get("git_dirty") else ""
        print(f"provenance: commit {prov['git_commit'][:12]}{dirty}")
    else:
        print("provenance: none linked (run freeze_provenance.py and pass --provenance)")
    print(f"VERDICT: {report['verdict']}   -> {args.out}")
    sys.exit(0 if report["verdict"] == "GATE2_PASS" else 1)

# ---------- plan ----------

def build_plan(run_dir, csv_name, dataset, emofan_ckpt):
    """Exact official evaluation/ pipeline, in order, with the real gotchas.
       Paths are relative to the C-MET/evaluation directory."""
    csv_rel = f"runs/{run_dir}/{csv_name}"
    frames = f"runs/{run_dir}/frames"
    steps = [
        ("prereq: generation",
         "# produce generated videos over dataset/{d}/test.csv and build {c} "
         "with a 'generated_path' 5th column (see evaluation/README.md; batch "
         "inference is inference_dataset_ref.py, outside evaluation/). "
         "AITV comes from its infer_times.".format(d=dataset, c=csv_name)),
        ("1 frames",
         f"python vide2frame_custom.py    # -> {frames}/ours and {frames}/ours_GT"),
        ("2 faces",
         "python frame2face_custom.py    # face crops; writes face_dir back into the csv"),
        ("3 FID (needs --csv_path or aggregator sees nothing)",
         f"python pytorch-fid/custom.py {frames}/ours {frames}/ours_GT "
         f"--csv_path {csv_rel} --batch-size 512 --video_batch_size 64"),
        ("4 FVD",
         f"python fvd.py runs/{run_dir}"),
        ("5 Accemo (inference uses --csv_file, not --train/--test_csv)",
         f"python Emotion-FAN/emotion-fan.py --csv_file {csv_rel} "
         f"--checkpoint {emofan_ckpt} --num_frames 16"),
        ("6 Sync (pairs generated video with SOURCE audio, not GT)",
         f"python syncnet_python/all_pipeline.py --csv_path {csv_rel} && "
         f"python syncnet_python/all_syncnet.py --csv_path {csv_rel} && "
         f"python syncnet_python/conf_mean.py --csv_path {csv_rel}"),
        ("7 aggregate (writes nothing; prints FID/FVD/Sync_conf/Accemo)",
         f"python check_quantitative_all.py --csv_path {csv_rel}"),
        ("8 Gate 2 verdict (this runner, from CAFE root)",
         f"python reproduce_cmet.py score --csv_path ./C-MET/evaluation/{csv_rel} "
         f"--dataset {dataset} --provenance ./C-MET/paper_artifacts/provenance/provenance.json"),
    ]
    return steps

def cmd_plan(args):
    steps = build_plan(args.run_dir, args.csv_name, args.dataset, args.emofan_ckpt)
    print(f"# C-MET reproduction plan  (dataset={args.dataset}, run={args.run_dir})")
    print("# run steps 1-7 from ./C-MET/evaluation ; step 8 from ./C-MET or CAFE root")
    print("# each step must succeed before the next; stop and fix on any error\n")
    for label, cmd in steps:
        print(f"## {label}")
        print(cmd + "\n")

# ---------- self-test ----------

def selftest():
    import tempfile
    hib = {"Sync_conf": True, "Accemo": True, "FID": False, "fvd": False}
    targets = {"FID": 90.804, "fvd": 329.862, "Sync_conf": 7.9996, "Accemo": 55.91}

    # A) aggregation matches a hand computation, including Accemo
    with tempfile.TemporaryDirectory() as td:
        csvp = os.path.join(td, "ours.csv")
        with open(csvp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gt_emotion", "predicted_emotion", "FID", "fvd", "Sync_conf"])
            w.writerow(["angry", "angry", "90", "330", "8.0"])   # correct
            w.writerow(["happy", "sad", "91", "329", "8.1"])     # wrong
            w.writerow(["sad", "sad", "", "331", ""])            # FID/Sync missing
        s = summarize_csv(csvp)
        assert abs(s["FID"] - 90.5) < 1e-9, s["FID"]              # mean of 90,91
        assert abs(s["fvd"] - 330.0) < 1e-9, s["fvd"]
        assert abs(s["Sync_conf"] - 8.05) < 1e-9, s["Sync_conf"]
        assert abs(s["Accemo"] - (200.0/3)) < 1e-9, s["Accemo"]   # 2/3 correct
        assert s["_n_rows"] == 3

    # B) grading directions
    # FID lower is better: reproduced 95 vs 90.8 -> worse by ~4.6% -> PASS at 10%
    assert grade("FID", 95.0, 90.804, hib, 0.10)[0] == "PASS"
    # FID 120 vs 90.8 -> ~32% worse -> FAIL
    assert grade("FID", 120.0, 90.804, hib, 0.10)[0] == "FAIL"
    # Accemo higher better: 50 vs 55.91 -> ~10.6% worse -> FAIL at 10%
    assert grade("Accemo", 50.0, 55.91, hib, 0.10)[0] == "FAIL"
    # Accemo 54 vs 55.91 -> ~3.4% worse -> PASS
    assert grade("Accemo", 54.0, 55.91, hib, 0.10)[0] == "PASS"
    # implausibly good FID (10 vs 90.8) -> SUSPICIOUS
    assert grade("FID", 10.0, 90.804, hib, 0.10)[0] == "SUSPICIOUS"
    # missing
    assert grade("FID", None, 90.804, hib, 0.10)[0] == "MISSING"

    # C) provenance attach
    with tempfile.TemporaryDirectory() as td:
        pj = os.path.join(td, "provenance.json")
        with open(pj, "w") as f:
            json.dump({"git": {"commit": "abc123def456", "dirty": False},
                       "checkpoints": {"checkpoints": [{"sha256": "deadbeef"}]}}, f)
        prov = load_provenance(pj)
        assert prov["git_commit"] == "abc123def456"
        assert prov["checkpoints"]["checkpoints"] == "deadbeef"
        assert load_provenance(os.path.join(td, "nope.json")) is None

    # D) plan generation contains the critical gotchas, in order
    steps = build_plan("mead_ours", "ours.csv", "MEAD",
                       "Emotion-FAN/checkpoints/Emotion-FAN_MEAD.pth")
    joined = "\n".join(c for _, c in steps)
    assert "--csv_path" in joined and "custom.py" in joined, "FID --csv_path gotcha missing"
    assert "--csv_file" in joined, "Emotion-FAN --csv_file gotcha missing"
    assert "all_pipeline.py" in joined and "conf_mean.py" in joined, "sync steps missing"
    labels = [l for l, _ in steps]
    assert labels.index("7 aggregate (writes nothing; prints FID/FVD/Sync_conf/Accemo)") \
        > labels.index("3 FID (needs --csv_path or aggregator sees nothing)"), "order wrong"

    # E) end-to-end score verdict on a synthetic PASS csv
    with tempfile.TemporaryDirectory() as td:
        csvp = os.path.join(td, "ours.csv")
        with open(csvp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gt_emotion", "predicted_emotion", "FID", "fvd", "Sync_conf"])
            # values close to MEAD targets, Accemo ~ 55.6 (5/9)
            emos = ["angry", "happy", "sad", "fear", "neutral"]
            for i in range(9):
                pred = emos[i % 5] if i < 5 else "wrong"
                gt = emos[i % 5]
                w.writerow([gt, gt if i < 5 else "wrong2", "91", "330", "8.0"])
        refp = os.path.join(td, "ref.json")
        with open(refp, "w") as f:
            json.dump({"_source": "test", "higher_is_better": hib,
                       "targets": {"MEAD": {"AITV": 2.643, **targets}}}, f)
        # reuse cmd_score via a tiny args shim
        class A: pass
        a = A(); a.csv_path=csvp; a.reference=refp; a.dataset="MEAD"
        a.tol=0.10; a.provenance=None; a.out=os.path.join(td, "g2.json")
        try:
            cmd_score(a)
        except SystemExit as e:
            code = e.code
        rep = json.load(open(a.out))
        assert rep["metrics"]["FID"]["status"] == "PASS"
        assert rep["metrics"]["fvd"]["status"] == "PASS"
        assert rep["metrics"]["Sync_conf"]["status"] == "PASS"
        # Accemo here is 5/9=55.6 vs 55.91 -> PASS
        assert rep["metrics"]["Accemo"]["status"] == "PASS", rep["metrics"]["Accemo"]
        assert rep["verdict"] == "GATE2_PASS", rep["verdict"]
        assert code == 0

    print("selftest OK: aggregation+Accemo, grade directions, SUSPICIOUS/MISSING, "
          "provenance attach, plan gotchas+order, end-to-end Gate 2 verdict")
    return True

# ---------- cli ----------

def main():
    ap = argparse.ArgumentParser(description="C-MET reproduction runner (Gate 2)")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("score", help="aggregate final csv, compare to paper, Gate 2 verdict")
    s.add_argument("--csv_path", required=True, help="final per-video ours.csv")
    s.add_argument("--dataset", default="MEAD", help="MEAD or CREMA_D")
    s.add_argument("--reference", default=os.path.join(os.path.dirname(__file__), "paper_reference.json"))
    s.add_argument("--provenance", default=None, help="Phase 0 provenance.json to link")
    s.add_argument("--tol", type=float, default=0.10, help="relative worse-than-paper tolerance")
    s.add_argument("--out", default="./paper_artifacts/gate2_report.json")
    s.set_defaults(func=cmd_score)

    p = sub.add_parser("plan", help="print the exact official reproduction command sequence")
    p.add_argument("--run_dir", default="mead_ours")
    p.add_argument("--csv_name", default="ours.csv")
    p.add_argument("--dataset", default="MEAD")
    p.add_argument("--emofan_ckpt", default="Emotion-FAN/checkpoints/Emotion-FAN_MEAD.pth")
    p.set_defaults(func=cmd_plan)

    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not getattr(args, "cmd", None):
        ap.print_help(); sys.exit(1)
    args.func(args)

if __name__ == "__main__":
    main()
