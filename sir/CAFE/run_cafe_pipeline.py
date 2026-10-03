#!/usr/bin/env python3
"""
CAFE unified experimental pipeline (single-command, resumable, auditable).

One orchestrator that runs the CAFE study as an ordered sequence of GATES.
Each gate runs only after the previous gate's validation passes. State is
checkpointed after every stage, so a crash or Ctrl-C resumes without repeating
completed work. Every gate logs to its own file, writes numerical results,
and (for implemented gates) produces paper-ready artifacts under an organized
run directory.

HONESTY CONTRACT (matches the project's no-scaffold rule):
  - Implemented gates call the real, already-tested CAFE scripts as subprocesses.
  - Gates whose model code does not exist yet (factorization, composition,
    falsification harness) are NOT faked. They raise NotImplementedError with a
    reason; the runner records them as PENDING_IMPL and stops. No invented
    numbers, no placeholder science.

The ORCHESTRATION LOGIC here (gate-gating, resume, safety checks, exception
handling, result aggregation) is unit-tested with --selftest using mock gates,
because the real gates need the DGX GPU, C-MET checkpoints, and the MEAD data.

Directory layout created per run (under --out_root):
  runs/<run_id>/
    checkpoints/     model states and per-gate pass markers
    logs/            per-gate and pipeline logs
    results/         numerical results (json/csv)
    tables/          paper-ready tables
    figures/         plots and curves
    visualizations/  qualitative videos / grids
    publication/     final publication-ready summary
    state.json       resume state (status of every gate)
    config_snapshot.json

Usage:
  python run_cafe_pipeline.py --selftest
  python run_cafe_pipeline.py --dry-run
  python run_cafe_pipeline.py --config pipeline_config.json
  python run_cafe_pipeline.py --config pipeline_config.json --resume
  python run_cafe_pipeline.py --config pipeline_config.json --only gate2_reproduce
  python run_cafe_pipeline.py --config pipeline_config.json --from preprocess
"""

import argparse, json, logging, os, shutil, subprocess, sys, time, traceback
from datetime import datetime, timezone

# ---- statuses -------------------------------------------------------------
PASSED = "PASSED"
FAILED = "FAILED"
BLOCKED = "SKIPPED_BLOCKED"       # a prerequisite gate did not pass
NO_INPUTS = "SKIPPED_INPUTS"      # required input files missing
RESUMED = "SKIPPED_RESUME"        # already passed on a previous run
PENDING_IMPL = "PENDING_IMPL"     # gate not implemented yet (honest stub)
PENDING = "PENDING"               # not yet reached
TERMINAL_OK = {PASSED, RESUMED}   # counts as "prerequisite satisfied"

SUBDIRS = ["checkpoints", "logs", "results", "tables", "figures",
           "visualizations", "publication"]


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- context --------------------------------------------------------------

class Ctx:
    """Everything a gate needs: resolved paths, config, run dirs, logger."""
    def __init__(self, config, run_dir):
        self.cfg = config
        self.run_dir = run_dir
        for d in SUBDIRS:
            setattr(self, d, os.path.join(run_dir, d))
        self.logger = logging.getLogger("cafe")

    def p(self, *parts):
        return os.path.join(self.run_dir, *parts)


# ---- gate definition ------------------------------------------------------

class Gate:
    def __init__(self, key, title, requires=None, inputs=None,
                 run=None, validate=None, implemented=True):
        self.key = key
        self.title = title
        self.requires = requires or []          # prior gate keys
        self.inputs = inputs or []               # paths that must exist (safety)
        self._run = run                          # callable(ctx) -> artifacts dict
        self._validate = validate                # callable(ctx, artifacts)->(ok,reason)
        self.implemented = implemented

    def run(self, ctx):
        if not self.implemented or self._run is None:
            raise NotImplementedError(
                f"{self.key} is not implemented yet: model code pending. "
                f"It will be built after the prior gate passes; the pipeline "
                f"does not fabricate its outputs.")
        return self._run(ctx)

    def validate(self, ctx, artifacts):
        if self._validate is None:
            return True, "no explicit validation"
        return self._validate(ctx, artifacts)


# ---- the runner (the unit-tested core) ------------------------------------

class Runner:
    def __init__(self, gates, ctx, state, keep_going=False, force=False):
        self.gates = gates
        self.ctx = ctx
        self.state = state              # {key: {status, ...}}
        self.keep_going = keep_going
        self.force = force

    def _save_state(self):
        with open(self.ctx.p("state.json"), "w") as f:
            json.dump(self.state, f, indent=2)

    def _mark(self, key, status, **extra):
        rec = self.state.get(key, {})
        rec.update({"status": status, "updated": now_iso(), **extra})
        self.state[key] = rec
        self._save_state()

    def _prereqs_ok(self, gate):
        missing = [r for r in gate.requires
                   if self.state.get(r, {}).get("status") not in TERMINAL_OK]
        return missing

    def _inputs_missing(self, gate):
        return [pth for pth in gate.inputs if not _exists(pth)]

    def run_all(self, selected=None):
        log = self.ctx.logger
        for gate in self.gates:
            if selected is not None and gate.key not in selected:
                continue

            # resume: already passed and not forced
            if not self.force and self.state.get(gate.key, {}).get("status") in TERMINAL_OK:
                self._mark(gate.key, RESUMED)
                log.info("[%s] already passed, skipping (resume)", gate.key)
                continue

            # gate-gating: prerequisites must have passed
            missing = self._prereqs_ok(gate)
            if missing:
                self._mark(gate.key, BLOCKED, blocked_by=missing)
                log.warning("[%s] BLOCKED: prerequisite(s) not passed: %s",
                            gate.key, missing)
                if not self.keep_going:
                    log.warning("stopping: a required gate is not satisfied")
                    break
                continue

            # safety: required inputs must exist
            miss_in = self._inputs_missing(gate)
            if miss_in:
                self._mark(gate.key, NO_INPUTS, missing_inputs=miss_in)
                log.warning("[%s] SKIPPED: missing inputs: %s", gate.key, miss_in)
                if not self.keep_going:
                    break
                continue

            # run
            self._mark(gate.key, "RUNNING", started=now_iso())
            log.info("[%s] %s -- running", gate.key, gate.title)
            try:
                artifacts = gate.run(self.ctx)
                ok, reason = gate.validate(self.ctx, artifacts)
                if ok:
                    self._mark(gate.key, PASSED, artifacts=artifacts, reason=reason)
                    log.info("[%s] PASSED (%s)", gate.key, reason)
                else:
                    self._mark(gate.key, FAILED, artifacts=artifacts, reason=reason)
                    log.error("[%s] FAILED validation: %s", gate.key, reason)
                    if not self.keep_going:
                        break
            except NotImplementedError as e:
                self._mark(gate.key, PENDING_IMPL, reason=str(e))
                log.warning("[%s] PENDING_IMPL: %s", gate.key, e)
                if not self.keep_going:
                    break
            except Exception as e:  # real failure, caught and recorded
                tb = traceback.format_exc()
                self._mark(gate.key, FAILED, reason=repr(e))
                _write(self.ctx.p("logs", f"{gate.key}.error.log"), tb)
                log.error("[%s] EXCEPTION (see logs/%s.error.log): %s",
                          gate.key, gate.key, e)
                if not self.keep_going:
                    break
        return self.state

    def summary(self):
        rows = []
        for gate in self.gates:
            st = self.state.get(gate.key, {}).get("status", PENDING)
            rows.append((gate.key, gate.title, st))
        return rows


# ---- small fs helpers -----------------------------------------------------

def _exists(path):
    if any(ch in path for ch in "*?[]"):
        import glob
        return len(glob.glob(path)) > 0
    return os.path.exists(path)

def _write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(text)

def _run_cmd(ctx, key, cmd, cwd=None):
    """Run a subprocess, tee stdout/err to the gate log, raise on non-zero."""
    logf = ctx.p("logs", f"{key}.log")
    os.makedirs(os.path.dirname(logf), exist_ok=True)
    ctx.logger.info("[%s] $ %s", key, " ".join(cmd))
    with open(logf, "a") as lf:
        lf.write(f"\n$ {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed (rc={proc.returncode}): {' '.join(cmd)} "
                           f"(see logs/{key}.log)")
    return proc.returncode


# ---- run directory + logging ----------------------------------------------

def setup_run_dir(out_root, run_id, config):
    run_dir = os.path.join(out_root, run_id)
    for d in SUBDIRS:
        os.makedirs(os.path.join(run_dir, d), exist_ok=True)
    with open(os.path.join(run_dir, "config_snapshot.json"), "w") as f:
        json.dump(config, f, indent=2)
    return run_dir

def setup_logging(run_dir):
    logger = logging.getLogger("cafe")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(run_dir, "logs", "pipeline.log"))
    fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt); logger.addHandler(ch)
    return logger

def load_state(run_dir):
    sp = os.path.join(run_dir, "state.json")
    if os.path.exists(sp):
        with open(sp) as f:
            return json.load(f)
    return {}


# ---- the real CAFE gates (thin subprocess wrappers + honest stubs) --------

def build_real_gates(ctx):
    cfg = ctx.cfg
    repo = cfg["cmet_repo"]                # e.g. /workspace/sir/CAFE/C-MET
    cafe = cfg["cafe_root"]                # e.g. /workspace/sir/CAFE
    dataset = cfg.get("dataset", "MEAD")
    test_csv = cfg["test_csv"]             # dataset/MEAD/test_present.csv (repo-relative)
    connector = cfg["connector_ckpt"]
    gen_dir = cfg.get("gen_dir", "evaluation/runs/mead_ours/gen")
    fps25 = cfg["fps25_root"]              # dataset/MEAD/FPS25

    def R(*a):  # repo-relative path
        return os.path.join(repo, *a)

    # Gate 0: environment + provenance freeze
    def run_env(c):
        _run_cmd(c, "gate0_env",
                 ["python", os.path.join(cafe, "freeze_provenance.py"),
                  "--root", repo, "--out", os.path.join(repo, "paper_artifacts", "provenance")])
        return {"provenance": os.path.join(repo, "paper_artifacts", "provenance", "provenance.json")}
    def val_env(c, art):
        ok = os.path.isfile(art["provenance"])
        return ok, "provenance.json written" if ok else "provenance.json missing"

    # data validation: checkpoints, test csv, videos, e2v all present
    def run_datacheck(c):
        need = [connector, test_csv, fps25]
        missing = [p for p in need if not _exists(p if os.path.isabs(p) else R(p))]
        report = {"checked": need, "missing": missing}
        _write(c.p("results", "data_check.json"), json.dumps(report, indent=2))
        return report
    def val_datacheck(c, art):
        ok = len(art["missing"]) == 0
        return ok, ("all data present" if ok else f"missing: {art['missing']}")

    # preprocess: crop_video.py (crop + 25fps) over all front videos, then prep_video.py
    def run_preprocess(c):
        # This is the real batch. It is heavy (GPU, ~1100 videos) and runs on DGX.
        script = os.path.join(cafe, "preprocess_crop_batch.py")
        if not os.path.isfile(script):
            raise NotImplementedError(
                "preprocess_crop_batch.py not present yet. It wraps crop_video.py "
                "(crop + 25fps) over the FPS25 tree, then prep_video.py for EDTalk "
                "features. Generate it before running this gate.")
        _run_cmd(c, "preprocess", ["python", script, "--fps25_root", R(fps25),
                                   "--cmet_repo", repo], cwd=repo)
        return {"features_root": R(fps25)}
    def val_preprocess(c, art):
        import glob
        feats = glob.glob(os.path.join(art["features_root"], "*", "front", "*", "*", "*_ED_exp.npy"))
        ok = len(feats) > 0
        return ok, f"{len(feats)} EDTalk feature files present" if ok else "no features produced"

    # Gate 2 part A: generate videos + assemble ours.csv
    def run_generate(c):
        bo = os.path.join(cafe, "build_ours.py")
        _run_cmd(c, "gate2_generate", ["python", bo, "emit",
                 "--test_csv", test_csv, "--dataset", dataset,
                 "--connector_ckpt", connector, "--gen_dir", gen_dir,
                 "--out_script", os.path.join(gen_dir, "generate_all.sh")], cwd=repo)
        _run_cmd(c, "gate2_generate", ["bash", os.path.join(gen_dir, "generate_all.sh")], cwd=repo)
        _run_cmd(c, "gate2_generate", ["python", bo, "assemble",
                 "--test_csv", test_csv, "--gen_dir", gen_dir,
                 "--out", os.path.join(gen_dir, "..", "ours.csv")], cwd=repo)
        return {"ours_csv": os.path.join(repo, gen_dir, "..", "ours.csv")}
    def val_generate(c, art):
        ok = os.path.isfile(art["ours_csv"])
        return ok, "ours.csv assembled" if ok else "ours.csv missing"

    # Gate 2 part B: eval pipeline + score vs paper -> Gate 2 verdict
    def run_reproduce(c):
        rc = os.path.join(cafe, "reproduce_cmet.py")
        ours = os.path.join(repo, gen_dir, "..", "ours.csv")
        prov = os.path.join(repo, "paper_artifacts", "provenance", "provenance.json")
        out = c.p("results", "gate2_report.json")
        _run_cmd(c, "gate2_reproduce", ["python", rc, "score", "--csv_path", ours,
                 "--dataset", dataset, "--provenance", prov, "--out", out], cwd=cafe)
        return {"gate2_report": out}
    def val_reproduce(c, art):
        if not os.path.isfile(art["gate2_report"]):
            return False, "gate2_report.json missing"
        with open(art["gate2_report"]) as f:
            rep = json.load(f)
        ok = rep.get("verdict") == "GATE2_PASS"
        # copy a paper-ready table into publication/
        _write(c.p("publication", "gate2_reproduction.json"), json.dumps(rep, indent=2))
        return ok, rep.get("verdict", "no verdict")

    return [
        Gate("gate0_env", "Environment and provenance freeze",
             inputs=[connector], run=run_env, validate=val_env),
        Gate("data_check", "Data validation (checkpoints, test set, videos, e2v)",
             requires=["gate0_env"], run=run_datacheck, validate=val_datacheck),
        Gate("preprocess", "Crop + 25fps + EDTalk feature extraction",
             requires=["data_check"], run=run_preprocess, validate=val_preprocess),
        Gate("gate2_generate", "Generate videos and assemble ours.csv",
             requires=["preprocess"], run=run_generate, validate=val_generate),
        Gate("gate2_reproduce", "Reproduce C-MET metrics and score Gate 2",
             requires=["gate2_generate"], run=run_reproduce, validate=val_reproduce),
        Gate("gate3_factorization", "Counterfactual content/affect factorization",
             requires=["gate2_reproduce"], implemented=False),
        Gate("gate4_composition", "Zero-compound-supervision affect composition",
             requires=["gate3_factorization"], implemented=False),
        Gate("gate5_harness", "Falsification harness + external baselines",
             requires=["gate4_composition"], implemented=False),
    ]


def write_final_summary(ctx, runner):
    rows = runner.summary()
    summary = {
        "run_id": os.path.basename(ctx.run_dir),
        "generated": now_iso(),
        "gates": [{"key": k, "title": t, "status": s} for k, t, s in rows],
    }
    _write(ctx.p("publication", "pipeline_summary.json"), json.dumps(summary, indent=2))
    # console table
    print("\n" + "=" * 66)
    print(f"{'gate':<22}{'status':<18}title")
    print("-" * 66)
    for k, t, s in rows:
        print(f"{k:<22}{s:<18}{t}")
    print("=" * 66)
    print(f"summary -> {ctx.p('publication', 'pipeline_summary.json')}")


# ---- default config -------------------------------------------------------

DEFAULT_CONFIG = {
    "cmet_repo": "/workspace/sir/CAFE/C-MET",
    "cafe_root": "/workspace/sir/CAFE",
    "dataset": "MEAD",
    "test_csv": "dataset/MEAD/test_present.csv",
    "connector_ckpt": "./checkpoints/_epoch_2105_checkpoint_step000200000.pth",
    "fps25_root": "dataset/MEAD/FPS25",
    "gen_dir": "evaluation/runs/mead_ours/gen",
    "out_root": "/workspace/sir/CAFE/runs",
    "seed": 0,
}


# ---- self-test (orchestration logic, no GPU/data) -------------------------

def selftest():
    import tempfile, random
    random.seed(0)
    with tempfile.TemporaryDirectory() as td:
        cfg = dict(DEFAULT_CONFIG); cfg["out_root"] = td
        run_dir = setup_run_dir(td, "selftest", cfg)
        logging.getLogger("cafe").handlers.clear()
        logging.getLogger("cafe").addHandler(logging.NullHandler())
        ctx = Ctx(cfg, run_dir)

        # mock gates: g1 pass, g2 pass (needs a real input file), g3 fail-validate,
        # g4 depends on g3 (must be BLOCKED), g5 stub (PENDING_IMPL)
        good_input = os.path.join(td, "exists.txt"); open(good_input, "w").close()

        calls = []
        def mk_run(name, make=None):
            def _r(c):
                calls.append(name)
                if make: _write(c.p("results", f"{name}.json"), "{}")
                return {"ran": name}
            return _r

        gates = [
            Gate("g1", "first", run=mk_run("g1", True), validate=lambda c, a: (True, "ok")),
            Gate("g2", "needs input", requires=["g1"], inputs=[good_input],
                 run=mk_run("g2", True), validate=lambda c, a: (True, "ok")),
            Gate("g3", "fails validation", requires=["g2"],
                 run=mk_run("g3"), validate=lambda c, a: (False, "bad metric")),
            Gate("g4", "blocked by g3", requires=["g3"], run=mk_run("g4"),
                 validate=lambda c, a: (True, "ok")),
            Gate("g5", "not implemented", requires=["g4"], implemented=False),
        ]

        # T1: linear run stops at g3 failure; g4/g5 never run
        st = {}
        r = Runner(gates, ctx, st)
        r.run_all()
        assert st["g1"]["status"] == PASSED, st["g1"]
        assert st["g2"]["status"] == PASSED, st["g2"]
        assert st["g3"]["status"] == FAILED, st["g3"]
        assert "g4" not in st or st["g4"]["status"] == PENDING, st.get("g4")
        assert calls == ["g1", "g2", "g3"], calls  # g4/g5 not executed

        # T2: results + dirs written
        assert os.path.isfile(ctx.p("results", "g1.json"))
        assert os.path.isfile(ctx.p("state.json"))
        for d in SUBDIRS:
            assert os.path.isdir(os.path.join(run_dir, d))

        # T3: resume -> g1,g2 skipped as RESUMED, not re-run
        calls.clear()
        # flip g3 to pass now, so the rest can proceed
        gates[2] = Gate("g3", "now passes", requires=["g2"], run=mk_run("g3"),
                        validate=lambda c, a: (True, "ok"))
        r2 = Runner(gates, ctx, st)
        r2.run_all()
        assert st["g1"]["status"] == RESUMED and st["g2"]["status"] == RESUMED
        assert "g1" not in calls and "g2" not in calls, calls  # not re-run
        assert st["g3"]["status"] == PASSED
        assert st["g4"]["status"] == PASSED  # g3 passed -> g4 unblocked
        assert st["g5"]["status"] == PENDING_IMPL  # honest stub, stops here

        # T4: gate-gating -- force a prereq to fail, dependent must be BLOCKED
        st2 = {"g1": {"status": PASSED}}
        gates_block = [
            Gate("g2", "will fail", requires=["g1"], inputs=[good_input],
                 run=mk_run("g2b"), validate=lambda c, a: (False, "x")),
            Gate("g3", "dependent", requires=["g2"], run=mk_run("g3b"),
                 validate=lambda c, a: (True, "ok")),
        ]
        r3 = Runner(gates_block, ctx, st2, keep_going=True)
        r3.run_all()
        assert st2["g2"]["status"] == FAILED
        assert st2["g3"]["status"] == BLOCKED, st2["g3"]

        # T5: missing-input safety -> NO_INPUTS
        st3 = {"a": {"status": PASSED}}
        gate_missing = [Gate("b", "needs missing file", requires=["a"],
                             inputs=[os.path.join(td, "nope.bin")],
                             run=mk_run("b"), validate=lambda c, a: (True, "ok"))]
        Runner(gate_missing, ctx, st3).run_all()
        assert st3["b"]["status"] == NO_INPUTS, st3["b"]
        assert "b" not in calls  # never executed

        # T6: exception in a gate is caught, recorded FAILED, error log written
        st4 = {}
        def boom(c):
            raise ValueError("kaboom")
        gate_boom = [Gate("x", "raises", run=boom, validate=lambda c, a: (True, "ok"))]
        Runner(gate_boom, ctx, st4).run_all()
        assert st4["x"]["status"] == FAILED
        assert os.path.isfile(ctx.p("logs", "x.error.log"))

        # T7: --only selection runs just that gate (prereqs already satisfied)
        st5 = {"g1": {"status": PASSED}, "g2": {"status": PASSED}}
        calls.clear()
        sel_gates = [
            Gate("g1", "a", run=mk_run("g1s"), validate=lambda c, a: (True, "ok")),
            Gate("g2", "b", requires=["g1"], run=mk_run("g2s"), validate=lambda c, a: (True, "ok")),
            Gate("g3", "c", requires=["g2"], run=mk_run("g3s"), validate=lambda c, a: (True, "ok")),
        ]
        Runner(sel_gates, ctx, st5).run_all(selected={"g3"})
        assert calls == ["g3s"], calls  # only g3 ran

    print("selftest OK: linear-stop-on-fail, results+dirs, resume-skip, "
          "gate-gating(block), missing-input safety, exception capture, --only selection")
    return True


# ---- cli ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="CAFE unified pipeline")
    ap.add_argument("--selftest", action="store_true", help="run orchestration self-test")
    ap.add_argument("--config", help="JSON config; defaults used if omitted")
    ap.add_argument("--run_id", default=None, help="run directory name (default: timestamp)")
    ap.add_argument("--resume", action="store_true", help="skip gates already passed")
    ap.add_argument("--force", action="store_true", help="re-run even passed gates")
    ap.add_argument("--keep-going", action="store_true", help="do not stop on first block/fail")
    ap.add_argument("--dry-run", action="store_true", help="print the gate plan and exit")
    ap.add_argument("--only", nargs="*", help="run only these gate keys")
    ap.add_argument("--from", dest="from_key", help="start from this gate key")
    ap.add_argument("--until", dest="until_key", help="stop after this gate key")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    config = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    run_id = args.run_id or datetime.now().strftime("cafe_%Y%m%d_%H%M%S")

    # dry-run is read-only: build gates from config, print plan, create nothing
    if args.dry_run:
        gates = build_real_gates(Ctx(config, os.path.join(config["out_root"], run_id)))
        keys = [g.key for g in gates]
        selected = set(args.only) if args.only else None
        if selected is None and (args.from_key or args.until_key):
            lo = keys.index(args.from_key) if args.from_key else 0
            hi = keys.index(args.until_key) + 1 if args.until_key else len(keys)
            selected = set(keys[lo:hi])
        print(f"# CAFE pipeline plan  (run_id={run_id})")
        print(f"# run dir (not created in dry-run): {os.path.join(config['out_root'], run_id)}\n")
        for g in gates:
            mark = "impl" if g.implemented else "PENDING_IMPL (stub)"
            sel = "" if (selected is None or g.key in selected) else "  [skipped by selection]"
            print(f"  {g.key:<22} requires={g.requires or '-'}  [{mark}]{sel}")
        print("\nvalidation gates run in order; each needs the previous to pass.")
        sys.exit(0)

    run_dir = setup_run_dir(config["out_root"], run_id, config)
    setup_logging(run_dir)
    state = load_state(run_dir) if (args.resume or args.force) else {}
    ctx = Ctx(config, run_dir)
    gates = build_real_gates(ctx)

    # --from / --until / --only selection
    keys = [g.key for g in gates]
    selected = None
    if args.only:
        selected = set(args.only)
    elif args.from_key or args.until_key:
        lo = keys.index(args.from_key) if args.from_key else 0
        hi = keys.index(args.until_key) + 1 if args.until_key else len(keys)
        selected = set(keys[lo:hi])

    runner = Runner(gates, ctx, state, keep_going=args.keep_going, force=args.force)
    runner.run_all(selected=selected)
    write_final_summary(ctx, runner)


if __name__ == "__main__":
    main()
