"""Regenerate the two LFC power curves from the committed simulator.

WHY THIS EXISTS. results/gates/lfc/power_soft/ and power_hard/ hold the curves
the paper's 8000 and 125 sample-size floors are read off, but NO COMMITTED CODE
PRODUCES THEM. Searching meridian-causalbench at 117a464 and 460a064 finds both
files only ever read (81_ranktest_oracle.py:396 and 83_assumption_battery.py:91)
and `git log -S` finds no commit that adds a producer. They came out of ad-hoc
runs that were never committed. This closes that gap.

THE ARITHMETIC IS NOT REIMPLEMENTED. Every draw comes from
81_ranktest_oracle.py, imported via importlib and called, never copied, edited
or re-derived. This script only chooses the grid and tabulates. The simulator's
own gate1() does the work; the k=3 slice of its runs IS the power curve.

RNG STREAM POSITION MATTERS, AND THIS IS THE ONE REAL UNKNOWN. gate1 advances a
single Generator through its (kind, k) loop, so the state the k=3 cell sees
depends on how many (kind, k) cells ran before it. The original ad-hoc runs are
not committed, so their loop shape is unrecoverable. Two modes:

    --rng-order full      (default) run gate1 with its own defaults,
                          k_set=(1,2,3,5), kinds=("hard","soft"), and take the
                          k=3 slice. Reproduces the stream of a full gate1 run.
    --rng-order isolated  run gate1 with k_set=(3,) and kinds=(arm,) only.
                          Cheaper, different stream position.

The step-0 gate below decides which is right before any grid runs: soft at
n_e = 8000 must reject 1.0 in both scalings, because that is what the committed
artefact records. If it does not, the import or the mode is wrong and this
aborts rather than writing a curve that looks plausible.

COMPARISON IS REPORTED, NOT GATED. Ten seeds per point means one or two tenths
of movement is sampling noise. Every differing (n_e, scaling) is listed with
both values; nothing exits non-zero on a difference.

Usage (A100). Long-running batch job; launch detached.

    nohup python -u scripts/93_regen_power_curves.py --arm soft \
        > logs/regen_soft.log 2>&1 &
    nohup python -u scripts/93_regen_power_curves.py --arm hard \
        > logs/regen_hard.log 2>&1 &

    bash scripts/run_regen.sh          # both arms, with the artefact assertion

Neither arm needs raw data: the simulator is self-contained. $PRECOND_DATA is
still checked when set, so a mistyped root is caught here rather than in 92.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_ROOT = REPO / "results" / "regen"
COMMITTED = REPO / "results" / "gates" / "lfc"
LOCK_DIR = REPO / "logs"

SIMULATOR = HERE / "81_ranktest_oracle.py"
REQUIRED_SCRIPTS = (SIMULATOR,
                    HERE / "80_ranktest_core.py",
                    HERE / "84_results_io.py")

# Configuration. Every value is the simulator's own default; none is overridden.
#   d_latent, D  ->  the gate1 config shape used for the published curves
#   edge_prob    ->  make_scm's default, 0.4, asserted below rather than passed
#   k            ->  3, the power slice of gate1's k_set
#   alpha, B_null, seeds, scalings -> module constants read off the simulator
D_LATENT = 10
D_OBS = 200
EDGE_PROB = 0.4                 # make_scm default; asserted, never passed
K_POWER = 3
SCALINGS = ("raw", "standardised")

GRIDS = {"soft": [500, 2000, 8000, 20000],
         "hard": [125, 250, 500, 1000, 2000, 8000]}

# Step-0 gate: the committed soft curve rejects 1.0 at n_e = 8000 in both
# scalings. Seeds 0 and 1 only, so a wrong import costs two seeds, not a grid.
GATE_ARM = "soft"
GATE_N_E = 8000
GATE_SEEDS = (0, 1)
GATE_EXPECTED = 1.0

COMMITTED_SOURCES = {
    "soft": COMMITTED / "power_soft" / "2026-08-10T11-00-08.json",
    "hard": COMMITTED / "power_hard" / "2026-08-10T11-39-05.json",
}
# The committed artefacts store their curve under different keys.
CURVE_KEY = {"soft": "data", "hard": "curve"}
# The floor rule, verbatim from power_hard/...::rule. The soft artefact records
# no rule; 83_assumption_battery.py:83-87 applies this same rule to the soft
# curve, so it is stated once here and applied to both.
FLOOR_RULE = "smallest n_e with reject>=0.80 in both scalings"
FLOOR_THRESHOLD = 0.80


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    real = os.makedirs
    os.makedirs = lambda *a, **k: None
    try:
        spec.loader.exec_module(mod)
    finally:
        os.makedirs = real
    return mod


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    import subprocess
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "uncommitted"


def data_roots():
    roots = []
    if os.environ.get("PRECOND_DATA"):
        roots.append(("$PRECOND_DATA", Path(os.environ["PRECOND_DATA"])))
    roots += [("repo data/", REPO / "data"),
              ("A100 project", Path("/workspace/precondition-audit/data")),
              ("A100 legacy", Path("/workspace/ranktest-diagnostics/data"))]
    return roots


# ===================================================================== step 0
def step0_locate_simulator():
    """The simulator must be IN THIS REPO, present and readable. Loud on miss."""
    print("=" * 70, flush=True)
    print(" STEP-0 GATE (a): simulator", flush=True)
    print("=" * 70, flush=True)
    fatal = []
    for p in REQUIRED_SCRIPTS:
        if not p.exists():
            fatal.append(
                f"{p.name} is not in this repository (looked in {p.parent}).\n"
                "        STOPPING. Do not vendor a copy from another repo and\n"
                "        do not reimplement it: this script exists precisely to\n"
                "        avoid a second implementation of that arithmetic.")
        else:
            print(f"  [code] {p.name:<28} {sha256(p)}", flush=True)
    print(f"  [code] simulator resolved path : {SIMULATOR.resolve()}", flush=True)

    root = os.environ.get("PRECOND_DATA")
    if root:
        print(f"  [env]  PRECOND_DATA = {root}"
              f"{'' if Path(root).is_dir() else '   (NOT A DIRECTORY)'}",
              flush=True)
        if not Path(root).is_dir():
            fatal.append(f"$PRECOND_DATA is set but is not a directory: {root}\n"
                         + "".join(f"        candidate {lbl:<14} {r}"
                                   f"{'' if r.is_dir() else '   (absent)'}\n"
                                   for lbl, r in data_roots()).rstrip())
    else:
        print("  [env]  PRECOND_DATA unset; not required by this script "
              "(the simulator needs no data)", flush=True)

    if fatal:
        print("\n[fatal] step-0 gate FAILED:", file=sys.stderr)
        for f in fatal:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(2)

    ORC = _load(SIMULATOR, "_oracle")
    for name in ("gate1", "make_scm", "sample_latent", "SEEDS", "ALPHA",
                 "B_NULL"):
        if not hasattr(ORC, name):
            sys.exit(f"[fatal] {SIMULATOR.name} has no {name!r}; this is not "
                     f"the simulator this script was written against.")
    # edge_prob is a default parameter, never passed by gate1. Assert it rather
    # than assume it, because a changed default silently changes every curve.
    import inspect
    got = inspect.signature(ORC.make_scm).parameters["edge_prob"].default
    if got != EDGE_PROB:
        sys.exit(f"[fatal] make_scm edge_prob default is {got!r}, expected "
                 f"{EDGE_PROB!r}. The graph sparsity behind the published "
                 f"curves has changed. Refusing.")
    print(f"  [conf] make_scm edge_prob default = {got}", flush=True)
    print(f"  [conf] ALPHA={ORC.ALPHA}  B_NULL={ORC.B_NULL}  "
          f"SEEDS={ORC.SEEDS}", flush=True)
    print("  [gate] PASS\n", flush=True)
    return ORC


def run_cell(ORC, arm, n_e, seeds, rng_order):
    """One (arm, n_e) point. gate1 does the arithmetic; this only selects.

    Returns {scaling: reject_rate} over `seeds`, plus the raw run records.
    """
    cfg = [dict(d_latent=D_LATENT, D=D_OBS, n=int(n_e))]
    if rng_order == "full":
        k_set, kinds = (1, 2, 3, 5), ("hard", "soft")
    else:
        k_set, kinds = (K_POWER,), (arm,)

    real_seeds = ORC.SEEDS
    ORC.SEEDS = list(seeds)                 # gate1 reads the module constant
    try:
        out = ORC.gate1(configs=cfg, k_set=k_set, kinds=kinds,
                        b_null=ORC.B_NULL)
    finally:
        ORC.SEEDS = real_seeds

    sel = [r for r in out["runs"]
           if r["kind"] == arm and int(r["k"]) == K_POWER]
    rates = {}
    for sc in SCALINGS:
        v = [bool(r["reject"]) for r in sel if r["scaling"] == sc]
        rates[sc] = (float(np.mean(v)) if v else None, len(v))
    return rates, sel


def step0_import_check(ORC, rng_order):
    """Soft at n_e=8000, two seeds. Anything but 1.0 means the import is wrong."""
    print("=" * 70, flush=True)
    print(f" STEP-0 GATE (b): {GATE_ARM} arm at n_e={GATE_N_E}, seeds "
          f"{list(GATE_SEEDS)}, rng-order={rng_order}", flush=True)
    print("=" * 70, flush=True)
    rates, _ = run_cell(ORC, GATE_ARM, GATE_N_E, GATE_SEEDS, rng_order)
    bad = []
    for sc in SCALINGS:
        got, n = rates[sc]
        print(f"  {sc:<14} reject = {got}  over {n} seed(s)  "
              f"(expected {GATE_EXPECTED})", flush=True)
        if got != GATE_EXPECTED:
            bad.append((sc, got))
    if bad:
        sys.exit(
            f"\n[fatal] STEP-0 IMPORT GATE FAILED\n"
            + "".join(f"        {sc}: got {got!r}, expected "
                      f"{GATE_EXPECTED!r}\n" for sc, got in bad)
            + f"        The committed power_soft artefact records 1.0 at "
              f"n_e={GATE_N_E} in both\n"
              f"        scalings. A disagreement here means the simulator "
              f"import or the RNG\n"
              f"        stream position is wrong, not that power has moved. "
              f"Try\n"
              f"          --rng-order {'isolated' if rng_order == 'full' else 'full'}\n"
              f"        ABORTING before the full grid.")
    print("  [gate] PASS\n", flush=True)


def floor_from(curve):
    """FLOOR_RULE applied to a curve: smallest n_e at >=0.80 in both scalings."""
    ns = sorted({c["n_e"] for c in curve})
    ok = [n for n in ns
          if all(c["reject"] >= FLOOR_THRESHOLD for c in curve if c["n_e"] == n)]
    return ok[0] if ok else None


def compare(arm, regen_curve):
    """Point-by-point against the committed artefact. Reported, never gated."""
    src = COMMITTED_SOURCES[arm]
    if not src.exists():
        return dict(status="committed artefact not found", path=str(src),
                    differences=None)
    doc = json.load(open(src))
    old = {(c["n_e"], c["scaling"]): float(c["reject"])
           for c in doc.get(CURVE_KEY[arm], [])}
    new = {(c["n_e"], c["scaling"]): float(c["reject"]) for c in regen_curve}
    diffs = []
    for key in sorted(set(old) | set(new)):
        o, n = old.get(key), new.get(key)
        if o is None or n is None or o != n:
            diffs.append(dict(n_e=key[0], scaling=key[1],
                              committed=o, regenerated=n,
                              delta=(None if o is None or n is None else n - o)))
    return dict(
        status="compared",
        committed_artefact=str(src.relative_to(REPO)),
        committed_commit=(doc.get("meta") or {}).get("git_commit"),
        n_points_committed=len(old), n_points_regenerated=len(new),
        n_differing=len(diffs),
        differences=diffs,
        note=("Reported, not gated. At 10 seeds per point a difference of one "
              "or two tenths is sampling noise, and the RNG stream position of "
              "the original uncommitted run is unrecoverable. This is "
              "information, not failure."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("soft", "hard"), required=True)
    ap.add_argument("--rng-order", choices=("full", "isolated"), default="full")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--skip-import-gate", action="store_true",
                    help="only after the gate has passed once in this session")
    a = ap.parse_args()

    ORC = step0_locate_simulator()
    if not a.skip_import_gate:
        step0_import_check(ORC, a.rng_order)

    lock = LOCK_DIR / f"regen_{a.arm}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        other = lock.read_text().strip()
        alive = False
        if other.isdigit():
            try:
                os.kill(int(other), 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            sys.exit(f"[fatal] another {a.arm} regen is live (pid {other}, "
                     f"lock {lock}). Two concurrent batches into one output "
                     f"directory is how the 11 Aug artefacts had to be "
                     f"discarded. Refusing.")
        print(f"[lock] stale lock from pid {other or 'unknown'}; reclaiming",
              flush=True)
        lock.unlink()
    lock.write_text(str(os.getpid()))

    try:
        out = Path(a.outdir) if a.outdir else OUT_ROOT / a.arm
        if out.exists():
            print(f"[clean] rm -rf {out}", flush=True)
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        grid = GRIDS[a.arm]
        print("=" * 70, flush=True)
        print(f" {a.arm} power curve, k={K_POWER}, d_latent={D_LATENT}, "
              f"D={D_OBS}, edge_prob={EDGE_PROB}", flush=True)
        print(f"   n_e grid  : {grid}", flush=True)
        print(f"   seeds     : {ORC.SEEDS}", flush=True)
        print(f"   rng-order : {a.rng_order}", flush=True)
        print("=" * 70, flush=True)

        curve, runs, failed = [], [], []
        for n_e in grid:
            print(f"\n--- [n_e={n_e}] ---", flush=True)
            try:
                rates, sel = run_cell(ORC, a.arm, n_e, ORC.SEEDS, a.rng_order)
            except Exception as e:                       # noqa: BLE001
                print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
                failed.append((n_e, f"{type(e).__name__}: {e}"))
                continue
            for sc in SCALINGS:
                rate, n_seeds = rates[sc]
                if rate is None:
                    failed.append((n_e, f"{sc}: no runs returned"))
                    continue
                curve.append(dict(n_e=int(n_e), scaling=sc,
                                  reject=rate, n_seeds=int(n_seeds)))
                print(f"    {sc:<14} reject={rate:.3f}  n_seeds={n_seeds}",
                      flush=True)
            runs.extend(sel)

        # ---- artefact assertion: N points produced for N requested
        want = len(grid) * len(SCALINGS)
        if failed or len(curve) != want:
            print(f"\n[fatal] SHORTFALL: {len(curve)} point(s) for {want} "
                  f"requested.", file=sys.stderr)
            for n_e, why in failed:
                print(f"  - n_e={n_e}: {why}", file=sys.stderr)
            print("        This run is INCOMPLETE. Do not read the artefact as "
                  "a full curve.", file=sys.stderr)
            sys.exit(1)

        payload = {
            CURVE_KEY[a.arm]: curve,
            "floor": floor_from(curve),
            "rule": FLOOR_RULE,
            "regenerated_from": dict(
                simulator_path=str(SIMULATOR.relative_to(REPO)),
                simulator_sha256=sha256(SIMULATOR),
                core_path="scripts/80_ranktest_core.py",
                core_sha256=sha256(HERE / "80_ranktest_core.py"),
                repo_commit=git_commit(),
                rng_order=a.rng_order,
                entry_point="81_ranktest_oracle.gate1",
                note=("The arithmetic is the simulator's, imported and called, "
                      "never copied or re-derived. This script chooses the grid "
                      "and tabulates the k=3 slice of gate1's runs. No "
                      "committed code produced the original curves; that is "
                      "why this exists."),
            ),
            "config": dict(
                d_latent=D_LATENT, D=D_OBS, edge_prob=EDGE_PROB, k=K_POWER,
                kind=a.arm, alpha=ORC.ALPHA, B_null=ORC.B_NULL,
                seeds=list(ORC.SEEDS), scalings=list(SCALINGS),
                n_e=list(grid), draws_per_point=1,
                intervention=("noise variance multiplied by U(2.5, 4.0), graph "
                              "intact" if a.arm == "soft" else
                              "incoming edges zeroed, noise variance replaced "
                              "by U(2.0, 4.0)"),
            ),
            "comparison": compare(a.arm, curve),
            "contains_no_test_on_real_data": True,
            "disclaimer": ("SIMULATOR ONLY. No real dataset was touched and no "
                           "assumption verdict is expressed or implied."),
        }
        ts = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        path = out / f"{ts}__power_{a.arm}_regen.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\n[write] {path}", flush=True)

        cmp_ = payload["comparison"]
        print("\n" + "=" * 70)
        print(f" comparison vs {cmp_.get('committed_artefact', 'n/a')}")
        print("-" * 70)
        if cmp_.get("differences") is None:
            print(f"  {cmp_['status']}")
        elif not cmp_["differences"]:
            print("  every (n_e, scaling) point agrees exactly")
        else:
            print(f"  {cmp_['n_differing']} of {cmp_['n_points_committed']} "
                  f"point(s) differ:")
            print(f"  {'n_e':>8} {'scaling':<14}{'committed':>11}"
                  f"{'regenerated':>13}{'delta':>9}")
            for d in cmp_["differences"]:
                print(f"  {d['n_e']:>8} {d['scaling']:<14}"
                      f"{str(d['committed']):>11}{str(d['regenerated']):>13}"
                      f"{'' if d['delta'] is None else format(d['delta'], '+.3f'):>9}")
            print("  Reported, not gated.")
        print("-" * 70)
        print(f" floor ({FLOOR_RULE}) = {payload['floor']}")
        print("=" * 70)
    finally:
        if lock.exists():
            lock.unlink()


if __name__ == "__main__":
    main()
