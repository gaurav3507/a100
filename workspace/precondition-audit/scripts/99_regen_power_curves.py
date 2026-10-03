"""Regenerate the Section III-D power curves from the committed simulator.

WHY THIS EXISTS. The soft and hard detection floors of 8000 and 125 cells are
read off two committed artefacts under results/gates/lfc/power_soft/ and
power_hard/. Those artefacts are complete, but the drivers that produced them
were never committed: searching the source repository finds both files only ever
READ, and no commit adds a producer. The manuscript claims the curves regenerate
from version-pinned code. This is that code.

NO SIMULATOR LOGIC IS WRITTEN HERE. Not the graph construction, not the edge
probability, not the weight or noise distributions, not the soft or hard
intervention operator. Every one of those is read out of the committed simulator
module at run time and printed, and the module's own gate1 does the arithmetic.
This file chooses which grid points to visit and tabulates what comes back.

WHERE EACH CONSTANT COMES FROM, and what happens if it cannot be reached:

    alpha, B_null, seeds        module globals
    edge_prob                   default of the module's make_scm signature
    k_set, kinds                defaults of the module's gate1 signature
    graph / weights / noise     printed verbatim from the module's source
    soft and hard operators     printed verbatim from the module's source
    d_latent, D, n_e grid       meta.config of the COMMITTED artefact per arm,
                                then asserted to be reachable from the module
    k (the power slice)         parsed out of the committed artefact's meta.note,
                                then asserted to be a member of the module k_set
    gate expectation            the committed soft artefact's own value at the
                                largest n_e on its grid

Nothing on that list is a literal in this file. If any of it cannot be reached,
the script exits non-zero naming what was missing rather than substituting a
number from memory. A structural constant typed here would make the
regeneration a re-implementation, which is the one thing it must not be.

COMPARISON IS REPORTED, NOT GATED. Ten seeds per grid point means a difference of
one or two tenths is sampling noise. The manuscript already states these curves
are ten-seed grid edges. Every differing (n_e, scaling) is listed with both
values and nothing exits non-zero on a difference.

Usage (A100). Long-running batch job; launch detached:

    nohup python -u scripts/99_regen_power_curves.py > logs/regen_power.log 2>&1 &

    python scripts/99_regen_power_curves.py --arm soft     # one arm
    python scripts/99_regen_power_curves.py --step0-only   # locate and pin, stop

Neither arm needs raw data: the simulator is self-contained. $PRECOND_DATA and
$PRECOND_EXTERNAL are still existence-checked when set, so a mistyped root is
caught here rather than three hours later.
"""
import argparse
import datetime
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_ROOT = REPO / "results" / "power"
COMMITTED_ROOT = REPO / "results" / "gates" / "lfc"
LOCK_DIR = REPO / "logs"

# Arm -> the committed gate directory holding that arm's artefact. These are
# directory names in this repository, not simulator constants.
ARM_GATE_DIR = {"soft": "power_soft", "hard": "power_hard"}
# The committed artefacts store their curve under different top-level keys.
CURVE_KEY = {"soft": "data", "hard": "curve"}
# Symbols a module must expose to be the simulator behind Section III-D.
SIM_SYMBOLS = ("gate1", "make_scm", "sample_latent", "SEEDS", "ALPHA", "B_NULL")
# Source of these two functions is printed as the operator definition of record.
SIM_SOURCE_OF = ("make_scm", "sample_latent")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H-%M-%SZ")


def git_commit():
    import subprocess
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "uncommitted"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    real = os.makedirs
    os.makedirs = lambda *a, **k: None          # loaders mkdir at import time
    try:
        spec.loader.exec_module(mod)
    finally:
        os.makedirs = real
    return mod


def data_roots():
    """Existence-checked candidate roots. No hardcoded path but the defaults."""
    roots = []
    if os.environ.get("PRECOND_DATA"):
        roots.append(("$PRECOND_DATA", Path(os.environ["PRECOND_DATA"])))
    if os.environ.get("PRECOND_EXTERNAL"):
        roots.append(("$PRECOND_EXTERNAL", Path(os.environ["PRECOND_EXTERNAL"])))
    roots += [("repo data/", REPO / "data"),
              ("A100 project", Path("/workspace/precondition-audit/data")),
              ("A100 legacy", Path("/workspace/ranktest-diagnostics/data"))]
    return roots


# ===================================================================== step 0
def find_simulator():
    """The committed module that defines the Section III-D simulator.

    Discovered by the symbols it exposes, not by filename, so a rename does not
    silently point this at nothing. Exactly one module must match.
    """
    hits = []
    for p in sorted(HERE.glob("*.py")):
        if p.name == Path(__file__).name:
            continue
        try:
            src = p.read_text()
        except OSError:
            continue
        if all(re.search(rf'^\s*(def\s+{s}\b|{s}\s*=)', src, re.M)
               for s in SIM_SYMBOLS):
            hits.append(p)
    if not hits:
        sys.exit(f"[fatal] no module under {HERE} exposes all of {SIM_SYMBOLS}.\n"
                 f"        The simulator behind Section III-D is not in this "
                 f"repository.\n"
                 f"        STOPPING. Do not import it from another repository "
                 f"and do not vendor a copy:\n"
                 f"        the committed artefacts were produced by THIS "
                 f"module, and anything else\n"
                 f"        would be a re-implementation wearing its name.")
    if len(hits) > 1:
        sys.exit(f"[fatal] {len(hits)} modules expose {SIM_SYMBOLS}: "
                 f"{[h.name for h in hits]}.\n"
                 f"        Which one produced the committed artefacts is "
                 f"ambiguous. Refusing to guess.")
    return hits[0]


def step0(verbose=True):
    """Locate, pin, import, and READ the configuration. Fails loudly."""
    print("=" * 78, flush=True)
    print(" STEP 0: locate and pin the simulator", flush=True)
    print("=" * 78, flush=True)

    sim = find_simulator()
    resolved = sim.resolve()
    digest = sha256(resolved)
    print(f"  resolved path : {resolved}", flush=True)
    print(f"  sha256        : {digest}", flush=True)

    root = REPO.resolve()
    inside = str(resolved).startswith(str(root) + os.sep)
    print(f"  repo root     : {root}", flush=True)
    print(f"  inside repo   : {'YES' if inside else 'NO'}", flush=True)
    if not inside:
        sys.exit(f"[fatal] the simulator resolved OUTSIDE this repository:\n"
                 f"        {resolved}\n"
                 f"        STOPPING. Not importing from another repo and not "
                 f"vendoring a copy.")

    for lbl, r in data_roots():
        if not r.exists():
            print(f"  [data] {lbl:<18} {r}   (absent, not required)", flush=True)
        else:
            print(f"  [data] {lbl:<18} {r.resolve()}", flush=True)

    ORC = _load(resolved, "_simulator")
    missing = [s for s in SIM_SYMBOLS if not hasattr(ORC, s)]
    if missing:
        sys.exit(f"[fatal] {resolved.name} imported but does not expose "
                 f"{missing}. Refusing to continue.")

    def sig_default(fn, name):
        try:
            v = inspect.signature(fn).parameters[name].default
        except (KeyError, ValueError, TypeError):
            v = inspect.Parameter.empty
        if v is inspect.Parameter.empty:
            sys.exit(f"[fatal] {name!r} has no reachable default in "
                     f"{fn.__name__}() of {resolved.name}.\n"
                     f"        It is a structural constant of the simulator and "
                     f"this script will not\n"
                     f"        supply one from memory. Refusing to continue.")
        return v

    cfg = dict(
        alpha=ORC.ALPHA,
        B_null=ORC.B_NULL,
        seeds=list(ORC.SEEDS),
        edge_prob=sig_default(ORC.make_scm, "edge_prob"),
        k_set=list(sig_default(ORC.gate1, "k_set")),
        kinds=list(sig_default(ORC.gate1, "kinds")),
    )
    print("\n  --- configuration READ from the module ---", flush=True)
    for key in ("alpha", "B_null", "seeds", "edge_prob", "k_set", "kinds"):
        print(f"    {key:<12} = {cfg[key]!r}", flush=True)

    src = {}
    print("\n  --- structural definitions, verbatim from the module ---",
          flush=True)
    for name in SIM_SOURCE_OF:
        try:
            s = inspect.getsource(getattr(ORC, name))
        except (OSError, TypeError):
            sys.exit(f"[fatal] cannot read the source of {name}() from "
                     f"{resolved.name}. The operator definitions are the point "
                     f"of this report. Refusing to continue.")
        src[name] = s
        if verbose:
            print(f"\n    # {name}(), {resolved.name}", flush=True)
            for line in s.rstrip().splitlines():
                print(f"    {line}", flush=True)

    print("\n  [step 0] PASS\n", flush=True)
    return ORC, dict(path=str(resolved.relative_to(root)),
                     absolute_path=str(resolved), sha256=digest,
                     inside_repo=True, config=cfg, source=src,
                     repo_commit=git_commit())


# ======================================================== committed artefacts
def committed(arm):
    """The committed artefact for one arm, and what it fixes about the run."""
    d = COMMITTED_ROOT / ARM_GATE_DIR[arm]
    files = sorted(d.glob("*.json"))
    if not files:
        sys.exit(f"[fatal] no committed {arm} power artefact under {d}.\n"
                 f"        The grid, d_latent, D and k are read from it. There "
                 f"is nothing to read\n"
                 f"        and this script will not supply them from memory.")
    cur = []
    for p in files:
        try:
            doc = json.load(open(p))
        except ValueError:
            continue
        if (doc.get("meta") or {}).get("status") == "CURRENT":
            cur.append((p, doc))
    if len(cur) != 1:
        sys.exit(f"[fatal] expected exactly one CURRENT {arm} artefact under "
                 f"{d}, found {len(cur)}. Refusing to guess which produced the "
                 f"published curve.")
    return cur[0]


def read_arm_config(arm, ORC, simcfg):
    """d_latent, D, n_e and k for one arm, from its artefact, module-checked."""
    path, doc = committed(arm)
    mc = (doc.get("meta") or {}).get("config") or {}
    note = str((doc.get("meta") or {}).get("note") or "")

    def need(field):
        v = mc.get(field)
        if v is None:
            sys.exit(f"[fatal] {path.name} meta.config.{field} is absent.\n"
                     f"        It fixes the regeneration and is not supplied "
                     f"from memory. Refusing.")
        return v

    d_latent, D, n_e = need("d_latent"), need("D"), need("n_e")
    m = re.search(r'\bk\s*=\s*(\d+)', note)
    if not m:
        sys.exit(f"[fatal] cannot read the intervention cardinality k from "
                 f"{path.name}.\n"
                 f"        meta.note = {note!r}\n"
                 f"        meta.config carries no k field. k is a structural "
                 f"choice and this script\n"
                 f"        will not assume one. Refusing to continue.")
    k = int(m.group(1))
    if k not in simcfg["config"]["k_set"]:
        sys.exit(f"[fatal] {path.name} records k={k}, which is not in the "
                 f"module's k_set {simcfg['config']['k_set']}. The artefact and "
                 f"the simulator disagree. Refusing.")
    if arm not in simcfg["config"]["kinds"]:
        sys.exit(f"[fatal] arm {arm!r} is not one of the module's intervention "
                 f"kinds {simcfg['config']['kinds']}. Refusing.")
    # d_latent and D must be reachable from the module's own default config set,
    # otherwise the artefact was produced under a configuration this module no
    # longer describes.
    defaults = inspect.signature(ORC.gate1).parameters["configs"].default
    if defaults is None:
        src = inspect.getsource(ORC.gate1)
        dl_ok = re.search(rf'\b{d_latent}\b', src) and re.search(rf'\b{D}\b', src)
        if not dl_ok:
            print(f"  [warn] d_latent={d_latent} or D={D} does not appear in "
                  f"gate1's default configs; the artefact may predate the "
                  f"current module.", flush=True)
    fm = re.search(r'\bfloor\s+(\d+)', note)
    floor = int(fm.group(1)) if fm else None
    return dict(floor=floor,
                artefact=str(path.relative_to(REPO)), artefact_name=path.name,
                doc=doc, d_latent=int(d_latent), D=int(D),
                n_e=[int(x) for x in n_e], k=int(k),
                git_commit=(doc.get("meta") or {}).get("git_commit"),
                note=note)


# ================================================================ the curves
def run_point(ORC, arm, n_e, d_latent, D, k, seeds):
    """One (arm, n_e). The module's gate1 does the arithmetic; this selects.

    gate1 advances a single Generator through its (kind, k) loop, so the k-slice
    it is asked for sees a stream position that depends on how many cells ran
    before it. The full default k_set and kinds are therefore requested, and the
    wanted slice taken from the result, so the stream matches a full gate1 run
    rather than an isolated call.
    """
    cfg = [dict(d_latent=int(d_latent), D=int(D), n=int(n_e))]
    real_seeds = ORC.SEEDS
    ORC.SEEDS = list(seeds)                    # gate1 reads the module global
    try:
        out = ORC.gate1(configs=cfg, b_null=ORC.B_NULL)
    finally:
        ORC.SEEDS = real_seeds
    sel = [r for r in out["runs"]
           if r["kind"] == arm and int(r["k"]) == int(k)]
    rates = {}
    for scaling in sorted({r["scaling"] for r in out["runs"]}):
        v = [bool(r["reject"]) for r in sel if r["scaling"] == scaling]
        rates[scaling] = (float(np.mean(v)) if v else None, len(v))
    return rates


def step1_gate(ORC, arm_cfg, simcfg):
    """Reproduction gate: one grid point, two seeds, against the artefact."""
    arm = "soft"
    a = arm_cfg[arm]
    # The gate point is the arm's OWN recorded floor, parsed from its artefact:
    # that is the smallest n_e at which the published curve reaches full power,
    # so it is the cheapest point that still distinguishes a correct call from a
    # broken one. Falls back to the largest grid point if no floor is recorded,
    # rather than to a number typed here.
    n_e, floor_src = a.get("floor"), "meta.note floor"
    if n_e is None or n_e not in a["n_e"]:
        n_e, floor_src = max(a["n_e"]), "largest grid point (no floor recorded)"
    seeds = simcfg["config"]["seeds"][:2]
    old = {(c["n_e"], c["scaling"]): float(c["reject"])
           for c in a["doc"].get(CURVE_KEY[arm], [])}
    expected = {sc: v for (n, sc), v in old.items() if n == n_e}
    if not expected:
        sys.exit(f"[fatal] the committed {arm} artefact has no point at "
                 f"n_e={n_e}; there is nothing to gate against. Refusing.")

    print("=" * 78, flush=True)
    print(f" STEP 1: reproduction gate, {arm} arm, n_e={n_e}, seeds {seeds}",
          flush=True)
    print(f"   gate point from {floor_src}", flush=True)
    print(f"   expected, from {a['artefact_name']}: "
          f"{ {k: v for k, v in sorted(expected.items())} }", flush=True)
    print("=" * 78, flush=True)

    rates = run_point(ORC, arm, n_e, a["d_latent"], a["D"], a["k"], seeds)
    bad = []
    for sc, exp in sorted(expected.items()):
        got, n = rates.get(sc, (None, 0))
        print(f"  {sc:<14} got {got}  over {n} seed(s)   expected {exp}",
              flush=True)
        if got != exp:
            bad.append((sc, exp, got))
    if bad:
        sys.exit(
            "\n[fatal] STEP-1 REPRODUCTION GATE FAILED\n"
            + "".join(f"        {sc}: expected {exp!r}, got {got!r}\n"
                      for sc, exp, got in bad)
            + "        The import or the call is wrong, not the power. Running "
              "the full grid now\n"
              "        would spend the night computing a curve nobody can "
              "trust. ABORTING.")
    print("\n  [step 1] PASS\n", flush=True)
    return dict(arm=arm, n_e=int(n_e), seeds=list(seeds),
                expected=expected,
                observed={sc: rates[sc][0] for sc in rates})


def compare(arm, regen, a):
    """Point-by-point against the committed artefact. Never gates."""
    old = {(c["n_e"], c["scaling"]): float(c["reject"])
           for c in a["doc"].get(CURVE_KEY[arm], [])}
    new = {(c["n_e"], c["scaling"]): float(c["reject"]) for c in regen}
    diffs = []
    for key in sorted(set(old) | set(new)):
        o, n = old.get(key), new.get(key)
        if o is None or n is None or o != n:
            diffs.append(dict(n_e=key[0], scaling=key[1], committed=o,
                              regenerated=n,
                              delta=(None if o is None or n is None
                                     else round(n - o, 6))))
    return dict(committed_artefact=a["artefact"],
                committed_git_commit=a["git_commit"],
                n_points_committed=len(old), n_points_regenerated=len(new),
                n_differing=len(diffs), differences=diffs,
                gated=False,
                note=("Reported, not gated. Ten seeds per point means one or "
                      "two tenths is sampling noise, and the manuscript states "
                      "these curves are ten-seed grid edges. A difference here "
                      "is information about resampling variability, not a "
                      "failure."))


def run_arm(ORC, arm, a, simcfg, RIO, outdir):
    seeds = simcfg["config"]["seeds"]
    print("=" * 78, flush=True)
    print(f" {arm} arm: n_e {a['n_e']}, k={a['k']}, d_latent={a['d_latent']}, "
          f"D={a['D']}", flush=True)
    print(f"   seeds {seeds}   alpha {simcfg['config']['alpha']}   "
          f"B_null {simcfg['config']['B_null']}   "
          f"edge_prob {simcfg['config']['edge_prob']}", flush=True)
    print("=" * 78, flush=True)

    curve, failed = [], []
    for n_e in a["n_e"]:
        print(f"\n--- [{arm} n_e={n_e}] ---", flush=True)
        try:
            rates = run_point(ORC, arm, n_e, a["d_latent"], a["D"], a["k"],
                              seeds)
        except Exception as e:                              # noqa: BLE001
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append((n_e, f"{type(e).__name__}: {e}"))
            continue
        for scaling in sorted(rates):
            rate, n_seeds = rates[scaling]
            if rate is None:
                failed.append((n_e, f"{scaling}: no runs returned"))
                continue
            curve.append(dict(n_e=int(n_e), scaling=scaling, reject=rate,
                              n_seeds=int(n_seeds)))
            print(f"    {scaling:<14} reject={rate:.3f}  n_seeds={n_seeds}",
                  flush=True)

    n_scalings = len({c["scaling"] for c in curve}) or 1
    want = len(a["n_e"]) * n_scalings
    if failed or len(curve) != want:
        print(f"\n[fatal] {arm}: SHORTFALL, {len(curve)} point(s) for {want} "
              f"requested.", file=sys.stderr)
        for n_e, why in failed:
            print(f"  - n_e={n_e}: {why}", file=sys.stderr)
        print("        This arm is INCOMPLETE and no artefact is written for "
              "it.", file=sys.stderr)
        return None

    payload = {
        CURVE_KEY[arm]: curve,
        "arm": arm,
        "n_seeds_per_point": len(seeds),
        "simulator": simcfg,
        "config_read_from_artefact": {
            k: a[k] for k in ("artefact", "artefact_name", "d_latent", "D",
                              "n_e", "k", "floor", "git_commit", "note")},
        "comparison": compare(arm, curve, a),
        "versions": RIO.versions(),
        "platform_tag": RIO.platform_tag(),
        "repo_commit": git_commit(),
        "contains_no_test_on_real_data": True,
        "disclaimer": ("SIMULATOR ONLY. No real dataset was touched and no "
                       "assumption verdict is expressed or implied."),
    }
    path = outdir / f"{utc_stamp()}__power_{arm}_regen.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n[write] {path}", flush=True)

    c = payload["comparison"]
    print(f"\n  comparison vs {c['committed_artefact']} "
          f"(commit {c['committed_git_commit']})")
    if not c["differences"]:
        print("    every (n_e, scaling) point agrees exactly")
    else:
        print(f"    {c['n_differing']} of {c['n_points_committed']} differ:")
        print(f"    {'n_e':>8} {'scaling':<14}{'committed':>11}"
              f"{'regenerated':>13}{'delta':>9}")
        for d in c["differences"]:
            print(f"    {d['n_e']:>8} {d['scaling']:<14}"
                  f"{str(d['committed']):>11}{str(d['regenerated']):>13}"
                  f"{('' if d['delta'] is None else format(d['delta'], '+.3f')):>9}")
        print("    Reported, not gated.")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARM_GATE_DIR),
                    help="one arm only (default: both)")
    ap.add_argument("--step0-only", action="store_true",
                    help="locate, pin and report the simulator, then stop")
    ap.add_argument("--outdir", default=str(OUT_ROOT))
    args = ap.parse_args()

    ORC, simcfg = step0()
    if args.step0_only:
        print("  --step0-only: nothing was run.")
        return

    arms = [args.arm] if args.arm else sorted(ARM_GATE_DIR)
    arm_cfg = {a: read_arm_config(a, ORC, simcfg) for a in arms}
    for a in arms:
        c = arm_cfg[a]
        print(f"  [artefact] {a:<6} {c['artefact_name']}  "
              f"n_e={c['n_e']}  k={c['k']}  d_latent={c['d_latent']}  "
              f"D={c['D']}", flush=True)
    print()

    if "soft" in arm_cfg:
        gate = step1_gate(ORC, arm_cfg, simcfg)
    else:
        gate = dict(skipped="soft arm not requested; step-1 gate needs it")
        print("  [step 1] skipped: the gate runs on the soft arm, which was "
              "not requested.\n", flush=True)
    simcfg["step1_gate"] = gate

    lock = LOCK_DIR / "regen_power.lock"
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
            sys.exit(f"[fatal] another regeneration is live (pid {other}, lock "
                     f"{lock}). Two concurrent batches into one output "
                     f"directory is how the 11 Aug artefacts had to be "
                     f"discarded. Refusing.")
        print(f"[lock] stale lock from pid {other or 'unknown'}; reclaiming",
              flush=True)
        lock.unlink()
    lock.write_text(str(os.getpid()))

    try:
        RIO = _load(HERE / "84_results_io.py", "_results_io")
        written = []
        for arm in arms:
            outdir = Path(args.outdir) / arm
            if outdir.exists():
                print(f"[clean] rm -rf {outdir}", flush=True)
                shutil.rmtree(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            p = run_arm(ORC, arm, arm_cfg[arm], simcfg, RIO, outdir)
            if p is not None:
                written.append(p)

        print("\n" + "=" * 78)
        print(f" artefacts written: {len(written)} of {len(arms)} arm(s)")
        for p in written:
            print(f"   {p}")
        print("=" * 78)
        if len(written) != len(arms):
            print(f"\n[fatal] SHORTFALL: {len(written)} artefact(s) for "
                  f"{len(arms)} arm(s). Rerun the missing arm with --arm.",
                  file=sys.stderr)
            sys.exit(1)
    finally:
        if lock.exists():
            lock.unlink()


if __name__ == "__main__":
    main()
