"""Results storage layout, mandatory meta block, and the schema gate.

LAYOUT
    causalbench/results/ranktest/<statistic>/<gate>/<timestamp>.json

    statistic in {diy_retired, cfa_kappa, cft, lfc, descriptive}
    gate      in {acceptance, gate0, gate1, gate2, envelope, power_soft,
                  power_hard, alpha_sensitivity, battery, descriptives,
                  preprocessing_sweep}

WHY THE META BLOCK IS MANDATORY. Twice in this lane a number was quoted from
a file whose producing statistic was not recoverable from its name, and once a
power curve computed under a retired statistic was nearly attached to a
current result. write_results() refuses to write anything missing a meta
field, so that failure mode is now impossible rather than merely discouraged.

Nothing in here is allowed to edit a results file in place. Migration writes
NEW files and records the old path in meta.migrated_from; the originals are
moved, never deleted.
"""
import json
import os
import platform as _platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# New-repo layout: scripts/ sits at the repo root, and the <statistic>/<gate>
# tree that this module manages holds the GATE artefacts.
RESULTS = HERE.parent / "results" / "gates"

# "descriptive" is not a test statistic: it labels artefacts that involve
# no hypothesis test at all, e.g. dataset profiling.
STATISTICS = ("diy_retired", "cfa_kappa", "cft", "lfc", "descriptive")
GATES = ("acceptance", "gate0", "gate1", "gate2", "envelope",
         "power_soft", "power_hard", "alpha_sensitivity", "battery",
         "descriptives", "preprocessing_sweep")

# Every one of these must be PRESENT. A value of None is allowed where the
# field genuinely does not apply to that run (e.g. d_latent for a pure
# counting check), but the KEY may never be absent -- an absent key is
# indistinguishable from a forgotten one.
CONFIG_FIELDS = ("alpha", "B", "n_e", "d", "d_latent", "D", "n_env",
                 "seeds", "draws_per_point")
META_FIELDS = ("statistic", "gate", "git_commit", "timestamp", "config",
               "versions", "superseded_by", "migrated_from")
STATUSES = ("CURRENT", "SUPERSEDED", "HISTORICAL-FOR-COMPARISON-TABLE")


class MetaSchemaError(ValueError):
    """Raised instead of writing a results file with an incomplete meta block."""


def git_commit(short=True):
    try:
        r = subprocess.run(["git", "-C", str(HERE), "rev-parse",
                            "--short" if short else "HEAD", "HEAD"],
                           capture_output=True, text=True)
        return r.stdout.split()[0] if r.returncode == 0 and r.stdout.strip() else None
    except OSError:
        return None


def versions():
    """Package versions AND platform.

    The A100 venv (cb) differs from the Mac venv, so a number that moves
    between machines has to be attributable either to the data or to the
    environment. Recording versions alone is not enough: SVD results depend on
    the BLAS backend, so the backend is captured too where numpy exposes it.
    """
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "pandas", "sklearn", "scanpy", "anndata"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    out["platform"] = {
        "system": _platform.system(),
        "release": _platform.release(),
        "machine": _platform.machine(),
        "processor": _platform.processor() or None,
        "python_implementation": _platform.python_implementation(),
    }
    try:                                   # BLAS backend, when numpy exposes it
        import numpy as _np
        cfg = getattr(_np, "__config__", None)
        blas = None
        if cfg is not None and hasattr(cfg, "show"):
            d = getattr(cfg, "_built_with_meson", None)
            try:
                info = cfg.show(mode="dicts")           # numpy >= 1.25
                blas = (info.get("Build Dependencies", {})
                          .get("blas", {}).get("name"))
            except Exception:
                blas = None
        out["platform"]["numpy_blas"] = blas
    except Exception:
        out["platform"]["numpy_blas"] = None
    return out


def platform_tag():
    """Short stable label for the machine, used to diff Mac vs A100 runs."""
    p = _platform
    return f"{p.system()}-{p.machine()}-py{sys.version.split()[0]}"


def make_meta(statistic, gate, timestamp, config, status="CURRENT",
              superseded_by=None, migrated_from=None, note=None,
              commit=None):
    """Build a meta block. Missing config keys are filled with None, present."""
    cfg = {k: (config or {}).get(k) for k in CONFIG_FIELDS}
    extra = {k: v for k, v in (config or {}).items() if k not in CONFIG_FIELDS}
    if extra:
        cfg["extra"] = extra
    return dict(statistic=statistic, gate=gate,
                git_commit=commit if commit is not None else git_commit(),
                timestamp=timestamp, config=cfg, versions=versions(),
                superseded_by=superseded_by, migrated_from=migrated_from,
                status=status, note=note)


def validate_meta(meta):
    """Raise MetaSchemaError unless every mandatory field is present and sane."""
    if not isinstance(meta, dict):
        raise MetaSchemaError("meta must be a dict")
    missing = [f for f in META_FIELDS if f not in meta]
    if missing:
        raise MetaSchemaError(f"meta is missing mandatory field(s): {missing}")
    if meta["statistic"] not in STATISTICS:
        raise MetaSchemaError(
            f"meta.statistic={meta['statistic']!r} not in {STATISTICS}")
    if meta["gate"] not in GATES:
        raise MetaSchemaError(f"meta.gate={meta['gate']!r} not in {GATES}")
    for f in ("git_commit", "timestamp"):
        if not meta.get(f):
            raise MetaSchemaError(f"meta.{f} is empty; it must identify the run")
    cfg = meta.get("config")
    if not isinstance(cfg, dict):
        raise MetaSchemaError("meta.config must be a dict")
    miss_cfg = [f for f in CONFIG_FIELDS if f not in cfg]
    if miss_cfg:
        raise MetaSchemaError(f"meta.config is missing key(s): {miss_cfg} "
                              f"(None is allowed, absence is not)")
    ver = meta.get("versions")
    if not isinstance(ver, dict) or not ver.get("python"):
        raise MetaSchemaError("meta.versions must record at least python")
    st = meta.get("status")
    if st is not None and st not in STATUSES:
        raise MetaSchemaError(f"meta.status={st!r} not in {STATUSES}")
    return True


def results_path(statistic, gate, timestamp, suffix="", results_dir=RESULTS):
    """<results>/<statistic>/<gate>/<timestamp><suffix>.json"""
    stamp = str(timestamp).replace(":", "-")
    name = f"{stamp}{suffix}.json"
    return Path(results_dir) / statistic / gate / name


def write_results(payload, meta, path=None, results_dir=RESULTS, suffix=""):
    """Validate meta, then write atomically. REFUSES on an incomplete meta."""
    validate_meta(meta)
    if path is None:
        path = results_path(meta["statistic"], meta["gate"], meta["timestamp"],
                            suffix=suffix, results_dir=results_dir)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Several artefacts are bare JSON lists (power curves, sweeps). A list
    # cannot carry a meta block, so it is wrapped under "data" rather than
    # silently losing its provenance.
    body = dict(payload) if isinstance(payload, dict) else {"data": payload}
    if "meta" in body and body["meta"] != meta:
        raise MetaSchemaError("payload already carries a different meta block")
    body["meta"] = meta
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(body, f, indent=2)
    os.rename(tmp, path)
    return path


def iter_results(results_dir=RESULTS, statistic=None, gate=None, status=None):
    """Walk the layout and yield (path, payload) for artefacts that match."""
    root = Path(results_dir)
    for stat in sorted(STATISTICS):
        if statistic and stat != statistic:
            continue
        for g in sorted(GATES):
            if gate and g != gate:
                continue
            d = root / stat / g
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.json")):
                try:
                    payload = json.load(open(p))
                except ValueError:
                    continue
                m = payload.get("meta") or {}
                if status and m.get("status") != status:
                    continue
                yield p, payload


def load_current(gate, statistic="lfc", results_dir=RESULTS):
    """The single CURRENT artefact for a gate, newest first. None if absent."""
    hits = [(p, d) for p, d in iter_results(results_dir, statistic, gate,
                                            status="CURRENT")]
    if not hits:
        return None, None
    hits.sort(key=lambda t: t[1]["meta"]["timestamp"], reverse=True)
    return hits[0]


def key_numbers(gate, payload):
    """One-line digest of an artefact, for the INDEX table."""
    d = payload
    try:
        if gate == "gate0":
            s = d.get("summary") or []
            v = [x.get("fpr_reject_rank2_pooled",
                       (sum(x["fpr_reject_rank2_per_seed"])
                        / len(x["fpr_reject_rank2_per_seed"]))
                       if x.get("fpr_reject_rank2_per_seed") else None)
                 for x in s]
            v = [x for x in v if x is not None]
            if not v and s:
                v = [x.get("fpr_gt0_median") for x in s if x.get("fpr_gt0_median") is not None]
            rng = f"{min(v):.4f}-{max(v):.4f}" if v else "n/a"
            return f"verdict={d.get('verdict')}; pooled FPR {rng}"
        if gate == "gate1":
            if "data" in d:                      # boundary n-sweep
                return "; ".join(f"n={o['n_e']} {o['scaling'][:3]} {o['reject']:.3f}"
                                 for o in d["data"])
            c = (d.get("summary") or {}).get("curve") or {}
            k1 = {k: v["1"]["frac_reject"] for k, v in c.items() if "1" in v}
            k3 = {k: v["3"]["frac_reject"] for k, v in c.items() if "3" in v}
            return (f"1a={d.get('verdict_1a')} 1b={d.get('verdict_1b')}; "
                    f"k=1 " + "/".join(f"{k1[k]:.3f}" for k in sorted(k1)) +
                    f"; k=3 " + "/".join(f"{k3[k]:.3f}" for k in sorted(k3)))
        if gate == "gate2":
            if "runs" in d and (d.get("summary") or {}).get("kill_criterion"):
                kc = d["summary"]["kill_criterion"]
                sc = d["summary"].get("largest_s_k1_noreject_90pct", {})
                return ("kill " + ("TRIGGERED" if any(v.get("triggered") for v in kc.values())
                                   else "NOT TRIGGERED")
                        + f"; scope s*={sc}")
            if "all_identical" in d:
                return f"s=0 reduction exact on all seeds: {d['all_identical']}"
        if gate in ("power_soft", "power_hard"):
            body = d.get("curve") or d.get("data") or d.get("soft_power_curve")
            if isinstance(body, dict):          # the superseded {n_e:[], raw:[], std:[]}
                return ("raw " + "/".join(f"{x:.3f}" for x in body.get("reject_raw", []))
                        + "; std " + "/".join(f"{x:.3f}" for x in body.get("reject_std", []))
                        + f"; n_e={body.get('n_e')}")
            if body:
                floor = d.get("floor")
                if floor is None:               # derive by the standard rule
                    ns = sorted({o["n_e"] for o in body})
                    ok = [n for n in ns
                          if all(o["reject"] >= 0.80 for o in body if o["n_e"] == n)]
                    floor = ok[0] if ok else None
                return "; ".join(f"n={o['n_e']} {o['scaling'][:3]} {o['reject']:.3f}"
                                 for o in body) + f"; floor={floor}"
        if gate == "envelope":
            body = d.get("sweep") or d.get("data")
            if isinstance(body, list) and body and isinstance(body[0], dict):
                if "s_star_interp" in body[0]:
                    return "; ".join(f"d={o['d_latent']} {o['scaling'][:3]} "
                                     f"s*={o['s_star_interp']:.3f}" for o in body)
                return "; ".join(f"s={o['s']} {o['scaling'][:3]} {o['reject']:.3f}"
                                 for o in body)
            if "raw_text" in d:
                return d["raw_text"].strip().splitlines()[0][:90]
        if gate == "alpha_sensitivity":
            return "; ".join(f"a={o['nominal_alpha']} {o['scaling'][:3]} "
                             f"{o['realised']:.3f}" for o in d.get("data", []))
        if gate == "acceptance":
            a = d.get("acceptance") or {}
            return (f"{a.get('verdict')}; cf "
                    + "/".join(f"{x:.3f}" for x in a.get("cf_reject", [])))
        if gate == "battery" and "raw_text" in d:
            ls = [l for l in d["raw_text"].strip().splitlines() if l.strip()]
            return (f"gap_ratio separation, AUC by n_e: "
                    + "; ".join(l.split()[0] + ":" + l.split()[-1]
                                for l in ls[1:] if len(l.split()) >= 7))
    except Exception as e:
        return f"(digest failed: {type(e).__name__})"
    return ""


def write_index(results_dir=RESULTS):
    """Regenerate INDEX.md from the artefacts themselves."""
    root = Path(results_dir)
    rows = []
    for p, d in iter_results(root):
        m = d["meta"]
        rows.append(dict(
            gate=m["gate"], statistic=m["statistic"],
            path=str(p.relative_to(root)),
            key=key_numbers(m["gate"], d).replace("|", "/"),
            commit=m["git_commit"] or "", status=m["status"] or "",
            superseded_by=m["superseded_by"] or "", timestamp=m["timestamp"]))
    order = {s: i for i, s in enumerate(
        ("CURRENT", "SUPERSEDED", "HISTORICAL-FOR-COMPARISON-TABLE"))}
    rows.sort(key=lambda r: (order.get(r["status"], 9), r["gate"], r["timestamp"]))

    L = ["# ranktest results index",
         "",
         "Generated by `84_results_io.py --index`. Do not hand-edit: rerun it.",
         "",
         "Layout is `<statistic>/<gate>/<timestamp>.json`. Every artefact carries a",
         "mandatory `meta` block (statistic, commit, timestamp, config, versions,",
         "`superseded_by`); `84_results_io.write_results` refuses to write without one.",
         "",
         "Statistics, in the order they were tried and retired:",
         "",
         "| statistic | what it was | why retired |",
         "|---|---|---|",
         "| `diy_retired` | marginal exceedance count, then the zero-signal band `lam[2] > band[2]` | count had no multiplicity control; band calibrated the composite null at its emptiest point and rejected on intervention STRENGTH |",
         "| `cfa_kappa` | Chen-Fang CF-A, kappa-tuned r_hat | r_hat under-selected at the rank-2 boundary |",
         "| `cft` | Chen-Fang CF-T, tuning-free sequential r_hat | r_hat under-selected badly (0 in 108/160 k=1 soft runs); Gate 1a regressed |",
         "| `lfc` | **current**: r_hat pinned to r0=2, least favourable configuration | in use |",
         "",
         "| gate | statistic | key numbers | commit | status | superseded by | path |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['gate']} | `{r['statistic']}` | {r['key']} | `{r['commit']}` "
                 f"| {r['status']} | {r['superseded_by'] or '-'} | `{r['path']}` |")
    L += ["",
          "## Reading rules",
          "",
          "- Quote only rows marked **CURRENT**.",
          "- **SUPERSEDED** rows are kept because deleting them would lose the audit",
          "  trail, not because they are usable. The `cfa_kappa/power_soft` curve in",
          "  particular OVERSTATES power (0.500 vs the LFC 0.300 at n_e=500) and must",
          "  never be quoted.",
          "- **HISTORICAL-FOR-COMPARISON-TABLE** rows exist so the paper can show how",
          "  each retired statistic behaved. They are not results about the method as",
          "  it now stands.",
          "- Regenerate the paper's CSVs with `87_make_gate_tables.py`. It reads only",
          "  CURRENT artefacts and never re-runs a gate.",
          "",
          "Originals from before the migration are preserved untouched in",
          "`_migrated_originals/`; each new artefact records its old path in",
          "`meta.migrated_from`.",
          ""]
    out = root / "INDEX.md"
    out.write_text("\n".join(L))
    return out, len(rows)


if __name__ == "__main__":
    import sys as _s
    if "--index" in _s.argv:
        p, n = write_index()
        print(f"wrote {p} ({n} artefacts)")
        raise SystemExit(0)
    print(__doc__)
    print("statistics:", STATISTICS)
    print("gates     :", GATES)
    print("meta      :", META_FIELDS)
    print("config    :", CONFIG_FIELDS)
    n = sum(1 for _ in iter_results())
    print(f"\nartefacts currently in the layout: {n}")
