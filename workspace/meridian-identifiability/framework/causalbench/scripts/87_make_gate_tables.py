"""Emit the paper's tables as CSV from the CURRENT results artefacts.

Reads only what is already on disk under results/ranktest/<statistic>/<gate>/.
REGENERATING TABLES NEVER RE-RUNS A GATE: if an artefact is missing the table
is skipped with a note naming the artefact, rather than silently computed or
silently emptied.

Tables:
  (i)   comparison_k1.csv        three statistics at k=1
  (ii)  gate1_full.csv           Gate 1, k x {hard,soft} x {raw,std}
  (iii) rhat_monotonicity.csv    rejection vs the r_hat used in the projection
  (iv)  boundary_nsweep.csv      rank-2 boundary excess vs n_e
  (v)   power_curves.csv         soft and hard curves with their floors
  (vi)  mixing_envelope.csv      s* by d_latent, with the s=0 linear control

Usage:
    python causalbench/scripts/87_make_gate_tables.py [--outdir paper/tables]
"""
import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_rio", HERE / "84_results_io.py")
RIO = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(RIO)

DEFAULT_OUT = HERE.parents[1] / "paper" / "tables"
SKIPPED = []


def _write(outdir, name, header, rows):
    p = Path(outdir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {name:<26} {len(rows)} rows")
    return p


def _need(payload, what):
    if payload is None:
        SKIPPED.append(what)
        print(f"  SKIP  {what}: artefact not found")
        return False
    return True


def _latest(gate, statistic, status=None):
    hits = [(p, d) for p, d in RIO.iter_results(statistic=statistic, gate=gate)
            if (status is None or d["meta"]["status"] == status)
            and "__" not in p.name]
    if not hits:
        return None
    hits.sort(key=lambda t: t[1]["meta"]["timestamp"], reverse=True)
    return hits[-1] if status is None else hits[0]


def _suffixed(gate, statistic, suffix):
    for p, d in RIO.iter_results(statistic=statistic, gate=gate):
        if p.name.endswith(suffix + ".json"):
            return p, d
    return None


# ------------------------------------------------------------------ (i)
def table_comparison_k1(outdir):
    rows = []
    for stat in ("diy_retired", "cfa_kappa", "cft", "lfc"):
        hit = _latest("gate1", stat)
        if hit is None:
            continue
        p, d = hit
        curve = (d.get("summary") or {}).get("curve") or {}
        for key, byk in sorted(curve.items()):
            if "1" not in byk:
                continue
            scaling, kind = key.split("|")
            rows.append([stat, scaling, kind, byk["1"]["frac_reject"],
                         d["meta"]["git_commit"], d["meta"]["timestamp"]])
    if not rows:
        SKIPPED.append("comparison_k1"); print("  SKIP  comparison_k1"); return
    _write(outdir, "comparison_k1.csv",
           ["statistic", "scaling", "kind", "reject_rate_k1", "commit", "timestamp"],
           rows)


# ------------------------------------------------------------------ (ii)
def table_gate1_full(outdir):
    hit = _latest("gate1", "lfc", status="CURRENT")
    if not _need(hit, "gate1_full"):
        return
    p, d = hit
    rows = []
    for key, byk in sorted((d["summary"]["curve"]).items()):
        scaling, kind = key.split("|")
        for k in sorted(byk, key=int):
            rows.append([scaling, kind, int(k), byk[k]["frac_reject"],
                         byk[k].get("n_runs")])
    _write(outdir, "gate1_full.csv",
           ["scaling", "kind", "k", "reject_rate", "n_runs"], rows)


# ----------------------------------------------------------------- (iii)
def table_rhat_monotonicity(outdir):
    """Rejection vs the r_hat used in the projection, pooled over k=1 runs.

    Taken from the CF-T artefact, which is the only one that varied r_hat: LFC
    pins it to 2 by construction and so cannot show the monotonicity, and that
    monotonicity is the entire justification for the LFC choice.
    """
    hit = _latest("gate1", "cft")
    if not _need(hit, "rhat_monotonicity"):
        return
    p, d = hit
    runs = [r for r in d.get("runs", [])
            if r.get("k") == 1 and r.get("kind") != "shift" and "cf_r_hat" in r]
    if not runs:
        SKIPPED.append("rhat_monotonicity"); print("  SKIP  rhat_monotonicity"); return
    rows = []
    for v in sorted({r["cf_r_hat"] for r in runs}):
        sel = [r for r in runs if r["cf_r_hat"] == v]
        rows.append([v, len(sel),
                     round(sum(bool(r["reject"]) for r in sel) / len(sel), 4)])
    _write(outdir, "rhat_monotonicity.csv",
           ["r_hat_used", "n_runs", "reject_rate"], rows)


# ------------------------------------------------------------------ (iv)
def table_boundary_nsweep(outdir):
    hit = _suffixed("gate1", "lfc", "__boundary_nscaling")
    if not _need(hit, "boundary_nsweep"):
        return
    p, d = hit
    rows = [[o["n_e"], o["scaling"], o["reject"], round(o["reject"] / 0.05, 3),
             o.get("n_draws")] for o in d["data"]]
    _write(outdir, "boundary_nsweep.csv",
           ["n_e", "scaling", "reject_rate", "multiple_of_nominal", "n_draws"], rows)


# ------------------------------------------------------------------ (v)
def table_power_curves(outdir):
    rows = []
    soft = _latest("power_soft", "lfc", status="CURRENT")
    if soft:
        p, d = soft
        cur = d["data"]
        ns = sorted({c["n_e"] for c in cur})
        ok = [n for n in ns if all(c["reject"] >= 0.80 for c in cur if c["n_e"] == n)]
        floor = ok[0] if ok else None
        for c in cur:
            rows.append(["soft", c["n_e"], c["scaling"], c["reject"], floor, "CURRENT"])
    hard = _latest("power_hard", "lfc", status="CURRENT")
    if hard:
        p, d = hard
        body = d.get("curve") or d.get("data")
        floor = d.get("floor")
        for c in body:
            rows.append(["hard", c["n_e"], c["scaling"], c["reject"], floor, "CURRENT"])
    sup = _latest("power_soft", "cfa_kappa")
    if sup:
        p, d = sup
        c = d["data"] if "data" in d else d["soft_power_curve"]
        for i, n in enumerate(c["n_e"]):
            rows.append(["soft_SUPERSEDED", n, "raw", c["reject_raw"][i], None,
                         "SUPERSEDED do not quote"])
            rows.append(["soft_SUPERSEDED", n, "standardised", c["reject_std"][i],
                         None, "SUPERSEDED do not quote"])
    if not rows:
        SKIPPED.append("power_curves"); print("  SKIP  power_curves"); return
    _write(outdir, "power_curves.csv",
           ["intervention", "n_e", "scaling", "reject_rate", "floor", "status"], rows)


# ------------------------------------------------------------------ (vi)
def table_mixing_envelope(outdir):
    hit = _latest("envelope", "lfc", status="CURRENT")
    if not _need(hit, "mixing_envelope"):
        return
    p, d = hit
    body = d.get("sweep") or d.get("data") or []
    ctrl = {(c["d_latent"], c["scaling"]): c["reject_at_s0"]
            for c in (d.get("s0_linear_control") or [])}
    rows = []
    for o in body:
        rows.append([o["d_latent"], o["d_proj"], o["scaling"],
                     o["s_star_grid"], round(o["s_star_interp"], 4),
                     ctrl.get((o["d_latent"], o["scaling"])),
                     o.get("s_star_valid"), o.get("s_star_note", "")])
    _write(outdir, "mixing_envelope.csv",
           ["d_latent", "d_proj", "scaling", "s_star_grid", "s_star_interp",
            "reject_at_s0_linear_control", "s_star_valid", "note"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    print(f"reading CURRENT artefacts from {RIO.RESULTS}")
    print(f"writing tables to {a.outdir}\n")
    table_comparison_k1(a.outdir)
    table_gate1_full(a.outdir)
    table_rhat_monotonicity(a.outdir)
    table_boundary_nsweep(a.outdir)
    table_power_curves(a.outdir)
    table_mixing_envelope(a.outdir)
    print(f"\n{6 - len(SKIPPED)}/6 tables written"
          + (f"; SKIPPED {SKIPPED}" if SKIPPED else ""))


if __name__ == "__main__":
    main()
