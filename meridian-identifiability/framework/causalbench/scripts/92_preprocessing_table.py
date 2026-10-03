"""Preprocessing-sweep tables: participation ratio and numerical rank by arm.

Reads only CURRENT artefacts under descriptive/preprocessing_sweep/. Runs
nothing. Emits markdown + CSV to causalbench/paper/tables/.

Two tables, same shape: rows = dataset, columns = the four arms.
  table_c_preprocessing_pr        participation ratio, plus a spread column
                                  (max/min across the coherent arms)
  table_d_preprocessing_numrank   numerical rank at tol 1e-10

An arm that was not coherent for a dataset renders "n/a", never 0 and never
blank, and the footnote names which arms were coherent where. That distinction
is the point of the sweep: log1p_std on already-log-normalised input is a
double log, not a preprocessing choice, and reporting it as a number would
invite exactly the misreading the sweep exists to prevent.

Usage:
    python causalbench/scripts/92_preprocessing_table.py
"""
import argparse
import csv
import datetime
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_DIR = REPO / "causalbench/results/ranktest/descriptive/preprocessing_sweep"
OUT_DIR = REPO / "causalbench/paper/tables"
NA = "n/a"
ARMS = ("raw", "standardise", "log1p_std", "rank_int")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def script_commit():
    r = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%h",
                        "--", str(Path(__file__).relative_to(REPO))],
                       capture_output=True, text=True)
    return r.stdout.strip() or "uncommitted"


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return NA if v is None else str(v)


def load(dirpath):
    kept, skipped = [], []
    for p in sorted(Path(dirpath).glob("*.json")):
        try:
            doc = json.load(open(p))
        except ValueError as e:
            skipped.append((p.name, f"unparseable: {e}"))
            continue
        if (doc.get("meta") or {}).get("status") != "CURRENT":
            skipped.append((p.name,
                            f"status={(doc.get('meta') or {}).get('status')!r}"))
            continue
        kept.append((p, doc))
    return kept, skipped


def build(kept, field):
    rows = []
    for _, doc in kept:
        res = doc.get("arms_result") or {}
        row = {"dataset": doc.get("dataset", NA)}
        vals = []
        for arm in ARMS:
            r = res.get(arm) or {}
            if r.get("status") == "ok" and r.get(field) is not None:
                row[arm] = r[field]
                if field == "participation_ratio":
                    vals.append(float(r[field]))
            else:
                row[arm] = NA
        if field == "participation_ratio":
            row["spread_max_over_min"] = (round(max(vals) / min(vals), 3)
                                          if len(vals) > 1 and min(vals) > 0
                                          else NA)
        rows.append(row)
    rows.sort(key=lambda r: str(r["dataset"]))
    return rows


def coherence_footnote(kept):
    bits = []
    for _, doc in kept:
        res = doc.get("arms_result") or {}
        ok = [a for a in ARMS if (res.get(a) or {}).get("status") == "ok"]
        bad = {a: (res.get(a) or {}).get("reason")
               for a in ARMS if (res.get(a) or {}).get("status") != "ok"}
        kind = doc.get("input_kind", "unknown")
        bit = (f"{doc.get('dataset')}: loader returns {kind}; coherent arms "
               f"{', '.join(ok) if ok else 'none'}")
        if bad:
            bit += "; " + "; ".join(f"{a} omitted ({r})" for a, r in bad.items())
        bits.append(bit)
    return ("Arm coherence is decided per dataset from a runtime probe of the "
            "control matrix, not assumed. " + " | ".join(bits))


def header_lines(kept, skipped, dirpath):
    commits = {}
    for p, doc in kept:
        commits.setdefault((doc.get("meta") or {}).get("git_commit", NA),
                           []).append(p.name)
    L = [f"generated  : {utc_now()}",
         f"generator  : 92_preprocessing_table.py @ {script_commit()}",
         f"source dir : {rel(dirpath)}",
         f"artefacts  : {len(kept)} CURRENT, {len(skipped)} skipped"]
    if len(commits) > 1:
        L.append("!! WARNING: MIXED PROVENANCE -- sources do not share one "
                 "meta.git_commit; rows below come from different versions.")
    for c, names in sorted(commits.items()):
        for n in sorted(names):
            L.append(f"  source   : {n}   meta.git_commit={c}")
    for n, why in skipped:
        L.append(f"  skipped  : {n}   ({why})")
    return L


def write_outputs(name, rows, header, footnotes, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else []
    with open(outdir / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        for line in header:
            w.writerow([f"# {line}"])
        if cols:
            w.writerow(cols)
            for r in rows:
                w.writerow([fmt(r[c]) for c in cols])
        for fn in footnotes:
            w.writerow([f"# {fn}"])
    L = [f"# {name}", "", "```"] + header + ["```", ""]
    if cols:
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in rows:
            L.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    else:
        L.append("_no CURRENT preprocessing-sweep artefacts found_")
    L.append("")
    for fn in footnotes:
        L += [f"> {fn}", ""]
    (outdir / f"{name}.md").write_text("\n".join(L))
    print(f"  wrote {name}.md / .csv   {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--outdir", default=str(OUT_DIR))
    a = ap.parse_args()
    kept, skipped = load(a.dir)
    head = header_lines(kept, skipped, a.dir)
    print("\n".join(head))
    if not kept:
        print("\nno CURRENT artefacts; tables not written")
        return
    fn = [coherence_footnote(kept)]
    write_outputs("table_c_preprocessing_pr",
                  build(kept, "participation_ratio"), head, fn, a.outdir)
    write_outputs("table_d_preprocessing_numrank",
                  build(kept, "numerical_rank"), head, fn, a.outdir)


if __name__ == "__main__":
    main()
