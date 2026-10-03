"""Regenerate causalbench/results/INDEX.md by walking results/ranktest/.

READ-ONLY. It opens artefacts and writes exactly one file, INDEX.md. It never
modifies, moves or deletes an artefact.

Timestamps are rendered verbatim. Most existing artefacts carry no timezone
marker at all, so they are shown as-is with " (tz unmarked)" appended rather
than being guessed at or rewritten. New artefacts are written in UTC with a Z.

Usage:
    python causalbench/scripts/89_make_index.py
    python causalbench/scripts/89_make_index.py --root <dir> --out <file>
"""
import argparse
import datetime
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
# scripts/ sits at the repo root in this repository, so the root is
# one level up, not two. (It was two when these lived in
# causalbench/scripts/ in the source repo.)
REPO = HERE.parent
DEFAULT_ROOT = REPO / "results"
DEFAULT_OUT = REPO / "results/INDEX.md"
NA = "n/a"

def rel(p):
    """Repo-relative when possible, absolute otherwise. --dir may be relative."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)



def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def script_commit():
    r = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%h",
                        "--", str(Path(__file__).relative_to(REPO))],
                       capture_output=True, text=True)
    return r.stdout.strip() or "uncommitted"


def render_ts(ts):
    if not ts:
        return NA
    s = str(ts)
    return s if s.endswith("Z") else f"{s} (tz unmarked)"


def g(d, *path, default=NA):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return default if cur is None else cur


def dataset_of(doc):
    if not isinstance(doc, dict):
        return NA
    ds = doc.get("dataset")
    arm = g(doc, "meta", "config", "extra", "arm", default=None)
    if ds and arm:
        return f"{ds} ({arm})"
    if ds:
        return str(ds)
    # non-descriptive artefacts have no dataset field
    return NA


def collect(root):
    rows = []
    for p in sorted(Path(root).rglob("*.json")):
        try:
            doc = json.load(open(p))
        except ValueError:
            rows.append(dict(statistic=NA, gate=NA, dataset=NA, timestamp=NA,
                             commit=NA, status="UNPARSEABLE", superseded_by=NA,
                             path=rel(p)))
            continue
        # Pre-migration originals under _migrated_originals/ are bare JSON
        # lists with no meta block. They are still artefacts on disk, so they
        # are listed rather than dropped: an index that silently omits files
        # is worse than one that flags them.
        if not isinstance(doc, dict) or not isinstance(doc.get("meta"), dict):
            rows.append(dict(statistic=NA, gate=NA, dataset=NA, timestamp=NA,
                             commit=NA, status="NO-META", superseded_by="-",
                             path=rel(p)))
            continue
        m = doc["meta"]
        rows.append(dict(
            statistic=g(m, "statistic"), gate=g(m, "gate"),
            dataset=dataset_of(doc),
            timestamp=render_ts(m.get("timestamp")),
            commit=g(m, "git_commit"), status=g(m, "status"),
            superseded_by=m.get("superseded_by") or "-",
            path=rel(p),
        ))
    rows.sort(key=lambda r: (str(r["statistic"]), str(r["gate"]),
                             str(r["timestamp"])))
    return rows


COLS = ["statistic", "gate", "dataset", "timestamp", "commit", "status",
        "superseded_by", "path"]


def section(title, rows):
    L = [f"## {title} ({len(rows)})", ""]
    if not rows:
        L += ["_none_", ""]
        return L
    L.append("| " + " | ".join(COLS) + " |")
    L.append("|" + "|".join(["---"] * len(COLS)) + "|")
    for r in rows:
        L.append("| " + " | ".join(
            f"`{r[c]}`" if c in ("commit", "path") else str(r[c])
            for c in COLS) + " |")
    L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()

    rows = collect(a.root)
    cur = [r for r in rows if r["status"] == "CURRENT"]
    sup = [r for r in rows if r["status"] == "SUPERSEDED"]
    other = [r for r in rows if r["status"] not in ("CURRENT", "SUPERSEDED")]

    L = ["# ranktest results index", "",
         "```",
         f"generated : {utc_now()}",
         f"generator : 89_make_index.py @ {script_commit()}",
         f"root      : {rel(a.root)}",
         f"artefacts : {len(rows)}",
         "```", "",
         "Regenerate with `python causalbench/scripts/89_make_index.py`. "
         "Do not hand-edit.", "",
         "Quote only rows in **CURRENT**. Timestamps without a `Z` carry no "
         "timezone marker in the artefact and are shown verbatim.", ""]
    L += section("CURRENT", cur)
    L += section("SUPERSEDED", sup)
    if other:
        # not dropped: an artefact that is neither CURRENT nor SUPERSEDED still
        # exists on disk, and omitting it would make the index a partial view
        L += section("OTHER (historical / unparseable)", other)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print(f"wrote {rel(out)}  "
          f"({len(cur)} CURRENT, {len(sup)} SUPERSEDED, {len(other)} other)")


if __name__ == "__main__":
    main()
