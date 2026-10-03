"""Report whether the target-column drop in project() was LIVE or a NO-OP
for K562, RPE1, and Norman.

03_screen.py's project() zeros the row of the PCA basis matrix that
corresponds to the perturbation target's own expression column, if the
target's name appears in var_names. If none of the target names appear as
expression columns, the drop is a no-op and every screen number for that
dataset was computed without the correction. Frangieh's target-column
drop is LIVE (239/248 targets present as columns), so any comparison
against a dataset where the drop was a no-op is asymmetric.

Norman already records target_column_drop_is_noop in norman.json (0/105).
K562 and RPE1 were run through 03_screen.py, which does not store that
field, so this diagnostic loads the .npz files and counts the overlap
directly.

Usage (A100):
    python causalbench/scripts/43_check_drop_status.py
"""
import json
import os
from pathlib import Path

import numpy as np

DATA = Path("/workspace/meridian-identifiability/causalbench/data")
SCREEN_DIR = Path("/workspace/meridian-identifiability/causalbench/results/screen")
OUT_JSON = SCREEN_DIR / "target_drop_status.json"

CTRL = "non-targeting"


def check_npz(ds, filt):
    name = f"dataset_{ds}" + ("_filtered" if filt else "") + ".npz"
    path = DATA / name
    if not path.exists():
        return dict(dataset=ds, filter=bool(filt), path=str(path),
                    aborted="file_not_found")
    d = np.load(path, allow_pickle=True)
    iv = np.asarray(d["interventions"])
    vn = [str(v) for v in d["var_names"]]
    gene_set = set(vn)
    targets = sorted({str(t) for t in iv if t not in (CTRL, "excluded")})
    in_cols = sorted(t for t in targets if t in gene_set)
    n_targets = len(targets)
    n_in = len(in_cols)
    return dict(
        dataset=ds, filter=bool(filt), path=str(path),
        n_var_names=int(len(vn)),
        n_unique_targets=int(n_targets),
        n_targets_in_expression_columns=int(n_in),
        target_column_drop_is_noop=bool(n_in == 0),
        drop_status="NO-OP" if n_in == 0 else "LIVE",
        example_overlapping=in_cols[:10],
    )


def read_norman_json():
    p = SCREEN_DIR / "norman.json"
    if not p.exists():
        return dict(dataset="norman", aborted="norman.json not found")
    j = json.load(open(p))
    return dict(
        dataset="norman",
        source="stored norman.json",
        n_var_names=int(j.get("n_genes", -1)),
        n_unique_targets=int(j.get("n_single_perturbation_genes", -1)),
        n_targets_in_expression_columns=int(j.get("n_targets_in_var_names",
                                                    -1)),
        target_column_drop_is_noop=bool(j.get("target_column_drop_is_noop",
                                                False)),
        drop_status=("NO-OP" if j.get("target_column_drop_is_noop")
                     else "LIVE"),
    )


def main():
    rows = []
    for ds in ("k562", "rpe1"):
        for filt in (False, True):
            rows.append(check_npz(ds, filt))
    rows.append(read_norman_json())

    out = dict(
        note="Reports whether project()'s target-column drop was LIVE or "
             "NO-OP in the runs that produced the screen numbers.",
        rows=rows,
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(OUT_JSON) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.rename(tmp, OUT_JSON)

    print("=" * 78)
    print("TARGET-COLUMN DROP STATUS")
    print("=" * 78)
    print(f"  {'dataset':<8}{'filt':>6}{'n_targets':>12}"
          f"{'n_in_cols':>12}{'status':>10}")
    for r in rows:
        if "aborted" in r:
            print(f"  {r['dataset']:<8}  ABORT: {r['aborted']}")
            continue
        print(f"  {r['dataset']:<8}{str(r.get('filter', '-')):>6}"
              f"{r['n_unique_targets']:>12}"
              f"{r['n_targets_in_expression_columns']:>12}"
              f"{r['drop_status']:>10}")
    print(f"\n[write] {OUT_JSON}")


if __name__ == "__main__":
    main()
