"""Remove one method's rows from a benchmark CSV so a resume-run recomputes
them under a revised protocol (e.g. MLEM early stopping). Keeps a backup.

  python experiments/drop_method_rows.py results/benchmark_mayo_fan.csv --method MLEM
"""
import argparse, csv, pathlib, shutil

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--method", required=True)
    ap.add_argument("--incident", type=float, default=None,
                    help="restrict removal to one dose level")
    a = ap.parse_args()
    p = pathlib.Path(a.csv)
    shutil.copy(p, p.with_suffix(".csv.bak"))
    rows = list(csv.DictReader(open(p)))
    keep = [r for r in rows if not (r["method"] == a.method and
            (a.incident is None or float(r["incident"]) == a.incident))]
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(keep)
    where = "" if a.incident is None else f" at inc={a.incident:g}"
    print(f"{p.name}: removed {len(rows)-len(keep)} '{a.method}'{where} rows, "
          f"kept {len(keep)} (backup: {p.with_suffix('.csv.bak').name})")
