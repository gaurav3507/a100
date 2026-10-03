"""Verify the WheelsEye repository.

Two modes:
  python tools/verify_repo.py                       # structure: required files exist and parse
  python tools/verify_repo.py --results DIR         # + recompute each result aggregate from its
                                                    #   per-fold values and compare with the paper

Exit code 0 = all checks passed, 1 = failures (listed). Designed to run in CI
without data (structure mode) and by reviewers with the result files (numbers mode).
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = [
    "README.md", "REPRODUCE.md", "LICENSE", "CITATION.cff", "requirements.txt",
    "paper/main.tex", "results/paper_numbers.json",
    "analysis/calibration/calibration_enroll.py",
    "analysis/calibration/patch_enroll_exact.py",
    "analysis/calibration/patch_meanonly_exact.py",
    "analysis/calibration/patch_dataloader_v1.py",
    "analysis/experiments/tier1_common.py",
    "analysis/experiments/exp_meanonly.py",
    "analysis/experiments/exp_modality_ablation.py",
    "analysis/experiments/exp_calibration_baselines.py",
    "analysis/experiments/exp_leakage_controls.py",
    "analysis/experiments/exp_mephy_calibration.py",
    "analysis/experiments/exp_ordinal_errors.py",
    "analysis/experiments/run_enrollment_curve.py",
    "analysis/experiments/run_multiseed.py",
    "analysis/experiments/uldd_enroll_excluded.py",
    "analysis/experiments/bench_efficiency.py",
    "analysis/stats/paired_tests.py",
    "analysis/stats/collect_v2.py",
    "analysis/stats/collect_ablation.py",
    "analysis/controls/revision_controls.py",
    "analysis/controls/make_controls_csv.py",
    "analysis/rldd/rldd_calibration.py",
]

# claims the manuscript must still make (guards against accidental regression while editing)
TEX_MUST_CONTAIN = [
    "clock cannot predict",              # RLDD clock-free defense
    "does not improve severity grading", # within-session negative result
    "hypothesis rather than a finding",  # unrun ablation not asserted
    "excluded from scoring",             # enrollment-exclusion convention
]
TEX_MUST_NOT_CONTAIN = ["\\tbd{", "\\tbdv", "[TBD", "Deployment-Faithful"]


def structure() -> list[str]:
    fails = []
    for rel in REQUIRED:
        p = ROOT / rel
        if not p.exists():
            fails.append(f"missing: {rel}"); continue
        if p.suffix == ".py":
            try:
                ast.parse(p.read_text(), feature_version=(3, 10))
            except SyntaxError as e:
                fails.append(f"syntax (py3.10): {rel}: {e}")
    tex = (ROOT / "paper/main.tex")
    if tex.exists():
        import re
        raw = tex.read_text()
        # drop comment lines and collapse whitespace so line-wrapped phrases still match
        s = "\n".join(l for l in raw.splitlines() if not l.lstrip().startswith("%"))
        s = re.sub(r"\\(emph|textbf|textit)\{([^}]*)\}", r"\2", s)   # unwrap inline markup
        s = re.sub(r"\s+", " ", s)
        for needle in TEX_MUST_CONTAIN:
            if needle not in s:
                fails.append(f"manuscript lost required statement: {needle!r}")
        for needle in TEX_MUST_NOT_CONTAIN:
            if needle in s:
                fails.append(f"manuscript contains forbidden token: {needle!r}")
    try:
        json.loads((ROOT / "results/paper_numbers.json").read_text())
    except Exception as e:
        fails.append(f"paper_numbers.json invalid: {e}")
    return fails


def _dig(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    return d


def numbers(results_dir: Path) -> tuple[list[str], list[str]]:
    spec = json.loads((ROOT / "results/paper_numbers.json").read_text())
    fails, notes = [], []
    for e in spec["entries"]:
        p = results_dir / e["file"]
        tol = e.get("tolerance", 0.05)
        if not p.exists():
            notes.append(f"  --  {e['table']:>4} {e['label']:<40} file absent ({e['file']})"); continue
        try:
            d = json.loads(p.read_text())
            if "per_fold" in d and "." not in e["metric"]:
                pf = d["per_fold"]
                got = 100.0 * sum(f["metrics"][e["metric"]] for f in pf) / len(pf)
            else:
                got = float(_dig(d, e["metric"]))
        except Exception as ex:
            fails.append(f"{e['file']}: cannot evaluate {e['metric']}: {ex}"); continue
        ok = abs(got - e["paper"]) <= tol
        line = f"  {'OK ' if ok else 'BAD'} {e['table']:>4} {e['label']:<40} paper={e['paper']:<7} got={got:.2f}"
        notes.append(line)
        if not ok:
            fails.append(line.strip())
    return fails, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, help="directory with result JSONs (enables number checks)")
    a = ap.parse_args()

    f = structure()
    print(f"structure: {len(REQUIRED)} required files, {len(f)} problems")
    for x in f: print("  -", x)

    if a.results:
        nf, notes = numbers(a.results)
        print(f"\nnumbers vs paper ({a.results}):")
        for n in notes: print(n)
        print(f"{len(nf)} mismatches")
        f += nf

    print("\nRESULT:", "PASS" if not f else f"FAIL ({len(f)})")
    return 0 if not f else 1


if __name__ == "__main__":
    raise SystemExit(main())
