"""Modality-lattice analysis over tier3_ablation_*_enroll{k}.json (all 2^G group combinations).
For each feature group: mean accuracy over all subsets WITH vs WITHOUT it (marginal contribution),
plus the sensor-level comparisons (vision = face_geom+fau+posture, bio = cardiac+autonomic, grip = grip_motion).
Usage: python collect_ablation.py [--dir data/processed/results] [--k 96]
"""
import argparse, json, itertools
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="data/processed/results"); ap.add_argument("--k", type=int, default=96); ap.add_argument("--suffix", default="", help="e.g. _mean"); a = ap.parse_args()
res = {}
for p in Path(a.dir).glob(f"tier3_ablation_*_enroll{a.k}{a.suffix}.json"):
    d = json.loads(p.read_text()); s = d["config"]["subset"]
    acc = np.mean([f["metrics"]["accuracy"] for f in d["per_fold"]]); f1 = np.mean([f["metrics"]["f1_macro"] for f in d["per_fold"]])
    res[s] = (acc * 100, f1 * 100, d["config"]["n_features"])
groups = sorted({g for s in res for g in s.split("+") if s not in ("all", "all+context")})
print(f"parsed {len(res)} subsets; groups: {groups}\n")
print("TABLE A -- marginal contribution of each group (mean acc over subsets with vs without it)")
print(f"{'group':<14}{'with':>8}{'without':>9}{'Δ acc':>8}{'n_with':>7}")
for g in groups:
    w = [v[0] for s, v in res.items() if s not in ("all", "all+context") and g in s.split("+")]
    wo = [v[0] for s, v in res.items() if s not in ("all", "all+context") and g not in s.split("+")]
    print(f"{g:<14}{np.mean(w):>8.2f}{np.mean(wo):>9.2f}{np.mean(w)-np.mean(wo):>+8.2f}{len(w):>7}")
def key(*gs): return None if any(g not in groups for g in gs) else "+".join(sorted(gs, key=groups.index))
combos = {"vision only (face_geom+fau+posture)": key("face_geom", "fau", "posture"), "bio only (cardiac+autonomic)": key("cardiac", "autonomic"),
          "grip only": "grip_motion", "bio + grip": key("cardiac", "autonomic", "grip_motion"),
          "bio + vision": key("face_geom", "fau", "posture", "cardiac", "autonomic"), "all six groups": "all", "all + context": "all+context"}
print("\nTABLE B -- sensor-level comparisons (LOSO, %d-window enrollment)" % a.k)
print(f"{'configuration':<40}{'features':>9}{'acc %':>8}{'F1 %':>8}")
for name, k in combos.items():
    # subsets were joined in group order; try both the canonical key and any permutation present
    hit = None if k is None else (res.get(k) or next((v for s, v in res.items() if set(s.split("+")) == set(k.split("+"))), None))
    print(f"{name:<40}" + (f"{hit[2]:>9}{hit[0]:>8.2f}{hit[1]:>8.2f}" if hit else f"{'(missing)':>25}"))
best = max(res.items(), key=lambda kv: kv[1][0]); print(f"\nbest subset: {best[0]}  acc={best[1][0]:.2f}  f1={best[1][1]:.2f}  ({best[1][2]} features)")
