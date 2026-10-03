"""Tier-2 #8: ordinal error analysis from existing result JSONs (no re-run).
Reports mean absolute error in class steps, adjacent-error share and two-step (Low<->High) error share.
Usage: python exp_ordinal_errors.py [--dir data/processed/results] [pattern]
"""
import argparse, json, numpy as np
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="data/processed/results"); ap.add_argument("pattern", nargs="?", default="tier3_*")
a = ap.parse_args()
print(f"{'result':<52}{'acc%':>7}{'MAE(steps)':>11}{'1-step err%':>12}{'2-step err%':>12}")
print("-" * 94)
for p in sorted(Path(a.dir).glob(a.pattern + ".json")):
    d = json.loads(p.read_text()); cm = np.zeros((3, 3))
    for f in d.get("per_fold", []):
        m = f["metrics"].get("confusion_matrix")
        if m is not None and len(m) == 3: cm += np.asarray(m)
    if cm.sum() == 0: continue
    i, j = np.indices(cm.shape); dist = np.abs(i - j)
    n = cm.sum(); mae = (cm * dist).sum() / n
    print(f"{p.stem[:52]:<52}{np.trace(cm)/n*100:>7.2f}{mae:>11.3f}{cm[dist==1].sum()/n*100:>12.2f}{cm[dist==2].sum()/n*100:>12.2f}")
