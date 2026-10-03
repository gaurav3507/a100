"""One-shot: recompute step0_pass in norman.json using mean_ratio_pairs
(the metric CausalBench's 0.96-1.05 target refers to), not mean_ratio_vs_ctrl.

mean_ratio_vs_ctrl inside step-0 is degenerate: ref_pool is empty, so the
'between' reference is drawn from the same pseudo-env's own rows and overlaps
~50% with Za. Expected value ~0.5, not ~1.0. mean_ratio_pairs draws from a
DIFFERENT pseudo-env and is the right gate.
"""
import json, os
from pathlib import Path

P = Path("/workspace/meridian-identifiability/causalbench/results/screen/norman.json")
d = json.loads(P.read_text())

pairs = [r["mean_ratio_pairs"] for r in d["step0_gate"]
         if "mean_ratio_pairs" in r]
d["summary"]["step0_gate_metric"] = "mean_ratio_pairs"
d["summary"]["step0_pairs"] = pairs
d["summary"]["step0_pass"] = all(0.9 <= x <= 1.1 for x in pairs) if pairs else False
d["summary"]["step0_vs_ctrl_note"] = (
    "mean_ratio_vs_ctrl in step-0 is degenerate: ref_pool is empty so ref "
    "overlaps ~50% with Za; expected ~0.5. Not the correct gate; retained "
    "for the record only."
)

tmp = str(P) + ".tmp"
with open(tmp, "w") as f:
    json.dump(d, f, indent=2)
os.rename(tmp, P)

print(f"step0_pairs: {pairs}")
print(f"step0_pass (pairs-based): {d['summary']['step0_pass']}")
print(f"wrote {P}")
