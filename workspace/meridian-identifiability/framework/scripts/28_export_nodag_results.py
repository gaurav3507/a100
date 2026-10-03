"""Export no-DAG ablation results to canonical JSON, next to the full-model results.

For each init in {shift, random}, in priority order:
  1. If results/model/zeroshot_nodag_{init}.json already has all required keys,
     re-emit it verbatim (source="existing_json"). Idempotent.
  2. Else parse logs/nodag_{init}.log for the final training block
     (source="parsed_log"). Per-gene R2 table is NOT recoverable this way.

Required keys: best_median_r2, final_median_r2, final_mean_r2,
               frac_beats_gmean, frac_beats_ridge.

Output: results/model/zeroshot_nodag_{init}.json  (two files total).

Usage:
    python scripts/28_export_nodag_results.py
"""
import os, json, re
from pathlib import Path

FW = Path("/workspace/meridian-identifiability/framework")
MODEL_DIR = FW / "results" / "model"
LOG_DIR = FW / "logs"
INITS = ("shift", "random")
REQUIRED = ("best_median_r2", "final_median_r2", "final_mean_r2",
            "frac_beats_gmean", "frac_beats_ridge")

RE_EP = re.compile(
    r"^ep\s+(\d+)\s+zeroshot R2 median\s+([+-]?\d+\.\d+)\s+mean\s+([+-]?\d+\.\d+)"
)
RE_BEST = re.compile(r"^\s*median R2\s+([+-]?\d+\.\d+)\s+vs ridge")
RE_BEATS = re.compile(r"^\s*beats gmean\s+(\d+)%,\s+beats ridge\s+(\d+)%")


def load_existing(init):
    p = MODEL_DIR / f"zeroshot_nodag_{init}.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        print(f"[{init}] existing {p.name} missing keys {missing}; "
              f"falling back to log parse", flush=True)
        return None
    d.setdefault("source", "existing_json")
    return d


def parse_log(init):
    p = LOG_DIR / f"nodag_{init}.log"
    if not p.exists():
        raise FileNotFoundError(f"log not found: {p}")
    ep_lines = []
    best_med = None
    beats_gmean = None
    beats_ridge = None
    with open(p) as f:
        for line in f:
            m = RE_EP.match(line)
            if m:
                ep_lines.append((int(m.group(1)),
                                 float(m.group(2)),
                                 float(m.group(3))))
                continue
            m = RE_BEST.match(line)
            if m:
                best_med = float(m.group(1))
                continue
            m = RE_BEATS.match(line)
            if m:
                beats_gmean = int(m.group(1)) / 100.0
                beats_ridge = int(m.group(2)) / 100.0
                continue
    if not ep_lines:
        raise RuntimeError(f"no 'ep XXX zeroshot R2' lines in {p}")
    ep_lines.sort()
    _, final_med, final_mean = ep_lines[-1]
    if best_med is None:
        best_med = max(e[1] for e in ep_lines)
    return dict(
        variant="no_causal_layer",
        init=init,
        source="parsed_log",
        best_median_r2=best_med,
        final_median_r2=final_med,
        final_mean_r2=final_mean,
        frac_beats_gmean=beats_gmean,
        frac_beats_ridge=beats_ridge,
        baselines=dict(global_mean=0.129, ridge=0.226),
    )


def atomic_write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.rename(tmp, path)


def _fmt(v):
    return f"{v:+.4f}" if isinstance(v, float) else str(v)


def main():
    for init in INITS:
        rec = load_existing(init) or parse_log(init)
        out = MODEL_DIR / f"zeroshot_nodag_{init}.json"
        atomic_write_json(out, rec)
        print(f"[{init}] source={rec['source']}", flush=True)
        print(f"[{init}]   best_median_r2  = {_fmt(rec['best_median_r2'])}", flush=True)
        print(f"[{init}]   final_median_r2 = {_fmt(rec['final_median_r2'])}", flush=True)
        print(f"[{init}]   final_mean_r2   = {_fmt(rec['final_mean_r2'])}", flush=True)
        print(f"[{init}]   frac_beats_gmean = {rec['frac_beats_gmean']}", flush=True)
        print(f"[{init}]   frac_beats_ridge = {rec['frac_beats_ridge']}", flush=True)
        print(f"[{init}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
