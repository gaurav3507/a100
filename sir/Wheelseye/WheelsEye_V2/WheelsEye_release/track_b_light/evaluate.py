"""Track B evaluation (tasks.md Task 4.3): same metrics format as Track A
(track_a_heavy/evaluate.py) for direct comparison. Reads the per-model
result JSONs written by track_b_light/train.py and prints one row per
option (MLP/LightGBM/GRU) plus which one wins the accuracy/latency
tradeoff (tasks.md 4.1: pick the best as "the" Track B deployment model).

Usage: .venv/Scripts/python.exe track_b_light/evaluate.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "track_b_results"
MODEL_LABELS = {"mlp": "MLP (64-32-16)", "lightgbm": "LightGBM", "gru": "GRU (context)"}


def main() -> int:
    results = {}
    for name in MODEL_LABELS:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            results[name] = json.loads(path.read_text())

    if not results:
        print(f"No Track B results found in {RESULTS_DIR} -- run track_b_light/train.py for each model first.")
        return 1

    header = f"{'Model':20s} {'Accuracy':>12s} {'Precision':>12s} {'Recall':>12s} {'F1':>12s} {'n_folds':>8s}"
    print(header)
    print("-" * len(header))
    best_name, best_acc = None, -1.0
    for name, agg in results.items():
        acc = agg["accuracy"]["mean"]
        print(
            f"{MODEL_LABELS[name]:20s} "
            f"{acc*100:10.2f}%  "
            f"{agg['precision_macro']['mean']*100:10.2f}%  "
            f"{agg['recall_macro']['mean']*100:10.2f}%  "
            f"{agg['f1_macro']['mean']*100:10.2f}%  "
            f"{agg['n_runs']:8d}"
        )
        if acc > best_acc:
            best_name, best_acc = name, acc

    print(f"\nBest by accuracy: {MODEL_LABELS[best_name]} ({best_acc*100:.2f}%)")
    print("Note: final Track B model choice must also weigh latency/model size/peak "
          "memory on the Jetson (benchmarks/profile_latency.py, profile_memory.py) -- "
          "not accuracy alone (tasks.md 4.1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
