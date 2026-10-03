"""Track A evaluation (tasks.md Task 3.3): aggregate metrics across folds
from track_a_heavy/train.py's saved results, produce a confusion matrix
figure and a results table in the same format as FatigueNet's Table 2/3
(Accuracy / Precision / Recall / F1 / Inference Time, one row per model
variant -- here just the one row for full Track A, extended by
track_a_heavy/ablation.py's variants).

Usage: .venv/Scripts/python.exe track_a_heavy/evaluate.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from common.labels import CLASS_NAMES

RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "track_a_results.json"
FIGURE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "track_a_confusion_matrix.png"


def print_results_table(agg: dict, model_name: str = "Track A (FatigueNet-style)") -> None:
    header = f"{'Model':30s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}"
    print(header)
    print("-" * len(header))
    print(
        f"{model_name:30s} "
        f"{agg['accuracy']['mean']*100:9.2f}% "
        f"{agg['precision_macro']['mean']*100:9.2f}% "
        f"{agg['recall_macro']['mean']*100:9.2f}% "
        f"{agg['f1_macro']['mean']*100:9.2f}%"
    )
    print(
        f"{'':30s} "
        f"(+/-{agg['accuracy']['std']*100:.2f}) "
        f"(+/-{agg['precision_macro']['std']*100:.2f}) "
        f"(+/-{agg['recall_macro']['std']*100:.2f}) "
        f"(+/-{agg['f1_macro']['std']*100:.2f})"
    )
    print(f"\nAggregated over {agg['n_runs']} LOSO fold(s).")


def plot_confusion_matrix(agg: dict, out_path: Path = FIGURE_PATH) -> None:
    cm = np.array(agg["confusion_matrix_sum"])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Purples", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, cbar_kws={"label": "row-normalized"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Track A confusion matrix (summed over LOSO folds)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix figure to {out_path}")


def main() -> int:
    if not RESULTS_PATH.exists():
        print(f"No results found at {RESULTS_PATH} -- run track_a_heavy/train.py first.")
        return 1
    agg = json.loads(RESULTS_PATH.read_text())
    print_results_table(agg)
    plot_confusion_matrix(agg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
