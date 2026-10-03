"""Shared evaluation metrics for both tracks: accuracy, precision/recall/F1
(macro), and confusion matrix, plus mean +/- std aggregation across folds
and seeds (tasks.md Section 0: every run must log these).

Defaults to UL-DD's 3-class Low/Medium/High scheme (common/labels.py),
Track A and Track B's primary use of this module -- but `labels`/
`class_names` are overridable so mephy_repro's 4-class rest/cognitive/
physical/combo reproduction can reuse the same aggregation and reporting
code instead of duplicating it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from common.labels import CLASS_NAMES as ULDD_CLASS_NAMES

DEFAULT_LABELS = (0, 1, 2)


@dataclass
class Metrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion: np.ndarray  # rows = true, cols = predicted
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    class_names: tuple[str, ...] = ULDD_CLASS_NAMES

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "confusion_matrix": self.confusion.tolist(),
            "per_class": {
                name: {"precision": float(p), "recall": float(r), "f1": float(f)}
                for name, p, r, f in zip(
                    self.class_names, self.per_class_precision, self.per_class_recall, self.per_class_f1
                )
            },
        }


def compute_metrics(y_true, y_pred, labels=DEFAULT_LABELS, class_names=None) -> Metrics:
    if class_names is not None:
        class_names = tuple(class_names)
    elif tuple(labels) == DEFAULT_LABELS:
        class_names = ULDD_CLASS_NAMES
    else:
        class_names = tuple(str(l) for l in labels)
    if len(class_names) != len(labels):
        raise ValueError(f"class_names ({len(class_names)}) must match labels ({len(labels)})")

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    accuracy = float(np.mean(y_true == y_pred))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return Metrics(accuracy, float(p_macro), float(r_macro), float(f_macro), cm, p, r, f, class_names)


def aggregate(fold_metrics: list[Metrics]) -> dict:
    """Mean +/- std across folds (and/or seeds) for each scalar metric,
    matching tasks.md 3.5/4.1's "report mean +/- std, not a single run"."""

    def stat(attr: str) -> dict:
        vals = np.array([getattr(m, attr) for m in fold_metrics], dtype=float)
        return {"mean": float(vals.mean()), "std": float(vals.std())}

    confusion_sum = sum(m.confusion for m in fold_metrics)
    return {
        "n_runs": len(fold_metrics),
        "accuracy": stat("accuracy"),
        "precision_macro": stat("precision_macro"),
        "recall_macro": stat("recall_macro"),
        "f1_macro": stat("f1_macro"),
        "confusion_matrix_sum": confusion_sum.tolist(),
        "class_names": list(fold_metrics[0].class_names) if fold_metrics else [],
    }
