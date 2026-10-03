"""Regression test for a real bug found while running the MePhy
reproduction: common/metrics.py originally hardcoded LABELS=(0,1,2) and
UL-DD's 3-class names, which silently dropped MePhy's 4th class
(combo-fatigue) from every metric instead of erroring -- the confusion
matrix came back 3x3 and accuracy looked near-random even though training
itself was fine. compute_metrics must support an arbitrary label set/class
name set while still defaulting to UL-DD's Low/Medium/High for Track A/B's
plain compute_metrics(y_true, y_pred) call sites.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.metrics import compute_metrics


def test_default_call_uses_uldd_three_class_names():
    m = compute_metrics([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 1])
    assert list(m.to_dict()["per_class"].keys()) == ["Low", "Medium", "High"]
    assert m.confusion.shape == (3, 3)


def test_four_class_labels_not_silently_dropped():
    y_true = [0, 1, 2, 3, 3, 2]
    y_pred = [0, 1, 2, 3, 2, 2]
    names = ("rest", "cognitive-fatigue", "physical-fatigue", "combo-fatigue")
    m = compute_metrics(y_true, y_pred, labels=(0, 1, 2, 3), class_names=names)
    assert m.confusion.shape == (4, 4)
    assert list(m.to_dict()["per_class"].keys()) == list(names)
    # class 3 (combo-fatigue) must actually be scored, not dropped
    assert m.confusion.sum() == len(y_true)
