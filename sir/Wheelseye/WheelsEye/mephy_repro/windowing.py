"""Fixed sliding-window slicing for MePhy. tasks.md 1.3 says "10s window,
5s step, matching FatigueNet's approach" -- but the FatigueNet paper's own
Methods section ("Feature extraction") states a **20s window, 5s step**.
Since this pipeline exists specifically to reproduce the paper's published
90.2% test accuracy as a sanity check (tasks.md 1.2/3.1), fidelity to the
paper's actual number matters more here than tasks.md's paraphrase of it --
so this uses 20s/5s. UL-DD's own windowing (common/windowing.py) stays at
10s/5s per tasks.md's separate, UL-DD-specific reasoning (matching the
4-minute KSS label granularity), which doesn't depend on this correction.

Each (user, condition) file is one short standalone recording (not a
continuous multi-hour session like UL-DD), and its own label is the
condition itself -- rest=0, cognitive-fatigue=1, physical-fatigue=2,
combo-fatigue=3 -- constant across every window in that file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mephy_repro import loaders
from mephy_repro.paths import CONDITION_LABEL

WINDOW_SECONDS = 20.0
STEP_SECONDS = 5.0

ALL_MODALITIES = ("ECG", "EDA", "EMG", "EyeBlinking")


@dataclass
class Signal:
    times: np.ndarray
    values: np.ndarray
    columns: list[str]


@dataclass
class Window:
    user: str
    condition: str
    window_id: int
    t_start: float
    t_end: float
    label: int
    signals: dict[str, Signal] = field(default_factory=dict)


def _slice(times: np.ndarray, values: np.ndarray, t_start: float, t_end: float):
    if len(times) == 0:
        return times, values
    mask = (times >= t_start) & (times < t_end)
    return times[mask], values[mask]


def session_duration_seconds(user: str, condition: str, modalities) -> float:
    durations = []
    for modality in modalities:
        times, _, _ = loaders.load(user, modality, condition)
        if len(times) == 0:
            raise ValueError(f"{user}/{condition}/{modality}: empty signal")
        durations.append(times[-1])
    return min(durations)


def iter_windows(
    user: str,
    condition: str,
    modalities: tuple[str, ...] = ALL_MODALITIES,
    window_s: float = WINDOW_SECONDS,
    step_s: float = STEP_SECONDS,
):
    label = CONDITION_LABEL[condition]
    duration = session_duration_seconds(user, condition, modalities)
    loaded = {m: loaders.load(user, m, condition) for m in modalities}

    window_id = 0
    t_start = 0.0
    while t_start + window_s <= duration:
        t_end = t_start + window_s
        signals = {}
        for modality, (times, values, columns) in loaded.items():
            s_times, s_values = _slice(times, values, t_start, t_end)
            signals[modality] = Signal(times=s_times, values=s_values, columns=columns)

        yield Window(
            user=user, condition=condition, window_id=window_id,
            t_start=t_start, t_end=t_end, label=label, signals=signals,
        )
        window_id += 1
        t_start += step_s
