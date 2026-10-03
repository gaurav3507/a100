"""Fixed sliding-window slicing across all UL-DD modalities (tasks.md Task 1.3).

10s window / 5s step by default, matching FatigueNet's approach and a
reasonable subdivision of the 4-minute KSS granularity (48 windows per
labeled interval).

Two data-quality quirks discovered while grounding this against the real
UL-DD files (see data/raw/ULDD_VERIFICATION_NOTES.md and Info.xlsx):
  - IBI (heartbeat interval) coverage is sparse within a session -- the
    wristband loses PPG lock for long stretches (e.g. subject A/Awake only
    has beat detections from t=131s to t=1231s out of a 2400s session).
    Windows outside that coverage will simply get an empty IBI slice; HRV
    features must handle that as a missing value, not crash or zero-fill.
  - Session duration is derived from the "full-coverage" modalities
    (ACC/BVP/EDA/HR/LGP/RGP/O2M/TEMP/FL/PL/FAU), which are 'a' (available)
    for every non-excluded subject/session. IBI and Telemetry are sparse or
    session-scoped in ways that don't reliably bound total duration, so
    they're sliced opportunistically but never used to size the window grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common import loaders
from common.labels import available_sessions, get_kss, get_label

WINDOW_SECONDS = 10.0
STEP_SECONDS = 5.0

FULL_COVERAGE_MODALITIES = (
    "ACC", "BVP", "EDA", "HR", "LGP", "RGP", "O2M", "TEMP", "FL", "PL", "FAU",
)
SPARSE_MODALITIES = ("IBI",)  # event-based, coverage can be partial
OPTIONAL_MODALITIES = ("Telemetry",)  # not present for every subject/session

ALL_MODALITIES = FULL_COVERAGE_MODALITIES + SPARSE_MODALITIES + OPTIONAL_MODALITIES


@dataclass
class Signal:
    times: np.ndarray
    values: np.ndarray
    columns: list[str]


@dataclass
class Window:
    subject: str
    session: str
    window_id: int
    t_start: float
    t_end: float
    label: int
    kss: float
    signals: dict[str, Signal] = field(default_factory=dict)


def _slice(times: np.ndarray, values: np.ndarray, t_start: float, t_end: float):
    if len(times) == 0:
        return times, values
    mask = (times >= t_start) & (times < t_end)
    return times[mask], values[mask]


def session_duration_seconds(subject: str, session: str) -> float:
    """Shortest span across full-coverage modalities, so the window grid
    never reaches past data that any of them actually has."""
    durations = []
    for modality in FULL_COVERAGE_MODALITIES:
        times, _, _ = loaders.load(subject, session, modality)
        if len(times) == 0:
            raise ValueError(
                f"{subject}/{session}: full-coverage modality {modality} is "
                f"empty -- expected always-available per Info.xlsx manifest"
            )
        durations.append(times[-1])
    return min(durations)


def iter_windows(
    subject: str,
    session: str,
    modalities: tuple[str, ...] = ALL_MODALITIES,
    window_s: float = WINDOW_SECONDS,
    step_s: float = STEP_SECONDS,
):
    """Yield Window objects covering the full session at the given
    window/step, each carrying raw signal slices for every requested
    modality plus the 3-class label for the containing 4-minute interval."""
    if session not in available_sessions(subject):
        raise KeyError(f"No KSS labels for subject={subject!r} session={session!r}")

    duration = session_duration_seconds(subject, session)
    loaded = {}
    for m in modalities:
        if m not in ALL_MODALITIES:
            continue
        try:
            loaded[m] = loaders.load(subject, session, m)
        except FileNotFoundError:
            if m not in OPTIONAL_MODALITIES:
                raise
            loaded[m] = (np.empty(0), np.empty((0, 1)), [m.lower()])

    window_id = 0
    t_start = 0.0
    while t_start + window_s <= duration:
        t_end = t_start + window_s
        minute = (t_start + t_end) / 2.0 / 60.0  # window midpoint, in minutes
        label = get_label(subject, session, minute)
        kss = get_kss(subject, session, minute)

        signals = {}
        for modality, (times, values, columns) in loaded.items():
            s_times, s_values = _slice(times, values, t_start, t_end)
            signals[modality] = Signal(times=s_times, values=s_values, columns=columns)

        yield Window(
            subject=subject,
            session=session,
            window_id=window_id,
            t_start=t_start,
            t_end=t_end,
            label=label,
            kss=kss,
            signals=signals,
        )
        window_id += 1
        t_start += step_s
