"""KSS -> 3-class label binning, shared by both tracks (tasks.md Task 1.2).

Bin boundaries match UL-DD's own validation methodology (tasks.md 1.3):
  Low    (Alert)  : KSS < 4
  Medium          : 4 <= KSS <= 6
  High   (Drowsy) : KSS > 6
"""
from __future__ import annotations

from functools import lru_cache

from common.paths import LABELS_CSV, SESSION_NAMES

LOW, MEDIUM, HIGH = 0, 1, 2
CLASS_NAMES = ("Low", "Medium", "High")

KSS_INTERVAL_MINUTES = 4  # KSS is scored once per 4-minute interval

_SESSION_CODE_BY_NAME = {name: code for code, name in SESSION_NAMES.items()}


def bin_kss(kss: float) -> int:
    """Map a raw KSS score (1-9) to {LOW, MEDIUM, HIGH}."""
    if kss < 4:
        return LOW
    if kss <= 6:
        return MEDIUM
    return HIGH


@lru_cache(maxsize=1)
def _load_raw() -> dict[tuple[str, str], list[float]]:
    """Parse Labels.csv into {(subject, session_code): [kss_per_4min_interval, ...]}."""
    table: dict[tuple[str, str], list[float]] = {}
    with open(LABELS_CSV, newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, *values = line.split(",")
            subject, session_name = key.split("_")
            session = _SESSION_CODE_BY_NAME[session_name]
            table[(subject, session)] = [float(v) for v in values]
    return table


def available_sessions(subject: str) -> tuple[str, ...]:
    """Which session codes ('A', 'D') have KSS labels for `subject`."""
    return tuple(sess for (subj, sess) in _load_raw() if subj == subject)


def get_kss(subject: str, session: str, minute: float) -> float:
    """Raw KSS (1-9) for the 4-minute interval containing `minute`."""
    series = _load_raw().get((subject, session))
    if series is None:
        raise KeyError(f"No KSS labels for subject={subject!r} session={session!r}")
    if minute < 0:
        raise ValueError(f"minute must be >= 0, got {minute}")
    idx = min(int(minute // KSS_INTERVAL_MINUTES), len(series) - 1)
    return series[idx]


def get_label(subject: str, session: str, minute: float) -> int:
    """3-class label (0=Low, 1=Medium, 2=High) for the interval containing `minute`."""
    return bin_kss(get_kss(subject, session, minute))
