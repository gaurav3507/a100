"""Shared HTTP session, polite rate limiting and retry/backoff.

BSE throttles aggressively and occasionally answers with an HTML error page under
a 200 status, so every caller gets the raw response back and decides for itself
whether the body is what it expected.
"""

from __future__ import annotations

import random
import threading
import time

import requests

from .config import (
    BACKOFF_BASE,
    BSE_MAX_DELAY,
    BSE_MIN_DELAY,
    HTTP_TIMEOUT,
    MAX_RETRIES,
)
from .logging_setup import get_logger

log = get_logger("http")


class RateLimiter:
    """Randomised minimum spacing between calls to one host."""

    def __init__(self, min_delay: float, max_delay: float):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            gap = random.uniform(self.min_delay, self.max_delay)
            elapsed = time.time() - self._last
            if elapsed < gap:
                time.sleep(gap - elapsed)
            self._last = time.time()


bse_limiter = RateLimiter(BSE_MIN_DELAY, BSE_MAX_DELAY)


def make_session() -> requests.Session:
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def get(
    session: requests.Session,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    limiter: RateLimiter | None = None,
    retries: int = MAX_RETRIES,
    timeout: int = HTTP_TIMEOUT,
    label: str = "",
) -> requests.Response | None:
    """GET with backoff. Returns the response, or None if every attempt failed.

    Retries on 429, 5xx and transport errors. A 4xx other than 429 is treated as
    a real answer and returned immediately so the caller can fall back (the
    AttachLive -> AttachHis chain depends on seeing the 404).
    """
    attempt = 0
    while True:
        attempt += 1
        if limiter:
            limiter.wait()
        try:
            r = session.get(url, params=params, headers=headers, timeout=timeout)
        except Exception as exc:  # transport-level failure
            if attempt > retries:
                log.warning("%s giving up after %d attempts: %s", label or url, attempt, exc)
                return None
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            log.warning("%s %s (attempt %d), retrying in %.0fs",
                        label or url, type(exc).__name__, attempt, wait)
            time.sleep(wait)
            continue

        if r.status_code == 429 or r.status_code >= 500:
            if attempt > retries:
                log.warning("%s HTTP %d, out of retries", label or url, r.status_code)
                return r
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            log.warning("%s HTTP %d (attempt %d), backing off %.0fs",
                        label or url, r.status_code, attempt, wait)
            time.sleep(wait)
            continue

        return r


def looks_like_pdf(content: bytes) -> bool:
    """BSE sometimes serves an HTML error page with status 200."""
    return bool(content) and content[:5] == b"%PDF-"
