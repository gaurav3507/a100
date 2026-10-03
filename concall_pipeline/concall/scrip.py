"""STEP 0 - map NSE symbol to BSE numeric scrip code, cached to disk.

indianapi is used here and nowhere else. After the first successful run every
symbol lives in scrip_codes.csv and the pipeline needs no API key at all.
"""

from __future__ import annotations

import csv
import os

from .config import INDIANAPI_BASE, SCRIP_CSV
from .http import RateLimiter, get, make_session
from .logging_setup import get_logger

log = get_logger("scrip")

_limiter = RateLimiter(1.0, 2.0)


def load_cache() -> dict[str, str]:
    if not SCRIP_CSV.exists():
        return {}
    out: dict[str, str] = {}
    try:
        with SCRIP_CSV.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                sym = (row.get("symbol") or "").strip().upper()
                code = (row.get("scrip_code") or "").strip()
                if sym and code:
                    out[sym] = code
    except Exception as exc:
        log.warning("could not read %s (%s), starting fresh", SCRIP_CSV.name, exc)
    return out


def save_cache(mapping: dict[str, str]) -> None:
    try:
        with SCRIP_CSV.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["symbol", "scrip_code"])
            for sym in sorted(mapping):
                w.writerow([sym, mapping[sym]])
        log.info("wrote %d mappings to %s", len(mapping), SCRIP_CSV.name)
    except Exception as exc:
        log.error("could not write %s: %s", SCRIP_CSV.name, exc)


def _pick(results: list[dict], symbol: str) -> dict | None:
    """Exact NSE ticker match first.

    Substring matching on the company name is actively dangerous here: a query
    for MARINE returns 'Knowledge Marine & Engineering Works' ahead of 'Marine
    Electricals (India)', and only the second one actually trades as MARINE.
    """
    up = symbol.upper()
    for r in results:
        if str(r.get("exchangeCodeNsi", "")).upper() == up:
            return r
    for r in results:
        if str(r.get("exchangeCodeBse", "")).upper() == up:
            return r
    # Single unambiguous result is acceptable; anything else we refuse to guess.
    if len(results) == 1:
        return results[0]
    return None


def resolve_symbol(session, symbol: str, api_key: str) -> str | None:
    """Return the BSE scrip code for one NSE symbol, or None."""
    r = get(
        session,
        f"{INDIANAPI_BASE}/industry_search",
        params={"query": symbol},
        headers={"X-Api-Key": api_key, "Accept": "application/json"},
        limiter=_limiter,
        label=f"industry_search[{symbol}]",
    )
    if r is None:
        log.warning("%s: industry_search unreachable", symbol)
        return None
    if r.status_code != 200:
        log.warning("%s: industry_search HTTP %d", symbol, r.status_code)
        return None
    try:
        payload = r.json()
    except ValueError:
        log.warning("%s: industry_search returned non-JSON", symbol)
        return None

    results = payload if isinstance(payload, list) else []
    results = [x for x in results if isinstance(x, dict)]
    if not results:
        log.warning("%s: no industry_search results", symbol)
        return None

    match = _pick(results, symbol)
    if not match:
        names = [r.get("commonName") for r in results[:5]]
        log.warning("%s: ambiguous, refusing to guess among %s", symbol, names)
        return None

    code = str(match.get("exchangeCodeBse") or "").strip()
    if not code:
        log.warning("%s: matched %r but it has no BSE code (NSE-only listing)",
                    symbol, match.get("commonName"))
        return None

    log.info("%s -> %s (%s)", symbol, code, match.get("commonName"))
    return code


def build_scrip_map(symbols: list[str]) -> dict[str, str]:
    """Load cache, resolve only what is missing, persist, return the full map."""
    mapping = load_cache()
    missing = [s for s in symbols if s.upper() not in mapping]

    if not missing:
        log.info("all %d symbols already cached, no API calls needed", len(symbols))
        return mapping

    api_key = os.environ.get("INDIANAPI_KEY", "").strip()
    if not api_key:
        log.error(
            "%d symbol(s) not in %s and INDIANAPI_KEY is unset: %s",
            len(missing), SCRIP_CSV.name, ", ".join(missing[:10]),
        )
        return mapping

    log.info("resolving %d uncached symbol(s) via indianapi", len(missing))
    session = make_session()
    resolved = 0
    for sym in missing:
        try:
            code = resolve_symbol(session, sym, api_key)
        except Exception as exc:
            log.error("%s: unexpected error during resolution: %s", sym, exc)
            code = None
        if code:
            mapping[sym.upper()] = code
            resolved += 1

    if resolved:
        save_cache(mapping)
    log.info("resolved %d of %d missing symbols", resolved, len(missing))
    return mapping
