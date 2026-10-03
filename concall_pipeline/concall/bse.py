"""STEPS 1-3 - announcement history, transcript identification, PDF download.

Everything here targets BSE. Two hard-won details are load bearing:

1. Attachments live under AttachLive when recent and AttachHis when older. You
   must try both; either path alone fails for most filings.
2. BSE will answer with an HTML error page under HTTP 200, so the magic bytes
   have to be checked before anything is written to disk.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

from .config import (
    ANN_DIR,
    ANN_PAGE_LIMIT,
    BSE_ANN_URL,
    BSE_ATTACH_PATHS,
    BSE_JSON_HEADERS,
    BSE_PDF_HEADERS,
    SUBCAT_MATCH,
    TITLE_EXCLUDE,
    TITLE_INCLUDE,
    TRANSCRIPT_DIR,
)
from .http import bse_limiter, get, looks_like_pdf
from .logging_setup import get_logger

log = get_logger("bse")


# ------------------------------------------------------------------ helpers
def infer_quarter(news_dt: str) -> str:
    """Map a filing date to the Indian fiscal quarter it reports on.

    Transcripts land a few weeks after quarter end, so the filing month tells you
    the quarter: Apr-Jun reports on Q4, Jul-Sep on Q1, Oct-Dec on Q2, Jan-Mar on
    Q3. FY2026 runs Apr 2025 to Mar 2026.
    """
    try:
        dt = datetime.fromisoformat(str(news_dt).replace("Z", "").split(".")[0])
    except Exception:
        return "UNKNOWN"
    m, y = dt.month, dt.year
    if 4 <= m <= 6:
        return f"FY{y % 100:02d}Q4"
    if 7 <= m <= 9:
        return f"FY{(y + 1) % 100:02d}Q1"
    if 10 <= m <= 12:
        return f"FY{(y + 1) % 100:02d}Q2"
    return f"FY{y % 100:02d}Q3"


def _text_of(row: dict) -> str:
    return f"{row.get('NEWSSUB') or ''} {row.get('HEADLINE') or ''}".lower()


def is_transcript_filing(row: dict) -> bool:
    """Primary signal is BSE's own SUBCATNAME; the title is only a backup."""
    subcat = str(row.get("SUBCATNAME") or "").lower()
    title = _text_of(row)

    # Exclusions win outright: these announce a call rather than transcribe one.
    if any(bad in title for bad in TITLE_EXCLUDE):
        # ...unless the subcategory explicitly says transcript, which outranks
        # a sloppy title.
        if SUBCAT_MATCH not in subcat:
            return False

    if SUBCAT_MATCH in subcat:
        return True
    return any(good in title for good in TITLE_INCLUDE)


# ------------------------------------------------------------------- step 1
def fetch_announcements(session, symbol: str, scrip_code: str,
                        lookback_months: int, force: bool = False) -> list[dict]:
    """Page through BSE announcements. Cached per company as raw JSON."""
    cache = ANN_DIR / f"{symbol}.json"
    if cache.exists() and not force:
        try:
            rows = json.loads(cache.read_text(encoding="utf-8"))
            log.info("%s: %d announcements from cache", symbol, len(rows))
            return rows
        except Exception as exc:
            log.warning("%s: unreadable announcement cache (%s), refetching", symbol, exc)

    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=int(lookback_months * 30.5))
    params_base = {
        "strCat": "-1",
        "strPrevDate": from_dt.strftime("%Y%m%d"),
        "strToDate": to_dt.strftime("%Y%m%d"),
        "strScrip": str(scrip_code),
        "strSearch": "P",
        "strType": "C",
    }

    rows: list[dict] = []
    seen_ids: set[str] = set()
    for page in range(1, ANN_PAGE_LIMIT + 1):
        params = dict(params_base, pageno=str(page))
        r = get(session, BSE_ANN_URL, params=params, headers=BSE_JSON_HEADERS,
                limiter=bse_limiter, label=f"ann[{symbol}p{page}]")
        if r is None:
            log.warning("%s: announcements unreachable at page %d", symbol, page)
            break
        if r.status_code != 200:
            log.warning("%s: announcements HTTP %d at page %d", symbol, r.status_code, page)
            break
        try:
            payload = r.json()
        except ValueError:
            log.warning("%s: non-JSON announcement body at page %d", symbol, page)
            break

        # BSE answers "No Record Found!" as a bare JSON string, not an error.
        if isinstance(payload, str):
            log.debug("%s: page %d returned %r", symbol, page, payload[:40])
            break
        table = payload.get("Table") if isinstance(payload, dict) else None
        if not table:
            break

        fresh = 0
        for row in table:
            nid = str(row.get("NEWSID") or row.get("ATTACHMENTNAME") or "")
            if nid and nid in seen_ids:
                continue
            seen_ids.add(nid)
            rows.append(row)
            fresh += 1
        log.debug("%s: page %d -> %d rows (%d new)", symbol, page, len(table), fresh)

        if len(table) < 20 or fresh == 0:
            break

    if rows:
        try:
            cache.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("%s: could not cache announcements: %s", symbol, exc)
    log.info("%s: %d announcements fetched", symbol, len(rows))
    return rows


# ------------------------------------------------------------------- step 2
def select_transcripts(symbol: str, rows: list[dict], n: int) -> list[dict]:
    """Filter to transcript filings, newest first, de-duplicated by quarter."""
    hits = [r for r in rows if is_transcript_filing(r)]

    def sort_key(r: dict) -> str:
        return str(r.get("NEWS_DT") or r.get("DT_TM") or "")

    hits.sort(key=sort_key, reverse=True)

    picked: list[dict] = []
    quarters_seen: set[str] = set()
    for r in hits:
        q = infer_quarter(r.get("NEWS_DT") or r.get("DT_TM") or "")
        if q in quarters_seen:
            continue
        quarters_seen.add(q)
        r["_quarter"] = q
        picked.append(r)
        if len(picked) >= n:
            break

    log.info("%s: %d transcript filing(s) matched, keeping %d",
             symbol, len(hits), len(picked))
    return picked


# ------------------------------------------------------------------- step 3
def download_pdf(session, symbol: str, row: dict, quarter: str) -> dict:
    """Download one attachment. Returns a result dict, never raises."""
    result = {
        "quarter": quarter,
        "news_dt": str(row.get("NEWS_DT") or row.get("DT_TM") or ""),
        "attachment": row.get("ATTACHMENTNAME") or "",
        "url_path": None,
        "pdf_path": None,
        "failure": None,
        "sha256": None,
    }
    att = result["attachment"]
    if not att:
        result["failure"] = "no ATTACHMENTNAME on announcement row"
        return result

    out_dir = TRANSCRIPT_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{symbol}_{quarter}.pdf"

    if dest.exists() and dest.stat().st_size > 0:
        try:
            head = dest.read_bytes()[:5]
        except Exception:
            head = b""
        if head == b"%PDF-":
            result["pdf_path"] = str(dest)
            result["url_path"] = "cache"
            result["sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
            log.debug("%s %s: pdf already on disk", symbol, quarter)
            return result

    last_err = None
    for template in BSE_ATTACH_PATHS:
        url = template.format(name=att)
        variant = "AttachLive" if "AttachLive" in template else "AttachHis"
        r = get(session, url, headers=BSE_PDF_HEADERS, limiter=bse_limiter,
                label=f"pdf[{symbol}/{quarter}/{variant}]")
        if r is None:
            last_err = f"{variant}: unreachable"
            continue
        if r.status_code != 200:
            last_err = f"{variant}: HTTP {r.status_code}"
            continue
        if not looks_like_pdf(r.content):
            last_err = f"{variant}: HTTP 200 but body is not a PDF ({len(r.content)}B)"
            continue
        try:
            dest.write_bytes(r.content)
        except Exception as exc:
            last_err = f"{variant}: write failed {exc}"
            continue
        result["pdf_path"] = str(dest)
        result["url_path"] = variant
        result["sha256"] = hashlib.sha256(r.content).hexdigest()
        log.info("%s %s: downloaded via %s (%d KB)",
                 symbol, quarter, variant, len(r.content) // 1024)
        return result

    result["failure"] = last_err or "download failed"
    log.warning("%s %s: %s", symbol, quarter, result["failure"])
    return result
