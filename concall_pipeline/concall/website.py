"""Optional second hop: fetch a transcript from the company's own website.

Some companies file a one-page cover letter with BSE saying the transcript "is
available on the Company's website" and give the URL. Federal Bank supplies a
direct .pdf link, which is trivially fetchable. Piramal supplies a landing page
that renders its links with JavaScript, which plain HTTP cannot follow.

Off by default. Enable with --follow-pointers.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from .config import BROWSER_UA, TRANSCRIPT_DIR
from .http import RateLimiter, get, looks_like_pdf
from .logging_setup import get_logger

log = get_logger("website")

_limiter = RateLimiter(2.0, 4.0)
_HEADERS = {"User-Agent": BROWSER_UA, "Accept": "application/pdf,text/html,*/*"}
_PDF_HREF = re.compile(r'href=["\']([^"\']+\.pdf[^"\']*)["\']', re.I)
_KEYWORDS = ("transcript", "earnings call", "concall", "conference call")


def _save(content: bytes, symbol: str, quarter: str) -> str | None:
    out_dir = TRANSCRIPT_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{symbol}_{quarter}_site.pdf"
    try:
        dest.write_bytes(content)
        return str(dest)
    except Exception as exc:
        log.warning("%s %s: could not write %s: %s", symbol, quarter, dest.name, exc)
        return None


def fetch_from_pointer(session, symbol: str, quarter: str, url: str) -> dict:
    """Try to retrieve the transcript named by a cover letter.

    Returns {"pdf_path": str|None, "failure": str|None, "source": str}.
    """
    res = {"pdf_path": None, "failure": None, "source": url}
    if not url:
        res["failure"] = "no URL found in cover letter"
        return res

    # Case 1: the letter gives the PDF directly.
    if urlparse(url).path.lower().endswith(".pdf"):
        r = get(session, url, headers=_HEADERS, limiter=_limiter,
                label=f"site[{symbol}/{quarter}]")
        if r is not None and r.status_code == 200 and looks_like_pdf(r.content):
            path = _save(r.content, symbol, quarter)
            if path:
                log.info("%s %s: recovered transcript from company website (%d KB)",
                         symbol, quarter, len(r.content) // 1024)
                res["pdf_path"] = path
                return res
        code = r.status_code if r is not None else "unreachable"
        res["failure"] = f"direct PDF link returned {code}"
        return res

    # Case 2: a landing page. Scan the static HTML for a transcript PDF.
    r = get(session, url, headers=_HEADERS, limiter=_limiter,
            label=f"site[{symbol}/{quarter}/page]")
    if r is None or r.status_code != 200:
        code = r.status_code if r is not None else "unreachable"
        res["failure"] = f"landing page returned {code}"
        return res

    hrefs = _PDF_HREF.findall(r.text or "")
    ranked = [h for h in hrefs if any(k in h.lower() for k in _KEYWORDS)]
    if not ranked:
        res["failure"] = (
            f"landing page exposed {len(hrefs)} PDF link(s) in static HTML but none "
            "named a transcript; the list is likely JavaScript-rendered and needs "
            "a browser fetch")
        log.warning("%s %s: %s", symbol, quarter, res["failure"])
        return res

    for href in ranked[:5]:
        pdf_url = urljoin(url, href)
        rr = get(session, pdf_url, headers=_HEADERS, limiter=_limiter,
                 label=f"site[{symbol}/{quarter}/pdf]")
        if rr is not None and rr.status_code == 200 and looks_like_pdf(rr.content):
            path = _save(rr.content, symbol, quarter)
            if path:
                log.info("%s %s: recovered transcript from landing page",
                         symbol, quarter)
                res["pdf_path"] = path
                res["source"] = pdf_url
                return res

    res["failure"] = f"found {len(ranked)} PDF link(s) but none fetched as a PDF"
    return res
