"""STEP 4 - PDF text extraction with an OCR fallback.

Also classifies what the PDF actually is. A "transcript" filing is not always a
transcript: several companies (Piramal, Federal Bank on some quarters) file a
one-page cover letter that merely points at their own website. Treating that as
a transcript silently poisons the LLM stage, so it is detected and reported.
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import (
    MIN_TRANSCRIPT_CHARS,
    MIN_TRANSCRIPT_PAGES,
    OCR_TRIGGER_CHARS,
    TRANSCRIPT_MARKERS,
)
from .logging_setup import get_logger

log = get_logger("extract")

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None

_URL_RE = re.compile(r"https?://[^\s\)\]<>\"']+", re.I)

# Boilerplate lines worth dropping. Deliberately conservative.
_BOILERPLATE = re.compile(
    r"^\s*(page \d+ of \d+"
    r"|e&oe[ \-–].*"
    r"|this transcript (has been|is) edited.*"
    r"|disclaimer\s*:?.*"
    r")\s*$",
    re.I,
)


def _dense(text: str) -> int:
    return len(re.sub(r"\s", "", text or ""))


def extract_with_pdfplumber(path: Path) -> tuple[str, int]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not installed")
    with pdfplumber.open(str(path)) as pdf:
        pages = [(p.extract_text() or "") for p in pdf.pages]
        return "\n".join(pages), len(pdf.pages)


def extract_with_ocr(path: Path) -> tuple[str, int]:
    """Scanned-PDF fallback. This is the only stage the GPU helps with."""
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(str(path), dpi=200)
    out = []
    for img in images:
        try:
            out.append(pytesseract.image_to_string(img))
        except Exception as exc:
            log.warning("OCR failed on a page of %s: %s", path.name, exc)
            out.append("")
    return "\n".join(out), len(images)


def clean_text(text: str) -> str:
    kept = [ln for ln in (text or "").splitlines() if not _BOILERPLATE.match(ln)]
    out = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def classify(text: str, n_pages: int) -> tuple[str, str]:
    """Return (kind, reason).

    kind is one of TRANSCRIPT, COVER_LETTER_ONLY, TOO_SHORT, EMPTY.
    """
    dense = _dense(text)
    if dense == 0:
        return "EMPTY", "no text extracted even after OCR"

    low = text.lower()
    markers = [m for m in TRANSCRIPT_MARKERS if m in low]

    if dense >= MIN_TRANSCRIPT_CHARS and n_pages >= MIN_TRANSCRIPT_PAGES:
        return "TRANSCRIPT", f"{n_pages} pages, {dense} chars, markers={len(markers)}"

    # Short document that names a URL is the classic "it's on our website" letter.
    if n_pages <= 3 and _URL_RE.search(text):
        urls = [u for u in _URL_RE.findall(text) if "bseindia" not in u.lower()]
        target = urls[0] if urls else "(url not isolated)"
        return "COVER_LETTER_ONLY", f"{n_pages}-page letter pointing to {target}"

    return "TOO_SHORT", f"only {n_pages} pages / {dense} chars, not a transcript"


def pointer_url(text: str) -> str | None:
    for u in _URL_RE.findall(text or ""):
        if "bseindia" in u.lower() or "nseindia" in u.lower():
            continue
        return u.rstrip(".,;")
    return None


def extract(pdf_path: str | Path, symbol: str, quarter: str) -> dict:
    """Extract, clean, classify and persist the text. Never raises."""
    path = Path(pdf_path)
    res = {
        "txt_path": None,
        "n_pages": 0,
        "n_chars": 0,
        "extract_method": None,
        "kind": None,
        "failure": None,
        "pointer_url": None,
    }

    txt_dest = path.with_suffix(".txt")
    if txt_dest.exists() and txt_dest.stat().st_size > 0:
        try:
            text = txt_dest.read_text(encoding="utf-8")
            dense = _dense(text)
            # Page count is not recoverable from cached text, so classify on
            # length alone and treat anything short with an off-exchange URL as
            # the cover-letter case.
            if dense >= MIN_TRANSCRIPT_CHARS:
                kind = "TRANSCRIPT"
            elif dense == 0:
                kind = "EMPTY"
            elif pointer_url(text):
                kind = "COVER_LETTER_ONLY"
            else:
                kind = "TOO_SHORT"
            res.update(
                txt_path=str(txt_dest),
                n_chars=dense,
                extract_method="cache",
                kind=kind,
            )
            if kind == "COVER_LETTER_ONLY":
                res["pointer_url"] = pointer_url(text)
                res["failure"] = (
                    f"cover letter only; transcript hosted at {res['pointer_url']}")
            log.debug("%s %s: text from cache (%d chars, %s)",
                      symbol, quarter, dense, kind)
            return res
        except Exception as exc:
            log.warning("%s %s: unreadable text cache (%s), re-extracting",
                        symbol, quarter, exc)

    try:
        text, n_pages = extract_with_pdfplumber(path)
        method = "pdfplumber"
    except Exception as exc:
        log.warning("%s %s: pdfplumber failed (%s), trying OCR", symbol, quarter, exc)
        text, n_pages, method = "", 0, None
        try:
            text, n_pages = extract_with_ocr(path)
            method = "ocr"
        except Exception as exc2:
            res["failure"] = f"pdfplumber and OCR both failed: {exc} / {exc2}"
            log.error("%s %s: %s", symbol, quarter, res["failure"])
            return res

    if _dense(text) < OCR_TRIGGER_CHARS:
        log.info("%s %s: only %d chars from pdfplumber, falling back to OCR",
                 symbol, quarter, _dense(text))
        try:
            ocr_text, ocr_pages = extract_with_ocr(path)
            if _dense(ocr_text) > _dense(text):
                text, n_pages, method = ocr_text, ocr_pages or n_pages, "ocr"
        except Exception as exc:
            log.warning("%s %s: OCR unavailable (%s), keeping pdfplumber output",
                        symbol, quarter, exc)

    text = clean_text(text)
    kind, reason = classify(text, n_pages)
    res.update(
        n_pages=n_pages,
        n_chars=_dense(text),
        extract_method=method,
        kind=kind,
    )
    if kind == "COVER_LETTER_ONLY":
        res["pointer_url"] = pointer_url(text)
        res["failure"] = f"cover letter only; transcript hosted at {res['pointer_url']}"

    try:
        txt_dest.write_text(text, encoding="utf-8")
        res["txt_path"] = str(txt_dest)
    except Exception as exc:
        log.warning("%s %s: could not write text file: %s", symbol, quarter, exc)

    log.info("%s %s: %s via %s (%s)", symbol, quarter, kind, method, reason)
    return res


def split_commentary_qa(text: str) -> tuple[str, str]:
    """Split a transcript into opening commentary and Q&A.

    Used only when a transcript will not fit the context window. The Q&A half is
    never discarded; both halves are analysed and merged.
    """
    patterns = [
        r"question[- ]and[- ]answer session",
        r"we will now begin the question",
        r"the floor is now open for question",
        r"open the floor for q\s*&\s*a",
        r"\bq\s*&\s*a session\b",
    ]
    low = text.lower()
    for pat in patterns:
        m = re.search(pat, low)
        if m and m.start() > len(text) * 0.1:
            return text[: m.start()], text[m.start():]
    # Fall back to a proportional split rather than losing anything.
    cut = int(len(text) * 0.4)
    return text[:cut], text[cut:]
