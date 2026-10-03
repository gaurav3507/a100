"""Paths, constants and tunables for the concall pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------- paths
RAW_DIR = ROOT / "raw"
ANN_DIR = RAW_DIR / "announcements"
TRANSCRIPT_DIR = ROOT / "transcripts"
LLM_CACHE_DIR = RAW_DIR / "llm"
LOG_DIR = ROOT / "logs"
DB_PATH = ROOT / "pipeline.db"
SCRIP_CSV = ROOT / "scrip_codes.csv"
COMPANIES_CSV = ROOT / "companies.csv"
EXCEL_OUT = ROOT / "concall_analysis.xlsx"

for _d in (RAW_DIR, ANN_DIR, TRANSCRIPT_DIR, LLM_CACHE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- endpoints
INDIANAPI_BASE = "https://stock.indianapi.in"
BSE_ANN_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
BSE_ATTACH_PATHS = (
    "https://www.bseindia.com/xml-data/corpfiling/AttachLive/{name}",
    "https://www.bseindia.com/xml-data/corpfiling/AttachHis/{name}",
)

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# BSE returns 403 without these. Not optional.
BSE_JSON_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Referer": "https://www.bseindia.com/corporates/ann.html",
    "Origin": "https://www.bseindia.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
BSE_PDF_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Referer": "https://www.bseindia.com/",
    "Accept": "application/pdf,*/*",
}

# ------------------------------------------------------------- filter tuning
# Primary signal: BSE's own subcategory field.
SUBCAT_MATCH = "transcript"

# Backup signal on the title, used only when SUBCATNAME is blank or unhelpful.
TITLE_INCLUDE = (
    "transcript",
    "earnings call",
    "conference call",
    "concall",
    "audio transcript",
)

# Filings *about* a call rather than the transcript of one.
TITLE_EXCLUDE = (
    "investor presentation",
    "earnings presentation",
    "audio recording",
    "intimation of conference call",
    "intimation of earnings call",
    "notice of earnings call",
    "intimation under regulation 30 of conference call",
    "analyst / investor meet - intimation",
)

# A real transcript is long. These separate it from a one-page cover letter that
# merely points at the company website (Piramal and Federal Bank both do this).
MIN_TRANSCRIPT_CHARS = 6000
MIN_TRANSCRIPT_PAGES = 5
OCR_TRIGGER_CHARS = 500

TRANSCRIPT_MARKERS = (
    "moderator",
    "ladies and gentlemen",
    "question-and-answer",
    "question and answer",
    "q&a",
)

# --------------------------------------------------------------- rate limits
BSE_MIN_DELAY = 2.0
BSE_MAX_DELAY = 4.0
MAX_RETRIES = 3
BACKOFF_BASE = 5.0
HTTP_TIMEOUT = 60

LOOKBACK_MONTHS = 24
TRANSCRIPTS_PER_COMPANY = 4
ANN_PAGE_LIMIT = 40  # hard stop so a pagination bug cannot loop forever

# ---------------------------------------------------------------------- LLM
VLLM_HOST = os.environ.get("VLLM_HOST", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
MAX_MODEL_LEN = 32768
LLM_MAX_CONCURRENCY = 8
# First launch downloads ~20GB of weights; be generous.
VLLM_START_TIMEOUT = 1800
LLM_RETRIES = 2

MODEL_80GB = "Qwen/Qwen2.5-32B-Instruct-AWQ"
MODEL_40GB = "Qwen/Qwen2.5-14B-Instruct-AWQ"

# Leave room for the generated answer inside the context window.
PROMPT_TOKEN_BUDGET = 26000
CHARS_PER_TOKEN = 3.6  # rough, deliberately conservative


@dataclass
class Settings:
    companies_csv: Path = COMPANIES_CSV
    limit: int | None = None
    only: list[str] = field(default_factory=list)
    skip_llm: bool = False
    force: bool = False
    lookback_months: int = LOOKBACK_MONTHS
    n_transcripts: int = TRANSCRIPTS_PER_COMPANY
    model: str | None = None
    autostart_vllm: bool = False
    redo_llm: bool = False
    follow_pointers: bool = False
