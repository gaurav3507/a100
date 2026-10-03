#!/usr/bin/env python3
"""CLI entrypoint for the concall pipeline.

    python run.py --limit 5
    python run.py --companies TIMEX,SHAILY,PRICOL
    python run.py --skip-llm            # fetch and extract only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from concall.config import (
    COMPANIES_CSV,
    LOOKBACK_MONTHS,
    TRANSCRIPTS_PER_COMPANY,
    Settings,
)
from concall.logging_setup import setup_logging
from concall.pipeline import run


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download the last N earnings concall transcripts for Indian "
                    "listed companies and extract insights with a local LLM.")
    ap.add_argument("--input", type=Path, default=COMPANIES_CSV,
                    help="CSV with a 'symbol' column (default: companies.csv)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N symbols (for testing)")
    ap.add_argument("--companies", type=str, default="",
                    help="comma-separated symbols, overrides --input")
    ap.add_argument("--transcripts", type=int, default=TRANSCRIPTS_PER_COMPANY,
                    help=f"transcripts per company (default {TRANSCRIPTS_PER_COMPANY})")
    ap.add_argument("--lookback-months", type=int, default=LOOKBACK_MONTHS,
                    help=f"BSE history window (default {LOOKBACK_MONTHS})")
    ap.add_argument("--skip-llm", action="store_true",
                    help="fetch and extract only, no GPU needed")
    ap.add_argument("--redo-llm", action="store_true",
                    help="discard cached LLM analysis and re-run it; keeps all "
                         "downloaded PDFs and extracted text")
    ap.add_argument("--force", action="store_true",
                    help="ignore caches and redo everything")
    ap.add_argument("--model", type=str, default=None,
                    help="pin a model instead of auto-selecting by VRAM")
    ap.add_argument("--autostart-vllm", action="store_true",
                    help="launch the vLLM server as a subprocess")
    ap.add_argument("--follow-pointers", action="store_true",
                    help="when a filing is only a cover letter, try to fetch the "
                         "transcript from the company website named in it")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging(args.verbose)

    settings = Settings(
        companies_csv=args.input,
        limit=args.limit,
        only=[s for s in args.companies.split(",") if s.strip()],
        skip_llm=args.skip_llm,
        force=args.force,
        lookback_months=args.lookback_months,
        n_transcripts=args.transcripts,
        model=args.model,
        autostart_vllm=args.autostart_vllm,
        redo_llm=args.redo_llm,
        follow_pointers=args.follow_pointers,
    )

    try:
        run(settings)
    except KeyboardInterrupt:
        log.warning("interrupted by user; progress is checkpointed in pipeline.db")
        return 130
    except SystemExit:
        raise
    except Exception:
        log.exception("pipeline failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
