"""Orchestration: wires steps 0-7 together with per-company isolation."""

from __future__ import annotations

import csv
import time
from collections import Counter

from . import bse, extraction, llm, output, scrip, website
from .config import Settings
from .http import make_session
from .logging_setup import get_logger
from .store import Store

log = get_logger("pipeline")


def read_companies(settings: Settings) -> list[str]:
    if settings.only:
        syms = [s.strip().upper() for s in settings.only if s.strip()]
        log.info("running %d symbol(s) from --companies", len(syms))
        return syms

    path = settings.companies_csv
    if not path.exists():
        raise SystemExit(f"input file not found: {path}")

    syms: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "symbol" not in [
            (f or "").strip().lower() for f in reader.fieldnames
        ]:
            raise SystemExit(f"{path.name} must have a 'symbol' column")
        key = next(f for f in reader.fieldnames if (f or "").strip().lower() == "symbol")
        for row in reader:
            s = (row.get(key) or "").strip().upper()
            if s and s not in syms:
                syms.append(s)

    if settings.limit:
        syms = syms[: settings.limit]
        log.info("--limit %d applied", settings.limit)
    log.info("%d company symbol(s) to process", len(syms))
    return syms


def process_company(session, store: Store, symbol: str, scrip_code: str,
                    settings: Settings) -> dict:
    """Steps 1-4 for one company. Never raises."""
    state = {"symbol": symbol, "scrip_code": scrip_code,
             "status": "FAILED", "transcripts_found": 0, "quarters": []}

    rows = bse.fetch_announcements(
        session, symbol, scrip_code, settings.lookback_months, force=settings.force)
    store.log_event(symbol, "", "announcements", f"{len(rows)} filings fetched")
    store.upsert_company(symbol, scrip_code=scrip_code, ann_count=len(rows))

    if not rows:
        state["status"] = "FAILED"
        store.log_event(symbol, "", "announcements", "no announcements returned by BSE")
        return state

    picked = bse.select_transcripts(symbol, rows, settings.n_transcripts)
    if not picked:
        state["status"] = "NO TRANSCRIPTS FILED"
        store.log_event(symbol, "", "select", "zero transcript filings in lookback window")
        log.info("%s: no transcript filings (this is a valid result)", symbol)
        return state

    seen_hashes: set[str] = set()
    good = 0
    for row in picked:
        quarter = row.get("_quarter", "UNKNOWN")
        try:
            dl = bse.download_pdf(session, symbol, row, quarter)
        except Exception as exc:
            log.error("%s %s: download crashed: %s", symbol, quarter, exc)
            store.log_event(symbol, quarter, "download", f"crashed: {exc}")
            store.upsert_transcript(symbol, quarter, failure=f"download crashed: {exc}")
            continue

        if dl.get("failure") or not dl.get("pdf_path"):
            store.log_event(symbol, quarter, "download", dl.get("failure") or "failed")
            store.upsert_transcript(symbol, quarter, news_dt=dl["news_dt"],
                                    attachment=dl["attachment"],
                                    failure=dl.get("failure") or "download failed")
            continue

        # BSE files one combined PDF under several announcement IDs; dedupe by
        # content so the same document is not analysed twice.
        h = dl.get("sha256")
        if h and h in seen_hashes:
            store.log_event(symbol, quarter, "download",
                            "duplicate content of an earlier quarter, skipped")
            continue
        if h:
            seen_hashes.add(h)

        store.log_event(symbol, quarter, "download",
                        f"ok via {dl['url_path']}")

        try:
            ex = extraction.extract(dl["pdf_path"], symbol, quarter)
        except Exception as exc:
            log.error("%s %s: extraction crashed: %s", symbol, quarter, exc)
            store.log_event(symbol, quarter, "extract", f"crashed: {exc}")
            continue

        fields = dict(
            news_dt=dl["news_dt"], attachment=dl["attachment"],
            url_path=dl["url_path"], pdf_path=dl["pdf_path"],
            txt_path=ex.get("txt_path"), n_chars=ex.get("n_chars"),
            extract_method=ex.get("extract_method"),
            kind=ex.get("kind"), failure=ex.get("failure"),
        )
        # The cached-text path cannot recover a page count; keep the stored one
        # rather than overwriting a real value with zero.
        if ex.get("n_pages"):
            fields["n_pages"] = ex["n_pages"]
        store.upsert_transcript(symbol, quarter, **fields)
        store.log_event(symbol, quarter, "extract",
                        f"{ex.get('kind')} via {ex.get('extract_method')} "
                        f"({ex.get('n_pages')}pg, {ex.get('n_chars')}ch)"
                        + (f" | {ex['failure']}" if ex.get("failure") else ""))

        # Optional second hop for cover-letter-only filings.
        if ex.get("kind") == "COVER_LETTER_ONLY" and settings.follow_pointers:
            hop = website.fetch_from_pointer(
                session, symbol, quarter, ex.get("pointer_url") or "")
            store.log_event(symbol, quarter, "website",
                            hop.get("failure") or f"recovered from {hop.get('source')}")
            if hop.get("pdf_path"):
                try:
                    ex2 = extraction.extract(hop["pdf_path"], symbol, quarter)
                except Exception as exc:
                    log.error("%s %s: re-extraction crashed: %s", symbol, quarter, exc)
                    ex2 = None
                if ex2 and ex2.get("kind") != "TRANSCRIPT":
                    store.log_event(symbol, quarter, "website",
                                    f"fetched file was {ex2.get('kind')}, not a "
                                    f"transcript; keeping BSE original")
                if ex2 and ex2.get("kind") == "TRANSCRIPT":
                    ex = ex2
                    store.upsert_transcript(
                        symbol, quarter, txt_path=ex.get("txt_path"),
                        n_pages=ex.get("n_pages"), n_chars=ex.get("n_chars"),
                        extract_method=ex.get("extract_method"),
                        kind=ex.get("kind"), failure=None,
                        url_path="company-website")
                    store.log_event(symbol, quarter, "extract",
                                    f"TRANSCRIPT recovered from company website "
                                    f"({ex.get('n_pages')}pg, {ex.get('n_chars')}ch)")

        if ex.get("kind") == "TRANSCRIPT":
            good += 1
            state["quarters"].append(quarter)

    state["transcripts_found"] = good
    if good >= settings.n_transcripts:
        state["status"] = "OK"
    elif good > 0:
        state["status"] = f"PARTIAL ({good} of {settings.n_transcripts})"
    else:
        # Distinguish "never holds calls" from "files a pointer, not the text".
        kinds = [t.get("kind") for t in store.get_transcripts(symbol)]
        if "COVER_LETTER_ONLY" in kinds:
            state["status"] = "TRANSCRIPTS NOT ATTACHED"
            urls = {t.get("failure", "") for t in store.get_transcripts(symbol)
                    if t.get("kind") == "COVER_LETTER_ONLY"}
            store.log_event(symbol, "", "select",
                            "company files a cover letter pointing at its own "
                            "website instead of attaching the transcript; "
                            + "; ".join(sorted(u for u in urls if u))[:400])
            log.warning("%s: transcripts exist but are hosted on the company site",
                        symbol)
        else:
            state["status"] = "NO TRANSCRIPTS FILED"
            store.log_event(symbol, "", "select",
                            "transcript filings existed but none yielded usable text")
    return state


def run(settings: Settings) -> None:
    t0 = time.time()
    store = Store()
    symbols = read_companies(settings)

    # ------------------------------------------------------------- step 0
    scrip_map = scrip.build_scrip_map(symbols)

    session = make_session()
    states: dict[str, dict] = {}

    # ---------------------------------------------------------- steps 1-4
    for i, sym in enumerate(symbols, 1):
        log.info("[%d/%d] === %s ===", i, len(symbols), sym)
        code = scrip_map.get(sym.upper())
        if not code:
            states[sym] = {"symbol": sym, "scrip_code": "",
                           "status": "SCRIP CODE NOT FOUND",
                           "transcripts_found": 0, "quarters": []}
            store.upsert_company(sym, status="SCRIP CODE NOT FOUND")
            store.log_event(sym, "", "scrip", "could not map symbol to BSE scrip code")
            log.warning("%s: no scrip code, carrying through to Excel", sym)
            continue
        try:
            states[sym] = process_company(session, store, sym, code, settings)
        except Exception as exc:
            log.exception("%s: unhandled error, continuing", sym)
            states[sym] = {"symbol": sym, "scrip_code": code, "status": "FAILED",
                           "transcripts_found": 0, "quarters": []}
            store.log_event(sym, "", "company", f"unhandled: {exc}")
        store.upsert_company(sym, scrip_code=states[sym].get("scrip_code", ""),
                             status=states[sym]["status"],
                             transcripts_found=states[sym]["transcripts_found"])

    # ---------------------------------------------------------- steps 5-6
    if settings.skip_llm:
        log.info("--skip-llm set, going straight to Excel")
    else:
        _run_llm(store, states, settings)

    # ------------------------------------------------------------- step 7
    _write_output(store, states, settings)
    _summarise(store, states, t0)
    store.close()


def _run_llm(store: Store, states: dict, settings: Settings) -> None:
    try:
        model = llm.pick_model(settings.model)
    except llm.GpuUnavailable as exc:
        log.error("GPU unusable, skipping LLM stages: %s", exc)
        log.error("The fetch and extract results are still valid; the Excel will "
                  "be written with empty analysis columns. Re-run without "
                  "--skip-llm once the GPU works and only the LLM stage will run.")
        return
    proc = None
    if settings.autostart_vllm:
        proc = llm.autostart_vllm(model)
        if proc is None:
            log.error("could not start vLLM; skipping analysis")
            return
    elif not llm.wait_for_server(llm.VLLM_HOST, timeout=30):
        log.error("no vLLM server at %s. Start it, or pass --autostart-vllm. "
                  "Skipping LLM stages.", llm.VLLM_HOST)
        return

    if settings.redo_llm:
        n = store.clear_analysis()
        log.info("--redo-llm: cleared %d cached analysis/diff row(s)", n)

    try:
        client = llm.LLMClient(model)
    except Exception as exc:
        log.error("could not build LLM client: %s", exc)
        return

    jobs: list[tuple[str, str, str]] = []
    for sym in states:
        for t in store.get_transcripts(sym):
            if t.get("kind") != "TRANSCRIPT" or not t.get("txt_path"):
                continue
            if (store.get_analysis(sym, t["quarter"])
                    and not (settings.force or settings.redo_llm)):
                log.debug("%s %s: analysis cached", sym, t["quarter"])
                continue
            try:
                text = open(t["txt_path"], encoding="utf-8").read()
            except Exception as exc:
                store.log_event(sym, t["quarter"], "llm", f"could not read text: {exc}")
                continue
            jobs.append((sym, t["quarter"], text))

    results = llm.analyse_batch(client, jobs)
    for (sym, q), payload in results.items():
        clean = llm.validated(payload)
        if clean is None:
            store.log_event(sym, q, "llm", "analysis failed schema validation")
            log.warning("%s %s: analysis unusable", sym, q)
            continue
        store.save_analysis(sym, q, clean)
        store.log_event(sym, q, "llm", "analysis ok")

    # ---- step 6, one diff call per company
    for sym in states:
        if store.get_diff(sym) and not (settings.force or settings.redo_llm):
            continue
        pairs = store.get_all_analysis(sym)
        if not pairs:
            continue
        pairs.sort(key=lambda p: p[0])  # chronological
        try:
            d = llm.diff_quarters(client, sym, pairs)
        except Exception as exc:
            log.error("%s: diff crashed: %s", sym, exc)
            store.log_event(sym, "", "diff", f"crashed: {exc}")
            continue
        store.save_diff(sym, d)
        store.log_event(sym, "", "diff",
                        f"shift_score={d.get('shift_score')}")

    if proc:
        log.info("stopping vLLM server")
        proc.terminate()


def _write_output(store: Store, states: dict, settings: Settings) -> None:
    summary_rows, quarter_rows = [], []
    for sym, st in states.items():
        diff = store.get_diff(sym) or {}
        pairs = sorted(store.get_all_analysis(sym), key=lambda p: p[0])
        summary_rows.append({
            "symbol": sym,
            "scrip_code": st.get("scrip_code", ""),
            "status": st.get("status", "FAILED"),
            "transcripts_found": st.get("transcripts_found", 0),
            "quarters_covered": ", ".join(q for q, _ in pairs) or
                                ", ".join(st.get("quarters", [])),
            "shift_score": diff.get("shift_score", ""),
            "narrative_shift": diff.get("narrative_shift", ""),
            "new_in_latest": diff.get("new_in_latest", []),
            "dropped": diff.get("dropped", []),
            "tone_trajectory": diff.get("tone_trajectory", ""),
        })
        tmeta = {t["quarter"]: t for t in store.get_transcripts(sym)}
        analysed = dict(pairs)
        # Emit a row for every quarter we touched, analysed or not, so the sheet
        # is still a useful inventory when --skip-llm was used or a call failed.
        for q in sorted(tmeta, reverse=True):
            meta = tmeta[q]
            row = {
                "symbol": sym, "quarter": q,
                "news_dt": meta.get("news_dt", ""),
                "kind": meta.get("kind", ""),
                "n_pages": meta.get("n_pages", ""),
                "n_chars": meta.get("n_chars", ""),
                "extract_method": meta.get("extract_method", ""),
            }
            a = analysed.get(q)
            if a:
                row.update(a)
            elif meta.get("failure"):
                row["business_summary"] = f"[not analysed] {meta['failure']}"
            quarter_rows.append(row)

    output.write_excel(summary_rows, quarter_rows, store.runlog())


def _summarise(store: Store, states: dict, t0: float) -> None:
    counts = Counter()
    for st in states.values():
        n = st.get("transcripts_found", 0)
        if st["status"].startswith("SCRIP"):
            counts["no_scrip"] += 1
        elif n >= 4:
            counts["full4"] += 1
        elif n > 0:
            counts["partial"] += 1
        else:
            counts["none"] += 1

    ocr = sum(1 for t in store.all_transcripts() if t.get("extract_method") == "ocr")
    cover = sum(1 for t in store.all_transcripts() if t.get("kind") == "COVER_LETTER_ONLY")
    mins = (time.time() - t0) / 60

    log.info("=" * 62)
    log.info("RUN COMPLETE")
    log.info("  companies processed      : %d", len(states))
    log.info("  got all 4 transcripts    : %d", counts["full4"])
    log.info("  got 1-3 transcripts      : %d", counts["partial"])
    log.info("  got 0 transcripts        : %d", counts["none"])
    log.info("  scrip code not found     : %d", counts["no_scrip"])
    log.info("  needed OCR               : %d", ocr)
    log.info("  cover-letter-only filings: %d", cover)
    log.info("  LLM calls made           : %d", llm.LLM_CALLS["n"])
    log.info("  wall clock               : %.1f min", mins)
    log.info("=" * 62)
