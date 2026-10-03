"""STEPS 5-6 - vLLM-backed extraction and the quarter-over-quarter diff.

The model is loaded once by the vLLM server and reused for the whole run. This
module only talks to the OpenAI-compatible endpoint, so the server can be started
by hand or by --autostart-vllm.
"""

from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .config import (
    CHARS_PER_TOKEN,
    LLM_MAX_CONCURRENCY,
    LLM_RETRIES,
    MAX_MODEL_LEN,
    MODEL_40GB,
    MODEL_80GB,
    PROMPT_TOKEN_BUDGET,
    VLLM_API_KEY,
    VLLM_HOST,
    VLLM_START_TIMEOUT,
)
from .extraction import split_commentary_qa
from .logging_setup import get_logger
from .schemas import (
    EMPTY_ANALYSIS,
    QuarterDiff,
    TranscriptAnalysis,
    json_schema,
)

log = get_logger("llm")

LLM_CALLS = {"n": 0}


# --------------------------------------------------------------- GPU / model
class GpuUnavailable(RuntimeError):
    """CUDA is not usable, so there is no point starting a model server."""


def detect_vram_gb() -> float:
    """Return max per-device VRAM in GB. Raises GpuUnavailable if CUDA is dead.

    The common failure here is a driver older than the CUDA build torch was
    compiled against. torch reports that as a warning and then quietly says no
    device is available, which must not be mistaken for a small GPU.
    """
    try:
        import torch
    except ImportError as exc:
        raise GpuUnavailable(f"torch is not installed ({exc})") from exc

    try:
        available = torch.cuda.is_available()
    except Exception as exc:
        raise GpuUnavailable(f"torch.cuda.is_available() raised {exc}") from exc

    if not available:
        drv = None
        try:
            drv = torch.version.cuda
        except Exception:
            pass
        raise GpuUnavailable(
            "torch cannot see any CUDA device. The usual cause is an NVIDIA "
            "driver older than the CUDA version torch was built for "
            f"(torch was built for CUDA {drv}). Check `nvidia-smi`, then either "
            "update the driver or install a vLLM/torch pair matching it, e.g. "
            "`pip install vllm==0.8.5.post1`."
        )

    total = max(
        torch.cuda.get_device_properties(i).total_memory
        for i in range(torch.cuda.device_count())
    )
    return total / (1024 ** 3)


def pick_model(explicit: str | None = None) -> str:
    """Choose a model by VRAM. Raises GpuUnavailable if the GPU is unusable."""
    vram = 0.0
    try:
        vram = detect_vram_gb()
        log.info("detected %.0f GB VRAM", vram)
    except GpuUnavailable as exc:
        if not explicit:
            raise
        # An explicit --model means the user knows what they want; the server
        # may even be remote. Warn but continue.
        log.warning("GPU check failed (%s) but --model was given, continuing", exc)

    if explicit:
        log.info("model pinned by --model: %s", explicit)
        return explicit

    if vram >= 70:
        model = MODEL_80GB
    elif vram >= 35:
        model = MODEL_40GB
    else:
        raise GpuUnavailable(
            f"only {vram:.0f} GB VRAM detected, which is too small for either "
            f"configured model. Pass --model to override if this is wrong."
        )
    log.info("selected model: %s", model)
    return model


def wait_for_server(base_url: str, timeout: int = 900) -> bool:
    health = base_url.rstrip("/").removesuffix("/v1") + "/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(health, timeout=5).status_code == 200:
                log.info("vLLM server is up after %.0fs", time.time() - start)
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def autostart_vllm(model: str) -> subprocess.Popen | None:
    """Launch vLLM as a subprocess, tee-ing its output to logs/vllm.log.

    The first launch downloads ~20GB of weights and can take 20 minutes with no
    visible progress, so its output goes to a file the user can tail rather than
    being discarded.
    """
    from .config import LOG_DIR

    cmd = [
        "vllm", "serve", model,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90",
        "--disable-log-requests",
    ]
    if model.endswith("AWQ"):
        cmd += ["--quantization", "awq"]

    vllm_log = LOG_DIR / "vllm.log"
    log.info("starting vLLM: %s", " ".join(cmd))
    log.info("vLLM output -> %s   (tail -f it to watch startup)", vllm_log)
    log.info("first launch downloads model weights; allow up to 30 minutes")

    try:
        fh = open(vllm_log, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        log.error("vllm not found on PATH; start the server manually and re-run "
                  "without --autostart-vllm")
        return None

    if not wait_for_server(VLLM_HOST, timeout=VLLM_START_TIMEOUT):
        rc = proc.poll()
        log.error("vLLM did not become healthy within %ds (process %s)",
                  VLLM_START_TIMEOUT,
                  f"exited with code {rc}" if rc is not None else "still running")
        try:
            tail = vllm_log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            log.error("last lines of %s:", vllm_log.name)
            for line in tail:
                log.error("  | %s", line)
        except Exception:
            pass
        if rc is None:
            proc.terminate()
        return None
    return proc


# ------------------------------------------------------------------- client
class LLMClient:
    """Client that negotiates whichever structured-output API this vLLM speaks.

    The parameter has been renamed twice: `guided_json` in extra_body (vLLM
    <=0.7), the OpenAI-compatible `response_format: json_schema` (vLLM v1), and
    `structured_outputs` in the newest builds. Rather than pin a version, we try
    each once, remember the one that works, and reuse it for the whole run.
    """

    # Ordered newest-first, since a fresh `pip install vllm` gets the newest.
    STRATEGIES = ("json_schema", "structured_outputs", "guided_json", "json_object")

    def __init__(self, model: str, base_url: str = VLLM_HOST):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=VLLM_API_KEY, timeout=600)
        self.strategy: str | None = None

    def _kwargs(self, strategy: str, schema: dict) -> dict:
        if strategy == "json_schema":
            return {"response_format": {
                "type": "json_schema",
                "json_schema": {"name": "analysis", "schema": schema, "strict": True},
            }}
        if strategy == "structured_outputs":
            return {"extra_body": {"structured_outputs": {"json": schema}}}
        if strategy == "guided_json":
            return {"extra_body": {"guided_json": schema}}
        # Last resort: JSON mode with the schema described in the prompt. Weaker,
        # but pydantic validation downstream still catches bad output.
        return {"response_format": {"type": "json_object"}}

    def _call(self, system: str, user: str, schema: dict,
              strategy: str, max_tokens: int) -> dict:
        if strategy == "json_object":
            user = (f"{user}\n\nReturn ONLY a JSON object matching this schema:\n"
                    f"{json.dumps(schema)}")
        LLM_CALLS["n"] += 1
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.1,
            max_tokens=max_tokens,
            **self._kwargs(strategy, schema),
        )
        return json.loads(resp.choices[0].message.content)

    def _negotiate(self, system: str, user: str, schema: dict,
                   label: str, max_tokens: int) -> dict | None:
        """First real call: find a structured-output API this server accepts."""
        for strategy in self.STRATEGIES:
            try:
                out = self._call(system, user, schema, strategy, max_tokens)
                self.strategy = strategy
                log.info("structured output negotiated: using %r", strategy)
                return out
            except Exception as exc:
                log.warning("structured-output mode %r rejected (%s), trying next",
                            strategy, str(exc)[:160])
        log.error("%s: no structured-output mode accepted by this vLLM server", label)
        return None

    def complete_json(self, system: str, user: str, schema: dict,
                      label: str = "", max_tokens: int = 2048) -> dict | None:
        """One structured call with retries. Returns parsed dict or None."""
        if self.strategy is None:
            return self._negotiate(system, user, schema, label, max_tokens)

        for attempt in range(1, LLM_RETRIES + 2):
            try:
                return self._call(system, user, schema, self.strategy, max_tokens)
            except Exception as exc:
                log.warning("%s: LLM attempt %d/%d failed: %s",
                            label, attempt, LLM_RETRIES + 1, str(exc)[:200])
                if attempt > LLM_RETRIES:
                    log.error("%s: giving up after %d attempts", label, attempt)
                    return None
                time.sleep(2 * attempt)
        return None


# ------------------------------------------------------------------ prompts
SYSTEM_ANALYST = """You are an equity research analyst taking margin notes on an \
Indian earnings call. You are writing for a reader who will NOT read the transcript.

HOW TO WRITE:
- Telegraphic notes, not prose. No full sentences except in notable_quotes.
- NEVER copy a sentence from the transcript. Compress it into your own shorthand.
- Numbers are the point. Always keep the figure, the unit and the direction.
- If something is not mentioned, write "n/a". Do not pad, infer or invent.

BAD (this is transcription, do not do this):
  "Revenue stood at INR251 crores, up 27% year-on-year, with EBITDA at INR66
   crores, up 43% year-on-year, resulting in a margin of 26.5%, representing an
   expansion of 310 basis points year-on-year."

GOOD (this is a note):
  key_numbers: ["Rev 251cr +27% YoY", "EBITDA 66cr +43%", "OPM 26.5% +310bps"]

BAD margin_commentary: "EBITDA margin was 31.8%, up 1,030 bps YoY."
GOOD margin_commentary: "Mix shift to healthcare; pen injectors carry higher realisation"
  (the number belongs in key_numbers; this field is for the CAUSE)

what_changed is the most important field. Put only genuine change there: a new
product line, a new customer type, a reversal, a first-time disclosure, a
guidance change. Routine growth is NOT a change. An empty list is a valid and
useful answer."""

SYSTEM_DIFF = """You are an equity research analyst comparing consecutive quarterly \
earnings calls from one company. Your reader wants to know if the story changed.

Be strict. Repetition of the same story with bigger numbers is NOT a shift, it is
execution. A shift means the company is describing a different business: new
end-market, new product category, changed strategy, reversed guidance, or a
segment that went from immaterial to dominant.

Write telegraphically. No preamble, no restating the summaries back.

shift_score calibration:
  0-2  same business, same drivers, numbers moved
  3-5  meaningful new initiative or a visible change in emphasis
  6-8  revenue mix or strategy materially changed within these quarters
  9-10 effectively a different company now

Most companies are 0-3. Reserve 6+ for cases where you can name the specific
quarter the change appears and what it displaced."""


def _fits(text: str) -> bool:
    return len(text) / CHARS_PER_TOKEN <= PROMPT_TOKEN_BUDGET


def _truncate(text: str) -> str:
    limit = int(PROMPT_TOKEN_BUDGET * CHARS_PER_TOKEN)
    return text[:limit]


def _merge(a: dict, b: dict) -> dict:
    """Merge two half-transcript analyses. Q&A half wins on tone and risks."""
    out = dict(a)
    for key in ("new_themes", "notable_quotes", "risks_flagged",
                "key_numbers", "what_changed"):
        merged, seen = [], set()
        for item in list(a.get(key) or []) + list(b.get(key) or []):
            k = str(item).strip().lower()
            if k and k not in seen:
                seen.add(k)
                merged.append(item)
        out[key] = merged[:3] if key == "notable_quotes" else merged

    seg, seen = [], set()
    for s in list(a.get("segments") or []) + list(b.get("segments") or []):
        name = str(s.get("name", "")).strip().lower()
        if name and name not in seen:
            seen.add(name)
            seg.append(s)
    out["segments"] = seg

    for key in ("order_book", "capex", "guidance", "margin_commentary",
                "demand_commentary", "business_summary"):
        av, bv = (a.get(key) or "").strip(), (b.get(key) or "").strip()
        blank = ("", "n/a", "not stated")
        if av in blank and bv not in blank:
            out[key] = bv
        elif av not in blank and bv not in blank and av != bv:
            out[key] = f"{av} | {bv}"[:140]

    # The Q&A section is where tone is actually revealed.
    out["management_tone"] = b.get("management_tone") or a.get("management_tone", "neutral")
    out["tone_justification"] = b.get("tone_justification") or a.get("tone_justification", "")
    return out


def analyse_transcript(client: LLMClient, symbol: str, quarter: str, text: str) -> dict | None:
    """STEP 5 for one transcript, splitting if it will not fit the window."""
    schema = json_schema(TranscriptAnalysis)
    label = f"{symbol}/{quarter}"

    if _fits(text):
        user = (f"Company NSE symbol: {symbol}\nQuarter: {quarter}\n\n"
                f"TRANSCRIPT:\n{text}")
        return client.complete_json(SYSTEM_ANALYST, user, schema, label=label)

    log.info("%s: transcript is ~%.0fk tokens, splitting commentary and Q&A",
             label, len(text) / CHARS_PER_TOKEN / 1000)
    commentary, qa = split_commentary_qa(text)
    parts = []
    for name, chunk in (("management commentary", commentary), ("Q&A", qa)):
        user = (f"Company NSE symbol: {symbol}\nQuarter: {quarter}\n"
                f"This is the {name} section of the call.\n\n"
                f"TRANSCRIPT SECTION:\n{_truncate(chunk)}")
        got = client.complete_json(SYSTEM_ANALYST, user, schema, label=f"{label}/{name}")
        parts.append(got)

    if parts[0] and parts[1]:
        return _merge(parts[0], parts[1])
    return parts[0] or parts[1]


def diff_quarters(client: LLMClient, symbol: str,
                  ordered: list[tuple[str, dict]]) -> dict:
    """STEP 6. `ordered` is [(quarter, analysis), ...] oldest first."""
    if len(ordered) < 2:
        log.info("%s: fewer than 2 transcripts, skipping diff", symbol)
        from .schemas import INSUFFICIENT_DIFF

        return dict(INSUFFICIENT_DIFF)

    blocks = []
    for q, a in ordered:
        blocks.append(
            f"=== {q} ===\n"
            f"Numbers: {a.get('key_numbers', [])}\n"
            f"Segments: {[s.get('name') for s in a.get('segments', [])]}\n"
            f"Changed this qtr: {a.get('what_changed', [])}\n"
            f"New themes: {a.get('new_themes', [])}\n"
            f"Guidance: {a.get('guidance','')}\n"
            f"Margin driver: {a.get('margin_commentary','')}\n"
            f"Demand: {a.get('demand_commentary','')}\n"
            f"Capex: {a.get('capex','')}\n"
            f"Tone: {a.get('management_tone','')} ({a.get('tone_justification','')})\n"
            f"Risks: {a.get('risks_flagged', [])}\n"
        )
    user = (
        f"Company NSE symbol: {symbol}\n"
        f"Below are structured summaries of {len(ordered)} consecutive earnings "
        f"calls, oldest first.\n\n" + "\n".join(blocks) +
        "\n\nshift_score guidance: 0 means business as usual across these quarters; "
        "10 means the company now describes itself as a fundamentally different "
        "business. Most companies score 0-3."
    )
    got = client.complete_json(SYSTEM_DIFF, user, json_schema(QuarterDiff),
                               label=f"{symbol}/diff", max_tokens=1536)
    if not got:
        from .schemas import INSUFFICIENT_DIFF

        out = dict(INSUFFICIENT_DIFF)
        out["narrative_shift"] = "FAILED"
        return out
    return got


def analyse_batch(client: LLMClient, jobs: list[tuple[str, str, str]]) -> dict:
    """Run many transcript analyses concurrently.

    jobs is [(symbol, quarter, text), ...]. vLLM batches server side, so the GPU
    stays busy instead of idling between sequential requests.
    """
    results: dict[tuple[str, str], dict | None] = {}
    if not jobs:
        return results
    log.info("dispatching %d transcript analyses (concurrency %d)",
             len(jobs), LLM_MAX_CONCURRENCY)
    with ThreadPoolExecutor(max_workers=LLM_MAX_CONCURRENCY) as pool:
        futs = {
            pool.submit(analyse_transcript, client, s, q, t): (s, q)
            for s, q, t in jobs
        }
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                results[key] = fut.result()
            except Exception as exc:
                log.error("%s/%s: analysis crashed: %s", key[0], key[1], exc)
                results[key] = None
    return results


def validated(payload: dict | None) -> dict | None:
    """Validate against the pydantic model; None means unusable."""
    if payload is None:
        return None
    try:
        return TranscriptAnalysis.model_validate(payload).model_dump()
    except Exception as exc:
        log.warning("schema validation failed: %s", exc)
        merged = dict(EMPTY_ANALYSIS)
        if isinstance(payload, dict):
            for k in merged:
                if k in payload:
                    merged[k] = payload[k]
        try:
            return TranscriptAnalysis.model_validate(merged).model_dump()
        except Exception:
            return None
