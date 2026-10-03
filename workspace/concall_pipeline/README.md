# Concall Pipeline

Downloads the last 4 earnings conference call transcripts for a list of Indian
listed companies from BSE, then extracts structured insights with a local LLM on
an A100 and writes everything to Excel.

Transcripts come from **BSE only**. indianapi.in is used once, to map NSE symbols
to BSE scrip codes, and never contributes a transcript.

---

## Install

Requirements are split so you do not drag in a GPU stack you may not need.

**Core, CPU only, about 50MB.** Enough for fetching, extraction and Excel:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**GPU stage, on the A100 only.** Install after the core file:

```bash
pip install -r requirements-gpu.txt
```

That second one is a big install. vLLM pulls torch, CUDA kernels, flashinfer and
related wheels: roughly 15GB of disk and 10 to 20 minutes of downloading. That is
normal, not a misconfiguration. torch is deliberately left unpinned so vLLM can
resolve the exact build it was compiled against.

OCR needs two system binaries, used only when a PDF turns out to be a scan:

```bash
sudo apt-get install -y tesseract-ocr poppler-utils
```

---

## Input

`companies.csv`, one NSE symbol per row:

```csv
symbol
TIMEX
SHAILY
PRICOL
```

---

## Run

**First run** needs an indianapi key to build the scrip code cache:

```bash
export INDIANAPI_KEY='your-key'
python run.py --limit 5 --skip-llm
```

**Every later run** reads `scrip_codes.csv` and needs no key at all:

```bash
python run.py --skip-llm
```

**Full run with LLM analysis on the A100.** Start the model server in one shell:

```bash
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --quantization awq --max-model-len 32768 --gpu-memory-utilization 0.90
```

Then in another:

```bash
python run.py
```

Or let the pipeline start and stop vLLM itself:

```bash
python run.py --autostart-vllm
```

### Useful flags

| flag | effect |
|---|---|
| `--limit N` | process only the first N symbols |
| `--companies A,B,C` | run specific symbols, ignores the CSV |
| `--skip-llm` | fetch and extract only, no GPU |
| `--follow-pointers` | chase transcripts hosted on company websites (see below) |
| `--force` | ignore all caches and redo everything |
| `--transcripts N` | how many quarters to fetch, default 4 |
| `--lookback-months N` | BSE history window, default 24 |
| `--model NAME` | pin a model instead of auto-selecting by VRAM |
| `-v` | debug logging |

---

## Output

```
concall_analysis.xlsx      Summary / By Quarter / Run Log
transcripts/<SYM>/         <SYM>_FY26Q4.pdf and .txt per quarter
raw/announcements/         cached BSE announcement JSON per company
scrip_codes.csv            symbol -> BSE scrip code cache
pipeline.db                SQLite checkpoint
logs/run_<timestamp>.log
```

**Summary** has one row per company including the ones with nothing, sorted by
`shift_score` descending with problem rows grouped at the bottom and the status
column colour coded.

**By Quarter** has one row per company-quarter with every extracted field. It is
populated even under `--skip-llm`, where it acts as a transcript inventory.

**Run Log** records what happened at each stage: announcements fetched, which URL
path worked, text versus OCR, and the reason for every failure.

---

## Status values

| status | meaning |
|---|---|
| `OK` | all 4 transcripts retrieved |
| `PARTIAL (n of 4)` | fewer than 4 exist or some failed |
| `TRANSCRIPTS NOT ATTACHED` | company files a cover letter pointing at its own website |
| `NO TRANSCRIPTS FILED` | company genuinely does not publish transcripts |
| `SCRIP CODE NOT FOUND` | symbol could not be mapped to a BSE code |
| `FAILED` | unexpected error, see Run Log |

`NO TRANSCRIPTS FILED` is a real result, not a bug. Plenty of small caps never
file one. Timex filed 220 announcements over two years and not a single
transcript among them.

---

## How the pieces work

**Scrip codes.** BSE keys everything by numeric code, not NSE symbol. The mapping
is resolved once via indianapi and cached. Resolution matches on the exact NSE
ticker and refuses to guess when a query is ambiguous, because substring matching
on company names silently picks the wrong company.

**Announcements.** `AnnSubCategoryGetData` returns full history with a date range.
Pagination is mandatory; one page silently truncates. BSE answers "No Record
Found!" as a bare JSON string rather than an error, which is handled.

**Identifying transcripts.** The primary filter is BSE's own `SUBCATNAME` field,
which is far more reliable than the title. Titles are a backup. Filings that
merely announce a call, such as investor presentations, audio recordings and
intimations, are excluded.

**Downloading.** Attachments live under `AttachLive` when recent and `AttachHis`
when older. Both are tried in that order. Either path alone fails for most
filings, so this is not optional. Responses are checked for the `%PDF` magic
bytes because BSE sometimes serves an HTML error page under HTTP 200.

**Deduplication.** Some companies file one combined PDF under several
announcement IDs. Files are hashed and duplicates skipped.

**Extraction.** pdfplumber first. Under 500 characters means it is a scan, so it
falls back to pytesseract. The method used is recorded per file.

**Cover letters.** Some companies, including Piramal and Federal Bank on some
quarters, file a one-page letter saying the transcript is on their website. That
is detected and reported as `COVER_LETTER_ONLY` rather than being passed to the
LLM as if it were a transcript. With `--follow-pointers` the pipeline extracts
the URL from the letter and tries to fetch the real document. Direct PDF links
work. Landing pages that render their file lists with JavaScript do not, and the
pipeline says so explicitly rather than silently grabbing the wrong file.

**LLM.** The model is chosen by detected VRAM: 32B AWQ at 80GB, 14B AWQ at 40GB.
It is loaded once by the vLLM server and reused for the whole run. Requests are
batched so the GPU is not idling on network I/O.

Structured output is negotiated at runtime rather than pinned. vLLM has renamed
this parameter twice, so on the first call the client tries
`response_format: json_schema`, then `structured_outputs`, then the older
`guided_json`, then plain JSON mode with the schema in the prompt. Whichever the
server accepts is logged and reused for the rest of the run. Output is then
validated with pydantic and retried twice before the row is marked failed. Transcripts that exceed the
context window are split into management commentary and Q&A, analysed
separately, and merged. The Q&A half is never dropped, and it wins on tone.

**Quarter diff.** One extra call per company compares the structured summaries in
chronological order and produces `narrative_shift`, `new_in_latest`, `dropped`,
`tone_trajectory` and `shift_score` from 0 to 10. Skipped with
`INSUFFICIENT DATA` when fewer than 2 transcripts exist.

---

## Resilience

One company failing never stops the run. Every company is wrapped, logged and
skipped on error.

Everything is cached and checkpointed to SQLite. Re-running skips completed work,
so an interrupted run resumes cheaply. `--force` overrides this.

BSE is rate limited to a random 2 to 4 second gap, with exponential backoff on
429 and 5xx and up to 3 retries. Do not lower this. BSE will throttle you.

---

## Known limits

**BSE only.** Companies listed solely on NSE return nothing. NSE has an
equivalent announcements API and would need a separate connector.

**JavaScript IR sites.** `--follow-pointers` handles direct PDF links and static
HTML. Piramal's page renders its list client side, so recovering those needs a
headless browser, which is not implemented.

**The A100 is only used for the LLM stage.** Fetching and extraction are network
and I/O bound. The one other place the GPU could help is OCR, which currently
runs on CPU via tesseract; swap in a GPU OCR engine if you hit many scans.

**Quarter inference is date based.** The fiscal quarter is derived from the
filing month, which is correct for the normal pattern of filing a few weeks after
quarter end. A company filing very late could be labelled one quarter off, so
check `news_dt` in the By Quarter sheet if a label looks wrong.

---

## Verified behaviour

Tested end to end on a 6 company watchlist:

| symbol | result |
|---|---|
| SHAILY | OK, 4 of 4 |
| PRICOL | OK, 4 of 4 |
| RPTECH | OK, 4 of 4 |
| FEDERALBNK | PARTIAL, 2 of 4, rest are website pointers |
| PPLPHARMA | TRANSCRIPTS NOT ATTACHED, all 4 are pointers to a JS-rendered page |
| TIMEX | NO TRANSCRIPTS FILED, 220 announcements, zero transcripts |

Both `AttachLive` and `AttachHis` paths were exercised. The LLM stage has not
been run here since this machine has no GPU.
