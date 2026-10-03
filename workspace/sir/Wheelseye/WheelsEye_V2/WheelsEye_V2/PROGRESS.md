# Progress

## v2 migration status (2026-08-18) — code written, NOTHING run yet

The v2 plan (see NOTES.md "v2 migration" for rationale) is implemented in
code but **deliberately not executed** — no feature build, no fold
generation, no training, no tests run. Execution order when approved:

1. `python -m pytest tests/ -v` (13 v1 tests + the new `tests/test_v2.py`)
2. `python scripts/verify_uldd.py` (re-verify the local dataset copy)
3. `python scripts/build_feature_table.py` (v2 columns: FAU, BVP-spectral,
   ACC, posture, motion, context — rebuilds `uldd_features.parquet`)
4. `python scripts/build_v2_folds.py` (tier1/tier2/LOSO-v2 fold files)
5. `python scripts/reproduce_uldd_baselines.py` (protocol anchor: expect
   ~83.75 SVM / ~75.14 RF; >5 pp off ⇒ stop and debug the protocol layer)
6. `python scripts/run_experiment.py --tier 1 --model {lightgbm,stacked,
   transformer,i2m2,gnn_v2,gnn_v2_nognn}` (Tier-1 head-to-head; the
   gnn_v2 vs gnn_v2_nognn delta is the GNN's fair-trial verdict)
7. Tier 2 for champions; Tier 3 `--calibrate` on/off for the
   personalization gap table; `--smooth 3` variants for tiers 2/3.

New/changed since v1: `common/{splits,calibration,context_features,
smoothing,features_fau,features_cardiac_spectral,features_acc,
features_posture}.py`, dataset v2 groups + `stats_source`, windowing
`context_slices`, `track_a_heavy/{gnn_v2,i2m2,transformer_fusion}.py`,
`track_b_light/stacked_fusion.py`, `scripts/{build_v2_folds,
run_experiment,reproduce_uldd_baselines}.py`, `tests/test_v2.py`.
v1 status below is retained for reference.

---

Tracks status against [`tasks.md`](tasks.md)'s task numbering. Legend:
✅ code complete and verified (via tests / forward-pass shape checks / brief
smoke tests) · 🟡 written, not yet exercised at all · ⬜ not started.

**No trained artifacts exist anywhere in this repo right now** — no
checkpoints, no ONNX exports, no results JSON, no ablation output.
Everything that had been produced by earlier smoke tests was deleted on
request, since real training hasn't started and shouldn't be represented
by leftover files. Everything marked ✅ below was verified either by
`pytest`, by a forward-pass-only check (`torch.no_grad()`, `model.eval()`,
no optimizer, no `.backward()` — confirmed no weight update occurs), or by
a brief smoke run whose *output* was then discarded (the code path is
proven to work; no number from it should be cited or reused). See
[`NOTES.md`](NOTES.md) for the reasoning behind every deviation from
tasks.md and every bug found along the way.

## Architecture change: Track A's node set is our sensors, not the paper's

`common/fatiguenet_model.py`'s GNN/Transformer/MGAF architecture is
generic (any number of named "nodes"), but what nodes get fed into it for
Track A has changed. It previously defaulted to four nodes mirroring
everything UL-DD happens to record (vision / bio / grip / telemetry). It
now defaults to **three**, matching the sensors the actual product has —
camera feed (vision), physiological sensor (bio: an ESP32-S3 PPG/GSR
breakout board in deployment, Empatica E4 + Checkme O2 Max in UL-DD), and
pressure sensor (grip: ESP32-S3 left/right FSR sensors). UL-DD's
driving-simulator Telemetry (heading/speed/rpm/gear) is excluded from the
default because no vehicle-telemetry sensor exists anywhere in the
deployed ESP32-S3/Jetson hardware — a Track A trained to depend on it
would measure a ceiling Track B could never approach even in principle.

- `common/dataset.py` now exposes `CORE_FEATURES`/`CORE_FEATURE_GROUPS`
  (vision+bio+grip, 37 features) as the default for both tracks, with
  telemetry reachable only via an explicit `include_telemetry=True` on the
  dataset classes and `track_a_heavy.model.build_track_a_model(...)`, for
  a clearly-separate, **Track-A-only** "what if we also had simulator
  telemetry" ceiling ablation — never wired into Track B.
- `track_b_light/model.py`'s `INPUT_DIM` was **37 dims short of a real
  deployability bug**: it was flattening all four groups (including
  telemetry) into Track B's actual input vector, meaning the deployment
  candidate itself was being trained on a feature the real hardware can
  never produce at inference time. Fixed to use `CORE_FEATURES` only.
  `track_b_light/train.py`'s LightGBM path had the identical bug, fixed
  the same way.
- `mephy_repro/` is unchanged and correctly still uses MePhy's own native
  four nodes (ECG/EDA/EMG/blink) — its entire purpose is faithfully
  reproducing FatigueNet on the paper's own dataset, not adapting to our
  hardware, so it's exempt from this change by design.

Verified with forward-pass-only checks: default 3-node Track A model,
`include_telemetry=True` 4-node variant, and Track B's 37-dim MLP/GRU all
produce correctly-shaped output with no crash. No training was run.

## Phase 1 — Data acquisition & preprocessing

- ✅ **Task 1.1** Verify UL-DD (`scripts/verify_uldd.py`) — cross-checks
  every file against `Info.xlsx`'s own availability manifest. Found one
  real anomaly: `J_IBI_A.csv` is 0 bytes despite being marked available.
- ✅ **Task 1.2** `common/labels.py` — KSS → 3-class binning, tested.
- ✅ **Task 1.3** `common/windowing.py` (+ `common/loaders.py`) — 10s/5s
  sliding windows across all UL-DD modalities, tested against real edge
  cases (sparse IBI coverage, missing Telemetry, no-drowsy subjects).
- ✅ **Task 1.4** `common/splits.py` — LOSO fold generation, 16 folds
  (excludes C/F/L), persisted to `data/processed/folds.json`.

## Phase 2 — Feature engineering

- ✅ **Task 2.1** `common/features_vision.py` — EAR/MAR/PERCLOS/head-pose.
- ✅ **Task 2.2** `common/features_bio.py` — HR/HRV, EDA tonic-phasic,
  SpO2, TEMP. Found and handled a `'--'` missing-value sentinel in 3
  O2M.csv files.
- ✅ **Task 2.3** `common/features_grip.py` — grip mean/std/asymmetry.
- ✅ **Task 2.4** `common/features_telemetry.py` (optional, kept for the
  Track-A-only ceiling ablation above) — speed variance, heading-change
  rate with correct 360° wraparound handling.
- ✅ **Task 2.5** `scripts/build_feature_table.py` —
  `data/processed/uldd_features.parquet`, 16,671 windows, all 19 subjects
  (kept on disk: this is a preprocessing artifact, not a trained one).

## Phase 3 — Track A (heavyweight GNN + Transformer + MGAF)

- ✅ **Task 3.1** `common/fatiguenet_model.py` + `track_a_heavy/model.py`
  — full architecture (dynamic correlation adjacency, multi-hop graph
  filter, node attention, Transformer, MetaNet+MGAF, reconstruction loss),
  now configured for our own 3-sensor node set (see above). Verified via
  forward-pass-only checks on real feature dimensions; a real training
  signal (loss decreasing, accuracy climbing) was previously confirmed on
  the prior 4-node version during bug-fixing, but that run's artifacts
  have since been deleted and the node set has changed since — treat this
  as shape-verified, not accuracy-verified, until the next real run.
- 🟡 **Task 3.2** `track_a_heavy/train.py` — per-fold LOSO training loop,
  checkpointing. Code correctness (folds-filtering bug fix) was confirmed
  by a smoke run whose output has since been deleted. **No training has
  been run since the node-set change** — next run is the first real
  signal on the current architecture.
- 🟡 **Task 3.3** `track_a_heavy/evaluate.py` — writes a results table +
  confusion matrix figure from `train.py`'s saved JSON. Simple and
  low-risk but has never run against a real result (none exists).
- 🟡 **Task 3.4** `track_a_heavy/ablation.py` — full / no-GNN /
  no-Transformer / no-MGAF / no-reconstruction-loss variants. Confirmed
  running without error pre-node-change; not re-verified since (should be
  fine, same code path as Task 3.2, but flagging honestly).
- ⬜ **Task 3.5** Hyperparameter sweep (`sweep.py`) — **not written yet**.
  This is the task that most needs the A100 instance: full LOSO × 3-5
  seeds × a GNN-hidden-dim/Transformer-config sweep is squarely the kind
  of job tasks.md says needs A100/H100-class compute.
- ⬜ **Task 3.6** (optional) Deep-vision node — not started.

## MePhy reproduction (tasks.md 1.2/3.1 sanity check — required per your instruction, not optional)

- ✅ Data pipeline (`mephy_repro/{paths,loaders,windowing,features,dataset}.py`)
  — fully built and validated against real MePhy data, including two real
  bugs found and fixed (an EDA/EMG timestamp-truncation logging bug in the
  source files, and the window-size correction from tasks.md's stated 10s
  to the paper's actual 20s). Unaffected by the node-set change (uses
  MePhy's own native 4 nodes by design — see above).
- 🟡 `mephy_repro/train.py` — runs both the paper's own random-80/20-split
  protocol and a rigorous LOSO protocol. Code correctness was confirmed
  via diagnostic partial runs while chasing the standardization bug; those
  runs' outputs have since been deleted per instruction. **No completed,
  retained run exists.** This remains the single most useful first job to
  run on the A100 — cheap (5,020 windows, small model), and it tells you
  immediately whether the reimplementation is sound before trusting Track
  A's UL-DD numbers.

## Phase 4 — Track B (lightweight edge model)

- ✅ **Task 4.1** `track_b_light/model.py` — MLP, GRU (torch), LightGBM
  (param builder). All three now correctly use `CORE_FEATURES` only (37
  dims) — see architecture-change note above.
- 🟡 **Task 4.2** `track_b_light/train.py` — MLP, GRU, and LightGBM paths
  all confirmed to run without error (including one full 16-fold LightGBM
  LOSO run) prior to the folds-filtering-bug fix and the node-set change;
  outputs deleted, not re-run since. Next run is the first real signal on
  the current, correct code.
- 🟡 **Task 4.3** `track_b_light/evaluate.py` — prints a side-by-side
  comparison table; has nothing to compare right now (no results exist).
- ✅ **Task 4.4** `track_b_light/export_onnx.py` — verified correct
  end-to-end (both MLP and GRU export to ONNX and matched PyTorch output
  to ~1e-6) prior to the node-set change; the exported `.onnx` files have
  since been deleted, but the export/verify code path itself needs no
  further changes now that `INPUT_DIM` is 37 instead of 40.
- ⬜ **Task 4.5** `export_tensorrt.py` — not started, needs to run on the
  Jetson itself.

## Phase 5 — ESP32-S3 firmware

⬜ Not started. Needs a decision on the actual PPG/GSR sensor part number
before `main.cpp` can be written meaningfully (tasks.md 5.1 explicitly
flags this — an Empatica E4 is not a raw-sensor breakout you can wire to
an ESP32, a different physical part must be sourced).

## Phase 6 — Jetson deployment pipeline

⬜ Not started. Depends on Track B's final model choice (Task 4.5) and
the ESP32 firmware's UART packet format (Task 5.4) existing first.

## Phase 7 — Comparison & benchmarking

⬜ Not started (`benchmarks/` is an empty scaffold directory). Depends on
both tracks having final trained models and Track B being deployed to
real Jetson hardware for honest latency numbers.

## Phase 8 — Testing & validation

- ✅ **Task 8.1** `tests/test_features.py` (11 tests) +
  `tests/test_metrics.py` (2 regression tests for the class-count bug) —
  13 tests, all passing, hand-computed examples for every feature
  function. Unaffected by the node-set change (features themselves didn't
  change, only which groups feed the model by default).
- ⬜ **Task 8.2** Python vs. ESP32-S3 C++ feature parity — blocked on
  firmware existing.
- ⬜ **Task 8.3** End-to-end integration test — blocked on the Jetson
  runtime existing.

## What needs to be corrected before a real A100 training run

Nothing outstanding is known-broken — everything below is either already
fixed or is a real "not started yet" item, not a bug:

1. **Fixed, this round**: Track A's node set and Track B's `INPUT_DIM`
   both included UL-DD's Telemetry, which no deployed sensor produces —
   would have made Track B's "deployment candidate" untrainable-on-real-
   hardware by construction, and Track A's ceiling incomparable to it.
2. **Fixed, prior round**: `--folds` subsets silently starving the
   training set in all three `train.py`/`ablation.py` entry points.
3. **Fixed, prior round**: ONNX export needed `onnxscript` (now in
   `requirements.txt`) and crashed on Windows console encoding with
   torch's default exporter (now uses `dynamo=False`).
4. **Fixed, prior round**: unstandardized features (up to 6 orders of
   magnitude apart) silently capped both neural architectures near chance
   accuracy — both dataset classes now z-score every feature by default.
5. **Not yet written**: Task 3.5's sweep script. Recommend writing this
   *before* the first A100 run so the LOSO×seeds×hyperparameter sweep can
   be launched directly, rather than doing a manual run first and writing
   the sweep after.
6. **Not yet decided**: the actual PPG/GSR sensor part number for the
   ESP32-S3 (blocks Phase 5, not Phase 3/4 — doesn't block training).

## Suggested order once the A100 instance is available

1. `mephy_repro/train.py --split both` — cheapest, fastest confirmation
   the reimplementation is sound, now that its artifacts are gone.
2. `track_a_heavy/train.py` (full 16-fold LOSO) + `ablation.py`, on the
   corrected 3-node architecture — this will be the first real numbers
   for the current codebase.
3. `track_b_light/train.py --model mlp` / `--model gru` / `--model
   lightgbm`, on the corrected 37-dim input — likewise, first real
   numbers for the current codebase.
4. Only after 1-3 look reasonable: write and run Task 3.5's sweep.
