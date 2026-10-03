# Implementation notes / deviations from tasks.md

This tracks where the actual build differs from tasks.md's assumptions, and
why, so it's not necessary to re-derive these decisions later.

## Compute: workstation is not an A100/H100

tasks.md sizes Track A training for an A100/H100 80GB and says not to
shrink anything to save VRAM. The actual training machine available right
now is a laptop RTX 4050 with **6GB VRAM**. Approach:

- All pipeline code (data loading, features, model architectures, training
  loops) is written to the full spec sizing in tasks.md -- hidden dims,
  context lengths, batch sizes are not pre-shrunk.
- Local runs on the RTX 4050 are used to validate correctness (does it
  train, does loss go down, do shapes work) at whatever batch size/precision
  fits in 6GB (bf16/fp16 autocast, gradient accumulation if needed).
- The full-scale runs tasks.md actually wants for reported numbers --
  Task 3.5's full LOSO x 3-5 seeds hyperparameter sweep across GNN hidden
  dim 256-512, and any run that doesn't fit in 6GB -- need to move to
  Colab/cloud A100/H100 when we get there. This will be called out
  explicitly at that point rather than silently reported as final numbers
  from local underpowered runs.

## Data quality issues found in UL-DD (Task 1.1)

Found by `scripts/verify_uldd.py` cross-checking every file against
`Info.xlsx`'s own availability manifest, and by the loaders in
`common/loaders.py` failing loudly on real data rather than silently
producing wrong numbers:

- `J_IBI_A.csv` (subject J, Awake session) is a genuine 0-byte file, despite
  being marked available in the manifest. Handled as "no beats detected".
- `F_O2M_A.csv`, `P_O2M_D.csv`, `R_O2M_A.csv` each have one row with the
  literal sentinel `--,--,0` for a dropped SpO2/pulse reading. Parsed as
  NaN, handled with NaN-aware reductions in `spo2_features`.
- IBI (heartbeat interval) coverage is sparse *within* a session even where
  the file isn't empty -- the wristband loses PPG lock for long stretches
  (e.g. subject A/Awake only has beat detections from t=131s to t=1231s out
  of a 2400s session). HRV features (RMSSD/SDNN) are NaN for windows
  outside whatever coverage a given session happens to have; this needs a
  documented imputation strategy before training (currently: per-feature
  median imputation in `common/dataset.py`, applied at load time).
- Telemetry.csv's own timestamp column doesn't necessarily span the same
  duration as the biometric/video modalities for the same subject/session
  (e.g. subject A/Drowsy: Telemetry spans ~2640s while the labeled/biometric
  session is 2400s). `common/windowing.py` never uses Telemetry to bound the
  session's window grid because of this -- only the always-full-coverage
  modalities (ACC/BVP/EDA/HR/LGP/RGP/O2M/TEMP/FL/PL/FAU) are used for that.

## MePhy reproduction (tasks.md 1.2/3.1 sanity check)

- The dataset's own ReadMe.pdf gives exact column semantics that had to be
  read before writing any loader (guessing would have been wrong): ECG is
  6 columns at 1Hz with a precomputed RR interval in both a 1/1024 format
  and milliseconds (cross-checked: col3*1024 == col6, confirming the
  reading); EDA/EMG are 2 columns at 1000Hz; EyeBlinking is 2 columns at
  30Hz with a real eyelid-closed boolean (confirmed non-trivial 1s occur).
  Modality coverage varies sharply by user ID: all 60 users have ECG, only
  user0-29 have EDA/EMG, only user0-18 have EyeBlinking -- the reproduction
  uses the 19 users (user0-18) with all four modalities, since FatigueNet's
  GNN needs all four as separate nodes.
- Found a real device-logging bug in MePhy's own EDA/EMG files: the
  per-row timestamp string truncates to just `HH:MM` (dropping seconds and
  milliseconds) at every 1000th row -- i.e. exactly at each one-second
  boundary of the 1000Hz recording. Fixed by deriving time from sample
  index / rate instead of parsing that column at all for these two
  modalities (they're confirmed fixed-rate, so this is exact, not an
  approximation).
- **tasks.md paraphrases the FatigueNet paper's window size incorrectly**:
  tasks.md 1.3 says "10s window, 5s step, matching FatigueNet's approach,"
  but the paper's own Methods section states a 20s window / 5s step.
  `mephy_repro/windowing.py` uses 20s to match the paper, since fidelity to
  the paper is the entire point of this reproduction; UL-DD's windowing
  stays at 10s per tasks.md's separate, UL-DD-specific rationale (matching
  the 4-minute KSS label granularity), which doesn't depend on this fix.
- **The paper's own reported 90.2% test accuracy comes from a random
  80/20 window-level split**, not a subject-independent one -- it states
  "the dataset was randomly partitioned into training and testing sets."
  tasks.md Section 0 explicitly forbids random splits in this project's
  own pipeline because they leak within-subject correlation across
  train/test and inflate accuracy. `mephy_repro/train.py` therefore runs
  *both* protocols: `--split random` replicates the paper's own (leaky)
  methodology as the actual "does the reimplementation hit ~90.2%" check,
  and `--split loso` runs the rigorous subject-independent version, which
  is expected to score lower than 90.2% for the same reason tasks.md warns
  about generally -- that gap is itself a real, reportable finding about
  the original paper's methodology, not evidence of a bug in this
  reimplementation.
- The GNN's dynamic adjacency (paper Eq. 4) is specified ambiguously in
  the paper ("correlation between modality feature vectors") and tasks.md
  only adds "within a batch." Implemented as: pool each node's embedding
  to one scalar per sample, then Pearson-correlate across the batch
  dimension between every node pair, giving one adjacency matrix shared
  by the batch (falls back to a uniform adjacency for batch size 1). Every
  other non-obvious equation-to-code mapping is documented inline in
  `common/fatiguenet_model.py`'s module docstring, including the one
  deliberate non-literal implementation (Eq. 9's transformer/adjacency
  hybrid is replaced with a standard Transformer encoder, matching
  tasks.md 3.1's own "4 layers, 4-8 heads" simplification).

## Critical bug: unstandardized features broke both neural models

While running the MePhy reproduction, the random-split test accuracy came
back at 34.6% against a paper target of 90.2%. Before assuming the
architecture or the simplified feature set was inadequate, this was
isolated by training a plain LightGBM classifier on the exact same
features: it hit **97.6%** accuracy on the same random split. That proved
the features carried plenty of signal and pointed straight at the neural
training path (`common/fatiguenet_model.py`) instead.

Root cause: raw feature scales span up to 6 orders of magnitude within a
single feature table (MePhy's `emg_waveform_length` ranges ~27,000-341,000
while `perclos` ranges 0-0.145; UL-DD has the same issue on a smaller
scale, ~3-4 orders of magnitude, e.g. `hrv_rmssd` ~0.05 vs
`blink_rate_per_min` ~124). These raw values were fed directly into
`nn.Linear` embedding layers with no normalization. Direct diagnosis
(training a `no_gnn` ablation variant) showed this wasn't just slow
learning -- it NaN'd out immediately without the GNN's forced
L2-normalization step; the "full" model didn't NaN only because that
L2-norm incidentally absorbed the scale explosion, but at the cost of
destroying signal (loss plateaued near chance-level accuracy).

Fix: both `common/dataset.py` (`ULDDWindowDataset`/`ULDDSequenceDataset`)
and `mephy_repro/dataset.py` (`MePhySequenceDataset`) now z-score every
feature column (mean/std computed from the dataset's own table) before
handing it to a model, default-on via a `standardize=True` parameter.
LightGBM (`track_b_light`'s Option 2) is unaffected either way since tree
splits are scale-invariant. After the fix, Track A's train loss actually
decreases session-over-session instead of plateauing (verified on real
UL-DD folds: train accuracy reaches 83-99%, up from ~45-53% stuck at
chance).

The takeaway for anything added later that touches these feature tables:
**never feed a raw feature column into a `nn.Linear` without checking its
scale first** -- this class of bug produces working code that trains
without errors (except in the more fragile no-GNN configuration) but
silently caps performance near chance, which looks like "the model just
isn't very good at this problem" rather than a numerical bug, and is easy
to misdiagnose as an architecture or feature-quality problem instead.

## Track A's node set is our sensors, not the paper's or UL-DD's full list

The instinct after "reproduce the paper's architecture on a new dataset" is
to keep the paper's per-modality node structure and just swap in whichever
signals the new dataset happens to ship. That's what `mephy_repro/` does
(and should: its entire purpose is faithfully reproducing FatigueNet on its
own native data). Track A must not do the same thing, on explicit
instruction: it is being built as the accuracy ceiling *for the product we
are actually building*, whose sensor set is fixed by Phases 5-6 (ESP32-S3
grip pressure + PPG/GSR breakout, Jetson camera) -- not by whatever UL-DD
happened to record.

Concretely, `common/dataset.py`'s default node set was three, not the four
it was previously (vision/bio/grip/telemetry): UL-DD's driving-simulator
Telemetry (heading/speed/rpm/gear) was dropped from the default because no
vehicle-telemetry sensor exists anywhere in the deployed hardware spec. A
Track A trained to rely on it would measure a ceiling Track B could never
approach even in principle, which breaks the entire point of comparing the
two tracks honestly. `track_b_light/model.py`'s `INPUT_DIM` had the same
problem more acutely -- it was already flattening all four groups
including telemetry into Track B's actual input vector, meaning the
*deployment candidate itself* was being trained on a feature it can never
receive from real hardware. Both are fixed now: `common/dataset.py`
exposes `CORE_FEATURES`/`CORE_FEATURE_GROUPS` (vision+bio+grip, 37 dims)
as the default for everything, with telemetry reachable only via an
explicit `include_telemetry=True` on the dataset/Track-A-model
constructors for a clearly-separate, Track-A-only "what if we also had
simulator telemetry" ceiling comparison -- never wired into Track B.

Verified with forward-pass-only checks (`torch.no_grad()`, `model.eval()`,
no optimizer, no `.backward()` -- confirmed no weight updates occur) rather
than a real training run, per instruction not to train anything yet: all
three node configurations (Track A default 3-node, Track A
`include_telemetry=True` 4-node, Track B's 37-dim flat MLP/GRU) produce
correctly-shaped output with no crash.

## `--folds` subsets were silently starving the training set

Found by smoke-testing (not full training) `track_a_heavy/train.py`,
`ablation.py`, and `track_b_light/train.py` before trusting them: all
three constructed their `Dataset` object pre-filtered to whatever subset
`--folds` requested (e.g. `--folds 0 1`), then computed each fold's
"train" set as "every row in that filtered dataset except the held-out
subject's." When only 2 of 16 folds are requested, that leaves exactly one
other subject's data to train on instead of the intended 15 -- any partial
-fold run (the normal way to do a quick check) was silently training on a
starved dataset while looking like it worked fine. Fixed by always loading
every foldable subject into the dataset regardless of `--folds`, and using
`--folds` only to choose which fold(s) get evaluated as the held-out test
set in the outer loop.

## ONNX export needs `onnxscript`, and torch's new exporter breaks on Windows

`track_b_light/export_onnx.py`, smoke-tested for the first time: `pip
install torch` alone is not sufficient for `torch.onnx.export` on torch
2.13 -- its new default dynamo-based exporter imports `onnxscript`, which
isn't a torch dependency and must be installed separately (added to
`requirements.txt`). Once that's fixed, the dynamo exporter itself crashes
on Windows with `UnicodeEncodeError` trying to print a unicode checkmark
through the default cp1252 console codepage. Both avoided by passing
`dynamo=False` to use the older, more mature TorchScript-based exporter,
which is also the more appropriate choice for models this architecturally
simple (plain MLP / single-layer GRU) regardless of the Windows issue.
Verified numerically: exported MLP and GRU ONNX outputs match PyTorch to
~1e-6 max absolute difference.

## Python version

Python 3.14 (only version available on this machine) turned out to have
working wheels for every package tasks.md's environment section lists for
the training environment (torch 2.13+cu126, torch-geometric, lightgbm,
optuna, onnx, onnxruntime, scikit-learn) as of Aug 2026 -- no compatibility
issues hit so far. `mediapipe` has not been installed/tested yet (not
needed until the Jetson runtime or Track A's optional deep-vision node);
flag it for a compatibility check against Python 3.14 wheels when that work
starts, per tasks.md 3.2's own note about verifying MediaPipe availability
for the installed Python version.
