# WheelsEye — Implementation Tasks

Two-track multimodal driver fatigue detection system. Track A is a heavyweight
GNN + Transformer + MGAF model (accuracy ceiling, trained/evaluated off-device
only). Track B is a lightweight feature-based model deployed on a Jetson Orin
Nano 8GB with an ESP32-S3 sensor hub. Both tracks share the same data,
preprocessing, features, splits, and evaluation harness so the final
comparison is apples-to-apples.

---

## 0. Ground rules for both tracks

- **Same data, same subject-independent splits, same features.** Track A and
  Track B must consume the exact same preprocessed feature files. Only the
  model architecture differs between tracks.
- **No random train/test split.** Use grouped/leave-one-subject-out
  cross-validation (group key = subject ID). Random splits leak
  within-subject correlation across train/test and inflate accuracy,
  especially for the heavier model.
- **Track A never touches the Jetson.** It is trained and evaluated entirely
  on a workstation/Colab GPU. Its only job is to produce an accuracy ceiling
  and an ablation study.
- **Track B is the only track exported to ONNX/TensorRT and benchmarked on
  real hardware.**
- **Compute is not a training constraint.** Training happens on an A100/H100
  80GB, for both tracks. Do not shrink model dimensions, batch sizes, or
  hyperparameter search to save training time or VRAM anywhere in this
  document unless a note explicitly says otherwise. The Jetson Orin Nano 8GB
  constraint applies only to Track B's **inference-time** deployed model —
  it has no bearing on how Track B is trained.
- Every experiment run must log: accuracy, precision, recall, F1 (macro),
  confusion matrix, and (for Track B) latency + model size + peak memory.

---

## 1. Datasets

### 1.1 Primary dataset — UL-DD (use this for both tracks)

- Name: University of Louisiana Drowsiness Dataset
- Source: Zenodo, DOI `10.5281/zenodo.17978727` (also referenced in-paper as
  `https://zenodo.org/records/17978727`)
- Why primary: its modalities match the hardware you actually have —
  IR/RGB/depth video, Empatica E4-class wristband (HR, EDA, TEMP, BVP, IBI,
  ACC), Checkme O2 Max (SpO2, pulse rate, motion), grip pressure (left/right),
  driving telemetry, plus **pre-extracted** facial landmarks (68-pt, Dlib),
  facial action units (30), and pose landmarks (33-pt, MediaPipe Pose).
- 19 subjects, two 40-minute sessions each (Awake / Drowsy), KSS label every
  4 minutes (9-level scale). 3 subjects (C, F, L) have no drowsy session —
  exclude them from subject-independent folds or handle as awake-only.
- Download everything: `Video_Data/`, `CSV_Files/`, `Extracted_Features/`,
  `Labels.csv`. You do **not** need to re-run Dlib/MediaPipe yourself — FL,
  FAU, and PL are already provided per subject/session as CSVs. This removes
  a large chunk of preprocessing work.
- Data is already time-synchronized across modalities (per-modality CSVs
  keyed by frame number or timestamp) — no manual sync step required.

### 1.2 Secondary/optional dataset — MePhy

- Referenced by the FatigueNet paper (DOI `10.1145/3610977.3637485` for the
  originating HRI paper). Modalities are ECG, EDA, EMG, eye-blink — **do not
  match your hardware** (no grip pressure, no camera-derived facial video,
  different biometric sensor set than Empatica E4).
- Worth doing now that training is cheap: reproduce FatigueNet on MePhy
  first, as a sanity check that your reimplementation (Task 3.1) hits
  roughly the published 90.2% test accuracy before adapting the same
  architecture to UL-DD for Track A. This validates the implementation
  independent of any modality/dataset differences, and an A100/H100 makes
  this a same-day side experiment rather than a real cost. Still optional,
  but recommended if time allows given the low cost.
- Do not train the final Track A/B models used for the report's comparison
  on MePhy — modality mismatch invalidates comparison to Track B, which
  depends on grip pressure and video that MePhy doesn't have.

### 1.3 Labeling scheme (apply identically to both tracks)

Bin KSS (1-9) into 3 classes, matching UL-DD's own validation methodology:
- `Low` (Alert): KSS < 4
- `Medium`: 4 <= KSS <= 6
- `High` (Drowsy): KSS > 6

Store this mapping in one shared module (`common/labels.py`) so both tracks
use identical class boundaries.

---

## 2. Repository structure

```
wheelseye/
├── data/
│   ├── raw/                      # untouched UL-DD download
│   └── processed/                # windowed feature files (parquet/npz)
├── common/
│   ├── labels.py                 # KSS -> 3-class binning
│   ├── splits.py                 # subject-independent CV fold generation
│   ├── features_vision.py        # EAR/MAR/PERCLOS/head-pose from FL/PL csv
│   ├── features_bio.py           # HRV, EDA tonic/phasic, SpO2 features
│   ├── features_grip.py          # grip mean/var/asymmetry
│   ├── windowing.py              # sliding window aggregation (shared)
│   ├── metrics.py                # shared eval: acc/prec/rec/F1/confmat
│   └── dataset.py                # PyTorch Dataset wrapping processed features
├── track_a_heavy/
│   ├── model.py                  # GNN + Transformer + MGAF + classifier head
│   ├── train.py
│   ├── evaluate.py
│   └── ablation.py               # remove GNN / remove Transformer / remove MGAF
├── track_b_light/
│   ├── model.py                  # MLP / small-GRU fusion head
│   ├── train.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   └── export_tensorrt.py        # runs ON the Jetson
├── firmware/
│   └── esp32s3/                  # PlatformIO/ESP-IDF project
│       ├── src/main.cpp
│       ├── src/filters.cpp       # Butterworth/IIR filtering
│       ├── src/features.cpp      # on-device HRV/EDA/grip feature extraction
│       └── src/uart_protocol.cpp
├── jetson_runtime/
│   ├── camera_pipeline.py        # capture + MediaPipe Face Mesh
│   ├── serial_reader.py          # reads ESP32-S3 feature packets over UART
│   ├── infer_loop.py             # main loop: fuse features -> TensorRT -> alert
│   └── buzzer_gpio.py
├── benchmarks/
│   ├── profile_latency.py
│   ├── profile_memory.py
│   └── compare_report.py         # generates Track A vs Track B comparison table/plot
└── tasks.md                      # this file
```

---

## 3. Environment & dependencies

### 3.1 Training environment — A100/H100 80GB (Track A + Track B training)

```
python>=3.10
torch>=2.1               # use bf16/fp16 mixed precision (torch.autocast) to
                          # exploit tensor cores; no need to shrink batch size
torch-geometric        # GNN layers for Track A
scikit-learn
lightgbm                 # optional alternative fusion head for Track B
optuna or wandb sweeps   # hyperparameter search — now cheap to run broadly
pandas, numpy, scipy
mediapipe                 # only needed if recomputing landmarks; UL-DD provides them
opencv-python
timm                       # pretrained CNN/ViT backbones, for optional Track A deep-vision node (Task 6.5)
matplotlib, seaborn
onnx, onnxruntime
tensorboard or wandb (experiment tracking)
```

With 80GB of VRAM, run full-batch or large-batch training, mixed precision,
and multi-seed runs freely. There is no reason to subsample subjects,
truncate window context length, or reduce hidden dimensions to fit a
smaller GPU anywhere in this pipeline.

### 3.2 Jetson Orin Nano 8GB runtime environment

```
JetPack (includes CUDA, cuDNN, TensorRT — install via NVIDIA SDK Manager)
python3
tensorrt (comes with JetPack)
pycuda
mediapipe (Jetson-compatible build, or use OpenCV + a lightweight landmark ONNX model if MediaPipe wheel unavailable for the JetPack Python version — verify first)
opencv-python
pyserial                  # UART communication with ESP32-S3
Jetson.GPIO                # buzzer control
```

### 3.3 ESP32-S3 firmware environment

```
PlatformIO (recommended) or ESP-IDF directly
Arduino core for ESP32-S3, or native ESP-IDF
CMSIS-DSP or arduinoFFT for on-device filtering (if not using ESP-DL)
```

---

## 4. Phase 1 — Data acquisition & preprocessing (shared)

**Task 1.1** — Download UL-DD, verify file counts against Table 5 of the
dataset paper (per-subject availability of IR/Depth/Pose video, ACC, BVP,
EDA, HR, IBI, grip left/right, SpO2, pulse rate, motion, TEMP, telemetry,
PL, FL, FAU). Flag and log any missing files per subject rather than
silently skipping.

**Task 1.2** — Write `common/labels.py`:
- Load `Labels.csv`
- Map each 4-minute KSS score to the 3-class scheme (Section 1.3)
- Expose `get_label(subject, session, minute) -> {0,1,2}`

**Task 1.3** — Write `common/windowing.py`:
- Fixed sliding window (recommend **10s window, 5s step**, matching
  FatigueNet's approach and reasonable for a 4-minute KSS granularity)
- For each window, pull the corresponding rows from every modality CSV
  (aligned by frame number / timestamp already provided), and assign the
  KSS label of the containing 4-minute interval
- Output one row per window per subject per session with all raw signal
  slices needed for feature extraction downstream

**Task 1.4** — Write `common/splits.py`:
- Implement `GroupKFold` / leave-one-subject-out using subject ID as the
  group key (use `sklearn.model_selection.GroupKFold` or manual LOSO loop)
- Persist fold assignments to disk so Track A and Track B use **identical**
  folds (critical for a fair comparison)

---

## 5. Phase 2 — Feature engineering (shared, computed once, used by both tracks)

**Task 2.1** — `common/features_vision.py` (from provided FL/PL CSVs, no
video decoding needed unless recomputing from scratch):
- Eye Aspect Ratio (EAR) per eye, per frame, from the 68-pt facial landmarks
  → blink detection, blink rate, PERCLOS (% time eyes below EAR threshold
  over the window)
- Mouth Aspect Ratio (MAR) → yawn detection/count
- Head pose (pitch/yaw/roll) from PL 33-pt landmarks or FL geometry →
  nodding/tilt frequency
- Aggregate per window: mean, std, rate-based counts (blinks/min,
  yawns/min), max PERCLOS

**Task 2.2** — `common/features_bio.py`:
- HR: mean, std over window
- HRV proxy from IBI: RMSSD, SDNN (standard formulas, same as FatigueNet
  Eq. reference — implement directly, do not need CWT/fractal features,
  those are FatigueNet-specific overkill for the lightweight track and
  optional for the heavy track)
- EDA: tonic/phasic decomposition (simple moving-average tonic estimate +
  phasic = raw − tonic is sufficient; do not need a full cvxEDA solver)
- SpO2/pulse rate: mean, std, min (min matters for detecting apnea-like dips)
- TEMP: mean, slope over window (captures the warming trend at higher
  drowsiness reported in the UL-DD technical validation)

**Task 2.3** — `common/features_grip.py`:
- Left/right grip: mean, std, and **asymmetry** (|left_mean − right_mean|)
  over the window — asymmetry is called out in the UL-DD paper as a
  drowsiness-relevant behavioral cue

**Task 2.4** — Telemetry features (optional, include if time permits):
- Speed variance, steering/heading change rate over window as an auxiliary
  behavioral signal

**Task 2.5** — Assemble the final feature table: one row per
`(subject, session, window)` with all vision + bio + grip (+ telemetry)
features and the 3-class label. Save as parquet. **This exact table is the
single input both Track A and Track B will consume** — Track A will
additionally need the raw/near-raw per-modality sequences (Task 6.1) but
should still use this table for labels/splits.

---

## 6. Phase 3 — Track A: heavyweight GNN + Transformer + MGAF

Purpose: establish an accuracy ceiling and an ablation study, adapted from
the FatigueNet architecture but re-targeted at UL-DD's modalities
(video-derived geometry, wristband biometrics, grip) instead of MePhy's
(ECG/EDA/EMG/blink).

**Task 3.1** — `track_a_heavy/model.py`:
- **Input**: per-window, per-modality feature vectors (vision, bio, grip —
  treat each as one "node" the way FatigueNet treats ECG/EDA/EMG/blink as
  nodes)
- **GNN block**: build the adjacency matrix dynamically from pairwise
  correlation between modality feature vectors within a batch (same
  approach as FatigueNet Eq. 4); match or exceed FatigueNet's own sizing —
  4 graph conv layers, hidden dim 256, since VRAM is no longer a limiting
  factor. Feel free to sweep 256-512 hidden dim as part of the
  hyperparameter search (Task 3.5) rather than picking one value up front.
- **Transformer block**: use a longer context window than the earlier
  resource-constrained plan — e.g. last 12-24 windows (2-4 minutes of
  context) instead of 6-12, since a longer temporal receptive field is
  plausibly useful for catching gradual fatigue onset and training cost is
  no longer a concern. 4 layers, 4-8 attention heads, 64-128 dim per head —
  sweep this too.
- **MGAF fusion**: MetaNet MLP producing per-modality relevance weights
  (Eq. 10-11), then attention-refined fusion (Eq. 12), same structure as
  FatigueNet Fig. 7 — use the full-size version, no need to compress it.
- **Classifier head**: replace FatigueNet's MSVM with a simple softmax
  linear layer — keep it a standard neural classifier so the whole model
  trains end-to-end with backprop; do not introduce a separate SVM stage.
- **Loss**: cross-entropy + L2 weight regularization + the
  reconstruction-consistency term from FatigueNet Eq. 15 (include it this
  time — with ample compute there's no reason to skip a component of the
  reference architecture; ablate it in Task 3.4 instead of omitting it
  up front).

**Task 3.2** — `track_a_heavy/train.py`:
- Train per-fold (LOSO or grouped k-fold from `common/splits.py`)
- Log per-fold and averaged accuracy/F1/confusion matrix
- Save best checkpoint per fold

**Task 3.3** — `track_a_heavy/evaluate.py`:
- Aggregate metrics across folds, produce the confusion matrix figure and
  a results table matching the format of FatigueNet's Table 2/3

**Task 3.4** — `track_a_heavy/ablation.py`:
- Variants: full model, no-GNN (concatenation instead), no-Transformer
  (skip temporal attention, use last-window features only), no-MGAF
  (simple average fusion instead), no-reconstruction-loss
- Report accuracy delta per component, same structure as FatigueNet Table 5

**Task 3.5** — Hyperparameter search (`track_a_heavy/sweep.py`), now
affordable to run properly:
- Use Optuna or W&B sweeps over GNN hidden dim (256/384/512), Transformer
  layers/heads/context length, dropout, learning rate, and the loss
  regularization coefficients (λ1, λ2)
- Run full **leave-one-subject-out** cross-validation (all ~16-19 subjects,
  not a reduced subset) for the final reported numbers — this was flagged
  as important in the earlier design discussion and is now cheap enough on
  an A100/H100 to run in full rather than approximating with k-fold
- Train each final configuration with **3-5 random seeds** and report
  mean ± std accuracy/F1, not a single run — this materially strengthens
  the credibility of the Track A vs Track B comparison

**Task 3.6 (optional, compute now allows it)** — Deep vision node instead
of/alongside geometric vision features:
- Fine-tune a small pretrained CNN or ViT backbone (via `timm`, e.g.
  ResNet-18 or ViT-Tiny) directly on face crops from UL-DD's IR video as an
  additional "deep visual" node feeding the GNN, alongside or instead of
  the EAR/MAR/PERCLOS geometric features
- This is *only* valid for Track A (never deployed to the Jetson) — it
  exists purely to test whether a learned visual representation raises the
  accuracy ceiling above what hand-engineered geometric features achieve,
  which is a legitimate and interesting ablation now that training compute
  isn't the bottleneck
- Report as a separate row in the ablation table: geometric-only vs.
  deep-vision-only vs. both concatenated

---

## 7. Phase 4 — Track B: lightweight edge model

Purpose: the actual deployment candidate. Must run inference on Jetson
Orin Nano 8GB within the latency budget.

**Task 4.1** — `track_b_light/model.py`:
- **Input**: the same per-window feature vector as Track A (vision + bio +
  grip), no raw signal processing at inference time — all feature
  extraction already happened in Phase 2 equivalent logic, which for
  deployment gets reimplemented on the ESP32-S3 (bio/grip) and Jetson
  (vision) directly, see Phases 8-9
- **Option 1 (recommended default): small MLP** — 2-3 hidden layers
  (e.g. 64 → 32 → 16), <50K params, single-window classification
- **Option 2: LightGBM/XGBoost** on the same feature vector — no GPU
  needed at all for inference, trivial to deploy, strong baseline for
  tabular features; benchmark this as a genuine competing lightweight
  option before assuming a neural net wins
- **Option 3 (only if Option 1/2 underperform): small GRU** over the last
  6-12 windows (single layer, hidden size 32-64, <100K params) to capture
  short-term temporal trend without Transformer-level cost
- Implement all three, compare on validation folds, pick the best
  accuracy/latency tradeoff as "the" Track B model for final deployment
  and comparison — this itself is a useful mini-ablation to report
- **The deployed model must stay small (Jetson-viable), but the search for
  it doesn't have to be.** Run a proper Optuna/W&B sweep over hidden sizes,
  depth, dropout, and (for LightGBM) tree count/depth on the A100/H100 —
  training a hundred small MLP candidates is nearly free on that hardware.
  Use full LOSO cross-validation for the final reported Track B numbers,
  matching Track A's evaluation rigor (Task 3.5), so the comparison between
  tracks isn't undermined by Track B having a weaker search budget.

**Task 4.2** — `track_b_light/train.py`:
- Same fold structure as Track A (identical splits — reuse
  `common/splits.py` output directly)

**Task 4.3** — `track_b_light/evaluate.py`:
- Same metrics format as Track A for direct comparison

**Task 4.4** — `track_b_light/export_onnx.py`:
- `torch.onnx.export(...)` for the MLP/GRU variant (skip for LightGBM —
  export via `onnxmltools` if you want a unified ONNX path, otherwise
  LightGBM has its own lightweight native inference path)
- Verify ONNX output matches PyTorch output numerically before proceeding

**Task 4.5** — `track_b_light/export_tensorrt.py` (run **on the Jetson**):
- Convert ONNX → TensorRT engine
- Build with INT8 calibration using a representative subset of validation
  windows (fall back to FP16 if INT8 calibration data or accuracy loss is
  a problem — document the tradeoff either way)
- Save the compiled `.engine` file

---

## 8. Phase 5 — ESP32-S3 firmware

Purpose: read biometric + grip sensors, extract compact features on-device,
stream to Jetson over UART. **Do not stream raw high-rate signals** — this
defeats the purpose of offloading work from the Jetson.

**Task 5.1** — Sensor interfacing (`firmware/esp32s3/src/main.cpp`):
- ADC reads for grip pressure FSR sensors (left/right)
- I2C/SPI driver for chosen biometric sensor (PPG + GSR breakout board —
  pick a specific part number based on what's actually sourced; do not
  assume an Empatica E4 is available, that device is not a raw-sensor
  breakout you can wire to an ESP32)
- Sampling rates: match what UL-DD used for meaningful feature parity —
  EDA/BVP/TEMP ~4Hz, ACC ~32Hz, grip ~3Hz (per UL-DD Data Records section)

**Task 5.2** — On-device filtering (`filters.cpp`):
- Lightweight IIR/Butterworth low-pass filtering matching the cutoffs used
  in the UL-DD technical validation (EDA 1.5Hz, TEMP/ACC 1.0Hz, grip 1.0Hz)
  — reuse those published cutoffs rather than re-deriving them

**Task 5.3** — On-device feature extraction (`features.cpp`):
- Compute the **same** feature set as `common/features_bio.py` and
  `common/features_grip.py` (HR mean/std, RMSSD/SDNN proxy, EDA
  tonic/phasic, grip mean/var/asymmetry) over a rolling 10s window —
  feature definitions must match training exactly, or the deployed model
  sees a distribution shift from what it was trained on
- Emit one feature packet every 5s (matching the sliding window step)

**Task 5.4** — UART protocol (`uart_protocol.cpp`):
- Define a fixed-size binary packet (e.g. struct of floats + a
  packet-start marker + checksum) sent over UART at a defined baud rate
  (115200 recommended)
- Document the exact packet layout in a shared header so
  `jetson_runtime/serial_reader.py` can decode it without ambiguity

---

## 9. Phase 6 — Jetson deployment pipeline

**Task 6.1** — `jetson_runtime/camera_pipeline.py`:
- Capture from IR/RGB camera (OpenCV `VideoCapture`)
- Run MediaPipe Face Mesh (or lightweight landmark ONNX model if MediaPipe
  is unavailable for the installed JetPack/Python combination — verify
  compatibility first and have this fallback ready)
- Compute the same EAR/MAR/head-pose features as
  `common/features_vision.py`, aggregated over the same 10s/5s window

**Task 6.2** — `jetson_runtime/serial_reader.py`:
- Read and decode UART packets from the ESP32-S3 per the protocol in
  Task 5.4
- Handle dropped/malformed packets gracefully (do not crash the alert loop
  on a bad packet — log and reuse last-known-good values with a staleness
  timeout)

**Task 6.3** — `jetson_runtime/infer_loop.py`:
- Main loop: every 5s, combine the latest vision feature vector with the
  latest ESP32-S3 feature packet into the exact input format the Track B
  model expects
- Run the TensorRT engine, get class probabilities
- Apply a debounce/hysteresis rule before triggering an alert (e.g.
  require 2 consecutive "High" classifications, not a single noisy frame)
  — this is a real-system requirement not present in the offline
  evaluation and should be explicitly implemented and justified
- Log every inference (timestamp, features, prediction) for post-hoc review

**Task 6.4** — `jetson_runtime/buzzer_gpio.py`:
- GPIO output to buzzer via `Jetson.GPIO`
- Escalating alert pattern (e.g. single beep at "Medium", sustained pattern
  at "High") rather than binary on/off

---

## 10. Phase 7 — Comparison & benchmarking

**Task 7.1** — `benchmarks/profile_latency.py`:
- Track A: measure inference latency on the training workstation GPU
  (report honestly as "desktop GPU latency", not claimed as real-time
  capable — it is the accuracy baseline, not a deployment candidate)
- Track B: measure end-to-end latency **on the Jetson**, from feature-ready
  to alert-decision, averaged over N runs with std dev reported (same
  methodology as FatigueNet's 200-run averaging)

**Task 7.2** — `benchmarks/profile_memory.py`:
- Peak GPU/CPU memory for both tracks, model file size on disk (raw
  checkpoint vs. TensorRT engine size for Track B)

**Task 7.3** — `benchmarks/compare_report.py`:
- Generate one results table: Accuracy / F1 / params / model size / latency
  / memory, one row per track
- Generate one accuracy-vs-latency scatter plot (this is the key figure —
  it visually shows the tradeoff rather than just declaring a winner)

---

## 11. Phase 8 — Testing & validation

**Task 8.1** — Unit tests for every feature function in `common/` — verify
against a small hand-computed example (e.g. a synthetic IBI sequence with
known RMSSD) before trusting it on real data.

**Task 8.2** — Cross-check that `common/features_*.py` (Python, used for
training) and the ESP32-S3 C++ equivalents (Task 5.3) produce matching
values on the same input signal — export a fixed test signal, run through
both implementations, diff the outputs. This is the single most likely
silent bug in the whole pipeline (train/deploy feature mismatch) and
deserves explicit test coverage.

**Task 8.3** — End-to-end integration test: simulate ESP32-S3 packets +
recorded video through the full `jetson_runtime` pipeline offline (no live
hardware needed) and confirm the alert logic fires on known-drowsy UL-DD
segments and stays quiet on known-alert segments.

---

## 12. Deliverables checklist

- [ ] Processed UL-DD feature table (parquet) + fold assignments, shared by both tracks
- [ ] Track A trained model + full-LOSO per-fold results (mean ± std over 3-5 seeds) + ablation table
- [ ] Track A hyperparameter sweep results (Task 3.5)
- [ ] (Optional) Track A deep-vision-node ablation (Task 3.6)
- [ ] (Optional) MePhy reproduction sanity check for the FatigueNet reimplementation
- [ ] Track B trained model (best of MLP/GBM/GRU) + full-LOSO per-fold results (mean ± std over 3-5 seeds)
- [ ] Track B hyperparameter sweep results
- [ ] Track B ONNX export + TensorRT engine, verified numerically equivalent
- [ ] ESP32-S3 firmware, flashed and verified sending correctly-formatted packets
- [ ] Jetson runtime pipeline, running live camera + serial input end-to-end
- [ ] Latency/memory benchmark results for both tracks
- [ ] Feature-parity test results (Python vs. ESP32-S3 C++)
- [ ] Final comparison table + accuracy-vs-latency plot
