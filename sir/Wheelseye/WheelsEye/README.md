# WheelsEye

Two-track multimodal driver fatigue detection. **Track A** is a heavyweight
GNN + Transformer + MGAF model (adapted from the FatigueNet paper) trained
and evaluated off-device to establish an accuracy ceiling. **Track B** is a
lightweight feature-based model (MLP / LightGBM / GRU) meant to actually
deploy on a Jetson Orin Nano 8GB with an ESP32-S3 sensor hub. Both tracks
share the same data, preprocessing, features, LOSO splits, and evaluation
harness, so the final Track A vs. Track B comparison is apples-to-apples.

The full task breakdown this repo implements against lives in
[`tasks.md`](tasks.md). Day-to-day decisions, deviations from that spec,
and real bugs found along the way are logged in [`NOTES.md`](NOTES.md).
Current status (what's done, what's left) is in
[`PROGRESS.md`](PROGRESS.md).

## Datasets

Neither dataset lives in this repo (both are large binary downloads) --
they're expected at fixed external paths:

| Dataset | Path | Role |
|---|---|---|
| UL-DD (University of Louisiana Drowsiness Dataset) | `E:\capstone\ul-dd\` | Primary dataset for both tracks |
| MePhy | `E:\capstone\mephy\MePhy Dataset\MePhy Dataset\` | Used only for the FatigueNet reimplementation sanity check (`mephy_repro/`) |

If your paths differ, edit `ULDD_ROOT`/`MEPHY_ROOT` in `common/paths.py`
and `mephy_repro/paths.py`.

## Repository layout

```
common/                   Shared by both tracks -- data, features, model architecture
  paths.py                 UL-DD on-disk path resolution
  labels.py                 KSS (1-9) -> 3-class Low/Medium/High binning
  loaders.py                 Per-modality raw signal loaders (fixed-rate, event-based, frame-indexed)
  windowing.py                 10s window / 5s step sliding-window slicing across all modalities
  features_vision.py            EAR/MAR/PERCLOS/head-pose from the provided FL/PL landmark CSVs
  features_bio.py                HR/HRV, EDA tonic-phasic, SpO2, TEMP
  features_grip.py                Grip mean/std/asymmetry
  features_telemetry.py            Speed variance, heading-change rate (UL-DD-only auxiliary signal,
                                       not part of our deployed sensor set -- see below)
  metrics.py                        Accuracy/precision/recall/F1(macro)/confusion matrix, fold aggregation
  dataset.py                         PyTorch Datasets wrapping the processed feature table
  fatiguenet_model.py                 The shared GNN+Transformer+MetaNet/MGAF architecture (Track A and
                                       mephy_repro both import this -- same code, different node configs)

mephy_repro/               FatigueNet reproduction sanity check on MePhy (optional per tasks.md, but
                            treated as required for this project)
  paths.py, loaders.py, windowing.py, features.py, dataset.py, train.py

track_a_heavy/             Track A: the accuracy-ceiling model, trained off-device only
  model.py                  Configures common/fatiguenet_model.py for OUR three sensor nodes -- vision
                             (camera), bio (physiological), grip (pressure) -- not UL-DD's full four;
                             Telemetry is deliberately excluded by default (no such sensor is ever
                             deployed), reachable only via an explicit include_telemetry=True for a
                             separate Track-A-only ceiling ablation, never for Track B
  train.py                    Per-LOSO-fold training loop
  evaluate.py                   Results table + confusion matrix figure
  ablation.py                     no-GNN / no-Transformer / no-MGAF / no-reconstruction-loss variants

track_b_light/              Track B: the actual Jetson deployment candidate
  model.py                   MLP / GRU (torch) + LightGBM param builder
  train.py                     Trains whichever of the three is selected, same LOSO folds as Track A
  evaluate.py                    Side-by-side comparison table across MLP/LightGBM/GRU
  export_onnx.py                  Trains a deploy model on all subjects, exports to ONNX, verifies
                                   numerical equivalence against the PyTorch model
  export_tensorrt.py              Not yet written -- runs on the Jetson itself (Task 4.5)

scripts/                    One-off pipeline scripts (not imported by anything else)
  verify_uldd.py              Cross-checks every UL-DD file against Info.xlsx's own availability manifest
  build_feature_table.py         Builds data/processed/uldd_features.parquet (the shared feature table)
  build_mephy_feature_table.py     Builds data/processed/mephy_features.parquet

tests/                      pytest unit tests, hand-computed examples (Task 8.1)

firmware/, jetson_runtime/, benchmarks/   Scaffolded directories, not yet implemented (Phases 5-7)

data/
  raw/                       Untouched dataset notes (not the datasets themselves -- those are external)
  processed/                 Generated artifacts: feature parquets, fold assignments, checkpoints,
                              training results (all gitignored except folds.json)
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or the CUDA build matching your GPU
pip install -r requirements.txt
```

Verified working on Python 3.14 with `torch==2.13.0+cu126`,
`torch-geometric==2.8.0.post1`, `lightgbm==4.7.0`, `onnx`/`onnxruntime`,
`onnxscript` (needed for `torch.onnx.export`). See `requirements.txt` for
the full pin list and what's deliberately not yet installed (`mediapipe`,
`timm`).

## Running the pipeline

Everything below assumes you're in the repo root with the venv active.

**1. Verify the raw UL-DD download** (checks every file against the
dataset's own availability manifest; found and documented two real data
issues already -- see NOTES.md):
```
python scripts/verify_uldd.py
```

**2. Build the shared feature tables** (only needs to be re-run if the
feature functions or windowing parameters change):
```
python scripts/build_feature_table.py         # UL-DD -> data/processed/uldd_features.parquet
python scripts/build_mephy_feature_table.py    # MePhy -> data/processed/mephy_features.parquet
```

**3. (Recommended first) Reproduce FatigueNet on MePhy** as a sanity check
that the shared architecture is implemented correctly before trusting it
on UL-DD:
```
python mephy_repro/train.py --split both --epochs-random 30 --epochs-loso 15
```
Runs two protocols and prints both: `random` replicates the paper's own
80/20 random-split methodology (the actual "do we hit ~90.2%" check),
`loso` runs the rigorous subject-independent version tasks.md's own rules
require for this project's real pipeline -- expect `loso` to score lower,
see NOTES.md for why that's not a bug.

**4. Train Track A** (GNN+Transformer+MGAF, full LOSO by default):
```
python track_a_heavy/train.py --epochs 15                # all 16 folds
python track_a_heavy/train.py --folds 0 1 2 --epochs 15   # subset, for a quicker check
python track_a_heavy/evaluate.py                          # results table + confusion matrix figure
python track_a_heavy/ablation.py --folds 0 1 2 --epochs 15   # component ablation (Task 3.4)
```

**5. Train Track B** (run each of the three options, then compare):
```
python track_b_light/train.py --model mlp
python track_b_light/train.py --model lightgbm
python track_b_light/train.py --model gru
python track_b_light/evaluate.py                          # side-by-side comparison, picks the best
python track_b_light/export_onnx.py --model mlp --epochs 30   # export + numerically verify
```

**6. Run tests**:
```
python -m pytest tests/ -v
```

## Compute note

This has been developed and validated on a laptop RTX 4050 (6GB VRAM), not
the A100/H100 80GB tasks.md assumes for Track A's full hyperparameter
sweep (Task 3.5: full LOSO x 3-5 seeds across a GNN hidden-dim sweep).
Everything above runs and produces real, correct results at this scale --
what hasn't happened yet is the large-scale sweep tasks.md wants for the
*final reported* Track A numbers. See PROGRESS.md for exactly what's
validated vs. what still needs bigger hardware.
