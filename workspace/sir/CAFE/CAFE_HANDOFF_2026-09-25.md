# CAFE Project Handoff (2026-09-25)

Continuation document for a new chat. Read fully before acting. Everything below is measured or decided; open questions are labeled as such.

## 1. Project identity

| Item | Value |
| --- | --- |
| Paper | CAFE: Counterfactual Affect Factorization and Composition, built on C-MET (Choi et al., CVPR 2026) |
| Thesis | Compound-emotion generalization with zero compound supervision, proven by a falsification-first, supervision-independent harness |
| Target venue | CVPR / ICCV / ECCV main track |
| C-MET repo | github.com/ChanHyeok-Choi/C-MET, pinned commit `44ccf3a` |

## 2. Working conventions (important, learned the hard way)

| Convention | Reason |
| --- | --- |
| Chat in Hinglish; all deliverables in formal American English; no em-dashes anywhere | user preference |
| One focused step per reply; one command at a time | user preference |
| Commands must be **single-line**. Multi-line pastes (heredocs, multi-line `python -c`) break in the JupyterLab terminal | observed repeatedly |
| Paste in the terminal with `Ctrl+Shift+V` (plain `Ctrl+V` fails) | observed |
| Uploads land in whatever folder is open in JupyterLab; **overwriting an existing file often silently fails**. Use **new versioned filenames** (`_v2`, `_v3`) and verify by exact byte size before use | three stale-version incidents |
| The assistant sandbox has **no access to the DGX**. The user runs every command | architecture |
| Long jobs: launch with `setsid nohup ... < /dev/null &` behind a `pgrep` guard (so a double paste cannot start duplicates). Plain `nohup` died when the JupyterLab terminal glitched | observed |
| Killer testing standard: sandbox self-tests, mutation tests (disable a guard, the test must fail), end-to-end integrations, and a regression test proving the previous algorithm fails on the new scenario | user requirement |
| DGX network: HuggingFace CDN, tfhub.dev, adrianbulat.com blocked or broken; GitHub, PyPI, dlib.net work; Google Drive large files need manual download; robots.ox.ac.uk partially works | measured |

## 3. Environment

| Item | Value |
| --- | --- |
| DGX node | dgxanode02, NVIDIA A100-SXM4-80GB, 256 CPU threads |
| Conda env | `C_MET`: Python 3.9, torch 2.1.0+cu121. Activate: `source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate C_MET` |
| Repo on DGX | `/workspace/sir/CAFE/C-MET` (run all C-MET and CAFE tools from here) |
| CAFE scripts | `/workspace/sir/CAFE/` (invoked as `python3 ../<script>.py` from the repo root) |
| Raw backup | ant-pc `/media/ant-pc/HDD2/CAFE/staging/<ID>/front/...` (raw 1080p, 30 fps). Keep until Gate 2 closes |
| Model caches | `/root/.cache/torch/hub/checkpoints/` (2DFAN4, s3fd, FID Inception), `/root/.cache/whisper/small.en.pt` (SHA256 verified), `/workspace/sir/CAFE/models/emotion2vec_plus_large/` |
| Full history | compacted transcript `/mnt/transcripts/2026-09-24-18-49-29-cafe-cmet-implementation.txt` (assistant side only) |

## 4. Status by phase

| Phase / Gate | Status |
| --- | --- |
| Phase 0 / Gate 0 (environment, provenance freeze, demo) | Closed |
| Phase 1 / Gate 1 (data audit and preprocessing) | Practically closed; formal audit document pending |
| Phase 2 / Gate 2 (C-MET reproduction) | Generation complete (1141/1141, 0 failures); metrics pending |
| Phases 3 to 7 | Not started |

## 5. Data state on the DGX (test identities M003, M030, W009, W015, front view)

| Artifact | State |
| --- | --- |
| `dataset/MEAD/FPS25/<ID>/front/<emo>/<level>/NNN.mp4` | 2641 clips, 256x256, exact 25 fps, audit clean |
| `NNN.wav` | 16 kHz mono PCM, extracted from the cropped clip (aligned with the trimmed video) |
| `NNN_ED_exp.npy`, `_ED_pose.npy`, `_ED_lip.npy` | 2641/2641, dims (10, 6, 20), rows equal frame count |
| `emotion2vec+large_features/NNN.npy` | 2641/2641, 1024-d; median cosine vs authors' released subset 0.9994 |
| Numbering | **Renumbered to the authors' numbering** (36 folders, atomic directory swaps). Originals kept outside the tree in `dataset/MEAD/renum_backup/<ID__emo__level>/` |
| Quarantined clips | 2, each renamed to `901` in its folder: `M030/sad/level_2` (whisper heard only "k", likely silent or corrupt) and `W009/disgusted/level_2` (garbled, unresolvable). C-MET loaders read only 001 to 040, so they are ignored |
| Transcripts | `dataset/MEAD/transcripts_all.csv` (rewritten to the new numbering; original in `.pre_renum`) |
| Ledgers | `crop_done.txt`, `fps_done.txt` rewritten to on-disk truth (originals in `.pre_renum`) |
| Test manifest | `dataset/MEAD/test_present.csv`: **1141 of 1143** rows (2 genuinely missing, listed in `test_missing_clips.txt`) |
| `prep_video.py` | still renamed to `prep_video.py.hold` (not used; restore for repo cleanliness) |

## 6. Research findings from Phase 1 (for the audit and the paper)

1. The official MEAD release has sentence-numbering errors at the folder level, including level_1: shifts, rotations, swaps, and level-specific permutations. In some emotions the error is the **majority** (angry: 3 of 4 speakers shifted by one from about 016; M003 correct).
2. Of the 36 test clips believed to be absent from the release, **34 were numbering artifacts**; the sentences exist at other numbers.
3. The authors' test set pairs gt 020 with neutral 030, but **020 never matches neutral 030 in any folder** (71/71). These rows are inherently content-mismatched in the authors' own protocol.
4. Evidence from the authors' released emotion2vec features indicates **partial renumbering by the authors**: contempt for M030, W009, W015 appears to follow the release numbering, while angry and M003 contempt follow the corrected numbering.
5. Some clips carry a sentence outside their script, and near-duplicate script sentences exist (for example "land based radar will / would help with this task").
6. The released `inference.py` cannot run the official evaluation direction (its `Dataset(...)` line is commented out) and `prep_video.py` is left in the CREMA-D configuration (finds 0 MEAD videos). `crop_video.py` places `-r 25` after the output file, so ffmpeg ignores it (clips stay 30 fps).

## 7. Renumbering method that finally passed (canonical_remap_v3, algorithm v5)

Reference (authors' numbering):
- 001 to 003 = neutral 001 to 003 (neutral scripts are identical across speakers);
- 021 to 030 = neutral 031 to 040 (C-MET's own rule);
- 004 to 020 = majority over **trusted** folders only (a folder is trusted if its 13 common and generic clips read as the expected neutral sentences, at most one miss), with the swap partner of any miss excluded from voting, plus an elimination fill when exactly one gap and one recurring unmatched sentence remain.

Final real-data validation: B (per-clip neutral check) 0/1070 misassigned; C (test-pair content) 0 bad on 001 to 003 and 021 to 030, 020 inherently bad (62 after, 71 before); LOSO contradictions 0; anchors 168 agree / 19 disagree. The e2v anchor error rate measured on positions with known text truth is 12%; 004 to 020 shows 8%, i.e. no excess error.

Approaches that failed and why (do not repeat): level_1 as reference (level_1 itself is shifted); plain majority (release errors can be the majority); emotion2vec anchors as a table source (e2v encodes emotion, not content; short sentences collide); absolute similarity thresholds (ASR noise; calibrated values: unrelated sentences max 0.60, same sentence with real whisper errors 0.83 to 0.92).

## 8. Decisions in force

| Decision | Detail |
| --- | --- |
| Option A for contempt | Use text-consistent numbering everywhere (documented deviation from the authors' partial renumbering). A sensitivity check (Branch B direction cosine, Option A vs C) is pending |
| Gate flags used | `--canonical --ignore_gt_numbers 20 --max_anchor_disagree 19` |
| Branch B direction | C-MET's own `Dataset('test', './dataset/MEAD/FPS25', mode='mean', direction='average', num_samples=10, audio_encoder='emotion2vec+large').get_raw_e2v('neutral', emo, level)` |
| Identity image | frame 0 of the neutral source video (the released code does not specify it; declared assumption) |
| emotion2vec weights | HuggingFace `emotion2vec/emotion2vec_plus_large` (model card states identical to the ModelScope release), loaded offline from the local folder |

## 9. Script inventory on the DGX (`/workspace/sir/CAFE/`)

Current:

| Script | Size (bytes) | Purpose |
| --- | --- | --- |
| `overnight_gate2_v4.py` | 42382 | Orchestrator: transcribe check, canonical gate, atomic renumber with rollback, post-verify, manifest, Branch B generation |
| `canonical_remap_v3.py` | 29532 | Canonical renumbering analysis (algorithm v5), report only |
| `align_mead_levels.py` | 16224 | Transcript alignment helpers (imported by canonical_remap) |
| `transcribe_mead.py` | 8801 | Whisper small.en transcription, resume-safe |
| `extract_edtalk_mead.py` | 15434 | EDTalk features with C-MET components, sharding support |
| `extract_e2v_mead.py` | 11781 | emotion2vec features (a 12489-byte version with an improved fidelity parser was never uploaded; the fidelity check was done with a one-liner) |
| `fps_wav_fix.py` | 11973 | 30 to 25 fps re-encode and wav extraction |
| `preprocess_crop_batch.py` | parallel version | Batch crop with `--workers`, ledger resume |

Superseded, do not use: `canonical_remap.py`, `canonical_remap_v1.py`, `canonical_remap_v2.py`, `overnight_gate2.py`, `_v1`, `_v2`, `_v3`; `build_ours.py` (uses Branch A, intensity-blind); the gates in `run_cafe_pipeline.py` (still wired to the old flow: `prep_video.py` and Branch A).

## 10. Gate 2 outputs

| Path (from repo root) | Content |
| --- | --- |
| `evaluation/runs/mead_cmet_repro/gen/*.mp4` | 1141 generated videos, named `ID_emotion_level_NNN.mp4`, each validated (256x256, frames, audio) |
| `evaluation/runs/mead_cmet_repro/ours.csv` | columns: source_video_path, gt_video_path, gt_emotion, intensity, generated_path, source_audio_path (paths relative to the repo root) |
| `dataset/MEAD/overnight_report.json`, `overnight.log`, `overnight_state.json` | run record (S3b done) |
| `dataset/MEAD/canonical_remap.json`, `remap_used.json`, `canonical_remap_after.json` | renumbering decisions and post-verify |
| `evaluation/runs/mead_cmet_repro_smoke/` | 2-row smoke test (ignore or delete) |

Generation speed: about 8 to 9.6 s per row including the ffmpeg mux (models loaded once).

## 11. Gate 2 metrics: plan and asset status

C-MET's `evaluation/README.md` pipeline (run from `evaluation/`): video to frames (`vide2frame_custom.py`), face crops (`frame2face_custom.py`), FID (`pytorch-fid/custom.py`), FVD (`fvd.py`), Acc_emo (`Emotion-FAN/emotion-fan.py --num_frames 16`), Sync_conf (`syncnet_python/all_pipeline.py`, `all_syncnet.py`, `conf_mean.py`, pairs generated video with **source** audio), summary (`check_quantitative_all.py`). AITV is timed inside inference (not in `evaluation/`); preprocessing included, mux and save excluded, no warm-up, no `cuda.synchronize`.

Critical gotchas found by reading the scripts:
- The scripts hardcode `./runs/mead_ours` and process **every CSV** in it. Place exactly one CSV at `evaluation/runs/mead_ours/ours.csv`.
- CSV paths are used as-is with `evaluation/` as the working directory. Write **absolute paths** into the evaluation copy of `ours.csv`.

Assets:

| Asset | Status | Location |
| --- | --- | --- |
| FID InceptionV3 `pt_inception-2015-12-05-6726825d.pth` | Done (95628359 bytes, SHA256 prefix 6726825d) | `/root/.cache/torch/hub/checkpoints/` |
| SyncNet face detector `sfd_face.pth` | Done (89844381 bytes) | `evaluation/syncnet_python/detectors/s3fd/weights/` |
| dlib `mmod_human_face_detector.dat` | Done (729940 bytes) | `evaluation/Emotion-FAN/data/face_alignment_code/lib/` |
| dlib `shape_predictor_5_face_landmarks.dat` | **Manual** (auto-download produced a 0-byte file) | same lib folder |
| `syncnet_v2.model` | **Manual** | `evaluation/syncnet_python/data/` |
| `Emotion-FAN_MEAD.pth` | **Manual** (Google Drive) | `evaluation/Emotion-FAN/checkpoints/` |
| `face_align_cuda.py`, S3FD code (`__init__.py`, `box_utils.py`, `nets.py`) | Fetched automatically by the placement command | lib folder, `detectors/s3fd/` |
| FVD: TensorFlow + tensorflow_gan + tensorflow_hub + I3D (kinetics-400) | **Hardest item**. tfhub.dev returns 404 (TF-Hub retired; the model moved to Kaggle). Plan: separate conda env for TensorFlow (do not install TF into `C_MET`), obtain the I3D module manually. Idea to verify: place it in `TFHUB_CACHE_DIR` under the SHA1 of the module URL so `fvd.py` runs unmodified | not started |
| Python packages (`pytorch_fid`, `natsort`, `scenedetect`, `python_speech_features`, `dlib`) | **Unknown**: the `PKGS` probe line was never captured. Rerun it | |

Manual download links (direct):
1. http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2 (upload the `.bz2`; the command extracts it)
2. https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model
3. https://drive.google.com/uc?export=download&id=1H0tqOEe5-EqlmomB_FujgbrG8C7dadf1 (click "Download anyway", rename to `Emotion-FAN_MEAD.pth`)

Upload all three to `/workspace/sir/CAFE/`, then run this placement command from the repo root (sandbox-tested):

```bash
U=/workspace/sir/CAFE; E=evaluation; L=$E/Emotion-FAN/data/face_alignment_code/lib; D=$E/syncnet_python/detectors/s3fd; mkdir -p $L $D $E/syncnet_python/data $E/Emotion-FAN/checkpoints; curl -fsSL --max-time 60 -o $L/face_align_cuda.py https://raw.githubusercontent.com/Open-Debin/Emotion-FAN/master/data/face_alignment_code/lib/face_align_cuda.py; for f in __init__.py box_utils.py nets.py; do curl -fsSL --max-time 60 -o $D/$f https://raw.githubusercontent.com/joonson/syncnet_python/master/detectors/s3fd/$f; done; f=$(find $U -maxdepth 2 -name "shape_predictor_5_face_landmarks.dat*" -size +0 | head -1); case "$f" in *.bz2) bunzip2 -c "$f" > $L/shape_predictor_5_face_landmarks.dat;; ?*) cp "$f" $L/shape_predictor_5_face_landmarks.dat;; esac; f=$(find $U -maxdepth 2 -name "syncnet_v2*.model" | head -1); [ -n "$f" ] && mv "$f" $E/syncnet_python/data/syncnet_v2.model; f=$(find $U -maxdepth 2 -name "Emotion-FAN_MEAD*.pth" | head -1); [ -n "$f" ] && mv "$f" $E/Emotion-FAN/checkpoints/Emotion-FAN_MEAD.pth; echo "=== assets (size, path; none may be 0) ==="; ls -l $L/*.dat $L/face_align_cuda.py $D/*.py $D/weights/sfd_face.pth $E/syncnet_python/data/syncnet_v2.model $E/Emotion-FAN/checkpoints/Emotion-FAN_MEAD.pth ~/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth 2>&1 | awk '{print $5, $NF}'; python3 -c "import importlib.util as u;print('PKGS', {m: bool(u.find_spec(m)) for m in ['pytorch_fid','tensorflow','dlib','natsort','scenedetect','python_speech_features','pandas','cv2']})"
```

Expected sizes: shape_predictor about 9 MB; face_align_cuda.py 4904; S3FD code 2142, 6698, 5573; syncnet_v2.model about 50 MB; none may be 0 bytes.

Paper reference (C-MET Table 1, MEAD): AITV 2.643, FID 90.804, FVD 329.862, Sync_conf 7.9996, Acc_emo 55.91. Gate 2 tolerance: within 10% of each paper value. Acc_emo is only comparable when computed with the same classifier (Emotion-FAN MEAD checkpoint).

## 12. Pending work, in order

1. Upload the three manual assets, run the placement command, capture the `PKGS` line; install missing packages into `C_MET` (not TensorFlow).
2. Build the evaluation CSV at `evaluation/runs/mead_ours/ours.csv` with absolute paths; run preprocessing, FID, Emotion-FAN, SyncNet (test on a small subset first, then full).
3. FVD: separate TensorFlow env plus manual I3D module.
4. AITV: time generation per the README methodology (report hardware; not strictly comparable).
5. Gate 2 decision against Table 1 (10% tolerance), including the known deviations below.
6. Sensitivity check: contempt Option A vs Option C (Branch B direction cosine).
7. Engineering debt: update `run_cafe_pipeline.py` gates to the new tools; restore `prep_video.py`; clean `renum_backup/` and the ant-pc staging backup only after Gate 2 closes.
8. Formal audit document (Phase 1 findings, methods, validations, deviations) for the reproducibility section.
9. Phase 3 prerequisite: MEAD **train identities** are not on the DGX. Plan storage, download, and the same preprocessing chain (all tools are general; the neutral-anchored renumbering works for any speaker).

## 13. Known deviations to report with Gate 2

- 2 of 1143 test rows unavailable (1 silent clip, 1 garbled clip; both quarantined).
- 020 test rows are content-mismatched in the authors' protocol itself (kept, as the authors did).
- Contempt numbering follows the content-consistent rule (Option A), which differs from the authors' apparent partial renumbering for M030, W009, W015.
- Identity image assumed to be frame 0 of the neutral source.
- Audio and wav files derive from the cropped (end-trimmed) clips; emotion2vec fidelity median cosine 0.9994 vs the authors' features.
- The official evaluation direction (Branch B) was reproduced with C-MET's own Dataset class because the released inference script cannot run it.

## 14. First message for the new chat (suggested)

"Continue the CAFE project from the attached handoff. We are at Gate 2 metrics: next step is placing the manual evaluation assets and capturing the PKGS line (Section 11). Follow the conventions in Section 2."
