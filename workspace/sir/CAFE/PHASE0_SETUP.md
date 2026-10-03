# CAFE Phase 0: Environment and Repository Freeze

Grounded in the official C-MET repo (`ChanHyeok-Choi/C-MET`, branch `main`) and its EDTalk dependency. Everything below is verified from source, not assumed. This phase produces the frozen baseline that every later result is traced back to. Do not modify any C-MET code before Gate 0 is closed.

Baseline coordinates:
- Paper: Choi et al., CVPR 2026, arXiv `2604.07786`.
- Code: `https://github.com/ChanHyeok-Choi/C-MET` (Python 3.9, PyTorch).
- Checkpoint and demo mirror: `https://huggingface.co/coldhyuk/C-MET`.
- Evaluation code released 2026-08-13 (in the repo `evaluation/` dir).

## 1. GPU verdict (resolved with the real dependency stack)

The earlier A4000-vs-DGX question is now settled at the environment level, not by guesswork.

- EDTalk's own repo targets `torch 1.10.0 + cu111`, which is Ampere-native and does not run on Hopper. This looks like it blocks the H100.
- It does not. C-MET's own `requirements.txt` pulls `audiocraft`, `funasr`, and `transformers==4.31.0`, which force a `torch 2.x`-class environment. C-MET re-uses only EDTalk's encoder and decoder weights under that newer env, so EDTalk's 1.10 pin is not binding for this pipeline.
- Therefore both GPUs are viable with `torch 2.1.2`: A4000 via the CUDA 11.8 wheel, H100 via the CUDA 12.1 wheel. `setup_cmet_env.sh` handles both.

| Question | A4000 ant-pc (16 GB, Ampere) | H100 DGX (80 GB, Hopper) |
| --- | --- | --- |
| Runs the C-MET torch 2.x stack | Yes (cu118 wheel) | Yes (cu121 wheel) |
| Reproduction (Gate 2) at full batch | Tight at 16 GB; may need batch or window trims | Comfortable |
| 29 to 35 run matrix (Section 10) | Serial, timeline bottleneck | 8-way parallel, tractable |
| MoEE and heavy external baselines | Painful or infeasible at 16 GB | Fine |
| Cost per single CAFE run | Sufficient (model is small, encoders frozen) | Overkill for one run |

Recommendation, unchanged and now reinforced: DGX (H100) as the primary machine for the full sweep and external baselines; A4000 for Phase 0 to 3 development and probe iteration to save DGX quota. This is the Section 10 "3090 default plus bigger overflow GPU" plan mapped to your actual hardware.

## 2. Verified dependency inventory

From C-MET `requirements.txt` and README (exact pins as published):

| Package | Pin | Notes |
| --- | --- | --- |
| python | 3.9 | `conda create -n C_MET python=3.9` |
| ffmpeg | 4.4 | `conda install -c conda-forge ffmpeg=4.4 pkg-config` |
| torch / torchvision / torchaudio | not pinned upstream | This project pins 2.1.2 / 0.16.2 / 2.1.2 deliberately (see below) |
| transformers | 4.31.0 | Old; watch for funasr conflicts |
| funasr | latest | Provides emotion2vec+large feature extraction |
| audiocraft | latest | Forces torch 2.x-class env |
| spacy | 3.7.5 | |
| timm | 1.0.9 | |
| face_alignment | 1.4.1 | |
| torchdiffeq | 0.2.5 | |
| gfpgan | 1.3.8 | Optional face super-resolution |
| basicsr | latest | torchvision import gotcha, see Section 5 |
| moviepy | 1.0.3 | |
| openai-whisper, deepfilternet, ninja, tyro, tensorboard, opencv-python, gradio 4.44.1 | as listed | |

Pin rationale for torch: `torch 2.1.2` supports both `sm_86` (A4000) and `sm_90` (H100). `torchvision 0.16.2` still ships `transforms.functional_tensor`, which `basicsr` imports; that module was removed in torchvision 0.17, so newer torchvision silently breaks `basicsr`. Pinning 0.16.2 avoids the break by construction.

## 3. Build the environment

Run on the target GPU node:

```bash
bash setup_cmet_env.sh a4000     # Ampere / A4000
bash setup_cmet_env.sh h100      # Hopper / H100 DGX
```

The script clones C-MET at a pinned commit, writes `CMET_PINNED_COMMIT.txt`, builds the conda env, installs torch first, then the requirements, guards the basicsr import, and runs an import smoke test that fails loudly if anything is wrong.

## 4. Checkpoint and data manifest (verified)

Download and place, then hash with the freeze script:

| Asset | Source | Destination |
| --- | --- | --- |
| C-MET connector checkpoint | Google Drive `1cTB3GqCDIigtedh-SNzUzYIyW0pBk_mQ` | `./checkpoints/` |
| EDTalk.pt and EDTalk Audio2Lip.pt | EDTalk Google Drive `1EKJXpq5gwFaRfkiAs6YUZ6YEiQ-8X3H3` | `./pretrained_weights/` |
| dlib shape predictor | `shape_predictor_68_face_landmarks.dat` (italojs mirror) | `./data_preprocess/` |
| GFPGAN weights (optional) | Google Drive `1SEWp_lnvxTHI1EIzurbNGYbPABmxih8A` | `./gfpgan/weights/` |
| MEAD | `wywu.github.io/projects/MEAD` | `./dataset/MEAD/FPS25/` |
| CREMA-D | `github.com/CheyneyComputerScience/CREMA-D` | `./dataset/CREMA_D/` |

Data layout expected by the repo:

```text
./dataset/MEAD/FPS25/<ID>/front/<emotion>/level_<n>/001.mp4
                                                    001.wav
                                                    001_ED_exp.npy
                                                    001_ED_pose.npy
                                                    001_ED_lip.npy
```

Frame rate must be 25 fps for every clip. Feature extraction after cropping:
- EDTalk expression features: `python prep_video.py --root ./dataset/MEAD/FPS25`
- emotion2vec+large features: `python extract_e2v+L.py --root ./dataset/MEAD/FPS25`
- Frozen test lists live at `./dataset/MEAD/test.csv` and `./dataset/CREMA_D/test.csv`.

## 5. Known failure checkpoints (concrete, verified)

| # | Symptom | Cause | Fix |
| --- | --- | --- | --- |
| F1 | `ImportError: cannot import name 'rgb_to_grayscale' from torchvision.transforms.functional_tensor` | torchvision >= 0.17 removed that module; basicsr still imports it | Keep torchvision 0.16.2 (pinned); the setup script also auto-patches basicsr if needed |
| F2 | torch installs as CPU-only or an old build | torch not pinned upstream; a transitive dep pulls it | Install torch first from the correct CUDA index (setup script does this) |
| F3 | H100 runs but ops error with `no kernel image` | cu118 wheel used on Hopper | Use the h100 target (cu121 wheel) |
| F4 | Reproduction numbers do not match the paper | MEAD generic-sentence file order differs from the paper appendix | Realign per Section 6; this is a Gate 1 item, flagged here because it originates at data download |
| F5 | funasr or emotion2vec import or download errors | transformers 4.31.0 is old; funasr may want newer | Pin transformers as-is first; if funasr breaks, bump only transformers and re-run the smoke test |
| F6 | Cropping fails | missing dlib `shape_predictor_68_face_landmarks.dat` | Place it in `./data_preprocess/` |

## 6. The MEAD sentence-order realignment (author-confirmed)

The C-MET README states it directly: the task uses each identity's Common Sentences (files 001 to 003) and Generic Sentences (021 to 030 per emotion, 031 to 040 for neutral), following the MEAD paper appendix order. The official MEAD Part0 release's actual generic-sentence file order does not always match that appendix, so the authors manually listened to every file and renumbered them. If you use the official release as-is, your numbering will not match theirs and results will not reproduce exactly. Producing the machine-readable remap file is a Gate 1 deliverable, but the risk originates the moment MEAD is downloaded, so budget for it now.

## 7. Close Gate 0

After the environment builds and checkpoints are placed, run from the C-MET repo root:

```bash
python /path/to/freeze_provenance.py --root . --out ./paper_artifacts/provenance
python /path/to/freeze_provenance.py --selftest   # optional CPU self-check
```

It writes `provenance.json`, `requirements.lock`, and `provenance.txt`, and exits non-zero until every expected checkpoint is present and hashed.

Gate 0 checklist (from the roadmap):

- [ ] Official demo runs end to end on at least 3 identities and 3 emotions.
- [ ] Generated outputs trace to one pinned commit (`CMET_PINNED_COMMIT.txt`) and one environment (`requirements.lock`).
- [ ] GPU, driver, CUDA, torch versions, and checkpoint hashes are logged automatically (`provenance.json`).
- [ ] No local C-MET code modification before the baseline artifacts are archived.

Only when all four are checked do you proceed to Phase 1 (dataset audit).

## 8. Exact first commands, in order

```bash
# on the GPU node
bash setup_cmet_env.sh h100          # or a4000
cd C-MET
# ... download checkpoints and dlib predictor into the paths in Section 4 ...
python inference.py --num_samples 10 \
  --connector_exp_path ./checkpoints/_epoch_2105_checkpoint_step000200000.pth \
  --source_path ./asset/identity/ChatGPT_man3_crop.png \
  --audio_driving_path ./asset/audio/W009_038.wav \
  --pose_driving_path ./asset/video/W009_038.mp4 \
  --save_path ./res/demo_happy.mp4 \
  --neu_e2v_path ./audios/MEAD/neutral/emotion2vec+large_features/ \
  --emo_e2v_path ./audios/MEAD/happy/emotion2vec+large_features/
python /path/to/freeze_provenance.py --root . --out ./paper_artifacts/provenance
```

If the demo renders and the freeze script reports all checkpoints hashed, Gate 0 is closed.
