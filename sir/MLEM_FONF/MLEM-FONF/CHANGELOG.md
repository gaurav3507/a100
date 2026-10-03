# Changelog

## v0.9.32 — 2026-09-09
- Alpha ladder extended to 409.6 so the discrepancy criterion, not the
  ladder end, terminates the search. Measured at 60 photons per ray: the
  selected order is 25.6 and lands within 0.02 dB of the optimum, while the
  old 2.0 ceiling cost about 4 dB.

## v0.9.31 — 2026-09-09
- CHO block resume is now cache-aware: a finished CSV block is recomputed if
  the ROI cache no longer holds every requested method (needed after the
  alpha-cap fix invalidated the A-FONF ROIs).

## v0.9.30 — 2026-09-09
- Fixed the alpha ceiling: the 0.2-2.0 selection grid capped the method
  wherever the data demanded stronger smoothing (photon-starved settings),
  costing ~0.55 dB and producing the "alpha saturates at 2.0" artefact.
  Replaced by an unbounded geometric ladder with discrepancy-driven early
  stopping (alpha > 2 admissible; H remains in [0,1], symmetric,
  nonexpansive). A-FONF rows of the gate, the low-dose campaign points and
  CHO must be regenerated. 44 checks.

## v0.9.29 — 2026-09-09
- CHO: --object-stripe task (random in-band directional interference per
  realization, shared within a pair) that actually exercises the notch
  mechanism; --select-iters to control the Morozov selection budget
  (default unchanged; 8.5x faster A-FONF cells when reduced).

## v0.9.28 — 2026-09-08
- Figure editorial pass on real data: no clipped curves (data-driven
  y-limits), panel labels clear of titles, violin panels scaled to their
  data, slope-chart label stack widened, stability annotation repositioned.

## v0.9.27 — 2026-09-08
- experiments/summarize_gate.py: per-dose mean ± 95% CI, paired Wilcoxon of
  A-FONF against each baseline, and an explicit PASS/REVIEW verdict for the
  venue gate. Verified on known-truth CSVs (PASS, REVIEW and null branches).

## v0.9.26 — 2026-09-08
- Gate runner: --append / --out so per-dose runs with different tuned TV
  weights accumulate in one CSV (method label records the weight).

## v0.9.25 — 2026-09-08
- repair_cho_cache.py (duplicate/mismatch audit and --fix repair using the
  per-ROI key files) and a campaign lock preventing two concurrent writers
  from corrupting a shared ROI cache. Both live-tested.

## v0.9.24 — 2026-09-08
- Gate audit: matrix-free A-FONF now reproduces the shipped pipeline
  bit-identically (exponent damping, undamped selection stage, warm-started
  persistence partner); FBP scale calibrated on a phantom instead of the
  test slice (oracle leak). Two permanent regression tests; 43 checks.

## v0.9.23 — 2026-09-08
- Gate runner: --tv-weight (baseline tuning for fairness; recorded in the
  method label) and --methods subsetting. TMI gate passed at dose-matched
  realistic geometry pending the tuned-TV comparison.

## v0.9.22 — 2026-09-08
- TMI gate made a genuine LOW-DOSE test: FBP and MLEM+TV comparators added
  over operator closures, and incidents default to dose-matched values
  (equal total photons to the 64x90 reference grid) instead of equal
  photons-per-ray, which at 736x576 carried 73x more flux.

## v0.9.21 — 2026-09-08
- CHO channel definition corrected to the standard Laguerre-Gauss form
  (mean subtraction removed; kept as a documented ablation). The deviation
  had depressed all d' by ~25% and manufactured an apparent TV advantage.
- Signal-matched channel width (a = 2 x lesion radius); --stats-only and
  --cache-dir to re-score cached ROIs without reconstructing; suite test
  now asserts the literature form pointwise.

## v0.9.20 — 2026-09-07
- Deterministic CHO ensembles (cell ids persisted; lexsort before
  statistics) — results no longer depend on worker completion order.
- Size-or-time cache flush (60 s) so a crash costs seconds.
- Permanent OPERATIONAL suite test: shrunken campaign run fresh, repeated
  and hard-killed+resumed must agree to 1e-9; parent memory measured flat.
  41 checks.

## v0.9.19 — 2026-09-07
- CHO campaign hardened after a silent death at 9.6k/14.4k cells: streaming
  ROI cache (parent holds nothing), cell-level resume index, request-aware
  block skip, per-cell fault isolation with NaN filtering, worker recycling.

## v0.9.18 — 2026-09-07
- Fixed silent DPS skip at non-3000 doses (dose-agnostic diffusion
  checkpoint fallback) and made unavailable --methods fail loudly with the
  checkpoint listing instead of writing zero rows.

## v0.9.17 — 2026-09-07
- CHO runner hardened: index-only tasks (15.1 GB -> 0.9 MB pool transfer),
  collision-free noise seeds (491 -> 800 unique), per-block resume,
  repeated split-half averaging so bootstrap CIs bracket the estimate,
  small-N warning.
- Realistic-geometry runner: operator preflight, actionable ASTRA error,
  incremental fsync'd CSV, empty-slice guard, --split. Both executed
  end-to-end (CHO on the synthetic tree; geometry runner on a stub backend).

## v0.9.16 — 2026-09-07
- CHO task-based evaluation: mlem_fonf.cho (LG channels, lesion model, ROI,
  split-half d' with bootstrap CI, AUC) + experiments/run_cho.py parallel
  SKE/BKS campaign. Four permanent validation tests (ideal-observer
  scaling, unbiased null, contrast monotonicity, channel/ROI exactness);
  38 checks total.

## v0.9.15 — 2026-09-07
- PAPER_PLAN.md (title, venue gate, contributions, scope) locked.
- run_realistic_geometry.py: matrix-free frozen pipeline over ASTRA fan
  operators (512^2 / 736 bins / 576 views) for the TMI decision run;
  loader gains `size`. Operator path executed on a sparse shim.

## v0.9.14 — 2026-09-07
- Figure identity v2: tier-coded palette, broken-axis violins with inner
  median/IQR boxes and collapsed-realization strip, stability-gap shading.

## v0.9.13 — 2026-09-07
- Figures redesigned to IEEE TMI/JBHI conventions: CI bands, family-coded
  identity, annotated deltas, distribution-shift slope chart, violin
  robustness with collapse %, dual-geometry stability trajectories; IEEE
  column widths, 8-pt fonts, vector PDF + 600-dpi PNG.

## v0.9.12 — 2026-09-07
- experiments/make_figures.py: PSNR-vs-dose 3-panel figure (consistent
  method identity, error bars), worst-case robustness bars, and the live
  MLEM-vs-spectral-step stability trajectory (fan, inc=60). PNG+PDF at
  300 dpi. Executed on synthetic benchmark CSVs (NaN/catastrophic rows
  handled) and the live trajectory.

## v0.9.11 — 2026-09-07
- MLEM early-stopping protocol: select_mlem_iters.py (validation-selected
  iteration per geometry x dose -> results/mlem_iters.json), read by both
  runners with 800-iteration fallback; drop_method_rows.py for protocol
  recomputes. Executed end-to-end on synthetic data.

## v0.9.10 — 2026-09-07
- DPS: clip_denoised (x0 clamped to [-1,1], sqrt(abar) floor) — fixes the
  abar_T explosion that turned trained-model samples into saturated noise.
  Trained-model fixture test added (tests/fixtures/score_phantom64.pt);
  34 checks.

## v0.9.9 — 2026-09-07
- Model-free DPS quality oracle (exact Gaussian-prior score): legacy step
  reproduces the bug (=prior mean), fixed step gains +10 dB; permanent
  suite test (33 checks). dps_infer accepts injected score models.

## v0.9.8 — 2026-09-07
- DPS sampler fixed to the Chung et al. formulation (residual-norm
  gradient step; the previous normalized step gave ~zero guidance).
  Shootout gains --zeta sweep and --split (validation-first protocol);
  runner gains --dps-zeta/--dps-steps.

## v0.9.7 — 2026-09-06
- experiments/dps_shootout.py: evaluates DPS from each diffusion candidate
  (tiny/L/XL) on held-out slices (PSNR/MSSIM/runtime) and names the winner
  for the benchmark runner. Executed on local checkpoints.

## v0.9.6 — 2026-09-06
- LoDoPaB tag-format unification (campaign "fN" vs runner "N" broke paired
  Wilcoxon for learned rows -> n/a): runner now emits f-tags; make_tables
  canonicalizes legacy digit tags. Killer-tested on a mixed-tag CSV
  (pairs restored, *** verdict).

## v0.9.5 — 2026-09-06
- Table generator hardened (double-audit): consistent dedup with warning,
  safe Wilcoxon (no crash on zero/degenerate diffs), slice-level paired
  test column (within-slice-correlation armor), partial-pair handling —
  all verified against adversarial known-truth CSVs.

## v0.9.4 — 2026-09-06
- experiments/make_tables.py: JBHI summary tables from benchmark CSVs —
  mean±std PSNR/MSSIM, median runtime, PAIRED Wilcoxon (A-FONF vs each,
  slice x seed pairs), significance stars; killer-tested on known-truth
  synthetic data (+2 dB -> p~1e-11, equal -> ns, NaN rows skipped);
  output name-guard (never overwrites input).

## v0.9.3 — 2026-09-06
- --init-from warm-start for the diffusion trainer (net+EMA, mismatch
  guard, EMA caveat documented); mechanism-verified. Incident rule logged:
  never act on assumed process state.

## v0.9.2 — 2026-09-06
- XL long-job GO-audit: sustained 30-epoch big-arch run, production-config
  (ch=128 @ 256^2) train-step, 256^2 DPS sampling, NaN-injection guard test
  (EMA-protecting fail-fast added to the diffusion trainer). 32/32.
- Ops rules logged: no-resume overwrite hazard; running-job nan-grep check.

## v0.9.1 — 2026-09-06
- RED-CNN trainer instrumentation: grad-norm / scaler-scale / skip-count /
  val-PSNR per epoch, non-finite fail-fast, grad clip, frozen-loss detector
  (sabotage-validated; detector tolerance bug caught by its own test).
  redcnn_3000 from the frozen run is quarantined pending diagnostic retrain.

## v0.9.0 — 2026-09-06
- Diffusion-XL baseline: BigUNet (4-scale residual score model, 42.7M @
  ch=128), --arch flag through trainer/checkpoint/DPS; executed battery
  (training, periodic checkpoint round-trip, shapes/backward); 32/32.

## v0.8.9 — 2026-09-06
- Pre-launch audit: unified _save_ckpt for periodic + final diffusion
  checkpoints (periodic block had hardcoded ch=64 from an interrupted,
  unexecuted turn — would have broken L-model restarts). Executed:
  periodic-file DPS round-trip, ch=128 L-config train + round-trip, 32/32.

## v0.8.8 — 2026-09-06
- Diffusion trainer: --ch width flag; checkpoint now persists the ACTUAL
  width (was hardcoded 64 — would have crashed DPS on any non-default
  model). Executed: ch=32 train -> checkpoint -> DPS load+sample round-trip.

## v0.8.8 — 2026-09-05
- Diffusion trainer: periodic checkpointing (--save-every, default 25
  epochs) for Jupyter-managed / restartable containers; executed-verified.

## v0.8.7 — 2026-09-05
- Bare-DGX hardening: npz training mode now requires ONLY torch + numpy
  (package import moved to root-mode branch). Proven by executing the
  trainer with scipy/skimage/pydicom/h5py/astra imports blocked.

## v0.8.6 — 2026-09-05
- Release audit of the DGX kit: exporter executed with EXACT round-trip;
  full export->train->DPS chain executed; zero-batch silent-training trap
  guarded in both trainers; permanent export round-trip suite test (32).

## v0.8.5 — 2026-09-05
- DGX launch kit: npz training mode (--npz) + export_train_npz.py (single
  ~300 MB transfer instead of the DICOM tree); diffusion trainer made
  device-agnostic. Executed in-container end-to-end: training (falling loss)
  -> checkpoint -> DPS round-trip (finite samples).

## v0.8.4 — 2026-09-05
- LPD dead-output fix: removed the training-time output ReLU (not part of
  the original LPD; it zeroed the output at init and killed all gradients —
  loss frozen at mean(ref^2)); positivity now enforced at inference.
  Dynamics test hardened against degenerate outputs.

## v0.8.3 — 2026-09-05
- DPS fix: model width persisted in diffusion checkpoints ("ch") and
  respected by dps_infer (root cause of the 30/31 suite failure).
- Campaign workers pin BLAS threads to 1 (OMP/MKL/OPENBLAS/NUMEXPR) —
  removes oversubscription thrash observed at 10 workers.

## v0.8.2 — 2026-09-05
- GradScaler version shim (torch 2.x torch.amp / torch 1.x torch.cuda.amp)
  in all trainers — future-proofs DGX and any legacy-env runs.

## v0.8.1 — 2026-09-05
- Campaign fault isolation (per-method try/except + finite checks, NaN rows
  logged loudly, run continues) — live-tested via NaN sabotage.
- End-to-end campaign verified on synthetic Mayo-DICOM/LoDoPaB trees;
  extreme-noise stress (inc=60, both geometries, 7 methods) all finite.
- Release zips exclude results CSVs (resume key-collision safeguard).

## v0.8.0 — 2026-09-05
- run_campaign.py: overnight multi-stage classical campaign — process-pool
  parallel cells, crash-proof per-row CSV flush, resume-skip of completed
  cells, staged mayo/lodopab/phantom x parallel/fan grids, 10-seed default,
  supplement tier default-on, --select-iters control.

## v0.7.4 — 2026-09-05
- Benchmark runner: --with-supplement flag exposes MLEM+FuzzyAD for the
  supplement-tier pass (was implemented but not wired to the grid).

## v0.7.3 — 2026-09-05
- LPD NaN fix: spectral-norm operator scaling (power-iteration sigma;
  network sees A/sigma, g/sigma; sigma persisted in checkpoint and applied
  at inference), gradient clipping, fail-fast non-finite guard.
- Suite: 31 checks — new torch-guarded LPD training-dynamics test
  (finite + decreasing loss under the exact training recipe).

## v0.7.2 — 2026-09-05
- Manual-gradient certified projector (custom autograd.Function; backward =
  exact transpose) — fixes DPS on real torch and pre-empts the same failure
  in LPD training; gradient-exactness test added to the suite.
- RED-CNN training volume raised to published scale (2500 slices x 16
  patches, 60 epochs) with --patches-per-slice; first 1.6k-patch checkpoint
  deprecated for reporting.

## v0.7.1 — 2026-09-05
- Pre-experiment audit fixes (5): AMP-safe fp32 sparse projector ops;
  TinyUNet moved to models.py (eval-importable); benchmark --append +
  --methods for true progressive fill; DPS sampler implemented
  (certified-projector likelihood guidance) with guarded mechanics test;
  per-incident checkpoint scheme {method}_{incident}.pt across trainers
  and runner. Suite: 30 checks.

## v0.7.0 — 2026-09-05
- M3 Delivery-1: FBP baseline (parallel + fan, certified-adjoint
  backprojector, frozen scale); torch-sparse port of the certified projector;
  RED-CNN / LPD / DIP models and training scripts (published configs, AMP,
  seed-locked, state_dict checkpoints); DGX diffusion training scaffold
  (cosine schedule, EMA, bring-back checklist); unified benchmark-grid
  runner with progressive learned-method fill and --quick smoke.
- Test battery: 29 checks (FBP parallel/fan sanity; torch-guarded certified
  adjoint and model-shape validation).

## v0.6.0 — 2026-09-05
- Algorithm 1' FROZEN: `pipeline.reconstruct` — fully automatic (Morozov
  alpha + persistence notch detection + gamma=0.9 damped core), zero run-time
  hand-tuning; warm-start `f0` added to the damped solver.
- ACQUISITION-BAND GUARD (physics fix discovered by end-to-end testing):
  in-loop notch detection restricted to the detector-Nyquist radius
  (r <= n_bins/2 - 2); kills parallel-beam aliasing false positives that are
  globally coherent and alpha-invariant. `select_notches_v2(..., r_max=)`.
- End-to-end battery 7/7: clean K=0 across contents x doses x geometries;
  object-domain stripe detected from projections at 0.23 px; beyond-band
  components correctly ignored.
- Test battery: 27 checks (pipeline determinism/report contract, in-band
  detection, band-guard rejection).

## v0.5.0 — 2026-09-05
- A-FONF (adaptive fractional order) SHIPPED: `alpha_morozov` — Poisson-
  discrepancy (Morozov) selection, content-adaptive, training-free; validated
  across 7 content x dose configs (4 held-out), worst mean oracle gap
  0.142 +/- 0.009 dB. Fast heuristic `alpha_from_counts` retained
  (known-physics logistic; bracket/initialization role).
- `poisson_sinogram(..., return_counts=True)` exposes raw Poisson counts.
- Scientific finding logged: PSNR-optimal alpha saturates at the integer
  bound under photon starvation; fractional/interior optima only in
  photon-rich, near-flat regimes. Full three-act ablation in THEORY_LOG.
- Test battery: 24 checks (adaptive-rule determinism, noise-proxy
  monotonicity added).

## v0.4.0 — 2026-09-05
- Fan-beam support (JBHI clinical-geometry axis): `build_fanbeam_matrix`
  (flat-detector, isocenter-equispaced bins, EXACT-adjoint certified backend;
  sod -> inf recovers the parallel matrix bin-for-bin) and `make_astra_fan`
  (ASTRA OpTomo wrapper, GPU scale backend; unmatched-adjoint caveat noted).
- Dual-backend policy documented: sparse = certification-grade, ASTRA = scale.
- Test battery extended to 22 checks: fan adjoint, fan->parallel limit
  (corr > 0.999 at sod = 2e5), fan-beam MLEM/FONF sanity, astra import-guard.

## v0.3.0 — 2026-09-05
- Robust notch selection v2: sectoral-MAD background + spatial-coherence gate
  (clean K=0 on 10/10 proxy slices with wide margin; stripe TP 10/10 at 0.26 px,
  including half-amplitude). `select_notches_v2`, `select_notches_persistent`.
- Damped Algorithm 1' (`mlem_fonf_damped`, gamma-relaxed MLEM step) + objective
  monitors (`kl_data`, `spectral_potential`); G1 equivalence/monotonicity study.
- Full-data gate script `experiments/run_notch_gate.py` (L506 1mm test set),
  `--limit` smoke flag, version banner in all experiment outputs.
- Killer test battery `tests/test_all.py` (18 checks: adjoint, H in [0,1],
  conjugate symmetry, nonexpansiveness, EM monotonicity, gamma=1 regression,
  synthetic coherence-gate separation, seed-pinned headline locks).

## v0.2.0 — 2026-09-05
- Data layer: Mayo LDCT DICOM loader (1mm-series preference, locked HU->mu
  preprocessing, patient splits) and LoDoPaB HDF5 loader (basename dedup);
  `experiments/verify_datasets.py` inventory with thickness/duplicate reports.

## v0.1.0 — 2026-09-04
- SPL reference implementation: sparse parallel-beam projector, FONF (Eq. 4),
  Sec. II-C notch rule, Algorithm 1, five training-free baselines, metrics,
  phantoms, four experiment scripts, Mayo real-slice runner.
