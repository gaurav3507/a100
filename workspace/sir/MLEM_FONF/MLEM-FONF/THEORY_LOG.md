# Research log — JBHI extension

## 2026-09-05 · Gate G1 (computational half): damped vs exact MLEM+FONF
Setup: gamma-relaxed multiplicative step f <- f * ratio^gamma (mirror-descent
step size on Poisson KL), gamma in {1.0, 0.9, 0.8, 0.5}; FONF relaxation
unchanged (lam_dt = 0.95). Monitors: PSNR trajectory, fixed-point residual,
objective F(f) = KL(g, Af) + c*psi(f), psi(f) = 0.5 f^T (I - A_s) f (Parseval).

Results (fig: results/theory_damped_equivalence.png):
- Mayo slice (inc=3000, alpha=1.2): gamma=0.9 max trajectory gap vs exact =
  0.029 dB (gate <= 0.05 PASSED); objective monotone for all gamma.
- Shepp-Logan (inc=60, alpha=0.5): damping IMPROVES final PSNR
  (18.98 -> 19.14 -> 19.31 -> 19.96 for gamma 1 -> 0.5); exact gamma=1
  objective is non-monotone (overshoot), all damped gammas monotone.
- Fixed-point residuals decay identically (geometric-like) for all gamma.

Verdict: G1 GREEN, upgraded. Decision (locked): journal Algorithm 1' uses
gamma = 0.9 by default; Theorem 1 targets gamma < 1 in the Bregman/BPG frame
(observed monotone descent of KL + c*psi is precisely the sufficient-decrease
signature the proof needs); exact-vs-damped becomes an ablation where damped
never loses. Next theory task: write the BPG sufficient-decrease lemma with
psi as the explicit convex spectral potential and the gamma-step as the
KL-Bregman step size.

Next session targets: (1) robust-notch v2 (sectoral median + MAD + persistence),
gate = K=0 on all 10 Mayo slices, K=2 on injected stripe; (2) A-FONF alpha-rule
derivation using existing oracle sweeps.

## 2026-09-05 · Robust-notch v2 (P3) — design + container calibration
v1 rule on real anatomy: K = 8 false positives per slice (isotropic-mean
background broken by anisotropic spectra). v2 design, in three steps:
1. Sectoral MEDIAN background + MAD threshold (kappa): FP 8 -> ~0.7/slice.
2. Persistence across perturbations: insufficient alone — three adjacent
   L109 slices share a REAL stationary component at (+10,+24) px.
3. Band-pass diagnosis showed that component is LOCALIZED anatomical texture
   (bowel wave-packets), not a global artifact -> added the
   SPATIAL-COHERENCE GATE: one-sided band-pass envelope coverage >= 0.35
   (global stripes cover the field; texture packets do not).
Container calibration (10 Mayo proxy slices): clean K = 0 on 10/10 across
coverage 0.25–0.45 (wide margin); stripe TP 10/10 at 0.26 px, and 10/10 even
at HALF amplitude. Locked: kappa = 10, coverage_min = 0.35, r0 = 8,
sectors = 16. Persistence retained as optional in-loop belt-and-braces.
Full-data gate: experiments/run_notch_gate.py on Mayo L506 1mm test set
(target: K=0 all slices, TP amp 0.10 = 100%). Awaiting ant-pc run.

## 2026-09-05 · GATE G2a: CLOSED (full-data, ant-pc run, v0.3.0)
Mayo L506 1mm full-dose test partition, every 4th slice = 263 slices:
  clean false activations: 0/263 (100% specificity;
    one-sided 95% Clopper-Pearson upper bound on FP rate: < 1.14%)
  injected-stripe sensitivity: 263/263 at amplitude 0.10 AND 0.05
  runtime: 68 s for the full set (~0.26 s/slice, CPU)
v1 -> v2 improvement: ~8 false notches/slice -> 0 across the entire test set.
Paper-ready claim (P3 detection half): "On the complete held-out patient
(263 slices), the robust rule produced zero false activations while
detecting synthetic narrowband artifacts with 100% sensitivity down to half
the nominal amplitude." Correction-on-real-rings half pending (TomoBank /
simulated detector-gain data, milestone M2 branch).

NEXT (locked): A-FONF alpha rule — derive closed-form alpha(noise level,
spectral decay) and validate against existing oracle sweeps (phantom + Mayo
at incident 60/600/3000); gate: |alpha_rule - alpha_oracle| <= 0.1 on all
validation points. Then freeze Algorithm 1' and open the benchmark grid.

## 2026-09-05 · A-FONF milestone: CLOSED (rule = Poisson-discrepancy selection)
Three-act ablation (all falsifications kept for the paper):
1. Spectral crossover rule (burn-in PSD plateau): direction INVERTED by
   convergence-speed confound at short burn-in; at 200-iter burn-in the
   plateau model is broken by streak (not white) noise. Falsified.
2. Counts-logistic rule (known-physics, zero estimation): 6/7 configs within
   0.024 dB incl. three held-out doses, but phantom600 exposed the counts x
   content interaction (same counts as mayo600, opposite optimum): 0.856 dB.
   Retained as fast heuristic only (alpha_from_counts).
3. FINAL: Morozov / Poisson-deviance selection (tau = 1 per ray, largest
   noise-consistent alpha on {0.2..2.0}): content-adaptive by construction.
   Seven configs (2 contents x 5 doses, 4 held-out): five gaps ~ 0;
   3-seed stats on the two nontrivial configs: phantom600 0.142 +/- 0.009 dB,
   mayo3000 0.075 +/- 0.030 dB. GATE (mean <= 0.15 dB) PASSED.
Key scientific finding for the paper: the PSNR-optimal order SATURATES at the
integer bound (alpha = 2) under photon starvation; the fractional regime and
interior optima appear only in photon-rich acquisitions where the objective
is nearly flat (<= 0.16 dB across the full range). Final operating points to
be re-examined under task-based CHO metrics in milestone M4 (pre-declared).
Algorithm 1-prime is now fully specified: gamma = 0.9 damped MLEM step +
Morozov-selected alpha + robust-notch v2. Next: freeze + end-to-end run.

## 2026-09-05 · Algorithm 1' FROZEN (pipeline.reconstruct) + end-to-end battery
Integration: gamma=0.9 damped step + Morozov alpha + notch-v2 persistence,
zero run-time hand-tuning. End-to-end testing exposed a genuine in-loop gap
component tests could not see: parallel-beam ALIASING/streak components on
reconstructions are globally coherent AND alpha-invariant (6/6 grid votes),
defeating both the coherence gate and ensemble persistence. Diagnostic showed
every false detection at r >= 50 px while the 64-bin acquisition's measurable
band ends at the detector Nyquist radius r = n_bins/2 = 32 px. FIX (physics,
not tuning): ACQUISITION-BAND GUARD — in-loop detection restricted to
r <= n_bins/2 - 2; frequencies beyond the measured band carried by an iterate
are inversion artifacts by construction.
Frozen battery (band-guarded), 7/7 OK:
  clean phantom/mayo x {60, 3000} x {parallel, fan}: K = 0 everywhere,
  PSNR >= milestone values (damped bonus retained; fan mayo3000 = 26.73);
  stripe IN THE OBJECT, detected from projections alone: K = 1 exactly,
  localization 0.23 px (stronger claim than the SPL image-domain demo);
  stripe beyond the measurable band: correctly ignored (K = 0) — the
  sampling-limit statement for the paper.
Suite: 27/27. Next: benchmark grid (M3) with the frozen pipeline as the
single FONF entry; learned baselines on the GPU box.

## 2026-09-05 · First Algorithm 1' run on REAL DICOM (ant-pc, v0.6.0)
Mayo L506 slice #0 (1mm full-dose DICOM via the locked loader), inc = 3000:
  auto alpha* = 1.2 (deviances 0.55/0.66/0.78/0.93/1.08/1.23 — monotone,
  tau = 1 crossed exactly between 1.2 and 1.6, selection as designed),
  K = 0 (band guard confirmed on real anatomy), PSNR = 27.04 dB
  (above all PNG-proxy results), runtime 67 s CPU, zero hand-tuning.
Frozen pipeline validated on real clinical data. M3 (benchmark grid +
learned baselines) is now unblocked.

## 2026-09-05 · Baseline audit (pre-M3): lineup LOCKED
Evidence scan (Eulig Med.Phys. 2024 benchmark; DenoMamba/TED-net/PPORLD
2024-25 comparison tables) against the roadmap's provisional 5+3 lineup.
Two gaps fixed: (1) FBP added (mandatory analytic reference, was missing);
(2) DIP(+TV) added as the training-free NEURAL comparator (Baguer 2020,
LoDoPaB baseline) — thesis-critical, closes the "training-free NNs exist"
reviewer attack. UNet-post-processing slot replaced by RED-CNN (Mayo
standard; Eulig 2024 shows it beats many newer nets — citation armor).
FuzzyAD demoted to supplement. GANs excluded with stated reason; optional
transformer slot behind a pre-decided feasibility gate. Final mixture:
5 classical (+1 suppl.) + 1 training-free-neural + 2 supervised + 1
generative (+1 optional) = 9-11 methods. See BASELINES.md.

## 2026-09-05 · Compute-allocation decision (pre-M3, LOCKED)
Dual-machine training, single-machine evaluation:
- DGX A100 (3-4 days): diffusion score model; optional transformer if the
  BASELINES.md gate opens. Deliverables back: state_dict checkpoints, seeds,
  loss curves, env export (env_dgx.yml), training-script git hash.
- RTX A4000 (ant-pc): RED-CNN, LPD, DIP (eval-time), all classical grids.
Rules: (1) ALL evaluation + ALL reported inference times on the A4000
workstation only; (2) published hyperparameters per baseline regardless of
hardware (grad-accumulation if memory-forced, documented); (3) identical
locked Mayo split everywhere; (4) allocation fixed now, not post-hoc.
Experimental-setup wording drafted for the paper.

## 2026-09-05 · A4000 16GB VRAM sizing (verified)
RED-CNN train ~2-3GB (published patch recipe) | LPD train ~3-5GB @ batch 4-8,
using a torch-sparse port of OUR certified projector (differentiable + exact
adjoint, no extra deps) | DIP eval ~2-4GB | diffusion INFERENCE (DGX-trained
checkpoints) ~5GB batch-1 | classical grids CPU. AMP default-on in all
training scripts. Confirms allocation: only diffusion/transformer TRAINING
(20-40GB class) requires the DGX. No memory-forced hyperparameter deviations
expected on the A4000.

## 2026-09-05 · M3 Delivery-1 (v0.7.0): FBP, torch stack, training scripts, benchmark runner
- FBP baseline shipped (Ram-Lak+Hann, fan cosine-weighting, one frozen scale
  constant): dense-view sanity 24.8/24.6 dB (par/fan), sparse 64x90 noiseless
  18.9 dB — the ill-posedness anchor for all tables.
- torch_ops.to_torch_projector: the SAME certified sparse matrix as torch CSR
  (forward + transpose), autograd-ready; suite verifies torch-side adjoint.
- training/: models.py (RED-CNN per Chen'17; LPD 10-unroll over the certified
  projector; DIPNet), train_redcnn.py (FBP-lowdose -> GT, 55x55 patches,
  batch 128, AMP), train_lpd.py (batch 4, AMP), train_diffusion_dgx.py
  (time-conditioned UNet, cosine schedule, EMA; DGX bring-back checklist in
  docstring), eval_learned.py (redcnn/lpd inference + per-image DIP).
- experiments/run_benchmark.py: unified grid (dataset x geometry x incidents
  x seeds x methods), learned methods auto-skip until checkpoints exist
  (progressive fill), --quick smoke. Smoke result (phantom, inc=3000):
  FBP 15.12 / MLEM 19.86 / TV 20.38 / AD 21.11 / PnP 20.05 / A-FONF 24.80.
- Suite: 29/29 (FBP sanity; torch-guarded adjoint + model-shape checks —
  live validation lands on ant-pc where torch exists).
Open decision (when training starts): single-incident checkpoints (3000)
vs per-dose — resolve at kick-off, both supported by scripts' --incident.

## 2026-09-05 · Pre-experiment killer audit (v0.7.1): 5 catches, all fixed
1. AMP-crasher: sparse mm would receive fp16 under autocast in LPD/DIP ->
   projector closures now force fp32 with autocast disabled.
2. TinyUNet was defined inside the trainer's __main__ (uninstantiable at
   eval) -> moved to models.py; trainer imports it.
3. Benchmark CSV opened in "w" every run, destroying progressive fill ->
   --append mode with header guard + --methods filter for learned-only passes.
4. Diffusion checkpoint check existed with NO sampler -> dps_infer
   implemented (likelihood guidance through the certified projector);
   torch-guarded mechanics test added (runs live on ant-pc).
5. Dose-mismatch design gap locked: PER-INCIDENT checkpoints
   {method}_{incident}.pt; trainers auto-name; runner per-incident lookup
   with generic fallback. Plan: RED-CNN at {60,300,1500,3000}; LPD at
   {60,3000}; diffusion dose-agnostic (trained on clean slices).
Suite: 30/30 (DPS mechanics guarded here, live on the GPU box).

## 2026-09-05 · Live-suite catch on ant-pc (v0.7.2): sparse-CSR backward
DPS mechanics test FAILED on the GPU box (torch present): torch's sparse-CSR
mm lacks reliable autograd backward — would also have killed LPD training
tonight (its backward passes through the same op; the shape-only test could
not see it). FIX: custom autograd.Function with MANUAL gradients via the
exact certified transpose (d/dx[Ax] = A^T; both directions), fp32 +
autocast-off retained. Suite hardened: gradient-exactness check
(d/dx sum(Ax) == A^T 1 vs scipy, atol 1e-3) now guards this class forever.
Also: first RED-CNN run converged cleanly (mse .0798 -> .0027) but on only
1,600 patches (~80 s) — an UNDER-TRAINED baseline is unfair-to-baseline;
defaults raised to 2,500 slices x 16 patches (~40k patches, 60 epochs,
~15-20 min/dose) for published-scale training volume. redcnn_3000 to be
retrained under the new defaults before any reported numbers.

## 2026-09-05 · LPD NaN on ant-pc (v0.7.3): operator-scale explosion
Raw line-integral sinograms (magnitude ~60-80) + A^T gain (~200x) exploded
the dual/primal blocks within the first batches; fp16 accelerated the
overflow; every epoch printed mse=nan (checkpoint lpd_3000 from that run is
garbage and is overwritten on retrain). ROOT FIX (Adler-Oektem practice):
spectral normalization — sigma = ||A||_2 by power iteration; the network
sees A/sigma, A^T/sigma, g/sigma (all O(1)); sigma stored in the checkpoint
and applied identically at inference. Added: gradient clipping (1.0),
non-finite-loss guard (fail fast, never save garbage), and a torch-guarded
TRAINING-DYNAMICS suite test (6 steps: finite + decreasing) — the missing
test class that forward-shape checks could not cover. Epoch time 18 s
=> full LPD retrain ~9 min; both dose checkpoints ~20 min total.

## 2026-09-05 · Campaign runner (v0.8.0): overnight JBHI-grade grids
run_campaign.py: process-pool parallel cells (auto workers = cpu-2, cap 10),
per-row CSV flush + fsync (crash-proof; verified live — a killed run's rows
survived and resume skipped them), resume from existing CSVs, staged
dataset x geometry execution, n=10 seeds default, supplement tier default-on,
--select-iters passthrough. Budget @8 workers, per-cell ~90 s (ant-pc):
mayo_parallel (20 sl x 4 inc x 10 seeds = 800 cells) ~2.5 h; lodopab_parallel
~2.5 h; mayo_fan ~2.5 h -> all three grids in one night alongside GPU
training. Single-process this would have been ~60 h.

## 2026-09-05 · Pre-night killer audit (v0.8.1): campaign hardened
1. Per-method fault isolation in the worker: exceptions/non-finite results
   are caught, logged loudly ("!!! FAILED ..."), written as NaN rows, and
   the campaign continues — LIVE-tested by sabotaging MLEM with NaNs
   (contained; healthy methods unaffected).
2. End-to-end campaign executed on synthetic Mayo-DICOM and LoDoPaB trees —
   the exact overnight code path (loader -> pickling -> workers -> CSV),
   both stages green at inc=60.
3. Extreme-noise stress, REAL texture, both geometries, all 7 methods at
   800 iters: all finite. Observed (honest data, not bugs): FuzzyAD and
   fan-MLEM pixel ranges explode to 4e3-6e3 at inc=60 while A-FONF stays in
   [0, 1.9] — a live stability contrast for the paper.
4. Packaging catch: junk benchmark CSVs from container smokes would have
   shipped in the zip and RESUME-SKIPPED real cells (key collision). CSVs
   purged and excluded from release zips henceforth.

## 2026-09-05 · Process upgrade: container now EXECUTES torch paths
Accountability note: several of tonight's bugs (LPD output-relu, TinyUNet
placement, CSV overwrite, DPS width hardcode) were plain authoring errors;
the rest were first-deployment environment boundaries. Root process gap:
torch code was shipped inspected, not executed (no torch in the dev
container). FIXED: CPU torch installed; the ACTUAL training scripts now run
end-to-end here before any release — train_lpd.py: mse 0.121 -> 0.103 ->
0.088 over 3 epochs (finite, decreasing, no dead output); train_redcnn.py
likewise; full suite with all torch tests LIVE in-container. New shipping
rule (locked): no torch-path change ships without an in-container execution
of the exact user-facing command.

## 2026-09-05 · v0.8.6 release audit (DGX launch kit)
Executed, not inspected: export_train_npz.py run against a synthetic DICOM
tree (round-trip vs loader EXACT, max|diff| = 0); full user-chain
export -> npz -> diffusion trainer (falling loss) -> checkpoint -> DPS
sampler (finite). New trap caught + guarded: batch > dataset previously
looped ZERO batches silently and saved an untrained checkpoint — now a
clear assertion in both epoch-loop trainers. Suite: 32 checks incl. a
permanent, self-contained export round-trip test.

## 2026-09-06 · v0.8.9 pre-launch audit (L-diffusion night run)
User-demanded release audit caught a resurrected bug: the periodic-save
block (written in an interrupted turn, never executed) hardcoded "ch": 64 —
the L run (ch=128) would have produced periodic checkpoints that crash DPS
on load, the exact width-bug fixed once before. FIX: single _save_ckpt
helper for periodic + final (identical dict, ch = a.ch, epoch key).
Executed battery: periodic path fires and its file DPS-round-trips (ch=32);
ch=128 L-config trains + round-trips; suite 32/32. Lesson logged: code from
interrupted turns is UNVERIFIED by definition — audit before use.

## 2026-09-06 · v0.9.0: Diffusion-XL (night-scale generative baseline)
User call: the A100 night deserved a real job (tiny finished in ~7 min).
Shipped BigUNet — 4-resolution residual UNet, channel mults (1,2,3,4),
2 res-blocks/level, learned time-embedding; 42.7M params @ ch=128 (~30x
tiny). --arch {tiny,big} wired through trainer, checkpoint (arch+ch keys)
and dps_infer. Executed battery: arch=big training (falling loss),
periodic-file DPS round-trip, 256-shape/backward checks, suite 32/32.
Night plan: XL on the FULL 5,092-slice export, 1200 epochs, batch 64,
periodic saves q25 -> any morning state is a valid checkpoint. Tiny vs XL
compared at eval; winner becomes the official DPS baseline (other ->
supplement). Closes the "toy generative baseline" reviewer attack.

## 2026-09-06 · redcnn_3000 retrain FROZEN (v0.9.1 instrumentation)
Retrain on the full 40k-patch volume froze bit-identical at mse 0.178922
(the 1.6k-patch run had converged) — same zero-update class as the LPD
relu bug; leading suspect: AMP scaler death from a non-finite gradient
somewhere in the larger patch population. Checkpoint redcnn_3000.pt marked
SUSPECT — excluded from any reported table until the diagnostic retrain.
v0.9.1: trainer instrumented (per-epoch grad-norm, scaler scale, skipped
non-finite count, val-PSNR), non-finite-loss fail-fast, grad clip 5.0, and
a FROZEN-LOSS DETECTOR (3 identical epochs at 1e-6 -> abort with
diagnostics). Detector validated by lr=0 sabotage (fires at epoch 3); the
sabotage test itself caught a detector bug (round-9 vs float-order noise ->
tolerance set to 1e-6). Healthy path re-executed; suite 32/32.

## 2026-09-06 · v0.9.2: XL long-job GO-audit (sandbox-executed)
Clarified: BigUNet is a residual-UNet score model (standard DPS class), NOT
a transformer; the transformer denoiser remains a separate un-opened gate
(offline DGX; Eulig-2024-justified omission).
Battery (all executed): B1 sustained 30-epoch big-arch run — falling loss,
periodic saves firing; B2 TRUE production config (ch=128, arch=big, 256^2)
fwd+bwd+opt step finite (15.5 s CPU => ~0.1-0.3 s A100-fp16, matches the
6-9 h ETA); B3 256^2 DPS sample from a big checkpoint — finite, [0,1];
B4 NaN-injection — new EMA-protecting guard aborts BEFORE any corrupted
periodic save. Suite 32/32.
Operational precautions (logged as rules): (1) trainer has NO resume —
relaunching over an existing --out retrains from scratch and will overwrite
a good periodic checkpoint after 25 epochs; ALWAYS move/rename the old
checkpoint before any relaunch. (2) The running XL job carries v0.9.0 code
(pre-guard): morning check = grep -i nan on its log; nan-free + falling
loss => keep; else kill, move ckpt, relaunch on v0.9.2.
Honest limit: CUDA-AMP numerics can't be sandbox-tested on CPU; mitigations
= guard + q25 periodic + morning log-grep.

## 2026-09-06 · Incident + fix (v0.9.3): XL murdered by my pkill; rescued
INCIDENT: the "premature" XL launch had in fact SURVIVED (206 epochs,
loss 0.0117, periodic-saved at ep200). My relaunch block assumed it dead
and pkill'ed it — killing a healthy 5 h run; the armed auto-launcher would
then have overwritten the ep200 checkpoint after 25 fresh epochs. Rescue
protocol issued (kill launcher + copy ckpt); root process error: ACTED ON
AN ASSUMED STATE WITHOUT CHECKING (nvidia-smi/log would have shown it
alive). Rule added: never kill/overwrite based on assumed process state.
FIX: --init-from warm-start (net+EMA from ckpt, arch/ch mismatch guard).
Verification story: loss-based test "failed" correctly, exposing the EMA
few-steps subtlety (24-step EMA is ~97% init) — mechanism-level test added
(ckpt weights provably present after warm start); real rescue ckpt has
~32k steps => EMA fully converged, warm start = true model resume
(optimizer state fresh, acceptable). Suite 32/32.

## 2026-09-06 · RED-CNN freeze ROOT-CAUSED + quarantine lifted
Instrumented retrain (v0.9.1+) on the full 40k-patch volume: mse 0.00126,
valPSNR climbing to ~26.9 dB, gnorm 0.078->0.052 — healthy convergence.
Smoking gun in the telemetry: scaler scale ~2^21 with skip-count 4/18,780
steps => rare non-finite gradients from extreme patches. In the original
un-clipped run this inf-cascade collapsed the AMP scale -> fp16 updates
underflowed to zero -> the bit-identical 0.178922 freeze. With
unscale+clip(5.0), rare bad steps are skipped and training proceeds.
redcnn_3000.pt (fresh) is REAL — quarantine lifted. Remaining doses
{60, 300, 1500} to train (~50 min each) with identical telemetry.

## 2026-09-06 · v0.9.5: table-generator double-audit (paper-number safety)
Second adversarial battery (user-demanded) on make_tables: (T1) partial
missing pairs — Wilcoxon on the common subset, n reported (75/90) ✓;
(T2) duplicate rows — loud warn + last-wins dedup applied CONSISTENTLY to
means and tests (was: means counted dups, test did not — silent-wrong-number
class, fixed) ✓; (T3) all-zero diffs — n/a, no scipy crash ✓; (T4) NEW
slice-level Wilcoxon column (seeds averaged within slice first, n=slices) —
answers the within-slice-correlation reviewer critique alongside seed-pair
tests ✓; (T5) known-truth verdicts intact (+2 dB -> ***, equal -> ns) ✓.
Third test-harness bug of the campaign caught by its own run (startswith
"MLEM" matched the version banner). Suite 32/32.

## 2026-09-06 · TABLE-1 (Mayo parallel, 800 cells x 7 methods, 0 failures)
Headline: A-FONF wins outright at inc=1500 (+0.92 dB) and inc=3000
(+1.24 dB) over the best classical baseline, *** at BOTH seed-pair (n=200)
and slice-level (n=20) Wilcoxon, with the best MSSIM — zero hand-tuning.
inc=300: AD/FuzzyAD +0.1-0.2 dB but slice-level ns (statistical tie).
inc=60 (photon starvation): TV/PnP lead (22.5 vs 20.6) — honest
regime-dependence for the paper. STABILITY finding: at inc=60 the
diffusion-type priors show catastrophic spread (AD +/-10.5, FuzzyAD
+/-12.8 dB) while A-FONF stays +/-1.37 — a worst-case-robustness table is
planned (catastrophic-cell counts). Saturated p-values (1.4e-34 / 1.9e-06)
are the Wilcoxon complete-separation floors, not artifacts. Runtime:
A-FONF 87.7 s median (Morozov stage dominates; one-time per acquisition,
reported honestly). Next: LoDoPaB + fan tables on stage completion;
learned tier joins after checkpoint pack-in.

## 2026-09-06 · Robustness finding (inc=60, 200 cells/method)
Catastrophic cells (PSNR < 10): FuzzyAD 16, AD 9, TV 4, PnP 3 — versus
A-FONF 0, and also 0 for unregularized FBP/MLEM (uniformly mediocre, never
collapsed). Reading: catastrophic failure is a pathology of NONLINEAR
SPATIAL priors under photon starvation; A-FONF delivers regularization
without it. Theory link for the paper: the spectral step is provably
nonexpansive (A_s 1-Lipschitz, Thm-1 machinery; suite-verified property)
and cannot amplify, whereas state-dependent diffusion coefficients can run
away — the theorem predicted the failure taxonomy that 200 real-data cells
confirmed. Planned: robustness table (catastrophic %, min PSNR, median)
per method x dose, generated once all three stage CSVs are sealed.

## 2026-09-06 · TABLE-2 (LoDoPaB parallel, 800 cells x 7): REPLICATION
The Mayo regime-structure replicates on a fully different dataset with
zero re-tuning: inc=60 TV/PnP-regime (22.8 vs 18.7); inc=300 AD-crossover
(~1.0 dB, TV tied ns); inc=1500 A-FONF #1 (+0.41, *** seed / * slice);
inc=3000 A-FONF #1 (+0.77, *** both). Stability taxonomy replicates too
(FuzzyAD +/-10.8, AD +/-8.2 vs A-FONF +/-2.6 at starvation). Honest
nuance: AD-family MSSIM marginally higher at 1500/3000 while A-FONF leads
PSNR — both metrics reported side by side. Cross-dataset consistency =
the paper's generalization paragraph.

## 2026-09-06 · TABLE-1 with learned tier (9 methods, 6800 rows)
In-distribution supervised ceiling: RED-CNN 24.87/26.45/27.75/28.12 across
doses (+1.5-1.9 dB over A-FONF at clinical doses; +4.2 at starvation where
LPD's per-dose training also excels, 24.83). A-FONF: best TRAINING-FREE
method at 1500/3000 (overall #2 at 1500), within 1.6 dB of the supervised
ceiling with zero training data/re-tuning; all classical baselines below
it with *** at both test levels. Honest runtime trade-off recorded
(learned inference 0.06-0.38 s GPU vs A-FONF 87.7 s CPU one-time
selection). Decision experiment now in flight: Mayo-trained RED-CNN/LPD on
LoDoPaB (OOD) vs A-FONF's already-replicated 26.19/26.73.

## 2026-09-06 · THESIS RESULT (Table-2 final, 6800 rows, all pairs valid)
The order INVERTS under domain shift. In-distribution (Mayo):
RED-CNN 28.12 > LPD 26.96 > A-FONF 26.57. OOD (LoDoPaB):
A-FONF 26.73 > LPD 23.25 > RED-CNN 21.17 — supervised post-processing
loses 6.95 dB and lands 5.56 dB BELOW A-FONF (*** at both test levels).
A-FONF is numerically invariant across datasets (26.57<->26.73;
26.19<->26.19 at inc=1500 — identical to 2 decimals). Degradation
taxonomy matches the physics-content of each architecture:
post-processing (-6..-7 dB) > unrolled LPD (-3.5..-3.7) > training-free
(~0). Honest exceptions recorded: OOD@60 LPD (21.28) > A-FONF (18.73);
RED-CNN@60 slice-level ns. Experimental core (classical + supervised
tiers, two datasets, 13,600 paired rows) is now COMPLETE pending: DPS
tier (DGX trio), fan-geometry table (campaign in flight), figures.

## 2026-09-07 · DPS sampler bug (v0.9.8): guidance effectively zero
First real shootout: tiny/L/XL gave 8.5/5.9/9.4 dB — below FBP, i.e.
unconditional prior samples. Root cause in dps_infer: guidance step was
grad/||grad|| * 0.3 — a fixed 0.3 L2 move in a 65k-dim image (~1e-3 per
pixel) = no data coupling. My "mechanics test" (finite/shape) could never
detect a quality bug; lesson: samplers need a QUALITY assertion on a real
score model, not just plumbing checks. FIX: Chung et al. formulation,
x_{t-1} <- x'_{t-1} - zeta * grad ||y - A x0_hat||_2 (gradient of the
residual norm). Protocol hygiene added: zeta swept on the VALIDATION
patient (L109) only, frozen, then the test shootout; runner takes
--dps-zeta/--dps-steps. Models themselves are unaffected (XL loss 0.010).

## 2026-09-07 · v0.9.9: DPS quality oracle (model-free killer test)
Built an exact analytic score model (Gaussian prior, closed-form posterior
mean) so DPS correctness can be asserted WITHOUT a trained network:
prior-mean baseline 7.63 dB; legacy step 7.65 dB (= baseline: the bug is
reproduced); fixed step 8.0/11.4/15.1/17.6 dB for zeta 0.03/0.1/0.3/1.0,
overshoot at 3.0 (11.4) — textbook DPS behaviour, validating the zeta
sweep. Test is now permanent in the suite (33 checks). Rule reinforced:
generative samplers ship only with a quality assertion, never plumbing
checks alone.

## 2026-09-07 · v0.9.10: DPS root cause #2 — abar_T explosion (clip_denoised)
Reproduced locally with a REAL small score model (100 ep, phantoms @64):
even unconditional sampling gave saturated noise (3.6 dB, std 0.497),
zeta-invariant. Cause: cosine schedule abar[T] ~ 1e-15; the first step's
x0 = (x - sqrt(1-abar) eps)/sqrt(abar) amplifies eps error by ~1e7. The
oracle passed because its eps is exact (numerator cancels). FIX: standard
clip_denoised — clamp x0 to [-1,1] (+ sqrt(abar) floor). Verified visually
and numerically: unconditional 10.6 dB (prior sample), zeta=0.1 -> 19.1 dB
clean reconstruction, over-guidance speckle at zeta>=1. A trained-model
fixture test is now permanent (34 checks). Lesson: oracle tests validate
formulas; only a trained model validates numerics.

## 2026-09-07 · DPS baseline established (v0.9.10)
Validation sweep (L109, inc=3000): zeta 0.03/0.1/0.3/1.0 -> 22.2/23.4/
25.9/22.1 dB — clean response curve; zeta=0.3 frozen. Test shootout
(L506, 3 slices, inc=3000): tiny 12.9 / L 13.0 / XL 27.05 dB (MSSIM 0.744)
— XL is the official DPS baseline (same weight class as A-FONF 26.57 and
LPD 26.96); tiny/L kept as a capacity ablation (supplement). 16 s/slice
on the A4000 at 200 steps. Per-dose zeta tuned on validation for baseline
fairness; overnight DPS grid on both datasets (~7 h).

## 2026-09-07 · Fan table + MLEM early-stopping protocol (v0.9.11)
TABLE-3 (Mayo fan, 5600 rows): regime structure IDENTICAL to parallel —
A-FONF #1 at 1500/3000 (26.08/26.42, *** both levels), crossover at 300,
TV/PnP regime at 60; stability gap widens (fan@60: A-FONF +/-1.4 vs
FuzzyAD +/-19, AD +/-15.5). Geometry-agnostic claim supported.
Red flag investigated: unregularized MLEM in fan gives -14 dB (60) /
4.7 dB (300). Hypothesis "FOV-corner sensitivity" REJECTED by data (AT1 as
uniform as parallel; explosion inside the FOV). Trajectories showed the
truth: fan-MLEM PEAKS above parallel (18.6/20.0 dB at 10-25 iters) then
diverges (noise runaway in the more ill-conditioned fan system), while
parallel stays bounded. => protocol issue: unregularized MLEM must be
EARLY-STOPPED (textbook). New protocol: select_mlem_iters.py picks the
iteration on the VALIDATION patient per geometry x dose (frozen in
results/mlem_iters.json; both runners read it); MLEM rows recomputed via
drop_method_rows.py + resume. Regularized methods keep 800 its. The
fixed-budget divergence becomes a STABILITY FIGURE: same MLEM engine, same
800 iterations — alone -14 dB, with the spectral step 20.1 dB (fan, inc=60).
DPS grid: tables were generated with only 10 DPS cells present — job
status to be verified before any DPS reading.

## 2026-09-07 · Figure redesign to TMI/JBHI conventions (v0.9.13)
User critique accepted: std error bars misrepresented bimodal collapse as
uncertainty and crushed the informative band; no annotations; robustness
%-bar broken by absolute threshold; no dedicated shift figure. Redesign:
Fig.1 2x3 PSNR/SSIM vs dose with mean ± 95% CI bands, family-coded
identity (proposed bold red / supervised dashed / classical thin cool),
data-driven delta annotations, panel labels, IEEE widths, 8-pt fonts,
vector PDF. Fig.2 slope chart Mayo -> LoDoPaB per method (the inversion in
one glance; label spreading with leaders; FBP excluded). Fig.3 per-
realization violins + strips at the lowest dose with method-relative
collapse % (> 6 dB below own median). Fig.4 stability trajectories in
both geometries with the early-stop optimum marked. All executed on
real-structured synthetic CSVs and reviewed visually. Pending: Fig.5
qualitative reconstruction grid (ROI zoom + difference maps) — needs the
GPU box (checkpoints).

## 2026-09-07 · Figure visual identity v2 (v0.9.14)
Tier palette: hue encodes tier (crimson = proposed; blue/indigo/violet =
supervised & generative; green/teal/ochre = classical priors; greys =
MLEM/FBP), luminance + marker distinguish members — tier recognition at a
glance. Fig.3 rebuilt as broken-axis violins (bodies >= 8 dB with white
median/IQR boxes; lower strip shows collapsed realizations; collapse share
labelled). Fig.4 adds shaded "stability gap" between MLEM and the spectral
step, early-stop optimum marked, divergence annotated. Reviewed visually
on real-structured synthetic data; final pass on the user's real renders.

## 2026-09-07 · Locked plan + TMI decision run (v0.9.15)
Title locked: "A-FONF: Adaptive Fractional-Order Notch Filtering for Robust
and Self-Tuning Low-Dose CT Reconstruction". Venue: TMI gated on a
realistic-geometry decision run; JBHI fallback without delay. CHO included;
TomoBank included (primary tomo_00072 soft tissue / severe rings,
replication 00076, stress 00071 broad rings; Globus transfer). Full plan in
PAPER_PLAN.md.
run_realistic_geometry.py: operator-closure (matrix-free) version of the
frozen pipeline — Morozov selection, persistence notches with band guard,
damped step — over an ASTRA fan operator at 512^2 / 736 bins / 576 views.
Loader now takes `size` (512 slices). Operator path EXECUTED in-container on
a sparse shim (128^2, 96 bins, 120 views): A-FONF 29.34 vs MLEM 29.03 dB,
finite, alpha selected, band guard active. First observation for the gate:
at near-complete sampling the classical gap narrows sharply (as expected;
the 64x90 advantage is a sparse-view/starvation phenomenon) — the run will
quantify exactly this.

## 2026-09-07 · CHO task-based evaluation implemented (v0.9.16)
src/mlem_fonf/cho.py: Laguerre-Gauss channels (unit-norm, zero-mean),
Gaussian low-contrast lesion, ROI extraction, Hotelling template, d' with
bootstrap CI and AUC. Validation battery (now permanent, 38 checks):
(1) on white noise the CHO tracks the analytic matched-filter d' = ||s||/sigma
    at ratio 0.99 across three noise levels and never exceeds it;
(2) NULL case exposed a real statistical trap — resubstitution gives
    d' = 0.13 where truth is 0; SPLIT-HALF estimation (now the DEFAULT)
    gives -0.02 with a CI covering zero;
(3) monotone in contrast; AUC mapping correct;
(4) end-to-end on MLEM reconstructions: d' = 3.4 / 12.4 / 31.7 at doses
    100 / 1000 / 10000 — dose-monotone as physics requires.
experiments/run_cho.py: parallel SKE/BKS campaign (methods x doses x
contrast x lesion size), per-cell signal-present/absent reconstructions,
split-half d' with bootstrap CI; smoke-executed. Sized for the DGX
(256 cores): the full grid is ~hours there.

## 2026-09-07 · Pre-launch audit of both new runners (v0.9.17)
run_cho.py — three real defects found by audit and FIXED, each measured:
  P1 MEMORY BOMB: tasks carried full 256^2 ref + lesion arrays
     (1.05 MB x 14,400 tasks = 15.1 GB through the pool) -> workers now hold
     slices/lesion in globals; task = indices only (63 bytes, 0.9 MB total).
  P2 SEED COLLISIONS: seed = 7000 + 97*si + 2k + present produced 800 seeds
     with only 491 unique (309 duplicate noise realizations across slices)
     -> disjoint scheme, verified 800/800 unique.
  P3 NO RESUME on a multi-hour job -> per-lesion-block resume from the CSV,
     live-tested (rerun skips completed blocks, new block appends).
  P4 STATISTICS: bootstrap CIs did not bracket the point estimate (single
     random half-split for the estimate vs different splits in replicates)
     -> repeated split-half averaging (n_splits=20) everywhere; verified
     bracketing at N = 20/60/200 and null still unbiased. Small-N warning.
run_realistic_geometry.py — A1 dead FBP dict removed; A2 actionable ASTRA
  error + OPERATOR PREFLIGHT (delta forward peak, A^T 1 range) before
  spending GPU hours; A3 incremental CSV with fsync (2 h run no longer
  all-or-nothing); A4 empty-refs guard + --split. Executed end-to-end
  against a stubbed ASTRA backend: MLEM 22.70 vs A-FONF 23.19 dB, rows
  written incrementally.
Suite: 40 checks (CI-bracketing and task/seed contract added).

## 2026-09-07 · Silent-skip bug: DPS ran at one dose only (v0.9.18)
The overnight DPS grid produced 400 cells instead of 1600, silently:
ckpt_for("diffusion", inc) looked for diffusion_{inc}.pt then diffusion.pt,
while the deployed file is diffusion_3000.pt (the XL winner). Only
inc = 3000 matched; the other doses dropped DPS from the method dict and
the runner reported "+0 rows, 0 errors". TWO fixes: (1) dose-agnostic
checkpoint fallback — the diffusion prior is trained on clean slices and
does not depend on dose, so any diffusion_*.pt is valid at every dose
(logged when used); (2) FAIL-LOUD: an explicitly requested --methods entry
that cannot run now aborts with the checkpoint listing instead of silently
producing nothing. Both executed. Lesson: "0 rows, 0 errors" must never be
a valid outcome of an explicit request.

## 2026-09-07 · CHO campaign died at 9,600/14,400 -> hardened (v0.9.19)
The DGX CHO run vanished with no error and no output (log froze mid-block,
process gone; timing had accelerated near the end — the signature of workers
dying one by one, most likely the OOM killer against the parent's growing
in-RAM ensemble plus 120 worker buffers).
Hardening, all executed:
1. STREAMING CACHE: ROIs are flushed to per-(method, dose, class) .npy files
   in batches of 50; the parent no longer holds the ensemble.
2. CELL-LEVEL RESUME: a JSON index records every completed
   (method, dose, slice, realization, class) cell; a restart recomputes only
   what is missing. Verified: rerun after deleting the CSV recomputed
   NOTHING and reproduced d' bit-identically (3.50 / 3.01); the ensemble
   stayed at 20 instead of doubling (the append bug the test caught).
3. REQUEST-AWARE BLOCK SKIP: enlarging the job (10 -> 30 realizations) now
   extends the ensemble (40 of 60 new cells) instead of silently skipping a
   "finished" block; stale CSV rows for that block are replaced.
4. FAULT ISOLATION per cell (NaN ROI + loud log) and NaN filtering before
   the statistics; maxtasksperchild=200 recycles workers (memory hygiene).

## 2026-09-07 · Operational testing added (v0.9.20) — the missing test class
Accountability: the CHO campaign was shipped with FUNCTIONAL tests (is d'
correct?) but no OPERATIONAL tests (does the process survive 14,400 tasks?).
That gap is what cost the DGX run. Operational testing is now part of the
suite, and it immediately found two more defects:
D1 NON-DETERMINISM: imap_unordered stored ROIs in completion order, and the
   split-half permutation runs over array order -> the SAME seeds gave
   d' = 3.1241 vs 3.1081 across runs. Paper numbers cannot depend on worker
   scheduling. FIX: cell ids (slice, realization) are persisted with every
   flush and the ensemble is lexsorted before statistics.
D2 FLUSH GRANULARITY: cache flushed only every 50 cells per key, so a kill
   12 s in left an EMPTY cache (160/160 recomputed). FIX: size-OR-time
   flush (60 s), verified — a kill at 25 s left 9 cache files and the
   resume recomputed 0 of 160.
Verification now permanent (41 checks): a shrunken campaign runs three
times — fresh, repeat, and hard-kill(SIGKILL)+resume — and all three must
agree to 1e-9 (they do: 3.0729 / 3.0523 every time). Parent peak RSS
measured flat at 124 MB over 960 cells.

## 2026-09-07 · Two negative results that reshape the artifact claim (P3)
(1) CHO block c=0.03, r=6 (large low-contrast lesion, inc=3000):
    MLEM+TV d' = 1.87 [1.70, 2.07] and PnP 1.74 [1.57, 1.92] are
    SIGNIFICANTLY ABOVE A-FONF 1.23 [1.04, 1.40] (= MLEM 1.25, AD 1.20,
    FBP 1.17; non-overlapping CIs). Interpretation: detecting a large,
    smooth, low-contrast disc rewards aggressive low-frequency noise
    suppression — TV's home ground — while our spectral step preserves
    noise texture and resolution. At r = 3 all methods tie (~0.84, and that
    operating point is near chance: d' < 1 with 400 pairs cannot separate
    anything). To be reported honestly; also motivates a task that actually
    probes our contribution.
(2) RING artifacts are NOT handled by the current rule. Multiplicative
    detector-gain drift (random channels, 20-30%; sinusoidal 10-20%) damages
    MLEM and A-FONF equally (23.0 -> 18.7 vs 26.5 -> 18.8 dB) and the notch
    rule returns K = 0. The reason is structural, not a bug: a concentric
    ring has energy spread over ALL orientations (Bessel-type radial
    spectrum), whereas our detector targets narrowband DIRECTIONAL peaks.
    In the SINOGRAM the same defect is a narrowband stripe (measured
    peak/median = 63.5), so a sinogram-domain variant is the natural
    extension — but a naive first attempt did not recover PSNR (17.0 vs
    18.7 uncorrected), and the classic column-gain normalization failed
    outright here (-4.7 dB) because per-channel medians are anatomy
    dependent at 90 views. This needs proper design, not a patch.
DECISION REQUIRED (paper scope): (a) restrict the artifact claim to
narrowband directional/stripe patterns, where we have the 263/263 gate and
the projection-domain detection result, and move rings + TomoBank to future
work; or (b) invest 1-2 days in a designed sinogram-domain ring extension
with its own validation, then keep TomoBank in scope.

## 2026-09-08 · CHO channel bug (v0.9.21) — user-requested formula audit
The user asked for a literature cross-examination of the CHO code. It found
a REAL deviation with a ranking-level consequence.
BUG: laguerre_gauss_channels subtracted each channel's mean. The standard
definition (Gallas & Barrett, JOSA A 2003; Barrett & Myers) is
    u_j(r) = (sqrt2/a) L_j(2 pi r^2/a^2) exp(-pi r^2/a^2)
with NO mean subtraction — the j = 0 channel is a Gaussian. Zero-meaning
the channels makes the observer blind to the ROI mean-shift cue, which is
the dominant cue for a smooth low-contrast lesion.
MEASURED EFFECT (70 pairs of real reconstructions, three channel widths):
  standard channels  MLEM 3.83 | TV 3.72 | A-FONF 3.81  (all tied)
  zero-mean (shipped) MLEM 2.94 | TV 3.14 | A-FONF 2.93 (TV appears best)
i.e. the bug depressed every d' by ~25% AND manufactured the TV advantage
that the DGX campaign reported. The earlier "TV/PnP significantly beat
A-FONF on the detection task" conclusion is therefore INVALID and is
withdrawn; the corrected comparison shows parity at this operating point.
White-noise efficiency with standard channels: 0.99 / 1.05 of the analytic
matched-filter d' (correct scale).
FIXES: standard channels by default (zero_mean kept only as a documented
ablation); channel width now signal-matched (a = 2 x lesion radius);
--stats-only + --cache-dir re-score cached ROIs in seconds, so a scoring
change never again costs a 20-hour campaign; suite test flipped to assert
the literature form pointwise (41 checks).
ACTION: the running DGX campaign (v0.9.17) is scoring-invalid and must be
restarted on v0.9.21, which also writes the ROI cache.

## 2026-09-08 · Process rule added after the CHO channel loss
Cost of the channel bug: ~20 h of DGX compute discarded (two lesion blocks
scored with non-standard channels). Root cause was not the code but the
process: algorithms in this project have always shipped with a mathematical
reference test (projector: adjoint identity; filter: H in [0,1], symmetry,
nonexpansiveness; DPS: analytic oracle), while EVALUATION METRICS shipped
with only self-consistency checks. New rule, effective now:
  Any metric, observer or statistic that produces a number for the paper
  must be validated against a published closed form or reference
  implementation BEFORE it is run at scale — the same standard applied to
  the reconstruction operators.
Mitigations now in place: --stats-only/--cache-dir make any future scoring
change cost seconds instead of a campaign; the suite asserts the LG channel
form pointwise against the Gallas-Barrett expression.

## 2026-09-08 · CHO (corrected channels) — the honest task-based result
Two lesion blocks (c = 0.06; r = 3, 6 px) x 3 doses x 6 methods, 400 pairs
each, standard Laguerre-Gauss channels, split-half d' with bootstrap CIs
(7.7 h on the DGX; ROI cache 256 MB retained for instant re-scoring).
FINDING: at EVERY operating point the 95% CIs of all six methods OVERLAP —
no method is statistically separable on low-contrast detection. Examples:
  r=6, inc=3000: A-FONF 2.89 [2.73,3.19] ~ AD 2.85 ~ TV 2.81 > FBP 2.61
  r=3, inc=3000: PnP 1.53 ~ TV 1.50 ~ FBP 1.32 ~ A-FONF 1.32 ~ MLEM 1.15
  r=3, inc=300 : FBP 0.65 ~ A-FONF 0.64 ~ AD 0.63 ~ TV 0.59 ~ MLEM 0.58
The yesterday "TV/PnP significantly beat A-FONF" conclusion is WITHDRAWN:
it was an artifact of the zero-mean channel bug (v0.9.21 fixed it).
Sanity of the observer is confirmed by the structure: monotone in dose
(0.6 -> 1.0 -> 1.3 and 1.3 -> 2.1 -> 2.9) and in lesion size.
Second, more interesting finding: FBP matches or leads several conditions
(r=3, inc=300 #1; equals A-FONF at inc=3000) despite a ~10 dB PSNR deficit
— task performance does NOT follow the PSNR ranking. Planned paper wording:
report the parity openly, use it to argue that our contribution is
stability / artifact suppression / distribution-shift invariance rather
than low-contrast detectability, and cite this as evidence for why
task-based evaluation belongs in reconstruction papers.
Next: the artifact-present task (--stripe 0.25) where the notch mechanism
is actually exercised.

## 2026-09-08 · TMI GATE: passed at dose-matched realistic geometry (v0.9.23)
First smoke ran at equal photons-PER-RAY, which at 736 x 576 carries 74x the
total flux of the 64 x 90 reference grid — not a low-dose test at all; there
MLEM alone reached 31.7 dB and the advantage was +0.08 dB. Corrected to
DOSE-MATCHED incidents (equal total photons: 41 / 82 / 326 photons per ray)
and added FBP and MLEM+TV comparators over the operator closures.
Result (512^2, 736 bins, 576 views, L506 slice 0):
  inc  41: FBP 20.14 | MLEM 19.01 | TV 28.01 | A-FONF 29.36  (+1.35)
  inc  82: FBP 21.58 | MLEM 21.80 | TV 28.02 | A-FONF 30.02  (+2.00)
  inc 326: FBP 25.21 | MLEM 27.00 | TV 28.08 | A-FONF 30.80  (+2.72)
A-FONF also leads SSIM at 82/326 and is ~2x faster than the TV baseline
(18 s vs 34 s). The advantage is LARGER here than in the sparse-view
setting, i.e. the method is not a sparse-view artefact.
FAIRNESS CAVEAT (being addressed before any claim): TV's PSNR is flat
(28.01/28.02/28.08) across a 8x dose range — a saturation signature showing
its fixed weight (0.05, tuned for 256^2 / 64x90) over-regularizes here.
--tv-weight and --methods were added so the TV baseline can be tuned on the
VALIDATION patient per dose, exactly as DPS (zeta) and RED-CNN (per-dose
weights) were tuned. The gate claim will be made only against tuned TV.

## 2026-09-08 · v0.9.24 — gate audit (user-requested): 3 defects, all real
Honest note on method: formulas here are written from knowledge, not fetched
from papers; the safeguard is that each must be validated against a closed
form or a reference implementation. That safeguard is what this audit ran.
Findings in the matrix-free gate (all would have produced wrong numbers):
D1 WRONG ALGORITHM: the gate damped with a convex combination
   f <- (1-gamma) f + gamma f_EM, while the shipped algorithm (and the
   Bregman/mirror-descent theory) uses the EXPONENT form f_ml = f ratio^gamma.
D2 WRONG SELECTION STAGE: alpha_morozov scores UNDAMPED (gamma = 1) runs;
   the gate scored damped ones -> it selected alpha = 0.2 where the shipped
   pipeline selects 0.5, i.e. a different method was being reported.
D3 ORACLE LEAK: fbp_op fitted its global scale to the TEST slice
   (least squares against the ground truth). Now calibrated once on a
   phantom through the same operator and cached; object-independence is
   asserted in the suite.
Also aligned: the persistence partner now warm-starts from the selected
reconstruction, as pipeline.reconstruct does. After the fixes the operator
gate reproduces the shipped pipeline BIT-IDENTICALLY (max |diff| = 0.0,
same alpha, same K) at two doses. Verified noise model identity as well:
gate and campaign both draw Poisson(proj/max(proj) * incident).
Consequence: the dose-matched gate numbers reported earlier
(A-FONF +1.35/+2.00/+2.72 dB over TV) were produced by the WRONG variant and
must be regenerated on v0.9.24 before any claim; the TV baseline still needs
its validation-tuned weight. Suite: 43 checks.

## 2026-09-08 · Concurrent-campaign hazard (v0.9.25)
Two stripe campaigns were launched against the SAME cache (pgrep showed 130
processes = 2 x 64 workers + 2 parents; the log had been truncated by the
second launch, which then reported "7993 cells already computed"). Two
writers append to the same per-class .npy files, so an ensemble can contain
duplicated realizations — silently inflating n and biasing the statistics.
Two mitigations, both executed:
1. repair_cho_cache.py — audits every cache file for duplicate cells and
   index/array mismatches and, with --fix, rewrites deduplicated, canonically
   sorted arrays and rebuilds the done-index (the per-ROI key files added in
   v0.9.20 for determinism are what make the repair possible). Verified on an
   injected 30-ROI/10-duplicate cache: repaired to 20 unique, sorted.
2. A campaign lock (.campaign.lock with the writer pid, stale-pid aware,
   released via atexit) refuses a second concurrent writer with an
   actionable message; --stats-only is exempt. Live-tested.

## 2026-09-08 · TV tuning sweep at realistic geometry (baseline fairness)
Validation patient (L109), 512^2 / 736 bins / 576 views, dose-matched:
  w      inc=41   inc=82   inc=326
  0.0005  17.25    21.08    27.54
  0.001   18.92    23.34    28.91
  0.002   22.04    26.46    28.96   <- best at 326
  0.003   24.58    27.54    28.55   <- best at 82
  0.005   26.73    27.49    27.83   <- best (so far) at 41
  0.01    26.38    26.53    26.65
  0.02    25.13    25.16    25.21
  0.05    23.88    23.90    24.14   (the value tuned for 256^2/64x90:
                                     4-5 dB below optimum here, and flat
                                     across dose = over-regularized)
Two consequences: (1) the gate will compare against PER-DOSE TUNED TV, so
the claim cannot be dismissed as an untuned baseline; (2) a paper-grade
observation for C1 — TV's optimal weight moves by 4x across dose and by 10x
across geometry, and a mis-set weight costs up to 4 dB, while A-FONF selects
its own alpha with no sweep at all. Runner gained --append/--out so the
per-dose runs accumulate in one CSV.

## 2026-09-08 · Figure editorial pass on REAL data (v0.9.28)
Reviewed the four production renders and fixed what only real values expose:
1. Fig.1 clipped the FBP curve in the OOD panel (a hard y-floor of 12 dB cut
   the 11.5 dB point) -> limits now follow the data.
2. Fig.3 panel labels collided with the (long) panel titles, and the body
   panel's y-range was set from the collapse floor, squashing the violins
   -> labels lifted above the title band; body range now spans the
   above-floor data; the collapsed strip is scaled to the collapsed points.
3. Fig.2 label stack widened (taller canvas, tighter leader spacing).
4. Fig.4 early-stop annotation moved off the curves (axes-fraction anchor).
Content review of the real renders: the story reads correctly at a glance —
learned methods dashed and on top in Mayo, below A-FONF in LoDoPaB; the
crossover at ~300 photons is visible in every panel; SSIM panels show the
honest weakness at the lowest dose; collapse shares (4-17%) sit only on the
nonlinear spatial priors.

## 2026-09-08 · Artifact-present CHO (detector-gain rings): still a tie
c = 0.06, r = 6, 400 pairs, gain drift 25% on 4 of 64 channels:
  inc 3000: TV 3.05 [2.87,3.35] ~ A-FONF 3.04 [2.89,3.33] ~ AD 2.89 ~
            MLEM 2.75 ~ PnP 2.74 ~ FBP 2.71
  inc 1500: TV 2.28 ~ PnP 2.28 ~ A-FONF 2.13 ~ AD 2.09 ~ MLEM 1.99 ~ FBP 1.94
All CIs overlap again. Diagnostic: every method's d' went slightly UP versus
the clean run (FBP 2.61 -> 2.71, A-FONF 2.89 -> 3.04), i.e. the perturbation
did not degrade the task at all. Two reasons, both known: (i) gain drift
produces concentric RINGS, which our notch rule provably does not target
(directional narrowband peaks only — the K = 0 finding); (ii) rings from 4 of
64 channels need not intersect the lesion ROI. The experiment therefore
tests nothing about the mechanism — a design error on my side, not a result
about the method. Honest position for the paper: detection performance is
statistically indistinguishable across methods with AND without
detector-gain artifacts; the contribution is stability, self-tuning and
distribution-shift invariance, and the artifact claim is restricted to
narrowband directional patterns (263/263 gate, projection-domain detection).

## 2026-09-09 · Artifact task redesigned + selection-budget control (v0.9.29)
The ring experiment failed to exercise the mechanism, so the artifact model
was changed to the class the notch rule actually targets and the 263/263
gate validated: a NARROWBAND DIRECTIONAL stripe added to the acquisition,
with frequency (inside the measurable band), orientation and phase redrawn
PER REALIZATION but identical within a signal-present/absent pair — so the
artifact is a background nuisance in the observer's covariance, not a cue.
(Had it been fixed across realizations it would cancel in the
difference-of-means, which is why the previous design was uninformative.)
Mechanism check on real anatomy, 4 realizations: the rule detects the
stripe every time (K = 1) and A-FONF gains +1.3 to +1.5 dB over MLEM.
Efficiency finding while piloting: run_cho left the selection stage at the
shipped default of 800 iterations while reconstructions used 400, so every
A-FONF cell ran ~5,000 iterations — this is why the A-FONF tail dominated
every campaign. Added --select-iters (default unchanged at 800 so previous
results stay comparable); measured 8.5x faster A-FONF cells at 60.

## 2026-09-09 · TMI GATE RESULT: REVIEW -> venue locked to JBHI
Dose-matched realistic fan geometry (512^2, 736 bins, 576 views), 5 slices x
3 seeds, TV tuned per dose on the validation patient:
  41  ph/ray: TV(w=0.0065) 27.65 +/- 0.65 | A-FONF 26.98 +/- 0.69 (-0.67 ***)
  82  ph/ray: TV(w=0.003)  27.74 +/- 0.64 | A-FONF 27.62 +/- 0.64 (-0.12 ***)
  326 ph/ray: TV(w=0.002)  29.09 +/- 0.65 | A-FONF 28.78 +/- 0.65 (-0.31 ***)
  (MLEM 15.0/17.8/23.1 and FBP 14.6/17.1/22.0 are far below both.)
Per the pre-committed rule the advantage does not hold at realistic dense
sampling, so the venue is JBHI and the sparse-view lead is reported as
regime-specific. The same table carries the C1 evidence: the TV baseline
needed a per-dose, per-geometry weight sweep (0.0065 / 0.003 / 0.002; its
256^2 value 0.05 loses 4-5 dB here), while A-FONF selected its own alpha at
every operating point and landed within 0.12-0.67 dB of the tuned baseline
with no tuning at all.
Revised claim set: training-free and self-tuning; invariant under
distribution shift (headline, learned methods lose 2.7-7.0 dB and the
ranking inverts); never catastrophically unstable (0/200 vs 3-16/200);
competitive with per-setting-tuned classical priors and ahead of them in the
sparse-view regime; detectability statistically tied across methods.

## 2026-09-09 · v0.9.30 — THE ALPHA CAP: a real bug behind the gate result
User insisted on re-auditing the gate. Audit chain, with the wrong turn kept:
1. HYPOTHESIS (mine): alpha must scale as (n/256)^2 because w is in
   radians/pixel. Implemented a scaled grid.
2. CONTROLLED TEST REFUTED IT: varying only n (128 vs 256) with the sampling
   scaled proportionally and dose/ray fixed gives the SAME optimum (1.0 at
   both) — alpha is resolution-invariant because h_fonf normalizes w by the
   Nyquist radius. The scaled-grid "fix" was reverted.
3. REAL CAUSE: DOSE. At n = 256 the optimal order rises from 1.0 at 300
   photons/ray to 4.0 at 75 — outside the fixed 0.2-2.0 grid, so the method
   ran under-regularized in every photon-starved setting. Measured cap loss
   0.55-0.56 dB at 75 photons/ray. The earlier log entry "alpha saturates at
   the integer bound under photon starvation" was this ceiling, not physics.
CONSEQUENCE: the TMI gate ran at 41/82/326 photons/ray — squarely in the
capped zone — so its A-FONF rows are invalid and the REVIEW verdict (a
0.12-0.67 dB deficit versus tuned TV, i.e. the same order as the cap loss)
must be re-decided after a re-run. The JBHI lock is provisional again.
FIX: the fixed grid is replaced by an unbounded geometric ladder
(0.2 x 2^k, k = 0..7) with the discrepancy criterion deciding where to stop;
deviance is monotone in alpha, so the search early-stops at the first
noise-inconsistent order and costs no more than before. alpha > 2 is
admissible: H stays in [0,1], symmetric and nonexpansive for any alpha > 0,
so every guarantee in the theory survives. Verified: 75 photons/ray now
selects 3.2 and recovers +0.56 dB; 300 photons/ray selects 1.6 (unchanged
optimum); n = 128 selects 0.8. Suite: 44 checks (ladder uncapped, monotone,
early-stopping).
OPEN QUESTION for the author: the SPL letter's admissible range for alpha —
if it was stated as (0, 2], the JBHI paper must present this explicitly as
an extended range rather than silently widening it.
RE-RUNS REQUIRED (A-FONF rows only; all other methods unaffected):
  (a) realistic-geometry gate, 3 doses;
  (b) main campaign at inc = 60 and 300 (both datasets + fan);
  (c) CHO A-FONF rows (now ~8x cheaper with --select-iters).

## 2026-09-09 · v0.9.31 — cache-aware block skip
Deleting the A-FONF ROIs (to regenerate them after the alpha-cap fix) was
not enough: the block-level resume trusted the CSV alone and skipped the
whole lesion block. The skip now verifies that the ROI cache still holds
every REQUESTED method and recomputes only what is missing; verified by
deleting one method's cache and observing "CSV says done but cache is
missing [...] — recomputing", 30 of 60 cells run, and bit-identical d' for
the untouched method.

## 2026-09-09 · CHO re-run with the uncapped ladder: conclusions unchanged
Regenerating every A-FONF ROI with the discrepancy ladder (2,400 cells,
68 min) gives d' = 1.29 / 2.04 / 2.87 at 300 / 1500 / 3000 photons per ray
(r = 6, c = 0.06), against 1.28 / 2.05 / 2.89 with the capped grid: the cap
did not bind at these operating points, so the task-based conclusion stands
unchanged, and all methods remain statistically tied at every dose (CIs
overlap; A-FONF leads numerically at 3000 with AD and TV inside its
interval). Practical consequence: only the inc = 60 rows of the main
campaign can be affected by the cap, which reduces the pending re-run from
all doses to one.

## 2026-09-09 · v0.9.32 — the cap cost 4 dB at the lowest dose
Diagnostic at the campaign geometry (256 x 256, 64 bins, 90 views, Mayo
slice, 300 iterations):
  alpha   deviance   PSNR
   2.0      0.672    19.04   <- the ceiling used until v0.9.29
   6.4      0.805    21.43
  25.6      0.978    23.09   <- selected by the discrepancy criterion
  51.2      1.061    23.11   (criterion violated; PSNR optimum)
 102.4      1.145    22.79
So the criterion selects within 0.02 dB of the optimum once it is allowed to,
and the previous ceiling cost about 4 dB at 60 photons per ray. At 300
photons the selected order is 6.4 (24.71 dB versus 24.62 dB capped), and at
the realistic dense sampling geometry the selection is unchanged, which is
why the gate result did not move. The ladder now runs to 409.6 so that the
criterion terminates the search rather than the ladder end; early stopping
keeps the added entries free elsewhere.
IMPLICATION FOR THE TABLES: the inc = 60 rows of every A-FONF entry are
expected to move by several decibels, which can change the low-dose ordering
against TV and PnP. The campaign re-run must be verified to have actually
regenerated those rows (the first attempt returned unchanged values, which
is inconsistent with this diagnostic).
