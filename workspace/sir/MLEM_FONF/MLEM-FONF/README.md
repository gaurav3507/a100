# MLEM + FONF: Fractional-Order Multi-Notch Spectral Regularization for CT Reconstruction

Reference implementation accompanying the paper:

> A. P. Singh, M. Khurana, and S. Tiwari, "MLEM Reconstruction with
> Fractional-Order Multi-Notch Spectral Regularization," *IEEE Signal
> Processing Letters*, 2026.

MLEM+FONF embeds a fractional-order multi-notch spectral filter (FONF)
**inside** the maximum-likelihood expectation-maximization (MLEM) iteration as
a training-free, geometry-agnostic prior. Each iteration applies the MLEM
multiplicative update, shapes the iterate in the frequency domain with the
transfer function

H(ω) = G_α(ω) · Π_m [1 − exp(−‖ω − ω_m‖² / 2σ_m²)]^(α/2),  G_α(ω) = exp(−α‖ω‖² / 4ω_N²),

relaxes toward the filtered image (0 < λΔt < 1), and projects to
non-negativity. A data-driven rule activates K ≥ 0 Gaussian notches only when
narrowband artifacts are detected, so a single framework covers broadband
Poisson noise (K = 0) and periodic stripe/ring artifacts (K > 0). The spectral
step is linear, averaged, and 1-Lipschitz; no training data are used anywhere.

## Repository layout

```
src/mlem_fonf/
  projector.py    sparse parallel-beam system matrix (64 bins x 90 angles, 256x256)
  fonf.py         H_FONF (Eq. 4) + data-driven notch selection (Sec. II-C)
  algorithms.py   MLEM, MLEM+FONF (Algorithm 1), MLEM+TV/AD/FuzzyAD, PnP-ADMM(TV)
  metrics.py      SNR, MSE, RMSE, PSNR, CP (edge correlation), MSSIM
  phantoms.py     modified Shepp-Logan, simulated CT, elliptical, thorax-like
experiments/
  run_comparison.py     Table-II-style benchmark (CSV output)
  run_convergence.py    PSNR/MSSIM vs iteration (Fig.-5 style)
  run_alpha_sweep.py    fractional-order ablation (Fig.-4a style)
  run_stripe_demo.py    multi-notch proof of mechanism (Fig.-6 style)
results/                generated tables and figures
```

## Quick start

```bash
pip install -r requirements.txt
cd experiments
python run_comparison.py --phantom shepp_logan --iters 1000 --realizations 10
python run_convergence.py
python run_alpha_sweep.py
python run_stripe_demo.py
```

A full comparison run takes a few minutes on a laptop CPU; the system matrix
(≈3.1 M non-zeros) is built once per realization.

## Representative results (this implementation)

Modified Shepp-Logan, 64 bins x 90 angles, photon-limited Poisson data,
800 iterations, mean over noise realizations (`run_comparison.py`):

| Method        |  SNR (dB) |    MSE |   RMSE | PSNR (dB) |    CP | MSSIM |
|---------------|----------:|-------:|-------:|----------:|------:|------:|
| MLEM          |      3.60 | .0257  | .1602  |     15.91 | 0.571 | 0.524 |
| MLEM+TV       |      5.62 | .0161  | .1270  |     17.92 | 0.466 | **0.739** |
| MLEM+AD       |      5.74 | .0157  | .1252  |     18.05 | 0.461 | 0.699 |
| MLEM+FuzzyAD  |      2.98 | .0298  | .1724  |     15.29 | 0.484 | 0.703 |
| PnP-ADMM(TV)  |      5.90 | .0153  | .1234  |     18.20 | 0.478 | 0.732 |
| **MLEM+FONF** |  **6.67** | **.0126** | **.1124** | **18.98** | **0.660** | 0.604 |

The qualitative picture of the paper reproduces: **MLEM+FONF attains the best
SNR, MSE, RMSE, and PSNR of all six training-free methods**, TV-based priors
lead on MSSIM, FuzzyAD sits near plain MLEM, and the FONF step costs about
**5x less runtime than PnP-ADMM(TV)** (one forward + one inverse FFT per
iteration). The fractional-order sweep is smooth and monotonic in α, the
PSNR/MSSIM trajectories rise and stabilize while MLEM stays at its noisy
ML level, and on a synthetic stripe the Sec. II-C rule fires at the artifact
frequency and recovers **+8.2 dB** (paper: +7.7 dB).

![convergence](results/convergence_shepp_logan.png)
![stripe demo](results/stripe_demo.png)

## Relation to the numbers in the paper

This is an independent, clean-room **Python** reimplementation of the paper's
equations (the experiments in the letter were run in MATLAB R2025b). The
projector, Poisson calibration, and baseline prior weights therefore differ in
implementation detail, so absolute values are *representative rather than
identical*: the MLEM operating point matches the paper closely
(15.91 dB PSNR / 3.60 dB SNR / 0.524 MSSIM here vs 16.03 / 3.86 / 0.513
in the letter), while the FONF margin over MLEM is +3.1 dB here vs +6.1 dB in
the letter. Rankings, monotonicity of the α ablation, convergence behaviour,
runtime ordering, and the notch-detection gain all reproduce. The real
thoracic CT slice used in the paper is not redistributable; a synthetic
thorax-like phantom is provided instead (`phantoms.thorax_like`), and any
256x256 slice can be dropped in via `PHANTOMS`.

## Citation

See `CITATION.cff`, or:

```bibtex
@article{singh2026mlemfonf,
  author  = {Singh, Akhil Pratap and Khurana, Manju and Tiwari, Shailendra},
  title   = {{MLEM} Reconstruction with Fractional-Order Multi-Notch Spectral Regularization},
  journal = {IEEE Signal Processing Letters},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`).
