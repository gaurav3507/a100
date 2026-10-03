# Task 6: finite-sample direction certificate

## Scope

The precision gate and the recovery backbone use the same matrix,

\[
\widehat\Delta_e = \widehat\Sigma_e^{-1}
                      - \widehat\Sigma_0^{-1}.
\]

The gate tests whether its largest eigenvalue is detectably positive. That test
does not establish that the leading eigenvector is stable. Direction recovery
also requires separation between the first and second eigenvalues. Task 6 adds
this second, explicitly testable condition.

## Assumptions

For each control or intervention environment, rows are iid Gaussian with a
positive-definite covariance. The working projection is fixed in advance or is
fitted on an independent basis sample. Let `K` be the number of distinct sample
covariance matrices to certify jointly, and allocate `delta / K` failure
probability to each matrix.

These assumptions hold conditionally in the linear-Gaussian synthetic
experiment because PCA is fitted on an independent basis environment. They are
not asserted for the real Perturb-seq or fMRI panels.

## Data-dependent bound

For a matrix with `n` rows and working dimension `d`, set `nu = n - 1` and

\[
r = \sqrt{d/\nu} + \sqrt{2\log(2K/\delta)/\nu}.
\]

Gaussian singular-value concentration implies, with the allocated probability,

\[
(1-r)^2\Sigma \preceq \widehat\Sigma
                   \preceq (1+r)^2\Sigma.
\]

When `r < 1`, inversion and the Loewner order give the observable precision
radius

\[
\|\widehat P-P\|_{op}
\leq (2r+r^2)\|\widehat P\|_{op} =: \beta.
\]

For one environment, let `epsilon = beta_e + beta_0`. Then
`||widehat Delta_e - Delta_e||_op <= epsilon`. If `widehat g` is the empirical
top eigengap, Weyl's inequality yields the population-gap lower bound

\[
g \geq \widehat g - 2\epsilon.
\]

Combining this with the population-gap Davis-Kahan variant gives

\[
\sin\angle(\widehat v_e,v_e)
\leq {2\epsilon \over \widehat g-2\epsilon}.
\]

The certificate therefore needs `widehat g > 2 epsilon` to separate the gap and
`widehat g > 4 epsilon` to produce an angle strictly below 90 degrees. The code
reports 90 degrees rather than pretending to certify when either condition
fails.

Allocating the error probability across the control and every candidate
environment makes the statement simultaneous. The gate may therefore select a
subset using the same data without invalidating the retained direction bounds:
on the joint event, the bound already holds for every candidate direction.

The singular-value step follows Corollary 5.35 in Roman Vershynin,
"Introduction to the non-asymptotic analysis of random matrices," 2012,
<https://arxiv.org/abs/1011.3027>. The eigenvector step uses Theorem 2 of Yi Yu,
Tengyao Wang, and Richard J. Samworth, "A useful variant of the Davis-Kahan
theorem for statisticians," Biometrika 102(2), 2015,
<https://arxiv.org/abs/1405.0680>.

## What is and is not certified

The theorem certifies the sample leading eigenvector against the population
leading eigenvector of the precision difference. It certifies a causal latent
direction only under an additional population-alignment statement.

In GRD's hard-intervention simulator, the population precision difference is a
positive target term minus a removed-parent term. The target term dominates in
the recovery regime, but the population leading eigenvector need not equal the
true unmixing row exactly. The feasibility experiment therefore reports three
angles separately:

1. sample direction to population direction, which the theorem covers;
2. population direction to true latent direction, which is model bias;
3. sample direction to true latent direction, which contains both effects.

This separation prevents a sampling theorem from being presented as a stronger
causal-identification theorem.
