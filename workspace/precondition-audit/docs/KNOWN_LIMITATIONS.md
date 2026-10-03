# Known limitations

Each entry is a limitation that survives the current evidence. None is
rhetorical hedging; every one changes what may be claimed.

## 1. Upstream preprocessing provenance is undocumented

For all six transcriptomic matrices, what transform produced the values is not
recorded by the loaders or the upstream deposits. The artefacts establish that
none is raw counts (non-integral float64 sums; per-entry means 0.11-0.82, RMS
0.48-1.96), which is consistent with a log1p-style transform but does not prove
one. Every dimension estimate in this work was computed on that unidentified
preprocessing. The sweep in `91_preprocessing_sweep.py` tests whether the band
survives re-normalisation; it cannot recover what the original transform was.

## 2. Marchenko-Pastur corroboration exists only for K562 and RPE1

The MP crossing index and the participation ratio are independent estimates,
and where both are meaningful they agree: K562 522 vs 456 at cap 2000 and 517
vs 553 at cap 8000; RPE1 284 vs 234 and 336 vs 262. That agreement is the
strongest evidence in the figure that the dimension is structure rather than
noise.

It does not extend to the other four panels. Where p >> n the aspect ratio
gamma = p/n pushes the MP edge far above the bulk — at Frangieh cap 2000,
gamma = 11.9 gives an edge about 20x sigma^2 and only 2-3 eigenvalues clear it,
against a participation ratio of 415-436. The MP edge is uninformative in that
regime and the panels are annotated as rank-limited. Corroboration therefore
covers two of six profiles.

## 3. Norman numerical rank 3516 against a bound of 5000

At cap 8000 Norman has n=8000 control cells and p=5000 genes, so the sample
covariance should be full rank at 5000. Its numerical rank at tol 1e-10 is
3516: **1484 gene directions are linearly dependent within the control cells**.
At cap 2000 the same dataset is rank 1999 = n-1, purely sample-size-limited, so
the deficiency only becomes visible once n exceeds p.

Norman is the one dataset loaded under an HVG cap, so the natural suspect is
that HVG selection — computed over all cells — admitted genes that are constant
or collinear within the control subset specifically. **This is flagged, not
explained.** Testing it requires the expression matrix, which is A100-only.

## 4. ABIDE sites are measurement shifts, not interventions

ABIDE environments are acquisition sites and HCP environments are task
conditions. A site changes the measurement apparatus, not the mechanism, so the
attrition curve for these datasets is a descriptive analogue and not an
interventional one. Neither has a control condition, so neither contributes a
control-covariance spectrum, and both are absent from the dimension table.

## 5. The 8000-cell soft power floor is simulator-derived

The floors quoted for environment attrition (hard 125, soft 8000) were measured
on the linear-Gaussian simulator under the LFC statistic, not on real data. They
are the right order of magnitude for planning and are the only measured numbers
available, but they are not guaranteed to transfer: the simulator's mixing is
linear, and the diagnostic's own level control degrades once it is not.

## 6. Phase B was written and deliberately never run

`82_ranktest_real.py` applies the rank test to real perturbation data. It is
not in this repository. It was written, gated, and never authorised to run,
because Gate 2 showed the test loses level control at any non-zero mixing
nonlinearity (0.150 at s=0.1, 0.300 at s=0.25), so a rejection on real
nonlinear data would be confounded with the mixing rather than attributable to
the intervention. No real-data rank-test result is claimed anywhere in this
work.
