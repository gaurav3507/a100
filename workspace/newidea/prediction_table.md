# Preregistered estimator predictions

Generated after the complete oracle gate and before any transformed-grid result. Predictions are encoded in the committed generator script and were not derived from transformed results.

| method | transformation | prediction | short reason | theoretical assumption | load-bearing metric |
|---|---|---|---|---|---|
| PC-Pearson | A1 | NOMINALLY_INVARIANT | Variable relabeling preserves Fisher-z PC output up to the same relabeling | Permutation equivariance | skeleton F1 |
| PC-Pearson | B | NOMINALLY_INVARIANT | Pearson correlations and frozen per-cell standardization remove positive component scaling | Correlation scale invariance | skeleton F1 |
| PC-Pearson | C | NOMINALLY_NON_INVARIANT | Pearson partial correlations need not survive nonlinear marginal warps | Gaussian partial-correlation test restriction | skeleton F1 |
| PC-Pearson | D | NOMINALLY_NON_INVARIANT | Within-group coordinate mixing changes the node-level CI graph | Node semantics and conditional independence | skeleton F1 |
| PC-Pearson | E | NOMINALLY_NON_INVARIANT | Measurement error can alter conditional independences | Causal sufficiency and measurement assumptions | skeleton F1 |
| PC-nonparanormal-preproc | A1 | NOMINALLY_INVARIANT | Variable relabeling preserves the CPDAG up to relabeling | Permutation equivariance | skeleton F1 |
| PC-nonparanormal-preproc | B | NOMINALLY_INVARIANT | Positive scaling preserves ranks and the normal-score data exactly | Rank invariance | skeleton F1 |
| PC-nonparanormal-preproc | C | NOMINALLY_INVARIANT | Strictly increasing component warps preserve ranks and the normal-score data exactly | Nonparanormal marginal invariance | skeleton F1 |
| PC-nonparanormal-preproc | D | NOMINALLY_NON_INVARIANT | Coordinate mixing is not a marginal monotone transform and can change CI structure | Nonparanormal model restriction | skeleton F1 |
| PC-nonparanormal-preproc | E | NOMINALLY_NON_INVARIANT | Independent measurement error changes ranks and can alter CI structure | Measurement assumptions | skeleton F1 |
| DirectLiNGAM | A1 | NOMINALLY_INVARIANT | Node relabeling preserves the fitted graph after truth remapping | Permutation equivariance | directed F1 (oracle-ineligible, descriptive only) |
| DirectLiNGAM | B | NOMINALLY_INVARIANT | Frozen per-cell standardization removes positive component scaling | Scale normalization and linear SEM | directed F1 (oracle-ineligible, descriptive only) |
| DirectLiNGAM | C | NOMINALLY_NON_INVARIANT | Nonlinear marginal warps generally destroy the linear additive SEM form | Linearity and non-Gaussian independent errors | directed F1 (oracle-ineligible, descriptive only) |
| DirectLiNGAM | D | NOMINALLY_NON_INVARIANT | Coordinate mixing changes the node-level linear SEM and graph | Linear SEM node semantics | directed F1 (oracle-ineligible, descriptive only) |
| DirectLiNGAM | E | NOMINALLY_NON_INVARIANT | Measurement noise violates the error and measurement model | Independent non-Gaussian error assumptions | directed F1 (oracle-ineligible, descriptive only) |
| NOTEARS-linear | A1 | NOMINALLY_INVARIANT | The score and acyclicity constraint are permutation equivariant | Permutation equivariance | directed F1 (oracle-ineligible, descriptive only) |
| NOTEARS-linear | B | NOMINALLY_INVARIANT | Frozen per-cell standardization removes positive component scaling | Standardized linear least-squares score | directed F1 (oracle-ineligible, descriptive only) |
| NOTEARS-linear | C | NOMINALLY_NON_INVARIANT | Nonlinear marginal warps leave the linear SEM score class | Linear functional-form restriction | directed F1 (oracle-ineligible, descriptive only) |
| NOTEARS-linear | D | NOMINALLY_NON_INVARIANT | Coordinate mixing changes node semantics and the linear graph | Linear SEM node semantics | directed F1 (oracle-ineligible, descriptive only) |
| NOTEARS-linear | E | NOMINALLY_NON_INVARIANT | Measurement error changes the least-squares causal model | Measurement assumptions | directed F1 (oracle-ineligible, descriptive only) |
| NOTEARS-MLP | A1 | NOMINALLY_INVARIANT | The nonlinear score and acyclicity constraint are permutation equivariant | Permutation equivariance | directed F1 (oracle-ineligible, descriptive only) |
| NOTEARS-MLP | B | NOMINALLY_INVARIANT | Frozen per-cell standardization removes positive component scaling | Scale normalization | directed F1 (oracle-ineligible, descriptive only) |
| NOTEARS-MLP | C | THEORETICALLY_UNCLEAR | Model flexibility helps but marginal reparameterization can change additive noise and optimization behavior | Nonlinear additive model and score restrictions | directed F1 (oracle-ineligible, descriptive only) |
| NOTEARS-MLP | D | NOMINALLY_NON_INVARIANT | Coordinate mixing changes the node-level graph target | Node semantics | directed F1 (oracle-ineligible, descriptive only) |
| NOTEARS-MLP | E | NOMINALLY_NON_INVARIANT | Measurement error is outside the fitted nonlinear SEM | Measurement assumptions | directed F1 (oracle-ineligible, descriptive only) |
| GOLEM | A1 | NOMINALLY_INVARIANT | The likelihood score is permutation equivariant | Permutation equivariance | directed F1 (oracle-ineligible, descriptive only) |
| GOLEM | B | NOMINALLY_INVARIANT | Frozen per-cell standardization removes positive component scaling | Scale normalization | directed F1 (oracle-ineligible, descriptive only) |
| GOLEM | C | NOMINALLY_NON_INVARIANT | Nonlinear marginal warps violate the linear Gaussian score model | Linear Gaussian likelihood restriction | directed F1 (oracle-ineligible, descriptive only) |
| GOLEM | D | NOMINALLY_NON_INVARIANT | Coordinate mixing changes the node-level linear graph | Linear SEM node semantics | directed F1 (oracle-ineligible, descriptive only) |
| GOLEM | E | NOMINALLY_NON_INVARIANT | Measurement error is outside the likelihood model | Measurement assumptions | directed F1 (oracle-ineligible, descriptive only) |

PC orientation is load-bearing only for a PC method whose separately preregistered mean oracle orientation F1 is at least 0.80. `THEORETICALLY_UNCLEAR` cells are exploratory and cannot alone establish an observed-vs-nominal contradiction.
