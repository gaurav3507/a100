# CRL-CD Interface Compatibility Audit: Phase 1

This repository implements the preregistered Phase 1 audit using only true latent variables from G-CaRL Simulation 1. It does not train G-CaRL, an encoder, or any learned representation. The primary target is the inter-group causal DAG. Every intra-group score is marked `EXPLORATORY_ONLY`.

## Scientific scope

The audit asks whether representations inside the G-CaRL-style equivalence class can have materially different compatibility with downstream causal-discovery estimators. The inside-class transformations are within-group permutation (A1), heterogeneous component scaling (B), and component-wise invertible cubic warping (C). Within-group entanglement (D) is a boundary control. Additive measurement noise (E) is a corruption control.

The preregistered oracle eligibility threshold is 0.80 on directed F1 for DAG estimators and skeleton F1 for PC estimators. PC orientation is load-bearing only if mean oracle orientation F1 is at least 0.80. Candidate mismatch materiality is an absolute primary F1 drop of at least 0.20 or normalized degradation of at least 0.25.

## Verified G-CaRL substrate

Pinned upstream commit: `0020bfce34736d61d70ab8175f061d02951a7ed4`.

The active Simulation 1 block in `gcarl_training.py` specifies 3 groups, 10 variables per group, 65536 samples, a DAG, Laplace noise, 2 inter-group neighbors, 1 intra-group neighbor, no latent confounders, 3 mixing layers, `ar_alpha=3`, `ar_beta=0.8`, and first-order coefficient ranges `[0.9, 1.0]`.

The pinned `generate_dataset(...)` returns `x, s, lam1, lam2, lamin1, lamin2`. Its `s` return is true latent Z with exact shape `(65536, 3, 10)`. Simulation 1 calls `gen_net_dag` and then `gen_s_lap`. In the DAG generator and sampler, matrix rows select parents and columns select children. `lam1` contains inter-group first-order coefficients in shape `[dim, dim, direction, group_pair]`. `lamin1` contains the within-group first-order blocks. The evaluator selects `A1` and `Ain1` for the Laplace model, confirming that `lam1` and `lamin1` are the graph-generating first-order objects for Simulation 1. This substrate passes B1.

## Discovery methods and frozen procedure

The final smoke-passed set is PC-Pearson, PC-nonparanormal-preproc, DirectLiNGAM, NOTEARS-linear, NOTEARS-MLP, and GOLEM. CAM is excluded because no R runtime was available for import and smoke fit. Import alone was not accepted; each included method completed a fit on a three-variable known DAG and returned a 3 by 3 graph. Linear NOTEARS pins the authors' code at `4a9ab19fe502e392503da331773c5223d82d3666` and evaluates its identical centered l2 objective through a cached covariance matrix. The dual ascent, exact matrix-exponential acyclicity constraint, penalty, and threshold remain unchanged.

Every input cell is standardized independently with the same frozen procedure. PC-nonparanormal-preproc additionally replaces each marginal by average ranks and applies the standard normal inverse CDF to `(rank - 0.5) / n`. This is named as preprocessing and is not presented as a native Rank-PC implementation. All fixed hyperparameters and thresholds are in `config/methods.yaml`. No transformed cell may mutate them.

The audit is explicitly labelled `ORACLE-TUNED COMPATIBILITY AUDIT`, although the checked-in method settings are fixed procedure choices rather than per-cell ground-truth optimization.

## Graph semantics and scoring

G-CaRL truth is adapted to parent-row, child-column form. DirectLiNGAM exposes effect-row, cause-column weights and is transposed. gCastle exposes parent-row, child-column `causal_matrix`. causal-learn uses endpoint pairs: `(-1, 1)` is row-node to column-node, `(-1, -1)` is undirected, and other nonzero endpoint pairs remain in the skeleton but are not asserted as a directed edge.

Directed estimators receive directed precision, recall, F1, and pairwise SHD on inter-group edges. PC receives skeleton precision, recall, F1, and skeleton SHD. PC orientation scoring considers true edges that are present in the estimated skeleton; an undirected endpoint is an orientation false negative and a wrong direction is both an orientation false positive and false negative. Directed false-positive skeleton edges contribute orientation false positives. Known-answer tests assert all transpose and indexing conventions. This scoring substrate passes B3.

## Reproduction

```bash
git submodule update --init --recursive
uv sync --python 3.11
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
python scripts/capability_check.py
pytest -q
python scripts/run_oracle.py
python scripts/write_prediction_table.py
# Commit prediction_table.md and oracle summaries before continuing.
python scripts/build_manifest.py
# Commit results/manifest.csv before continuing.
scripts/launch_phase1_nohup.sh
scripts/check_phase1.sh
```

The full transformed grid refuses to run before `prediction_table.md` has been committed. Generated cell JSON, latent arrays, and logs are ignored. Small capability, oracle, manifest, and summary files are tracked.

If a configuration changes, archive or remove the old ignored result directories before rerunning. Never append a new schema to stale results.
