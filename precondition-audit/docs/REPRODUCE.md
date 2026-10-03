# Reproducing every table and figure

Two classes of step. **Anywhere** steps run from committed artefacts on any
machine with the Python environment below. **A100** steps need the raw data,
which is not redistributable and is never committed.

## Environment

Two environments, pinned separately because they differ and the difference
matters. Every quantity in the dimension tables comes from an SVD, and SVD
results depend on the BLAS backend at the last digits.

```bash
# rebuilding tables, figures and the index from committed artefacts
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt      # numpy 2.4.6, scipy 1.17.1, Accelerate

# running the loaders against raw data
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements-a100.txt     # numpy 2.2.6, scipy 1.15.3, scipy-openblas
```

`pandas`, `scanpy` and `anndata` are needed only for the A100 loaders and are
pinned in `requirements-a100.txt` only.

## The two constants that must reproduce

Any run that regenerates the descriptives must reproduce these exactly. They
are the control on the whole pipeline; if either moves, stop and investigate
rather than proceeding.

| quantity | value |
|---|---|
| k562 participation ratio, cap 8000 | `552.5243597491797` |
| frangieh_coculture participation ratio, cap 8000 | `491.85699635911675` |

`scripts/91_preprocessing_sweep.py` enforces the cap-2000 equivalents
automatically as a step-0 gate and exits non-zero on any disagreement beyond
1e-10.

## Anywhere — from committed artefacts, no data required

```bash
# 1. descriptive tables A and B  ->  paper/tables/
python scripts/88_make_descriptive_tables.py

# 2. spectrum figures, primary + sensitivity  ->  paper/figures/
python scripts/90_spectrum_figure.py

# 3. gate tables  ->  paper/tables/
python scripts/87_make_gate_tables.py

# 4. artefact index  ->  results/INDEX.md
python scripts/89_make_index.py
```

Step 2 verifies each eigenvalue sidecar against the `lam_len` / `lam_sum` /
`lam_sumsq` triple recorded in its JSON before plotting, and exits non-zero
naming both values on any mismatch.

## A100 only — raw data required

Point the loaders at the data with `$PRECOND_DATA` (or pass an explicit path).
No path is hardcoded; a missing file names every candidate tried.

```bash
export PRECOND_DATA=/workspace/precondition-audit/data

# 5. descriptives: 8 jobs, serial, artefact assertion at the end
bash scripts/run_descriptives.sh

# 6. preprocessing sweep: 6 jobs, PID-locked against concurrent batches
bash scripts/run_preprocessing_sweep.sh

# 7. preprocessing tables (writes nothing while the sweep is unrun)
python scripts/92_preprocessing_table.py
```

Both runners refuse to look complete when they are not: each asserts one new
artefact per requested job and exits non-zero with a PASS/FAIL table otherwise.

## Simulator gates — anywhere, but slow

```bash
python scripts/81_ranktest_oracle.py --gate 0     # ~50 min
python scripts/81_ranktest_oracle.py --gate 1     # ~20 min
python scripts/81_ranktest_oracle.py --gate 2     # ~10 min
python scripts/85_dataset_descriptives.py --selftest   # cross-machine gate
```

The selftest pins three constants measured on Darwin-arm64 with BLAS
`accelerate`. It checks the data fingerprint first, so a numpy-stream
difference is reported separately from a BLAS difference in the SVD.
