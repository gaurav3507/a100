# preprocessing sweep — artefacts pending

**This directory is deliberately empty.**

The preprocessing sweep (`scripts/91_preprocessing_sweep.py`) has been written
and its gates tested, but it has only ever run on the A100, and those artefacts
have **not** been copied to this repository. Nothing here cites their numbers,
and no table in `paper/tables/` reports them: `92_preprocessing_table.py` finds
no CURRENT artefact and writes nothing rather than emitting an empty table.

To populate it, run on the A100 and copy the JSON + `__sweep_spectra.npz` pairs
back into this directory:

```bash
bash scripts/run_preprocessing_sweep.sh
python scripts/92_preprocessing_table.py
```

The sweep refuses to proceed unless its `raw` arm reproduces the published
cap-2000 participation ratio for that dataset to 1e-10, read at runtime from
`results/descriptives/`. A sweep whose control arm does not reproduce is
measuring the pipeline, not the preprocessing.
