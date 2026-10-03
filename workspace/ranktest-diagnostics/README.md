# ranktest-diagnostics

Descriptive profiling and diagnostic gates for the multi-environment CRL
assumption bundle. Descriptives only here: no assumption verdicts.

## Claude Code cannot run on this machine

University policy. Every script in `scripts/` is written on the Mac, pushed to
`origin/ranktest-lane`, pulled here, and run by hand. If something needs
changing, change it on the Mac and pull again. Do not edit in place here, or
the A100 copy and the repo will silently diverge.

## Environment

Use the `cb` venv (numpy / scipy / sklearn / anndata / scanpy):

    source /workspace/venvs/cb/bin/activate

Do NOT use `dvae` (torch, numpy < 2) for anything in this lane.

## Launch convention

There is no tmux on this box. Long runs go through `nohup`, and `python -u` so
the log is not lost to buffering if the job is killed:

    cd /workspace/ranktest-diagnostics
    nohup python -u scripts/85_dataset_descriptives.py --dataset k562 \
        > logs/k562_desc.log 2>&1 &

Check progress with `tail -f logs/<name>.log`.

## Results

`results/INDEX.md` is the index: one row per artefact, with its statistic,
key numbers, commit and status. Quote only rows marked CURRENT.

Every results file carries a mandatory `meta` block recording the statistic,
git commit, timestamp, full config, package versions AND platform. The
platform block exists so a number that moves between this machine and the Mac
can be attributed to the environment rather than mistaken for a finding: the
BLAS backend differs, and SVD-derived quantities depend on it.

To check the two machines agree before trusting any cross-machine comparison:

    python -u scripts/85_dataset_descriptives.py --selftest
    python -u scripts/85_dataset_descriptives.py --overlap-check abide

## Not this project

`/workspace/meridian-identifiability/` is a separate project. Nothing here
writes to it. Its data is read-only input for some loaders, never a target.
