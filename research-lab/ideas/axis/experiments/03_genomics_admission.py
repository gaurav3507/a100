"""TEMPLATE --- real-data admission test on Perturb-seq (genomics).

This script is a scaffold, not a finished experiment. It only means anything
AFTER gates 01 and 02 both pass. It never fabricates data: if no genomics data
path is set in configs/default.yaml, it prints a message and exits.

------------------------------------------------------------------------------
CAVEAT --- read before trusting anything this produces
------------------------------------------------------------------------------
AXIS was built on a TIME SERIES readout: fit VAR(1) per environment, then ask
whether a change between environments sits on the diagonal (per-channel =
measurement) or off-diagonal (channel-to-channel = mechanism).

Single-cell Perturb-seq data is NOT a time series. It is a cells x genes matrix.
So there is no x_{t-1} -> x_t to regress, and the VAR(1) machinery does not apply
as-is. A contrast has to be constructed. The intended construction (to be
implemented) is:

  1. Split cells into two ENVIRONMENTS. An environment here is a setting we
     suspect is only a measurement difference --- e.g. two sequencing batches,
     two labs, two 10x lanes --- NOT the perturbation itself.
  2. Within each environment, estimate a gene-gene linear coupling matrix M
     (each gene regressed on the others; a linear-SEM / precision-style estimate).
     M plays the role A played for VAR(1): its diagonal is the per-gene self term
     (variance / scale), its off-diagonal is gene-gene coupling.
  3. Match sample sizes across the two environments (subsample the larger), get
     per-coefficient SEs by bootstrap, and run the SAME dissociation readout.

  DIRECTION IS UNCERTAIN AND MUST BE REPORTED, NOT ASSUMED.
  CRISPRi knockdown is a SOFT change: it partially lowers the expression of a
  single targeted gene. That is close to a per-gene noise / scale change, which
  loads on the DIAGONAL. So if a perturbation leaks into the environment
  contrast, a genuine mechanistic intervention can masquerade as "measurement".
  Whatever diagonal-vs-off-diagonal split comes out here must be reported as
  observed, with the environment definition stated, and never assumed to mean
  measurement just because it landed on the diagonal.
------------------------------------------------------------------------------

Run on the A100. Imports only from ideas/axis/.

    python experiments/03_genomics_admission.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IDEA_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(IDEA_ROOT, "src"))

import yaml  # noqa: E402
# When implemented, these are the only imports needed (all within ideas/axis/):
# import var_fit
# import dissociation


def load_config():
    path = os.path.join(IDEA_ROOT, "configs", "default.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg, path


def load_perturbseq(data_path):
    """TODO: load the Perturb-seq matrix and the environment / perturbation labels.

    Returns (to be defined):
        counts : cells x genes array (normalized as appropriate).
        env    : per-cell environment label (batch / lab / lane).
        pert   : per-cell perturbation label (which gene was knocked down).
    Must record the data path and a hash of the loaded array for provenance,
    exactly like gates 01 and 02.
    """
    raise NotImplementedError(
        "TODO: implement Perturb-seq loading for path: %r" % data_path
    )


def gene_coupling_matrix(counts_env):
    """TODO: estimate the gene-gene linear coupling matrix M for one environment.

    Each gene regressed on the others (linear-SEM / precision-style). Diagonal =
    per-gene self/scale term; off-diagonal = gene-gene coupling. Return M and a
    per-coefficient SE (bootstrap), matching what dissociation.dissociate expects.
    """
    raise NotImplementedError("TODO: implement the gene-gene coupling estimator")


def main():
    cfg, cfg_path = load_config()
    data_path = (cfg.get("genomics_data_path") or "").strip()

    if not data_path:
        print("03_genomics_admission: no genomics_data_path set in", cfg_path)
        print("Set genomics_data_path in configs/default.yaml to a real Perturb-seq")
        print("dataset before running. This template will NOT fabricate data.")
        print("Also: gates 01 and 02 must pass first (see README run order).")
        sys.exit(0)

    if not os.path.exists(data_path):
        print(f"03_genomics_admission: genomics_data_path does not exist: {data_path}")
        sys.exit(1)

    # --- Everything below is a TODO sketch; it is not runnable yet. ---
    # counts, env, pert = load_perturbseq(data_path)
    # env_a, env_b = two chosen environments (batches/labs), sample-size matched
    # M_a, se_a, n_a = gene_coupling_matrix(counts[env == env_a])
    # M_b, se_b, n_b = gene_coupling_matrix(counts[env == env_b])
    # ratio, summary = dissociation.dissociate(M_a, se_a, n_a, M_b, se_b, n_b,
    #                                          threshold=cfg["threshold"])
    # Report ratio + summary AND the environment/perturbation definitions.
    # Report the diagonal/off-diagonal split as observed; do not assume its meaning.
    raise NotImplementedError(
        "03_genomics_admission is a template. Implement load_perturbseq and "
        "gene_coupling_matrix, and the environment contrast, before running."
    )


if __name__ == "__main__":
    main()
