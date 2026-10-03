# AXIS

**Hypothesis (one line):** when data is gathered across different settings (scanners,
lab batches, sites), a change in setting usually shifts only *how the system is
measured* — not *how the system works* — so causal methods that assume a real
mechanism change are leaning on something that isn't there.

AXIS fits a simple linear time model (VAR(1)) in each setting, then compares the
fitted coefficients. Changes concentrated on the diagonal (each channel's own
autocorrelation / noise scale) read as *measurement*; changes on the off-diagonal
(how channels drive each other) read as *mechanism*.

## Run order

Gate scripts must pass before the real-data script means anything:

1. `experiments/01_oracle_measurement.py` — GATE 1. A simulated pure *measurement*
   shift must load onto the diagonal only.
2. `experiments/02_oracle_mechanism.py` — positive control. A simulated pure
   *mechanism* shift must load onto the off-diagonal.
3. `experiments/03_genomics_admission.py` — real data (Perturb-seq). Only run this,
   and only trust its readout, after 01 and 02 both pass.

Everything imports only from `ideas/axis/` (see `src/`). No shared code, no
cross-idea imports.
