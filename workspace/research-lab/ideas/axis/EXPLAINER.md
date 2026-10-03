Append-only. Newest entry at the bottom. Never edit past entries.

---

## 2026-09-10 — AXIS

**What this idea is**

Some methods learn cause-and-effect from data that was gathered in different
settings — different brain scanners, different lab batches, different sites. To do
that, they quietly assume that changing the setting changes how the underlying
system actually works. AXIS is a simple test that checks whether a change in setting
really changes how the system works, or whether it only changes how the system is
measured. The bet behind AXIS: in real data the change is usually just measurement,
so those methods are leaning on something that isn't really there.

**Why it might matter**

If the test works, it tells you — before you spend time and compute running an
expensive method — whether your data even meets that method's most basic assumption.
And it comes with evidence, drawn from brain scans and genomics, that the assumption
often fails in practice. That would let people stop trusting results that were built
on a shaky footing.

**Current status**

NOT STARTED. Two safety checks have to pass before anything on real data means
anything:
1. A simulated change that is *pure measurement* must be flagged by the test as
   measurement.
2. The test must actually predict when existing methods fail.
Nothing has been run yet.

**Dead ends (don't reopen)**

- "Treat the measurement shift as a free bonus signal." Dead. The multi-view theorem
  (Yao et al. 2024) only hands back the shared / harmonized part of the data, and the
  method that recovers the invariant part is already taken (Kong et al. 2022). What
  survives from that direction is *the test itself*, plus the cross-domain evidence
  that the assumption fails. Nothing else.
