"""E2b: operating characteristics of the three-way attribution verdict.

Runs the subspace-attribution readout (precision_readout.attribute_environment,
Section X of the paper) on five synthetic environment types x SEEDS seeds, using the
committed simulator, backbone projection, and gate without modification. This is the
knob-free Option B resolution of the boundary behaviour seen at low B: the hard
threshold stays, and the false-attribution rates are MEASURED at full power and
reported as the verdict's operating characteristics.

PRE-REGISTERED EXPECTATIONS (written before the full run; a contradiction is reported
as a finding, not tuned away):
  mechanism (hard iv)        detected ~1.0; false DETECTABLE_BUT_UNATTRIBUTED ~alpha
                             (the subspace is preserved, Prop 3(i), so UNATTRIBUTED
                             verdicts on mechanism envs are the null's 5% by
                             construction).
  mixed gain U(0.5,1.5)      detected high (Prop 2, seed-dependent, ~0.8 at these
                             sizes); of the detected, UNATTRIBUTED ~1.0 (subspace
                             rotated ~18 deg vs ~3 deg null, Prop 3(ii)).
  uniform gain 0.7 (shrink)  detected 1.0 (Prop 1, c<1); verdict MECHANISM_SUPPORTED
                             ~1-alpha. THIS IS THE DOCUMENTED BLIND SPOT (Prop 3
                             Remark): a scalar gain preserves the subspace, so the
                             attributor passes it. Measured here so the limitation is
                             quantified, not hidden.
  uniform gain 1.0           detected ~alpha (identity sanity check).
  uniform gain 1.4 (inflate) detected ~0 (Prop 1, screen blind to inflation).

Cost note: the subspace null depends only on (X_obs, d, n_env). All five environment
types within a seed share one control and one sample size, so the null is computed
ONCE per seed and passed to attribute_environment via subspace_crit (a precomputed
value of the same statistic, not a tuning constant).

SMOKE NOTE (recorded before the full run, after a reduced-power wiring check at
B=100 / 3 seeds): two candidate deviations from the pre-registration surfaced and the
full run explicitly tests them. (i) uniform_gain_0.7 landed DETECTABLE_BUT_UNATTRIBUTED
3/3 rather than MECHANISM_SUPPORTED: the gain shrinks signal but not observation
noise, so the environment's subspace is estimated at lower SNR and its sampling angle
can exceed a null built from full-SNR control resamples; the population blind spot
(Prop 3 Remark) stands, but finite samples may catch the shrink through this SNR
side-channel. (ii) mechanism false-UNATTRIBUTED was 1/3: possibly the alpha-level
boundary, possibly genuinely elevated because the hard intervention weakens one latent
direction 10x and the environment's fifth subspace direction is then estimated
noisily (the M2.6 anisotropy effect in the attributor). Both are reported as measured,
not tuned.

Writes results/e2b/attribution_report.json; prints a per-type table. Exit code is 0
unless the wiring itself fails; expectation contradictions are reported in the JSON
and table, never silently absorbed.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SIM = _load(REPO / "sim" / "simulator.py", "grd_sim_simulator")
PR = _load(REPO / "src" / "gate" / "precision_readout.py", "grd_gate_precision_readout")
BK = _load(REPO / "src" / "recover" / "backbone.py", "grd_recover_backbone")

D_LATENT = 5
D_OBS = 200
D_PROJ = 5
EDGE_PROB = 0.4
ALPHA = 0.05
B_BOOT = 500
IV_BASE = 0.1
N_BASE = 2000
SEEDS = list(range(10))

OUT = Path("results/e2b")


def build(seed):
    specs = [SIM.EnvSpec("basis", None, ()), SIM.EnvSpec("obs", None, ())] + [
        SIM.EnvSpec(f"iv{i}", "hard", (i,), IV_BASE) for i in range(D_LATENT)]
    ds = SIM.simulate(D_LATENT, D_OBS, N_BASE, specs, seed, mixing="linear",
                      edge_prob=EDGE_PROB)
    mu, Wp = BK.fit_pca(ds.environments["basis"].X, D_PROJ)
    Y_obs = BK.project(ds.environments["obs"].X, mu, Wp)
    return ds, mu, Wp, Y_obs


def gain_env(ds, g, rng):
    """Definition 2 environment: unintervened latents through a diagonal gain g."""
    Zc = SIM.sample_latent(ds.B, ds.noise_var, N_BASE, rng)
    Xc = SIM.mix_linear(Zc, ds.A) * g
    return SIM.add_obs_noise(Xc, ds.sd_obs, rng)


CASES = [
    ("mechanism_hard_iv", None),
    ("mixed_gain_u05_15", lambda rng: rng.uniform(0.5, 1.5, D_OBS)),
    ("uniform_gain_0.7", lambda rng: np.full(D_OBS, 0.7)),
    ("uniform_gain_1.0", lambda rng: np.full(D_OBS, 1.0)),
    ("uniform_gain_1.4", lambda rng: np.full(D_OBS, 1.4)),
]

EXPECT = {
    "mechanism_hard_iv": "detected ~1.0; UNATTRIBUTED-rate ~alpha (Prop 3(i))",
    "mixed_gain_u05_15": "detected high; of detected, UNATTRIBUTED ~1.0 (Prop 3(ii))",
    "uniform_gain_0.7": "detected 1.0; MECHANISM_SUPPORTED (documented Prop 3 blind spot)",
    "uniform_gain_1.0": "detected ~alpha (identity sanity)",
    "uniform_gain_1.4": "detected ~0 (Prop 1, inflation invisible)",
}


def run(readout="precision"):
    """All seeds and cases; returns the report dict (main() writes it).

    readout selects only the detection statistic (precision_readout readout=);
    the default is the precision screen. The subspace attribution step, the
    environments and every RNG stream are the same for every readout. Tier 2 uses
    readout="frobenius" for Backbone C (experiments/e4_second_backbone.py).
    """
    t0 = time.time()
    rows = []
    for seed in SEEDS:
        ds, mu, Wp, Y_obs = build(seed)
        X_obs = ds.environments["obs"].X
        # Precision null: shared across cases (all envs have N_BASE rows).
        prec_crit = PR.null_threshold(Y_obs, alpha=ALPHA, B=B_BOOT,
                                      rng=np.random.default_rng(910_000 + seed),
                                      n_env=N_BASE, readout=readout)
        # Subspace null: shared across cases (same X_obs, same n_env). Computed once.
        sub_crit = PR.subspace_null(X_obs, D_LATENT, alpha=ALPHA, B=B_BOOT,
                                    rng=np.random.default_rng(920_000 + seed),
                                    n_env=N_BASE)
        for name, gfn in CASES:
            rng = np.random.default_rng(930_000 + seed)
            if gfn is None:
                X_env = ds.environments["iv0"].X
            else:
                X_env = gain_env(ds, gfn(rng), rng)
            Y_env = BK.project(X_env, mu, Wp)
            fired = bool(PR.readout_signal(Y_env, Y_obs, readout) > prec_crit)
            r = PR.attribute_environment(X_env, X_obs, D_LATENT, fired,
                                         alpha=ALPHA, B=B_BOOT,
                                         subspace_crit=sub_crit)
            rows.append(dict(seed=seed, case=name, **r,
                             precision_crit=round(float(prec_crit), 5),
                             subspace_null_deg_shared=round(
                                 float(np.degrees(sub_crit)), 3)))
        print(f"[seed {seed}] done ({time.time() - t0:.0f}s)", flush=True)

    # ---- aggregate ----
    summary = {}
    for name, _ in CASES:
        rs = [r for r in rows if r["case"] == name]
        n = len(rs)
        det = [r for r in rs if r["detected"]]
        unatt = [r for r in det if r["verdict"] == "DETECTABLE_BUT_UNATTRIBUTED"]
        mech = [r for r in det if r["verdict"] == "MECHANISM_SUPPORTED"]
        angs = [r["subspace_angle_deg"] for r in det]
        summary[name] = dict(
            n_seeds=n,
            detected_rate=round(len(det) / n, 3),
            unattributed_rate_given_detected=(round(len(unatt) / len(det), 3)
                                              if det else None),
            mechanism_supported_rate_given_detected=(round(len(mech) / len(det), 3)
                                                     if det else None),
            angle_deg_mean=(round(float(np.mean(angs)), 2) if angs else None),
            angle_deg_sd=(round(float(np.std(angs)), 2) if angs else None),
            expected=EXPECT[name],
        )

    report = dict(
        experiment="e2b_attribution",
        alpha=ALPHA, B=B_BOOT, seeds=SEEDS,
        d_latent=D_LATENT, d_obs=D_OBS, d_proj=D_PROJ, n_base=N_BASE,
        iv_base=IV_BASE, edge_prob=EDGE_PROB,
        note=("Three-way verdict operating characteristics. Subspace null computed "
              "once per seed and shared across cases (same control, same n_env). "
              "Uniform-shrink landing MECHANISM_SUPPORTED is the documented Prop 3 "
              "blind spot, measured deliberately."),
        summary=summary, rows=rows,
        wall_seconds=round(time.time() - t0, 1),
    )
    return report


def print_table(report):
    summary = report["summary"]
    print(f"\n{'case':22s} {'det':>5s} {'unatt|det':>9s} {'mech|det':>8s} "
          f"{'angle':>8s}  expected", flush=True)
    for name, _ in CASES:
        s = summary[name]
        ua = "-" if s["unattributed_rate_given_detected"] is None else \
            f"{s['unattributed_rate_given_detected']:.2f}"
        ms = "-" if s["mechanism_supported_rate_given_detected"] is None else \
            f"{s['mechanism_supported_rate_given_detected']:.2f}"
        ang = "-" if s["angle_deg_mean"] is None else \
            f"{s['angle_deg_mean']:.1f}d"
        print(f"{name:22s} {s['detected_rate']:5.2f} {ua:>9s} {ms:>8s} {ang:>8s}  "
              f"{s['expected']}", flush=True)


def main():
    report = run()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attribution_report.json").write_text(json.dumps(report, indent=2))
    print_table(report)
    print(f"\nwritten {OUT / 'attribution_report.json'} "
          f"({report['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
