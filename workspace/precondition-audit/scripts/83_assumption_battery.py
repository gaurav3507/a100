"""Nonlinearity-immune components of the CRL assumption battery.

WHY THIS FILE EXISTS
--------------------
The rank diagnostic in 80_ranktest_core.py is only valid under LINEAR mixing.
Gate 2 and the fine-grid follow-up measured how fast that fails on a
leaky-ReLU mixing sweep of scale s, and crucially the operating envelope
SHRINKS AS THE LATENT DIMENSION GROWS (400 draws/point, k=1, n_e=2000):

    s              0.01   0.02   0.05   0.075   0.10
    d_latent=10    0.050  0.050  0.050  0.050   0.062   raw
    d_latent=20    0.037  0.037  0.147  0.230   0.230   raw
    d_latent=20    0.022  0.022  0.045  0.193   0.215   standardised

At d_latent=10 the test holds level to s=0.1; at d_latent=20 raw already
breaks by s=0.05. The largest s holding rejection <= 0.075 in both scalings
is 0.1 at d_latent=10 but only 0.02 at d_latent=20. Real latent dimension is
larger than 20, so the usable envelope on real data is smaller still than
0.02 and cannot be quoted as a single number.

On real data the mixing is not linear, so a rank rejection is confounded with
mixing nonlinearity and cannot be attributed to the intervention. Phase B is
consequently NOT AUTHORISED.

The three checks here are deliberately chosen so that NONE of them depends on
the mixing being linear:

  env_count_check          pure counting on the experimental design. No model.
  intervention_type_check  reads the SHAPE of the observed second-moment
                           shift (how concentrated its spectrum is), not its
                           magnitude and not its rank in a latent space.
                           Monotone reparameterisations of the mixing change
                           the numbers but not the hard-vs-soft ordering.
  power_floor_check        pure arithmetic against sample-size floors that
                           were measured, under linear mixing, on the
                           simulator. See the caveat on that function.

Each returns a dict with a "verdict" and the number the verdict rests on.
None of them is a substitute for the rank test; they bound what can be
claimed, they do not certify the bundle.

INTERPRETATION RULE, carried over unchanged: a FAIL is informative, a PASS is
not evidence that the assumption holds.

Oracle tests: run this file directly.
    python causalbench/scripts/83_assumption_battery.py --self-test
Nothing here touches real data.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "ranktest"

# Measured LFC power floors: smallest n_e reaching 0.80 power at k=3 in BOTH
# scalings, from the simulator sweeps. Loaded from disk when available so the
# battery cannot drift from the curves it claims to encode.
SOFT_FLOOR_DEFAULT = 8000
HARD_FLOOR_DEFAULT = None          # filled from hard_power_curve_lfc.json


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    real = os.makedirs
    os.makedirs = lambda *a, **k: None
    try:
        spec.loader.exec_module(mod)
    finally:
        os.makedirs = real
    return mod


def load_power_floors(results_dir=RESULTS):
    """Read the measured floors off disk; fall back to the recorded defaults."""
    soft, hard = SOFT_FLOOR_DEFAULT, HARD_FLOOR_DEFAULT
    try:
        curve = json.load(open(Path(results_dir) / "soft_power_curve_lfc.json"))
        ns = sorted({c["n_e"] for c in curve})
        ok = [n for n in ns
              if all(c["reject"] >= 0.80 for c in curve if c["n_e"] == n)]
        soft = ok[0] if ok else None
    except (OSError, ValueError, KeyError):
        pass
    try:
        h = json.load(open(Path(results_dir) / "hard_power_curve_lfc.json"))
        hard = h.get("floor", hard)
    except (OSError, ValueError, KeyError):
        pass
    return dict(soft=soft, hard=hard)


# --------------------------------------------------------------------- (1)
def env_count_check(n_env, d_latent_est, includes_observational=True):
    """Does the environment count meet the one-per-latent-node requirement?

    Multi-environment CRL identifiability theorems require at least one
    intervention per latent node, so with an observational environment the
    design needs d_latent + 1 environments in total (d_latent interventional
    plus the observational reference). Without an observational environment
    the requirement is stated on the interventional count alone.

    This is pure counting on the experimental design. It involves no model of
    the mixing at all, so nonlinearity cannot affect it.

    A FAIL here is decisive and cheap: no amount of data fixes a design that
    does not perturb enough distinct latent directions, and every downstream
    identifiability claim is void regardless of what the rank test says.
    A PASS says only that the COUNT is sufficient. It does not check that the
    interventions hit DISTINCT nodes, which is a separate and harder question.
    """
    n_env = int(n_env)
    d_latent_est = int(d_latent_est)
    required = d_latent_est + 1 if includes_observational else d_latent_est
    deficit = required - n_env
    return dict(
        check="env_count",
        verdict="PASS" if n_env >= required else "FAIL",
        n_env=n_env,
        d_latent_est=d_latent_est,
        required=required,
        deficit=max(0, deficit),
        includes_observational=bool(includes_observational),
        basis="one intervention per latent node, plus the observational "
              "reference when present",
        caveat="counts environments only; does not verify that they perturb "
               "DISTINCT latent nodes",
    )


# --------------------------------------------------------------------- (2)
# Measured usable range for intervention_type_check; see its docstring table.
GAP_RATIO_USABLE_N = 8000

def intervention_type_check(Y_env, Y_ref, gap_ratio_threshold=0.25,
                            target_expression=None):
    """Hard vs soft, from the spectrum of the second-moment shift.

    A soft intervention rescales one node's noise variance, which is a RANK-1
    update to the latent covariance. A hard intervention additionally cuts the
    node's incoming edges, which makes it RANK-2 (rank 1 only when the node is
    already a source). So the discriminator is how much of the shift lives in
    the second eigendirection:

        gap_ratio = sigma_2(Delta) / sigma_1(Delta)

    gap_ratio near 0  -> one direction carries the shift  -> SOFT (or a hard
                         intervention on a source node, which is genuinely
                         indistinguishable from soft at second order)
    gap_ratio large   -> two comparable directions        -> HARD

    WHY THIS SURVIVES NONLINEAR MIXING. The verdict uses a RATIO of singular
    values of the observed shift, not its rank in any latent space and not its
    magnitude. A smooth invertible mixing distorts the two directions but does
    not manufacture a second comparable direction where there was one, nor
    collapse two into one, except in degenerate cases. The ordering
    soft < hard in gap_ratio is preserved; the threshold value is not exact
    under nonlinearity, which is why the result is reported with its number
    and an INCONCLUSIVE band rather than as a bare label.

    gap_ratio_threshold = 0.25 is a CHOSEN CONSTANT, not derived from theory.
    Measured on the simulator it turns out to be close to the empirically
    optimal split at large n (best threshold 0.269 at n=8000, 0.254 at
    n=20000), but well off it at small n (0.498 at n=2000). Values within
    +/- 0.10 of the threshold return INCONCLUSIVE rather than a label.

    THIS CHECK IS ONLY USABLE AT LARGE n_e. Measured separation of soft from
    hard-on-non-source, 30 seeds per point, linear mixing:

        n_e     soft med   hard med   accuracy@0.25   AUC
        500       0.401      0.483        0.600       0.602
        2000      0.264      0.545        0.633       0.717
        8000      0.113      0.467        0.800       0.831
        20000     0.107      0.499        0.767       0.834

    At n_e=500 this is barely better than a coin flip (AUC 0.602) and MUST
    NOT be used per-environment. It becomes informative around n_e>=8000
    (AUC ~0.83, accuracy ~0.78-0.80), and even there roughly one environment
    in five is mislabelled. Treat the verdict as a population-level summary,
    not a per-environment fact, and prefer the target-expression cross-check
    below whenever the target gene is present as an expression column. The
    returned dict carries n_env and a usable flag so a caller cannot silently
    apply it below the usable range.

    target_expression: OPTIONAL CROSS-CHECK HOOK for Perturb-seq. Pass
    (mean_expression_in_env, mean_expression_in_control) for the TARGETED gene
    when the target appears as an expression column. A CRISPR knockout should
    drive the target's own expression sharply down; CRISPRi knockdown and
    CRISPRa give partial or upward shifts. When supplied, the log2 fold change
    is returned as target_lfc alongside a concordance flag, so a spectral
    "HARD" that is not accompanied by strong target depletion is visible.
    NOTE for this project specifically: Norman had 0 of 105 targets present as
    expression columns, so the hook is inert there; Frangieh's targets ARE
    present, so it is live. See 41_screen_frangieh.py.
    """
    Y_env = np.asarray(Y_env)
    Y_ref = np.asarray(Y_ref)
    if Y_env.shape[1] != Y_ref.shape[1]:
        raise ValueError("environment and reference have different widths")
    Delta = np.cov(Y_env, rowvar=False) - np.cov(Y_ref, rowvar=False)
    sv = np.linalg.svd(Delta, compute_uv=False)
    if sv[0] <= 0:
        return dict(check="intervention_type", verdict="INCONCLUSIVE",
                    gap_ratio=None, reason="degenerate: sigma_1 = 0")
    gap_ratio = float(sv[1] / sv[0])

    lo, hi = gap_ratio_threshold - 0.10, gap_ratio_threshold + 0.10
    if gap_ratio < lo:
        verdict = "SOFT"
    elif gap_ratio > hi:
        verdict = "HARD"
    else:
        verdict = "INCONCLUSIVE"

    n_env = int(Y_env.shape[0])
    usable = n_env >= GAP_RATIO_USABLE_N
    if not usable and verdict != "INCONCLUSIVE":
        # Below the measured usable range the classifier is near chance
        # (AUC 0.602 at n=500). Refuse to emit a confident label rather than
        # let a caller act on one.
        verdict = "INCONCLUSIVE"
    out = dict(
        check="intervention_type",
        verdict=verdict,
        raw_verdict_before_n_guard=(
            "SOFT" if gap_ratio < lo else ("HARD" if gap_ratio > hi else "INCONCLUSIVE")),
        n_env=n_env,
        usable_n=bool(usable),
        usable_n_threshold=GAP_RATIO_USABLE_N,
        gap_ratio=gap_ratio,
        threshold=float(gap_ratio_threshold),
        inconclusive_band=[float(lo), float(hi)],
        sigma_1=float(sv[0]), sigma_2=float(sv[1]),
        caveat="a hard intervention on a SOURCE node is rank 1 and reads as "
               "SOFT; this is a real ambiguity, not a defect",
    )
    if target_expression is not None:
        env_mean, ctrl_mean = (float(x) for x in target_expression)
        eps = 1e-9
        lfc = float(np.log2((env_mean + eps) / (ctrl_mean + eps)))
        out["target_lfc"] = lfc
        out["target_depleted"] = bool(lfc <= -1.0)
        out["spectral_target_concordant"] = bool(
            (verdict == "HARD") == (lfc <= -1.0)) if verdict != "INCONCLUSIVE" else None
        out["target_hook"] = ("log2 fold change of the TARGETED gene's own "
                              "expression, env vs control; <= -1 read as "
                              "strong depletion consistent with knockout")
    return out


# --------------------------------------------------------------------- (3)
def power_floor_check(n_per_env, intervention_type, floors=None):
    """Are there enough cells per environment for the rank test to have power?

    Compares n_per_env against the LFC-derived floors measured on the
    simulator: the smallest n_e at which k=3 power reached 0.80 in BOTH raw
    and standardised scalings.

    Returns POWERED, UNDERPOWERED, or INCONCLUSIVE. INCONCLUSIVE is returned
    when the intervention type is unknown or when the relevant floor has not
    been measured, because guessing either way would be worse than saying so:
    an UNDERPOWERED verdict wrongly issued would discard usable environments,
    and a POWERED one wrongly issued would license a non-rejection that means
    nothing.

    CAVEAT, stated because it matters. These floors were measured under LINEAR
    mixing on the simulator. They are the right order of magnitude for
    planning and are the only measured numbers available, but they are not
    guaranteed under the nonlinear mixing of real data, and Gate 2 shows the
    test's level itself degrades there. Treat a POWERED verdict as necessary,
    not sufficient.
    """
    floors = load_power_floors() if floors is None else dict(floors)
    t = (intervention_type or "").strip().upper()
    key = {"SOFT": "soft", "HARD": "hard"}.get(t)

    if key is None:
        return dict(check="power_floor", verdict="INCONCLUSIVE",
                    n_per_env=int(n_per_env), intervention_type=t or None,
                    floors=floors,
                    reason="intervention type unknown or INCONCLUSIVE; the "
                           "soft and hard floors differ by orders of "
                           "magnitude so one cannot stand in for the other")
    floor = floors.get(key)
    if floor is None:
        return dict(check="power_floor", verdict="INCONCLUSIVE",
                    n_per_env=int(n_per_env), intervention_type=t,
                    floors=floors,
                    reason=f"the {key} power floor has not been measured")

    n = int(n_per_env)
    return dict(
        check="power_floor",
        verdict="POWERED" if n >= floor else "UNDERPOWERED",
        n_per_env=n,
        intervention_type=t,
        floor=int(floor),
        shortfall=max(0, int(floor) - n),
        floors=floors,
        basis="smallest n_e reaching 0.80 power at k=3 in both scalings, "
              "LFC statistic, linear-mixing simulator",
        caveat="floors measured under LINEAR mixing; necessary, not sufficient",
    )


# ------------------------------------------------------------- oracle tests
def _self_test():
    """Oracle tests on the existing simulator, where the answer is known."""
    orc = _load_module(HERE / "81_ranktest_oracle.py", "_orc_for_battery")
    core = _load_module(HERE / "80_ranktest_core.py", "_core_for_battery")
    fails = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<52} got={got!r} want={want!r}")
        if not ok:
            fails.append(name)

    print("\n(1) env_count_check -- pure counting, answer known by construction")
    check("d_latent=10, 11 envs incl observational",
          env_count_check(11, 10)["verdict"], "PASS")
    check("d_latent=10, 10 envs incl observational",
          env_count_check(10, 10)["verdict"], "FAIL")
    check("d_latent=10, 10 envs, no observational",
          env_count_check(10, 10, includes_observational=False)["verdict"], "PASS")
    check("deficit reported correctly",
          env_count_check(4, 10)["deficit"], 7)

    print("\n(2) intervention_type_check -- simulator, true type known")
    dl, D, n, d = 10, 200, 8000, 10   # at/above GAP_RATIO_USABLE_N
    got = {"soft": [], "hard_nonsource": [], "hard_source": []}
    medians = {}
    for seed in range(8):
        rng = np.random.default_rng(70_000 + seed)
        B_, nv, is_src = orc.make_scm(dl, rng)
        A = rng.standard_normal((D, dl))
        Zc = orc.sample_latent(B_, nv, 4 * n, rng)
        Xc = orc.mix_linear(Zc, A)
        sd = 0.1 * float(np.mean(Xc.std(0)))
        Xc = orc.add_obs_noise(Xc, sd, rng)
        mu, W = core.fit_pca(Xc[:2 * n], d)
        Yref = core.project(Xc[2 * n:], mu, W, ())
        nonsrc = [i for i in range(dl) if not is_src[i]]
        srcs = [i for i in range(dl) if is_src[i]]
        cases = [("soft", "soft", nonsrc), ("hard_nonsource", "hard", nonsrc),
                 ("hard_source", "hard", srcs)]
        for label, kind, pool in cases:
            if not pool:
                continue
            riv = np.random.default_rng(80_000 + seed)
            node = int(riv.choice(pool, size=1)[0])
            Ze = orc.sample_latent(B_, nv, n, rng, kind=kind, nodes=[node],
                                   rng_iv=riv)
            Xe = orc.add_obs_noise(orc.mix_linear(Ze, A), sd, rng)
            Ye = core.project(Xe, mu, W, ())
            got[label].append(intervention_type_check(Ye, Yref[:n]))
    for label in got:
        if not got[label]:
            continue
        gr = np.median([g["gap_ratio"] for g in got[label]])
        vs = [g["verdict"] for g in got[label]]
        mode = max(set(vs), key=vs.count)
        medians[label] = gr
        print(f"    {label:<16} median gap_ratio={gr:.3f}  verdicts={ {v: vs.count(v) for v in set(vs)} }")
        if label == "soft":
            check("soft -> SOFT (rank-1 shift)", mode, "SOFT")
        elif label == "hard_nonsource":
            check("hard on non-source -> HARD (rank-2 shift)", mode, "HARD")
        elif label == "hard_source":
            # documented ambiguity: a source node has no incoming edges to cut
            check("hard on SOURCE -> SOFT (documented ambiguity)", mode, "SOFT")

    # The mode alone is a weak test: it passes even at near-chance accuracy.
    # Assert the DIRECTION of separation, which is what the check rests on.
    check("soft median gap_ratio < hard non-source median",
          medians["soft"] < medians["hard_nonsource"], True)
    check("n-guard fires below the usable range",
          intervention_type_check(np.random.default_rng(0).standard_normal((500, 10)),
                                  np.random.default_rng(1).standard_normal((500, 10)))["usable_n"],
          False)

    print("\n    target-expression cross-check hook:")
    h = intervention_type_check(Ye, Yref[:n], target_expression=(0.2, 4.0))
    print(f"    knockout-like target lfc={h['target_lfc']:.2f} "
          f"depleted={h['target_depleted']}")
    check("strong target depletion detected", h["target_depleted"], True)
    h2 = intervention_type_check(Ye, Yref[:n], target_expression=(3.9, 4.0))
    check("no depletion when target unchanged", h2["target_depleted"], False)

    print("\n(3) power_floor_check -- against the measured floors")
    fl = load_power_floors()
    print(f"    floors loaded from disk: {fl}")
    if fl["soft"]:
        check("soft, n below floor -> UNDERPOWERED",
              power_floor_check(fl["soft"] - 1, "SOFT")["verdict"], "UNDERPOWERED")
        check("soft, n at floor -> POWERED",
              power_floor_check(fl["soft"], "SOFT")["verdict"], "POWERED")
    if fl["hard"]:
        check("hard, n below floor -> UNDERPOWERED",
              power_floor_check(fl["hard"] - 1, "HARD")["verdict"], "UNDERPOWERED")
        check("hard, n at floor -> POWERED",
              power_floor_check(fl["hard"], "HARD")["verdict"], "POWERED")
    check("unknown type -> INCONCLUSIVE",
          power_floor_check(10_000, "INCONCLUSIVE")["verdict"], "INCONCLUSIVE")
    check("missing floor -> INCONCLUSIVE",
          power_floor_check(10_000, "HARD",
                            floors=dict(soft=8000, hard=None))["verdict"],
          "INCONCLUSIVE")

    print(f"\n{'ALL ORACLE TESTS PASSED' if not fails else 'FAILURES: ' + ', '.join(fails)}")
    return not fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if _self_test() else 1)
    print(__doc__)
    print("floors:", load_power_floors())
