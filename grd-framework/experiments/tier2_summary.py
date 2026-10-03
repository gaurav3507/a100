"""Tier 2 artifact checks and the paste-ready summary table.

    python experiments/tier2_summary.py                  # print the summary table
    python experiments/tier2_summary.py --check e4_real:K562

--check validates one stage's artifact (exists, parses, expected shape) and prints
a one-line artifact count; exit 1 if it is missing or malformed. It judges the
artifact, never the scientific outcome. Used by experiments/run_tier2_a100.sh.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
E4 = REPO / "results" / "e4_second_backbone"
E5 = REPO / "results" / "e5_split_control"
N_SEEDS_SYNTH = 10
N_SEEDS_REAL = 5


def _load(path):
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------ checks
# Stage names: e4_* is Backbone C (the second backbone); e4nc_* is the covariance
# negative control (files e4_negctrl_*).
E4_PREFIX = {"e4": "e4", "e4nc": "e4_negctrl"}


def check(stage, root_e4=E4, root_e5=E5):
    """Return a one-line artifact description; raise on a bad artifact."""
    name, _, dataset = stage.partition(":")
    kind, _, part = name.partition("_")
    if name == "e4_popcheck":
        d = _load(root_e4 / "e4_population_check.json")
        assert len(d["per_seed"]) == N_SEEDS_SYNTH
        assert d["passed"], (f"population check FAILED (mean {d['population_mcc_mean']}, "
                             f"min {d['population_mcc_min']}, rule {d['rule']})")
        return (f"{N_SEEDS_SYNTH} seeds, population MCC mean {d['population_mcc_mean']} "
                f"min {d['population_mcc_min']}: PASS")
    if kind in E4_PREFIX and part == "calibration":
        d = _load(root_e4 / f"{E4_PREFIX[kind]}_calibration_report.json")
        arms = d["arms"]
        assert len(arms) == 4, f"expected 4 arms, got {len(arms)}"
        for a in arms.values():
            assert len(a["levels"]) == 5
            assert all(len(lv["per_seed"]) == N_SEEDS_SYNTH for lv in a["levels"])
        assert "population_decomposition" in d and "precision_same_code" in d
        assert ("headline" in d) if kind == "e4nc" else ("population_check" in d)
        return (f"{d['role']}: 4 arms x 5 levels x {N_SEEDS_SYNTH} seeds; contract "
                f"status {d['status']}")
    if kind in E4_PREFIX and part == "starvation":
        d = _load(root_e4 / f"{E4_PREFIX[kind]}_starvation_report.json")
        assert d["result_eligible"] and d["status"] == "PASS"
        assert len(d["per_seed"]) == N_SEEDS_SYNTH
        return (f"{d['role']}: {len(d['per_seed'])} seeds x "
                f"{len(d['aggregate']['levels'])} levels; restricts at or before MCC "
                f"crossover: {d['aggregate']['gate_restricts_at_or_before_mcc_crossover']}")
    if (kind in E4_PREFIX and part == "real") or name == "e5_real":
        path = (root_e5 / "e5_real_panel.json" if name == "e5_real"
                else root_e4 / f"{E4_PREFIX[kind]}_real_panel.json")
        block = _load(path)["datasets"][dataset]
        assert not block["selftest"], "selftest block in a results artifact"
        assert len(block["per_seed"]) == N_SEEDS_REAL
        n = block.get("n_environments", block.get("n_perts"))
        return f"{dataset}: {N_SEEDS_REAL} seeds x {n} environments"
    if name == "e5_confound":
        d = _load(root_e5 / "e5_rpe1_confound.json")
        assert not d["selftest"]
        fams = d["shared_vs_split"]
        assert set(fams) == {"perturbations", "random_controls", "structured_controls"}
        return "3 families: " + ", ".join(
            f"{k} {v['n_environments']}" for k, v in fams.items())
    if name == "e5_poscontrol":
        out = []
        for fname, n_obs in (("e5_poscontrol.json", 200),
                             ("e5_poscontrol_obs400.json", 400)):
            d = _load(root_e5 / fname)
            assert not d["selftest"] and len(d["dose_response"]) == 5
            assert d["n_obs"] == n_obs
            out.append(f"{n_obs} obs cells: {len(d['dose_response']) + 1} ratios")
        return "; ".join(out)
    raise SystemExit(f"unknown stage {stage!r}")


# ------------------------------------------------------------------ summary
def _jaccard(a, b):
    a, b = set(a), set(b)
    return None if not (a | b) else len(a & b) / len(a | b)


def _fmt_counts(values):
    mean = sum(values) / len(values)
    return f"{mean:6.1f} [{min(values)}-{max(values)}]"


def _jaccard_mean(e4_block, e5_block):
    """Mean BH Jaccard of an E4 readout against the precision gate, same seeds.

    Uses E5's shared design (the precision gate on the same data in the same run)
    when present, else the committed results/e3_stability sets stored in E4.
    """
    if e5_block is not None:
        ref = {r["seed"]: r["bh_selected_ids"]["corrected_disjoint"]
               for r in e5_block["per_seed"]}
        jac = [_jaccard(r["bh_selected_ids"], ref[r["seed"]])
               for r in e4_block["per_seed"] if r["seed"] in ref]
    else:
        jac = [r["jaccard_bh_vs_precision"]["e3_stability_same_seed"]
               for r in e4_block["per_seed"]]
    vals = [j for j in jac if j is not None]
    return "both empty" if not vals else f"{sum(vals) / len(vals):.3f}"


def summary():
    lines = ["", "TIER 2 SUMMARY (real panel: mean over gate seeds [min-max])", ""]
    header = (f"{'dataset':8} {'readout':22} {'design':14} {'n_env':>5}  "
              f"{'raw':>14}  {'BH':>14}  {'BH Jaccard vs precision'}")
    lines += [header, "-" * len(header)]
    real = {}
    for key, fname in (("C", E4 / "e4_real_panel.json"),
                       ("nc", E4 / "e4_negctrl_real_panel.json"),
                       ("e5", E5 / "e5_real_panel.json")):
        real[key] = _load(fname)["datasets"] if fname.exists() else {}
    for ds in ("K562", "RPE1", "Norman"):
        first = True

        def label():
            nonlocal first
            out = ds if first else ""
            first = False
            return out
        e5 = real["e5"].get(ds)
        if e5 is not None:
            per = e5["shared_vs_split"]["per_seed"]
            n = e5["n_perts"]
            lines.append(f"{label():8} {'precision':22} {'shared (E3)':14} {n:>5}  "
                         f"{_fmt_counts([e['shared']['raw_count'] for e in per])}  "
                         f"{_fmt_counts([e['shared']['bh_count'] for e in per])}  (reference)")
            jac = e5["shared_vs_split"]["summary"]["jaccard_bh_mean"]
            lines.append(f"{label():8} {'precision':22} {'split control':14} {n:>5}  "
                         f"{_fmt_counts([e['split']['raw_count'] for e in per])}  "
                         f"{_fmt_counts([e['split']['bh_count'] for e in per])}  "
                         f"{'both empty' if jac is None else f'{jac:.3f}'}")
        for key, text in (("C", "Backbone C frobenius"),
                          ("nc", "covariance (neg. ctrl)")):
            b = real[key].get(ds)
            if b is None:
                continue
            lines.append(f"{label():8} {text:22} {'shared':14} {b['n_environments']:>5}  "
                         f"{_fmt_counts(b['raw_count_per_seed'])}  "
                         f"{_fmt_counts(b['bh_count_per_seed'])}  {_jaccard_mean(b, e5)}")
    if not any(real.values()):
        lines.append("(no real-panel artifacts yet)")

    conf_path = E5 / "e5_rpe1_confound.json"
    if conf_path.exists():
        c = _load(conf_path)["shared_vs_split"]
        lines += ["", "RPE1 control splits (seed 0), raw/BH of n: shared -> split"]
        for fam, e in c.items():
            lines.append(f"  {fam:20} {e['shared']['raw_count']}/{e['shared']['bh_count']}"
                         f" -> {e['split']['raw_count']}/{e['split']['bh_count']} of "
                         f"{e['n_environments']}  (BH Jaccard {e['jaccard_bh']})")

    for fname in ("e5_poscontrol.json", "e5_poscontrol_obs400.json"):
        path = E5 / fname
        if not path.exists():
            continue
        doc = _load(path)
        p = doc["shared_vs_split"]
        lines += ["", f"Positive control, {doc['n_obs']} observational cells: BH-selected "
                  f"planted of 3 (null draws BH of 30), shared -> split"]
        for snr in sorted(p, key=float):
            e = p[snr]
            lines.append(f"  ratio {float(snr):>4g}: planted {e['planted']['shared']['bh_count']}"
                         f" -> {e['planted']['split']['bh_count']} | null "
                         f"{e['null_draws']['shared']['bh_count']} -> "
                         f"{e['null_draws']['split']['bh_count']}")

    pop_path = E4 / "e4_population_check.json"
    if pop_path.exists():
        pop = _load(pop_path)
        lines += ["", f"Backbone C population check: MCC mean {pop['population_mcc_mean']}, "
                  f"min {pop['population_mcc_min']} (rule {pop['rule']}): "
                  f"{'PASS' if pop['passed'] else 'FAIL'}"]
    for prefix, title in (("e4", "Backbone C (second backbone)"),
                          ("e4_negctrl", "Covariance readout (negative control)")):
        cal_path = E4 / f"{prefix}_calibration_report.json"
        if not cal_path.exists():
            continue
        cal = _load(cal_path)
        lines += ["", f"{title}: synthetic arms, this readout vs same-code precision"]
        for arm, cmp in cal["comparison"].items():
            lines.append(f"  {arm:30} {cmp['this_readout']['gate_behavior']:20} vs "
                         f"{cmp['precision']['gate_behavior']}")
        dec = cal["population_decomposition"]
        rule = "jad_rule" if prefix == "e4" else "covariance_rule"
        lines.append(f"  recovery MCC: population {dec[rule]['population']['mean']}, sample "
                     f"{dec[rule]['sample']['mean']} (precision rule: population "
                     f"{dec['precision_rule']['population']['mean']})")
        if "headline" in cal:
            lines.append(f"  headline: {cal['headline']['statement']}")
        st_path = E4 / f"{prefix}_starvation_report.json"
        if st_path.exists():
            a = _load(st_path)["aggregate"]
            lines.append(f"  starvation: gate first restricts at "
                         f"m={a['crossover_gate_first_restricts']}, MCC < 0.90 from "
                         f"m={a['crossover_random_subset_mcc_below_0p90']}, restricts at "
                         f"or before: {a['gate_restricts_at_or_before_mcc_crossover']}")
    attr_path = E4 / "e4_attribution.json"
    if attr_path.exists():
        attr = _load(attr_path)
        comp = (attr.get("comparison_to_precision_e2b") or {}).get("per_case", {})
        lines += ["", "Backbone C attribution cases (detected rate Frobenius / precision; "
                  "UNATTRIBUTED given detected, Frobenius / precision)"]
        for case, c in comp.items():
            lines.append(f"  {case:22} {c['detected_rate']['frobenius']} / "
                         f"{c['detected_rate']['precision']}   "
                         f"{c['unattributed_rate_given_detected']['frobenius']} / "
                         f"{c['unattributed_rate_given_detected']['precision']}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", default=None)
    args = p.parse_args()
    if args.check:
        try:
            print(check(args.check))
        except SystemExit:
            raise
        except Exception as exc:  # missing file, bad JSON, wrong shape
            print(f"bad artifact: {type(exc).__name__}: {exc}")
            sys.exit(1)
        return
    print(summary())


if __name__ == "__main__":
    main()
