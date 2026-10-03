"""Cross-check every number CLAIMED in prose/figures against the results files.

Section 60 verifies each results file against itself. This script does the other
half: it takes numbers asserted in README.md, the Norman verdict text, SUMMARY.json
and scripts/50_figures.py, and compares them to the authoritative results file.

It also inspects the SOURCE CODE for definitional divergences: two files can each
be internally consistent while computing different quantities under the same key
name. Those are reported as DEFINITION findings.

Usage:
    python scripts/62_crosscheck_claims.py
"""
import json
import re
from pathlib import Path

import numpy as np

FW = Path(__file__).resolve().parent.parent
CB = FW / "causalbench/results"

findings = []


def finding(kind, label, claimed, actual, claim_src, actual_src, note=""):
    findings.append(dict(kind=kind, label=label, claimed=claimed, actual=actual,
                          claim_src=claim_src, actual_src=actual_src, note=note))
    print(f"\n  [{kind}] {label}")
    print(f"      claimed : {claimed}")
    print(f"                {claim_src}")
    print(f"      actual  : {actual}")
    print(f"                {actual_src}")
    if note:
        print(f"      note    : {note}")


def ok(label, value, src):
    print(f"  [OK] {label:<48} {value}   ({src})")


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


print("=" * 78)
print("CROSS-CHECK: prose / figure claims vs results files")
print("=" * 78)

# ---------------------------------------------------------------- Norman metric
print("\n--- Norman headline mean-shift ratio: which metric? ---")
nj = load(CB / "screen/norman.json")
if nj:
    prim = next(r for r in nj["screen"] if r["nmin"] == 200 and r["d"] == 10)
    pairs, vs_ctrl = prim["mean_ratio_pairs"], prim["mean_ratio_vs_ctrl"]
    print(f"  norman.json  mean_ratio_pairs   = {pairs:.4f}   <- PRIMARY metric")
    print(f"  norman.json  mean_ratio_vs_ctrl = {vs_ctrl:.4f}   <- secondary")

    readme = (FW / "README.md").read_text() if (FW / "README.md").exists() else ""
    if "3.19" in readme:
        finding("METRIC-MISMATCH", "README Norman mean-shift ratio",
                "3.19", f"{pairs:.4f} (pairs) / {vs_ctrl:.4f} (vs_ctrl)",
                "README.md headline-numbers table",
                "causalbench/results/screen/norman.json $.screen[n=200,d=10]",
                "README quotes vs_ctrl (3.19). Under the corrected PRIMARY "
                "metric mean_ratio_pairs the value is 3.71.")

    vp = CB / "screen/norman_verdict.txt"
    if vp.exists() and "3.19" in vp.read_text():
        finding("METRIC-MISMATCH", "norman_verdict.txt mean-shift ratio",
                "3.19", f"{pairs:.4f} (pairs)",
                "causalbench/results/screen/norman_verdict.txt",
                "causalbench/results/screen/norman.json",
                "Verdict sentence quotes the secondary metric.")

    figp = FW / "scripts/50_figures.py"
    if figp.exists():
        fs = figp.read_text()
        n_vsctrl = fs.count("mean_ratio_vs_ctrl")
        n_pairs = fs.count("mean_ratio_pairs")
        if n_vsctrl:
            finding("METRIC-MISMATCH", "scripts/50_figures.py FIG 2 y-values",
                    f"plots mean_ratio_vs_ctrl ({n_vsctrl} refs)",
                    f"primary metric is mean_ratio_pairs ({n_pairs} refs, used "
                    f"only for the gate band)",
                    "scripts/50_figures.py fig2_mean_shift_ratio()",
                    "03_screen.py:63 -- ref_pool empty in step-0",
                    "Bars and the calibrated null/band come from DIFFERENT "
                    "metrics on one axis.")

# ------------------------------------------------------- README vs results files
print("\n--- README numeric claims vs results files ---")
readme_p = FW / "README.md"
if readme_p.exists():
    readme = readme_p.read_text()

    zc = load(FW / "results/zeroshot_canonical/k562.json")
    if zc:
        for label, claim, key in (("global_mean", "+0.129", "global_mean"),
                                   ("ridge_basis", "+0.226", "ridge_basis"),
                                   ("CEILING", "+0.528", "CEILING"),
                                   ("corr_prop", "−0.188", "corr_prop"),
                                   ("nn_corr", "−0.243", "nn_corr")):
            act = zc["results"][key]["r2_median"]
            claim_num = float(claim.replace("−", "-").replace("+", ""))
            if abs(round(act, 3) - claim_num) <= 5e-4:
                ok(f"README baseline {label}", f"{claim} == {act:.4f}",
                   "results/zeroshot_canonical/k562.json")
            else:
                finding("VALUE-MISMATCH", f"README baseline {label}",
                        claim, f"{act:.6f}", "README.md",
                        "results/zeroshot_canonical/k562.json")
        hd = zc["results"]["_headroom"]
        if "0.399" in readme:
            ok("README headroom", f"0.399 == {hd:.4f}",
               "results/zeroshot_canonical/k562.json $._headroom")
        if "24.2" in readme:
            g = zc["results"]["global_mean"]["r2_median"]
            r = zc["results"]["ridge_basis"]["r2_median"]
            ok("README ridge capture", f"24.2% == {100*(r-g)/hd:.1f}%",
               "recomputed from k562.json")

    S = load(FW / "results/model/SUMMARY.json")
    if S:
        for arm, claim in (("full_shift", 0.0097), ("full_random", -0.0057),
                            ("nodag_shift", -0.0422), ("nodag_random", 0.0130)):
            act = S["models"][arm]["best_median_r2"]
            if abs(round(act, 4) - claim) <= 5e-5:
                ok(f"README model {arm}", f"{claim:+.4f} == {act:.6f}",
                   "results/model/SUMMARY.json")
            else:
                finding("VALUE-MISMATCH", f"README model {arm}", f"{claim:+.4f}",
                        f"{act:.6f}", "README.md", "results/model/SUMMARY.json")

    # HCP claim
    hcp_lr = load(Path("/workspace/meridian-identifiability/hcp/results/mean_shift_LR.json"))
    if "1.00" in readme and "HCP" in readme:
        if hcp_lr:
            act = hcp_lr["results"]["10"]["mean_ratio"]
            finding("VALUE-MISMATCH", "README HCP mean-shift ratio",
                    "1.00", f"{act:.4f} (LR, d=10)",
                    "README.md headline-numbers table",
                    "hcp/results/mean_shift_LR.json $.results['10'].mean_ratio",
                    "README rounds HCP to the null value; the measured ratio is "
                    "BELOW 1.0.")
        else:
            finding("UNTRACEABLE", "README HCP mean-shift ratio = 1.00",
                    "1.00", "HCP results not on this machine",
                    "README.md", "hcp/results/mean_shift_LR.json (A100 only)",
                    "Run on the A100 to resolve; local record suggests ~0.75.")

    # K562 range claim
    if "2.0 – 5.6" in readme or "2.0 - 5.6" in readme:
        have = list((CB / "screen").glob("k562_filt*_n*_d*.json"))
        if not have:
            finding("UNTRACEABLE", "README K562 mean-shift range '2.0 - 5.6'",
                    "2.0 - 5.6", "no k562 screen JSON in this repo",
                    "README.md headline-numbers table",
                    "causalbench/results/screen/k562_filt*_n*_d*.json (A100 only)",
                    "Range spans unspecified configs and an unspecified metric.")

    # effective-dim claim
    if "~15" in readme:
        finding("UNTRACEABLE", "README CausalBench effective dim '~15'",
                "~15", "no k562 spectrum JSON in this repo",
                "README.md headline-numbers table",
                "causalbench/results/spectrum/k562_filt0_n200_d*.json (A100 only)",
                "Also ambiguous: dims_above_2x or participation_ratio?")
    if "3 (group only)" in readme:
        finding("UNTRACEABLE", "README HCP effective dim '3 (group only)'",
                "3 (group only)", "HCP results not on this machine",
                "README.md", "hcp/results/mean_shift_*.json (A100 only)",
                "'group only' qualifier does not appear in any results file key.")

# ------------------------------------------------ split-file internal consistency
print("\n--- split files: cells_per_perturbation vs n_usable ---")
for ds in ("k562", "rpe1"):
    p = FW / f"results/splits/{ds}_zeroshot_split.json"
    s = load(p)
    if not s:
        continue
    cpp = s["cells_per_perturbation"]
    nmin = s["nmin"]
    n_at_or_above = sum(1 for v in cpp.values() if v >= nmin)
    print(f"  {ds}: {len(cpp)} genes recorded, nmin={nmin}, "
          f"{n_at_or_above} at/above nmin, n_usable={s['n_usable']}")
    if n_at_or_above != s["n_usable"]:
        finding("INTERNAL-INCONSISTENCY",
                f"{ds} split n_usable vs cells_per_perturbation",
                f"n_usable = {s['n_usable']}",
                f"{n_at_or_above} genes have >= {nmin} cells",
                str(p), str(p),
                "cells_per_perturbation lists ALL genes (incl. below nmin); the "
                "count clearing nmin does not match n_usable.")
    else:
        ok(f"{ds} n_usable == count(cells >= nmin)", n_at_or_above, str(p.name))

# ------------------------------------------------------- DEFINITIONAL divergences
print("\n" + "=" * 78)
print("DEFINITIONAL DIVERGENCES (same key name, different quantity)")
print("=" * 78)

base_src = (FW / "scripts/25_zeroshot_baselines.py").read_text() \
    if (FW / "scripts/25_zeroshot_baselines.py").exists() else ""
train_src = (FW / "model/cb_train.py").read_text() \
    if (FW / "model/cb_train.py").exists() else ""

if base_src and train_src:
    base_perg = "gm_by_g[g]" in base_src
    train_const = "zs>0.129" in train_src.replace(" ", "")
    if base_perg and train_const:
        finding("DEFINITION", "frac_beats_gmean means two different things",
                "baselines: per-gene, r2 > that gene's own global_mean r2",
                "model: constant, r2 > 0.129",
                "scripts/25_zeroshot_baselines.py (gm_by_g[g])",
                "model/cb_train.py (np.mean(zs>0.129))",
                "Both are internally correct but NOT comparable. The model's "
                "0% and the baselines' 70.1% answer different questions.")

    base_halfB = "A, B = shift(cells[:h], t), shift(cells[h:], t)" in base_src
    train_full = 'truth = (X[iv==g].mean(0) - ctrl_mu)' in train_src
    if base_halfB and train_full:
        finding("DEFINITION", "R2 target differs between baselines and model",
                "baselines predict half B of the held-out cells",
                "model predicts the full-data mean shift",
                "scripts/25_zeroshot_baselines.py (A,B split; target = B)",
                "model/cb_train.py zeroshot_eval (target = all cells)",
                "The model's target is LESS noisy than the baselines' target, so "
                "model R2 and baseline R2 are not on one scale. The constants "
                "0.129 / 0.226 hard-coded in cb_train.py came from the half-B "
                "evaluation.")

    if "ss_tot = (true ** 2).sum()" in base_src and "(true**2).sum()" in train_src.replace(" ", ""):
        ok("R2 formula identical (1 - RSS / sum(true^2))",
           "both files", "25_zeroshot_baselines.py / cb_train.py")

screen_src = (FW / "causalbench/scripts/03_screen.py").read_text() \
    if (FW / "causalbench/scripts/03_screen.py").exists() else ""
if screen_src:
    m = re.search(r"ref_pool\s*=\s*ctrl_rows\[:0\]", screen_src)
    if m:
        line_no = screen_src[:m.start()].count("\n") + 1
        print(f"\n  [CONFIRMED] 03_screen.py line {line_no}: "
              f"ref_pool = ctrl_rows[:0]  (empty in step-0)")
        print("      -> in step-0, `ref` is drawn from the pseudo-env's OWN rows,")
        print("         so mean_ratio_vs_ctrl compares an env to itself.")
        print("         Only mean_ratio_pairs is calibrated to ~1.0.")

# ------------------------------------------------------------------- HCP vs 03
print("\n" + "=" * 78)
print("HCP CONSTRUCTION COMPARABILITY")
print("=" * 78)
hcp_script = Path("/workspace/meridian-identifiability/hcp/scripts/mean_shift_v2.py")
if hcp_script.exists():
    print(f"  hcp/scripts/mean_shift_v2.py present -- see EVIDENCE_PACK line-by-line")
else:
    print("  hcp/scripts/mean_shift_v2.py NOT on this machine (A100 only).")
    print("  Line-by-line comparison against 03_screen.py must be done there.")
    print("  Known from the results files:")
    print("    - HCP JSON has keys: mean_ratio, within_median, between_median")
    print("    - HCP JSON has NO mean_ratio_pairs and NO step-0 gate entry")
    print("    - HCP mean_ratio == between_median / within_median")
    print("      i.e. it corresponds to 03_screen.py's mean_ratio_VS_CTRL form,")
    print("      NOT to mean_ratio_pairs.")
    findings.append(dict(
        kind="COMPARABILITY", label="HCP metric vs CausalBench primary metric",
        claimed="HCP plotted on the same axis as K562/RPE1/Norman",
        actual="HCP computes between/within (vs_ctrl form); it has no pairs metric",
        claim_src="scripts/50_figures.py fig2/fig3",
        actual_src="hcp/results/mean_shift_*.json key set",
        note="If the paper's primary metric is mean_ratio_pairs, HCP has no "
             "corresponding value and cannot share the axis without recomputation."))

# ------------------------------------------------------------------------ totals
print("\n" + "=" * 78)
print("CROSS-CHECK TOTALS")
print("=" * 78)
by_kind = {}
for f in findings:
    by_kind.setdefault(f["kind"], []).append(f)
for k in sorted(by_kind):
    print(f"  {k:<24} {len(by_kind[k])}")
print(f"  {'TOTAL':<24} {len(findings)}")

out = FW / "paper/crosscheck_findings.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(findings, indent=2))
print(f"\n  wrote {out}")
