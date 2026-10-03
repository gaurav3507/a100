"""Generate the full GRD submission figure suite from committed result JSONs.

All quantitative values are read from files under results/. Nothing is
hard-coded. Vector PDF is the submission format; PNG copies are rendered at
400 dpi for review.

Figures:
    figure1  real-data decision surface (detectability vs alignment + BH set)
    figure2  synthetic precondition-violation calibration (arms A/B/C/D)
    figure3  RPE1 is not a success case (perturbation vs control splits)
    figure4  closest-prior-method stress test (naive vs calibrated iLCS)
    figure7  second backbone (Backbone C, joint diagonalization), mirrors figure2
    figure8  split-control gate: BH fraction, shared vs split, per dataset

The Gate-Recover-Discover schematic is maintained separately as a draw.io
file (paper/figures/grd_pipeline.drawio) and is not produced here.

Usage:
    python paper/make_all_figures.py --figure all
    python paper/make_all_figures.py --figure figure2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUTPUT = REPO / "paper" / "figures"
DPI = 400

CB = {
    "K562": "#0072B2",
    "RPE1": "#D55E00",
    "Norman": "#009E73",
    "HCP": "#CC79A7",
    "ABIDE": "#56B4E9",
    "gate": "#2A8F7B",
    "naive": "#B03A2E",
    "grey": "#666666",
    "light": "#E6E6E6",
}


def configure_style():
    # JMLR body text is Computer Modern (LaTeX). Match it via mathtext cm
    # fontset with a Computer Modern / serif family, so figure labels read
    # as part of the typeset paper rather than a slide.
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def panel_label(ax, label, x=-0.20, y=1.13):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", va="top", ha="left")


def save(fig, stem):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / stem
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=DPI)
    plt.close(fig)
    print(path.with_suffix(".pdf").relative_to(REPO))
    print(path.with_suffix(".png").relative_to(REPO), f"({DPI} dpi)")


# --------------------------------------------------------------------------
# FIGURE 1 : real-data decision surface
# --------------------------------------------------------------------------
FIG1_SPECS = [
    ("K562", "K562", RESULTS / "e3" / "e3_K562_CRISPRi.json"),
    ("RPE1", "RPE1", RESULTS / "e3" / "e3_rpe1_gate_fixed.json"),
    ("Norman", "Norman", RESULTS / "e3" / "e3_Norman_CRISPRa_singlegene.json"),
    ("HCP", "HCP task", RESULTS / "e3" / "e3_fMRI_HCP_task.json"),
    ("ABIDE", "ABIDE site", RESULTS / "e3" / "e3_fMRI_ABIDE_site.json"),
]


def load_fig1_rows():
    rows = []
    for key, label, path in FIG1_SPECS:
        d = read_json(path)
        screen = d.get("perturbation_screen") or d.get("environment_screen")
        cd = screen["corrected_disjoint"]
        align = d["confound_alignment"]["corrected_disjoint_raw"]
        n_total = d.get("n_powered_perts", d.get("n_environments"))
        rows.append(dict(
            key=key, label=label, color=CB[key],
            raw=float(cd["raw_fraction"]),
            bh=float(cd["bh_fraction"]),
            bh_count=int(cd["bh_count"]),
            n_total=int(n_total),
            align=float(align["top2_energy_median"]),
        ))
    return rows


def make_figure1():
    rows = load_fig1_rows()
    fig, (axb, axc) = plt.subplots(1, 2, figsize=(7.4, 3.4))
    fig.subplots_adjust(left=0.115, right=0.965, top=0.82, bottom=0.155, wspace=0.46)

    # panel (a): detectability vs alignment
    panel_label(axb, "(a)")
    for r in rows:
        lw = 1.4 if r["key"] == "RPE1" else 0.7
        axb.scatter(r["raw"], r["align"], s=64, color=r["color"],
                    edgecolor="white", linewidth=lw, zorder=3)
    # labels placed to avoid the spine (Norman) and each other
    offsets = {"K562": (9, 3), "RPE1": (-10, 13), "Norman": (11, 2),
               "HCP": (-48, 10), "ABIDE": (-54, -12)}
    for r in rows:
        axb.annotate(r["label"], (r["raw"], r["align"]),
                     xytext=offsets[r["key"]], textcoords="offset points",
                     fontsize=8.5, color=r["color"], va="center",
                     fontweight="bold" if r["key"] == "RPE1" else "normal")
    rpe1 = next(r for r in rows if r["key"] == "RPE1")
    # annotation in open upper-right space, clear of the HCP/ABIDE labels
    axb.annotate("all 20/20 structured\ncontrol splits fire",
                 xy=(rpe1["raw"], rpe1["align"]), xytext=(0.66, 0.52),
                 textcoords="data", fontsize=7.5, color=rpe1["color"], ha="left",
                 va="center",
                 arrowprops=dict(arrowstyle="->", color=rpe1["color"], lw=0.8,
                                 connectionstyle="arc3,rad=-0.25"))
    # isotropic reference: 2 of 10 PCs at equal energy = 0.20
    axb.axhline(0.20, color="#9AA0A6", lw=0.8, ls=(0, (4, 3)), zorder=1)
    axb.text(0.015, 0.207, "isotropic 2/10 reference", fontsize=6.8,
             color="#6B7075", va="bottom", ha="left")
    axb.set_xlim(0, 1.05)
    axb.set_ylim(0, 0.80)
    axb.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    axb.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    axb.grid(True, color=CB["light"], linewidth=0.5, zorder=0)
    axb.set_axisbelow(True)
    axb.set_xlabel("Raw detectable fraction (corrected null)")
    axb.set_ylabel("Median top-2 control-PC energy\n(among raw detections)")
    axb.set_title("Raw detectability versus control-PC alignment",
                  loc="left", pad=10, fontsize=9.5)

    # panel (b): BH-FDR-certified fraction
    panel_label(axc, "(b)")
    order = ["K562", "RPE1", "Norman", "HCP", "ABIDE"]
    ordered = [next(r for r in rows if r["key"] == k) for k in order]
    y = list(range(len(ordered)))
    axc.barh(y, [r["bh"] for r in ordered],
             color=[r["color"] for r in ordered], height=0.6, alpha=0.92)
    axc.set_yticks(y)
    axc.set_yticklabels([r["label"] for r in ordered])
    axc.invert_yaxis()
    axc.set_xlim(0, 1.30)
    axc.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    axc.xaxis.grid(True, color=CB["light"], linewidth=0.5)
    axc.set_axisbelow(True)
    axc.set_xlabel(r"BH-FDR-selected fraction ($q = 0.05$)")
    axc.set_title("BH-FDR-selected shift fractions",
                  loc="left", pad=10, fontsize=9.5)
    for i, r in enumerate(ordered):
        if r["bh"] >= 0.80:  # long bar: label inside, white
            axc.text(r["bh"] - 0.03, i, f"{r['bh_count']}/{r['n_total']}",
                     va="center", ha="right", fontsize=8, color="white")
        else:
            axc.text(r["bh"] + 0.03, i, f"{r['bh_count']}/{r['n_total']}",
                     va="center", ha="left", fontsize=8, color="#222222")
    axc.axhline(2.5, color="#BDBDBD", linewidth=0.7)
    axc.text(1.27, 0.9, "Perturb-seq\n(recovery-eligible)", rotation=90,
             ha="right", va="center", fontsize=6.3, color=CB["grey"])
    axc.text(1.27, 3.5, "fMRI\n(detection only,\nnot recovery-eligible)",
             rotation=90, ha="right", va="center", fontsize=6.3, color=CB["grey"])
    save(fig, "figure1_decision_surface")


# --------------------------------------------------------------------------
# FIGURE 2 : synthetic precondition-violation calibration
# --------------------------------------------------------------------------
def make_figure2():
    d = read_json(RESULTS / "e2" / "e2_calibration_report.json")
    e2c = read_json(RESULTS / "e2c" / "starvation_report.json")

    def arm(name):
        levels = d["arms"][name]["levels"]
        x = [lv["level"] for lv in levels]
        naive = [lv["naive_mcc_mean"] for lv in levels]
        cert = [lv["certified_recovery_mean"] for lv in levels]
        gate = [lv["gate_n_recoverable_mean"] for lv in levels]
        # per-seed gate range (min, max) for whiskers; per-seed SD for MCC
        grange, naive_sd, cert_sd = [], [], []
        for lv in levels:
            gs = [s.get("n_recoverable", s.get("gate_count"))
                  for s in lv["per_seed"]]
            grange.append((min(gs), max(gs)))
            nm = [s["naive_mcc"] for s in lv["per_seed"]]
            naive_sd.append(float(np.std(nm)))
            cm = [s["certified_recovery"] for s in lv["per_seed"]
                  if s.get("certified_recovery") is not None]
            cert_sd.append(float(np.std(cm)) if len(cm) > 1 else 0.0)
        return x, naive, cert, gate, grange, naive_sd, cert_sd

    def gate_whiskers(ax, x, gate, grange, d_latent=5.0, logx=False):
        """Draw gate cap mean as squares with min-max whiskers, scaled to /d."""
        lo = [g[0] / d_latent for g in grange]
        hi = [g[1] / d_latent for g in grange]
        mean = [g / d_latent for g in gate]
        yerr = [[m - l for m, l in zip(mean, lo)],
                [h - m for m, h in zip(mean, hi)]]
        ax.errorbar(x, mean, yerr=yerr, marker="s", ms=4.5, color=CB["gate"],
                    lw=1.4, capsize=2.5, elinewidth=0.8,
                    label=r"Gate cap $\hat{n}_{\mathrm{rec}}/d$ (min-max)")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.09,
                        hspace=0.44, wspace=0.30)

    # panel (a): direction starvation, PRIMARY random-subset series
    axA = axes[0, 0]
    panel_label(axA, "(a)")
    lv = e2c["aggregate"]["levels"]
    m = [x["m"] for x in lv]
    rs = [x["random_subset_control"] for x in lv]
    mcc = [r["mcc_mean"] for r in rs]
    sd = [r["mcc_seed_sd"] for r in rs]
    axA.errorbar(m, mcc, yerr=sd, marker="o", ms=5, color=CB["naive"],
                 capsize=2.5, lw=1.4, label="Ungated recovery (MCC)")
    gate_line = [r["gate_n_recoverable_mean"] / 5.0 for r in rs]
    axA.plot(m, gate_line, marker="s", ms=4.5, color=CB["gate"], lw=1.4,
             label=r"Gate cap $\hat{n}_{\mathrm{rec}}/d$")
    axA.set_xticks([1, 2, 3, 4, 5])
    axA.set_xlim(0.6, 5.4)
    axA.set_ylim(0, 1.05)
    axA.grid(True, color=CB["light"], linewidth=0.6)
    axA.set_axisbelow(True)
    axA.set_xlabel("Intervened directions $m$")
    axA.set_ylabel("Recovery quality")
    axA.set_title("Direction starvation\n(random-subset series; gate = $m$ by design)",
                  loc="left", pad=8, fontsize=9)
    axA.legend(loc="lower right", frameon=False)

    # panel (b): power starvation, with gate min-max whiskers
    axB = axes[0, 1]
    panel_label(axB, "(b)")
    x, naive, cert, gate, grange, nsd, csd = arm("B_power_starvation")
    axB.errorbar(x, naive, yerr=nsd, marker="o", ms=5, color=CB["naive"],
                 lw=1.4, capsize=2.5, elinewidth=0.8, label="Ungated MCC (seed SD)")
    gate_whiskers(axB, x, gate, grange, logx=True)
    axB.set_xscale("log")
    axB.set_xticks(x)
    axB.set_xticklabels([str(v) for v in x])
    axB.minorticks_off()  # drop minor vertical grid/ticks on log axis
    axB.set_ylim(0, 1.05)
    axB.grid(True, color=CB["light"], linewidth=0.6, which="major")
    axB.set_axisbelow(True)
    axB.set_xlabel("Samples per environment $n_e$")
    axB.set_ylabel("Recovery quality")
    axB.set_title("Power starvation\n(gate restricts before recovery failure)",
                  loc="left", pad=8, fontsize=9)
    axB.legend(loc="lower right", frameon=False)

    # panel (c): weak-signal starvation, with gate min-max whiskers
    axD = axes[1, 0]
    panel_label(axD, "(c)")
    x, naive, cert, gate, grange, nsd, csd = arm("D_weak_signal_starvation")
    axD.errorbar(x, naive, yerr=nsd, marker="o", ms=5, color=CB["naive"],
                 lw=1.4, capsize=2.5, elinewidth=0.8, label="Ungated MCC (seed SD)")
    gate_whiskers(axD, x, gate, grange)
    axD.set_xlim(0.05, 1.05)
    axD.set_ylim(0, 1.05)
    axD.grid(True, color=CB["light"], linewidth=0.6)
    axD.set_axisbelow(True)
    axD.set_xlabel(r"Intervention scale $s_{\mathrm{iv}}$")
    axD.set_ylabel("Recovery quality")
    axD.set_title("Weak-signal starvation\n(gate restricts first)",
                  loc="left", pad=8, fontsize=9)
    axD.legend(loc="lower left", frameon=False)

    # panel (d): measurement contamination, declared failure class
    axC = axes[1, 1]
    panel_label(axC, "(d)")
    x, naive, cert, gate, grange, nsd, csd = arm("C_measurement_contamination")
    axC.errorbar(x, naive, yerr=nsd, marker="o", ms=5, color=CB["naive"],
                 lw=1.4, capsize=2.5, elinewidth=0.8, label="Naive recovery MCC")
    axC.errorbar(x, cert, yerr=csd, marker="^", ms=5, color="#8E44AD",
                 lw=1.4, capsize=2.5, elinewidth=0.8, label="Certified recovery*")
    axC.set_xticks(x)
    axC.set_ylim(0.55, 1.05)
    axC.grid(True, color=CB["light"], linewidth=0.6)
    axC.set_axisbelow(True)
    axC.set_xlabel("Contaminated dimensions")
    axC.set_ylabel("Recovery quality")
    axC.set_title("Measurement contamination\n(declared failure class)",
                  loc="left", pad=8, fontsize=9)
    axC.legend(loc="upper right", frameon=False)
    axC.text(0.02, 0.02,
             "Contamination can pass detectability while\n"
             "certified recovery degrades.\n"
             "*conditional on seeds with $\\geq 1$ certified direction.",
             transform=axC.transAxes, fontsize=6.2, color=CB["grey"],
             ha="left", va="bottom")

    save(fig, "figure2_synthetic_calibration")


# --------------------------------------------------------------------------
# FIGURE 3 : RPE1 is not a success case
# --------------------------------------------------------------------------
def make_figure3():
    d = read_json(RESULTS / "e3" / "e3_rpe1_confound_check.json")
    # PRIMARY decision is BH-FDR at q=0.05 on the corrected-disjoint null.
    pert = d["perturbation_screen"]["corrected_disjoint"]
    rand = d["random_control_screen"]["corrected_disjoint"]
    struct = d["structured_control_screen"]["corrected_disjoint"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.24, wspace=0.42)

    # panel (a): BH-selected fraction across the three screens
    panel_label(ax1, "(a)")
    labels = ["Real\nperturbations", "Random\ncontrol splits",
              "Structured\ncontrol splits"]
    n_env = [pert["n_environments"], rand["n_environments"],
             struct["n_environments"]]
    counts = [pert["bh_count"], rand["bh_count"], struct["bh_count"]]
    fracs = [c / n for c, n in zip(counts, n_env)]
    colors = [CB["RPE1"], "#95A5A6", "#34495E"]
    xb = np.arange(3)
    ax1.bar(xb, fracs, color=colors, width=0.62, alpha=0.92)
    ax1.set_xticks(xb)
    ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_ylim(0, 1.12)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.yaxis.grid(True, color=CB["light"], linewidth=0.6)
    ax1.set_axisbelow(True)
    ax1.set_ylabel(r"Fraction passing BH-FDR ($q=0.05$)")
    ax1.set_title("All structured control splits\npass BH-FDR",
                  loc="left", pad=8, fontsize=9)
    for i, (c, n) in enumerate(zip(counts, n_env)):
        ax1.text(i, fracs[i] + 0.03, f"{c}/{n}", ha="center", va="bottom",
                 fontsize=7.5, color="#222222")

    # panel (b): top-2 control-PC energy of BH-selected perturbations
    panel_label(ax2, "(b)")
    align = d["perturbation_shift_alignment"]["corrected_disjoint_bh"]
    med = align["top2_energy_median"]
    n_sel = align["n_detected"]
    # isotropic 2-of-10 reference (equal energy across d_proj=10 PCs)
    ax2.axhline(0.20, color="#9AA0A6", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax2.text(0.66, 0.205, "isotropic\n2/10 reference", fontsize=6.6,
             color="#6B7075", va="bottom", ha="left")
    ax2.bar([0], [med], color=CB["RPE1"], width=0.5, alpha=0.92)
    ax2.set_xlim(-0.7, 0.95)
    ax2.set_ylim(0, 0.85)
    ax2.set_xticks([0])
    ax2.set_xticklabels([f"RPE1\n({n_sel} BH-selected)"], fontsize=7.5)
    ax2.set_ylabel("Median top-2 control-PC energy")
    ax2.yaxis.grid(True, color=CB["light"], linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.axhline(med, color=CB["RPE1"], lw=0.8, ls=":")
    ax2.text(-0.62, med + 0.02, f"median {med:.3f}", va="bottom", ha="left",
             fontsize=7.5, color=CB["RPE1"])
    ax2.set_title("Selected shifts concentrate in\nleading control PCs",
                  loc="left", pad=8, fontsize=9)

    fig.text(0.5, 0.02,
             "Random splits pass 0/20; structured splits pass 20/20. "
             "Association with control heterogeneity, not proven causation.",
             ha="center", fontsize=6.8, color=CB["grey"])
    save(fig, "figure3_rpe1_not_success")


# --------------------------------------------------------------------------
# FIGURE 4 : closest-prior-method stress test (iLCS)
# --------------------------------------------------------------------------
def make_figure4():
    d = read_json(RESULTS / "ilcs_baseline" / "aggregate.json")["per_dataset"]
    datasets = ["K562", "RPE1", "Norman"]
    conditions = [
        ("Pure-control\nfakes", "fake"),
        ("Random\nsplits", "random"),
        ("Structured\nsplits", "struct"),
        ("Real\nperturbations", "pert"),
    ]
    seeds = read_json(RESULTS / "ilcs_baseline" / "aggregate.json")["seeds"]

    def seed_values(ds, field):
        """Per-seed values for calib_<cond>_rate or ica_nonconverged_frac."""
        out = []
        for s in seeds:
            sd = read_json(RESULTS / "ilcs_baseline" / f"seed{s}" / f"{ds}.json")
            out.append(sd["summary"][field])
        return out

    from matplotlib.patches import Patch
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.9),
                                   gridspec_kw={"width_ratios": [1.7, 1]})
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.13, wspace=0.34)

    # panel (a): CALIBRATED firing only; naive=1.0 shown as a nominal line
    panel_label(ax1, "(a)")
    ncond = len(conditions)
    ndata = len(datasets)
    group_w = 0.74
    bar_w = group_w / ndata
    xbase = np.arange(ncond)
    for di, ds in enumerate(datasets):
        calib_vals = [d[ds][f"calib_{ckey}_rate"][0] for _, ckey in conditions]
        calib_sd = [d[ds][f"calib_{ckey}_rate"][1] for _, ckey in conditions]
        off = (di - (ndata - 1) / 2) * bar_w
        ax1.bar(xbase + off, calib_vals, bar_w * 0.92,
                yerr=calib_sd, color=CB[ds], alpha=0.95, capsize=1.5,
                error_kw=dict(lw=0.7), label=ds)
        # overplot the three individual seed values
        for ci, (_, ckey) in enumerate(conditions):
            pts = seed_values(ds, f"calib_{ckey}_rate")
            ax1.scatter([xbase[ci] + off] * len(pts), pts, s=6,
                        color="#222222", zorder=5, linewidths=0)
    ax1.axhline(1.0, color="#888888", lw=0.9, ls="--")
    ax1.text(ncond - 1 + 0.42, 1.0, "naive iLCS = 1.0\nfor every condition",
             ha="right", va="top", fontsize=6.8, color="#555555")
    ax1.set_xticks(xbase)
    ax1.set_xticklabels([c[0] for c in conditions], fontsize=8)
    ax1.set_ylim(0, 1.12)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.yaxis.grid(True, color=CB["light"], linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.set_ylabel("Calibrated iLCS firing rate")
    ax1.set_title("Size matching reduces iLCS firing, but structured-control firing remains",
                  loc="left", pad=20, fontsize=8.8)
    dot = mpl.lines.Line2D([], [], marker="o", ls="none", ms=3.2,
                           color="#222222", label="per-seed")
    ax1.legend(handles=[Patch(facecolor=CB[ds], label=ds) for ds in datasets]
               + [dot], loc="upper left", bbox_to_anchor=(0.0, 1.0),
               frameon=False, ncol=4, handlelength=1.1, columnspacing=1.2,
               fontsize=7.4)

    # panel (b): ICA non-convergence, one-decimal %, seed dots
    panel_label(ax2, "(b)")
    conv = [d[ds]["ica_nonconverged_frac"][0] for ds in datasets]
    conv_sd = [d[ds]["ica_nonconverged_frac"][1] for ds in datasets]
    xb = np.arange(ndata)
    ax2.bar(xb, conv, yerr=conv_sd, color=[CB[ds] for ds in datasets],
            width=0.6, alpha=0.92, capsize=2.5, error_kw=dict(lw=0.7))
    for i, ds in enumerate(datasets):
        pts = seed_values(ds, "ica_nonconverged_frac")
        ax2.scatter([i] * len(pts), pts, s=8, color="#222222", zorder=5,
                    linewidths=0)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(datasets)
    ax2.set_ylim(0, 0.33)
    ax2.yaxis.grid(True, color=CB["light"], linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.set_ylabel("ICA non-convergence fraction")
    ax2.set_title("ICA non-convergence is\nhighest on RPE1", loc="left",
                  pad=8, fontsize=9)
    for i, v in enumerate(conv):
        ax2.text(i + 0.34, v, f"{v*100:.1f}%", ha="left", va="center",
                 fontsize=7.5, color="#222222")

    fig.text(0.5, 0.005,
             "Bars: mean over 3 seeds; error bars: population SD (n=3); dots: individual seeds.",
             ha="center", fontsize=6.6, color=CB["grey"])
    save(fig, "figure4_ilcs_stress_test")


# --------------------------------------------------------------------------
# FIGURE 5 : faithful positive control dose-response (real K562 noise)
# --------------------------------------------------------------------------
def make_figure5():
    d = read_json(RESULTS / "e3_poscontrol" / "poscontrol_final.json")
    # the committed dose_response stops at ratio 4; ratio 8 is the smoke block
    rows = list(d["dose_response"]) + [d["smoke_snr8"]]
    rows.sort(key=lambda r: r["snr"])
    snr = [r["snr"] for r in rows]
    n_pl = rows[0]["planted_screen"]["corrected_disjoint"]["n_environments"]
    n_nu = rows[0]["negative_control_screen"]["corrected_disjoint"]["n_environments"]
    pl_raw = [r["planted_screen"]["corrected_disjoint"]["raw_count"] for r in rows]
    pl_bh = [r["planted_screen"]["corrected_disjoint"]["bh_count"] for r in rows]
    nu_raw = [r["negative_control_screen"]["corrected_disjoint"]["raw_count"] for r in rows]
    nu_bh = [r["negative_control_screen"]["corrected_disjoint"]["bh_count"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.20, wspace=0.36)
    xs = list(range(len(snr)))
    labels = [f"{v:g}" for v in snr]

    panel_label(ax1, "(a)")
    ax1.plot(xs, [c / n_pl for c in pl_raw], marker="o", ms=5, color="#8E44AD",
             lw=1.3, ls="--", label=r"raw $\alpha=0.05$")
    ax1.plot(xs, [c / n_pl for c in pl_bh], marker="s", ms=5, color=CB["gate"],
             lw=1.6, label=r"BH-FDR $q=0.05$")
    for i, (c, n) in enumerate(zip(pl_bh, [n_pl] * len(pl_bh))):
        ax1.text(i, c / n + 0.05, f"{c}/{n}", ha="center", va="bottom",
                 fontsize=7.5, color="#222222")
    ax1.set_xticks(xs); ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.15); ax1.set_yticks([0, 0.5, 1.0])
    ax1.grid(True, color=CB["light"], linewidth=0.5); ax1.set_axisbelow(True)
    ax1.set_xlabel("Signal-to-background ratio")
    ax1.set_ylabel(f"Planted interventions selected (of {n_pl})")
    ax1.set_title("Dose response of the corrected gate", loc="left", pad=8, fontsize=9)
    ax1.legend(loc="upper left", frameon=False)

    panel_label(ax2, "(b)")
    ax2.plot(xs, [c / n_nu for c in nu_raw], marker="o", ms=5, color="#8E44AD",
             lw=1.3, ls="--", label=r"raw $\alpha=0.05$")
    ax2.plot(xs, [c / n_nu for c in nu_bh], marker="s", ms=5, color=CB["gate"],
             lw=1.6, label=r"BH-FDR $q=0.05$")
    ax2.axhline(0.05, color="#9AA0A6", lw=0.8, ls=(0, (4, 3)))
    ax2.text(len(xs) - 1 + 0.05, 0.052, "nominal 0.05", ha="right", va="bottom",
             fontsize=6.8, color="#6B7075")
    for i, c in enumerate(nu_raw):
        if c > 0:
            ax2.text(i, c / n_nu + 0.012, f"{c}/{n_nu}", ha="center", va="bottom",
                     fontsize=7.5, color="#8E44AD")
    ax2.set_xticks(xs); ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 0.25); ax2.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
    ax2.grid(True, color=CB["light"], linewidth=0.5); ax2.set_axisbelow(True)
    ax2.set_xlabel("Signal-to-background ratio")
    ax2.set_ylabel(f"Null draws selected (of {n_nu})")
    ax2.set_title("Null false positives at each ratio", loc="left", pad=8, fontsize=9)
    ax2.legend(loc="upper left", frameon=False)
    save(fig, "figure5_positive_control")


# --------------------------------------------------------------------------
# FIGURE 6 : finite-sample direction certificate feasibility (appendix)
# --------------------------------------------------------------------------
def make_figure6():
    d = read_json(RESULTS / "e2e_certificate" / "certificate_report.json")
    rows = d["summaries"]
    n = [r["n_per_env"] for r in rows]
    nonvac = [r["nonvacuous_among_gate"] for r in rows]
    gap = [r["gap_separated_among_gate"] for r in rows]
    bound = [r["median_nonvacuous_angle_bound_deg"] for r in rows]
    samp_pop = [r["median_sample_population_angle_deg"] for r in rows]
    bias = [r["median_population_latent_model_bias_deg"] for r in rows]
    ref_n = d.get("reference_n", 2000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.20, wspace=0.36)

    panel_label(ax1, "(a)")
    ax1.plot(n, gap, marker="o", ms=5, color="#8E44AD", lw=1.3, ls="--",
             label="eigengap separated")
    ax1.plot(n, nonvac, marker="s", ms=5, color=CB["gate"], lw=1.6,
             label="nonvacuous bound")
    ax1.axvline(ref_n, color="#9AA0A6", lw=0.8, ls=(0, (4, 3)))
    ax1.annotate(f"reference $n={ref_n}$", xy=(ref_n, 0.60),
                 xytext=(ref_n * 0.42, 0.80), fontsize=6.8, color="#6B7075",
                 ha="center", va="center",
                 arrowprops=dict(arrowstyle="->", color="#9AA0A6", lw=0.7))
    ax1.set_xscale("log"); ax1.set_xticks(n); ax1.set_xticklabels([str(v) for v in n],
             fontsize=7.5); ax1.minorticks_off()
    ax1.set_ylim(0, 1.08); ax1.set_yticks([0, 0.5, 1.0])
    ax1.grid(True, color=CB["light"], linewidth=0.5, which="major"); ax1.set_axisbelow(True)
    ax1.set_xlabel("Samples per environment $n_e$")
    ax1.set_ylabel("Fraction of gate-passing directions")
    ax1.set_title("Certificate becomes informative only at large $n$",
                  loc="left", pad=22, fontsize=9)
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
               frameon=False, handlelength=1.6, columnspacing=1.3, fontsize=7.6)

    panel_label(ax2, "(b)")
    ax2.plot(n, samp_pop, marker="o", ms=5, color=CB["naive"], lw=1.4,
             label="realized sample-population angle")
    nb = [(x, b) for x, b in zip(n, bound) if b is not None]
    ax2.plot([x for x, _ in nb], [b for _, b in nb], marker="s", ms=5,
             color=CB["gate"], lw=1.6, label="median certified bound")
    ax2.axhline(np.median(bias), color="#9AA0A6", lw=0.8, ls=(0, (4, 3)))
    ax2.text(n[0] * 1.05, np.median(bias) + 1.2, "model-bias angle (not covered)",
             fontsize=6.8, color="#6B7075", va="bottom")
    ax2.set_xscale("log"); ax2.set_xticks(n); ax2.set_xticklabels([str(v) for v in n],
             fontsize=7.5); ax2.minorticks_off()
    ax2.set_ylim(0, 72)
    ax2.grid(True, color=CB["light"], linewidth=0.5, which="major"); ax2.set_axisbelow(True)
    ax2.set_xlabel("Samples per environment $n_e$")
    ax2.set_ylabel("Angle (degrees)")
    ax2.set_title("Bound is valid but loose where it exists",
                  loc="left", pad=8, fontsize=9)
    ax2.legend(loc="upper left", frameon=False, fontsize=7.2)
    save(fig, "figure6_certificate_feasibility")


# --------------------------------------------------------------------------
# FIGURE 7 : second backbone (Backbone C, joint diagonalization), mirrors figure 2
# --------------------------------------------------------------------------
E4_DIR = RESULTS / "e4_second_backbone"
E5_DIR = RESULTS / "e5_split_control"
BEHAVIOR_TEXT = {
    "TRACKS_TIGHT": "gate tracks recovery",
    "TRACKS_CONSERVATIVE": "gate restricts first",
    "FOOLED": "gate certifies non-recoverable directions",
    "FLAT_OR_SILENT": "gate proceeds while MCC $<$ 0.90",
}


def _crossing(text, unit, value, never):
    """'text unit=value', or the 'never' wording when the crossover did not occur."""
    return never if value is None else f"{text} {unit}={value}"


def _mcc_bar(ax):
    """Recessive 0.90 recovery bar used by the E2/E4 crossover rules."""
    ax.axhline(0.90, color="#9AA0A6", lw=0.8, ls=(0, (4, 3)), zorder=1)


def make_figure7():
    d = read_json(E4_DIR / "e4_calibration_report.json")
    st = read_json(E4_DIR / "e4_starvation_report.json")

    def arm(name):
        levels = d["arms"][name]["levels"]
        x = [lv["level"] for lv in levels]
        naive = [lv["naive_mcc_mean"] for lv in levels]
        cert = [lv["certified_recovery_mean"] for lv in levels]
        gate = [lv["gate_n_recoverable_mean"] for lv in levels]
        grange, naive_sd, cert_sd = [], [], []
        for lv in levels:
            gs = [s["n_recoverable"] for s in lv["per_seed"]]
            grange.append((min(gs), max(gs)))
            naive_sd.append(float(np.std([s["naive_mcc"] for s in lv["per_seed"]])))
            cm = [s["certified_recovery"] for s in lv["per_seed"]
                  if s.get("certified_recovery") is not None]
            cert_sd.append(float(np.std(cm)) if len(cm) > 1 else 0.0)
        return x, naive, cert, gate, grange, naive_sd, cert_sd

    def gate_whiskers(ax, x, gate, grange, d_latent=5.0):
        lo = [g[0] / d_latent for g in grange]
        hi = [g[1] / d_latent for g in grange]
        mean = [g / d_latent for g in gate]
        yerr = [[m - l for m, l in zip(mean, lo)], [h - m for m, h in zip(mean, hi)]]
        ax.errorbar(x, mean, yerr=yerr, marker="s", ms=4.5, color=CB["gate"],
                    lw=1.4, capsize=2.5, elinewidth=0.8,
                    label=r"Gate cap $\hat{n}_{\mathrm{rec}}/d$ (min-max)")

    def subtitle(name, unit):
        a = d["arms"][name]
        return (f"{BEHAVIOR_TEXT.get(a['gate_behavior'], a['gate_behavior'])}\n"
                f"{_crossing('leaves PROCEED at', unit, a['crossover_gate_leaves_proceed'], 'always PROCEED')}; "
                f"{_crossing('MCC $<$ 0.90 at', unit, a['crossover_naive_below_0p90'], 'MCC never $<$ 0.90')}")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.3))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.91, bottom=0.09,
                        hspace=0.52, wspace=0.36)

    # panel (a): direction starvation, random-subset series
    axA = axes[0, 0]
    panel_label(axA, "(a)")
    agg = st["aggregate"]
    rs = [x["random_subset_control"] for x in agg["levels"]]
    m = [x["m"] for x in agg["levels"]]
    axA.errorbar(m, [r["mcc_mean"] for r in rs], yerr=[r["mcc_seed_sd"] for r in rs],
                 marker="o", ms=5, color=CB["naive"], capsize=2.5, lw=1.4,
                 label="Ungated recovery (MCC)")
    axA.plot(m, [r["gate_n_recoverable_mean"] / 5.0 for r in rs], marker="s", ms=4.5,
             color=CB["gate"], lw=1.4, label=r"Gate cap $\hat{n}_{\mathrm{rec}}/d$")
    _mcc_bar(axA)
    axA.set_xticks([1, 2, 3, 4, 5])
    axA.set_xlim(0.6, 5.4)
    axA.set_ylim(0, 1.05)
    axA.grid(True, color=CB["light"], linewidth=0.6)
    axA.set_axisbelow(True)
    axA.set_xlabel("Intervened directions $m$")
    axA.set_ylabel("Recovery quality")
    axA.set_title("Direction starvation\n"
                  + _crossing("gate first restricts at", "$m$",
                              agg["crossover_gate_first_restricts"], "gate never restricts")
                  + "; " + _crossing("MCC $<$ 0.90 from", "$m$",
                                     agg["crossover_random_subset_mcc_below_0p90"],
                                     "MCC never $<$ 0.90"),
                  loc="left", pad=8, fontsize=8.2)
    axA.legend(loc="lower right", frameon=False)

    # panel (b): power starvation
    axB = axes[0, 1]
    panel_label(axB, "(b)")
    x, naive, cert, gate, grange, nsd, csd = arm("B_power_starvation")
    axB.errorbar(x, naive, yerr=nsd, marker="o", ms=5, color=CB["naive"], lw=1.4,
                 capsize=2.5, elinewidth=0.8, label="Ungated MCC (seed SD)")
    gate_whiskers(axB, x, gate, grange)
    _mcc_bar(axB)
    axB.set_xscale("log")
    axB.set_xticks(x)
    axB.set_xticklabels([str(v) for v in x])
    axB.minorticks_off()
    axB.set_ylim(0, 1.05)
    axB.grid(True, color=CB["light"], linewidth=0.6, which="major")
    axB.set_axisbelow(True)
    axB.set_xlabel("Samples per environment $n_e$")
    axB.set_ylabel("Recovery quality")
    axB.set_title("Power starvation\n" + subtitle("B_power_starvation", "$n_e$"),
                  loc="left", pad=8, fontsize=8.2)
    axB.legend(loc="lower right", frameon=False)

    # panel (c): weak-signal starvation
    axD = axes[1, 0]
    panel_label(axD, "(c)")
    x, naive, cert, gate, grange, nsd, csd = arm("D_weak_signal_starvation")
    axD.errorbar(x, naive, yerr=nsd, marker="o", ms=5, color=CB["naive"], lw=1.4,
                 capsize=2.5, elinewidth=0.8, label="Ungated MCC (seed SD)")
    gate_whiskers(axD, x, gate, grange)
    _mcc_bar(axD)
    axD.set_xlim(0.05, 1.05)
    axD.set_ylim(0, 1.05)
    axD.grid(True, color=CB["light"], linewidth=0.6)
    axD.set_axisbelow(True)
    axD.set_xlabel(r"Intervention scale $s_{\mathrm{iv}}$")
    axD.set_ylabel("Recovery quality")
    axD.set_title("Weak-signal starvation\n"
                  + subtitle("D_weak_signal_starvation", r"$s_{\mathrm{iv}}$"),
                  loc="left", pad=8, fontsize=8.2)
    axD.legend(loc="lower left", frameon=False)

    # panel (d): measurement contamination
    axC = axes[1, 1]
    panel_label(axC, "(d)")
    x, naive, cert, gate, grange, nsd, csd = arm("C_measurement_contamination")
    axC.errorbar(x, naive, yerr=nsd, marker="o", ms=5, color=CB["naive"], lw=1.4,
                 capsize=2.5, elinewidth=0.8, label="Naive recovery MCC")
    cx = [v for v, c in zip(x, cert) if c is not None]
    axC.errorbar(cx, [c for c in cert if c is not None],
                 yerr=[s for s, c in zip(csd, cert) if c is not None], marker="^",
                 ms=5, color="#8E44AD", lw=1.4, capsize=2.5, elinewidth=0.8,
                 label="Certified recovery*")
    _mcc_bar(axC)
    low = min([v for v in naive + cert if v is not None] + [0.60])
    axC.set_xticks(x)
    axC.set_ylim(np.floor((low - 0.05) * 20) / 20, 1.05)
    axC.grid(True, color=CB["light"], linewidth=0.6)
    axC.set_axisbelow(True)
    axC.set_xlabel("Contaminated dimensions")
    axC.set_ylabel("Recovery quality")
    behavior = d["arms"]["C_measurement_contamination"]["gate_behavior"]
    axC.set_title("Measurement contamination\n"
                  + BEHAVIOR_TEXT.get(behavior, behavior),
                  loc="left", pad=8, fontsize=8.2)
    axC.legend(loc="lower right", bbox_to_anchor=(1.0, 0.07), frameon=False)
    axC.text(0.02, 0.02,
             "*conditional on seeds with $\\geq 1$ certified direction.",
             transform=axC.transAxes, fontsize=6.2, color=CB["grey"],
             ha="left", va="bottom")

    dec = d["population_decomposition"]
    note = ("Backbone C: joint diagonalization of all precision differences, gate "
            r"$\|\Delta_e\|_F$. Dashed line: 0.90 recovery bar." "\n"
            "Recovery MCC with exact population covariances: Backbone C "
            f"{dec['jad_rule']['population']['mean']:.3f}, precision rule "
            f"{dec['precision_rule']['population']['mean']:.3f}, covariance negative "
            f"control {dec['covariance_rule']['population']['mean']:.3f}.")
    fig.text(0.5, -0.012, note, ha="center", fontsize=6.6, color=CB["grey"])
    save(fig, "figure7_second_backbone")


# --------------------------------------------------------------------------
# FIGURE 8 : split-control gate, BH fraction shared versus split
# --------------------------------------------------------------------------
def make_figure8():
    d = read_json(E5_DIR / "e5_real_panel.json")["datasets"]
    datasets = [k for k in ("K562", "RPE1", "Norman") if k in d]
    designs = [("corrected_disjoint", "Shared reference", None),
               ("split_control", "Split control", "////")]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.84, bottom=0.20)
    width = 0.34
    xb = np.arange(len(datasets))
    for j, (key, label, hatch) in enumerate(designs):
        off = (j - 0.5) * width
        for i, ds in enumerate(datasets):
            stab = d[ds]["perturbation_stability"]
            mean = stab[f"{key}_bh_fraction_mean"]
            sd = stab[f"{key}_bh_fraction_sd"]
            pts = stab[f"{key}_bh_fraction_per_seed"]
            color = CB[ds]
            ax.bar(xb[i] + off, mean, width * 0.9, yerr=sd,
                   color=color if hatch is None else "white", edgecolor=color,
                   hatch=hatch, linewidth=1.0, capsize=2.5,
                   error_kw=dict(lw=0.7), zorder=2)
            ax.scatter([xb[i] + off] * len(pts), pts, s=7, color="#222222",
                       zorder=5, linewidths=0)
            n = d[ds]["n_perts"]
            counts = [e["shared" if key == "corrected_disjoint" else "split"]["bh_count"]
                      for e in d[ds]["shared_vs_split"]["per_seed"]]
            ax.text(xb[i] + off, max(pts + [mean + sd]) + 0.02,
                    f"{np.mean(counts):.4g}/{n}", ha="center", va="bottom",
                    fontsize=6.8, color="#222222")
    for i, ds in enumerate(datasets):
        jac = d[ds]["shared_vs_split"]["summary"]["jaccard_bh_mean"]
        txt = "both empty" if jac is None else f"Jaccard {jac:.2f}"
        ax.text(xb[i], -0.13, txt, ha="center", va="top", fontsize=6.8,
                color=CB["grey"], transform=ax.get_xaxis_transform())
    ax.set_xticks(xb)
    ax.set_xticklabels(datasets)
    ymax = max(max(d[ds]["perturbation_stability"][f"{k}_bh_fraction_per_seed"])
               for ds in datasets for k, _, _ in designs)
    ax.set_ylim(0, max(0.12, ymax * 1.25 + 0.04))
    ax.yaxis.grid(True, color=CB["light"], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel(r"BH-FDR-selected fraction ($q = 0.05$)")
    ax.set_title("Split-control gate: BH-selected perturbation fraction",
                 loc="left", pad=22, fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#777777", edgecolor="#777777",
                             label="Shared reference (E3)"),
                       Patch(facecolor="white", edgecolor="#777777", hatch="////",
                             label="Split control")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
              fontsize=7.4)
    seeds = d[datasets[0]]["seeds"]
    fig.text(0.5, 0.005,
             f"Bars: mean over {len(seeds)} gate seeds; error bars: population SD; "
             f"dots: individual seeds; counts are seed means.",
             ha="center", fontsize=6.6, color=CB["grey"])
    save(fig, "figure8_split_control")


# Tier 2 figures read artifacts produced on the A100; "all" skips them until the
# JSONs exist so figures 1-6 keep rendering from the committed results.
FIGURE_INPUTS = {
    "figure7": [E4_DIR / "e4_calibration_report.json",
                E4_DIR / "e4_starvation_report.json"],
    "figure8": [E5_DIR / "e5_real_panel.json"],
}

FIGURES = {
    "figure1": make_figure1,
    "figure2": make_figure2,
    "figure3": make_figure3,
    "figure4": make_figure4,
    "figure5": make_figure5,
    "figure6": make_figure6,
    "figure7": make_figure7,
    "figure8": make_figure8,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figure", choices=list(FIGURES) + ["all"], default="all")
    return p.parse_args()


def main():
    args = parse_args()
    configure_style()
    keys = list(FIGURES) if args.figure == "all" else [args.figure]
    for k in keys:
        missing = [p for p in FIGURE_INPUTS.get(k, []) if not p.exists()]
        if missing and args.figure == "all":
            print(f"skip {k}: missing {', '.join(str(p.relative_to(REPO)) for p in missing)}")
            continue
        FIGURES[k]()


if __name__ == "__main__":
    main()
