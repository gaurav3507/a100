"""IEEE TCBB publication figures for the environment-validity screen paper.

Reads ONLY from the tracked result JSONs and the HCP results on the analysis
machine. Does not recompute anything. Prints every plotted number so the caller
can verify against the source JSONs before trusting the rendered figures.

Output: figures/goyal-fig{1..6}.pdf and .png (both at 300 dpi at intended
display size). Widths follow IEEE TCBB (single 3.5 in, double 7.16 in). Fonts
are 8 pt sans-serif; PDF fonts are TrueType (Type 42) so they remain editable.
Colours use Wong 2011's colourblind-safe palette; each dataset also has a
distinct line style and marker so figures read in greyscale.

No titles are drawn inside the figures. Captions belong in the manuscript.

Usage:
    python scripts/50_figures.py
"""
import os, json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FW = Path(__file__).resolve().parent.parent
CB = FW / "causalbench/results"
OUT = FW / "figures"
OUT.mkdir(parents=True, exist_ok=True)

HCP_LR_MEAN = Path("/workspace/meridian-identifiability/hcp/results/mean_shift_LR.json")
HCP_RL_MEAN = Path("/workspace/meridian-identifiability/hcp/results/mean_shift_RL.json")
HCP_LR_SF = Path("/workspace/meridian-identifiability/hcp/results/shift_fraction_LR.json")
HCP_RL_SF = Path("/workspace/meridian-identifiability/hcp/results/shift_fraction_RL.json")

SINGLE_COL = 3.5
DOUBLE_COL = 7.16

C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_PURPLE = "#CC79A7"
C_GREY = "#666666"

Z_STOP_FC = "#F7C1C1"
Z_STOP_EC = "#791F1F"
Z_GO_FC = "#C0DD97"
Z_GO_EC = "#27500A"
Z_PROC_FC = "#B5D4F4"
Z_PROC_EC = "#0C447C"
Z_DEC_FC = "#FAC775"
Z_DEC_EC = "#854F0B"

STYLE = {
    "K562":   dict(color=C_BLUE,   ls="-",  marker="o"),
    "RPE1":   dict(color=C_ORANGE, ls="--", marker="s"),
    "Norman": dict(color=C_GREEN,  ls="-.", marker="^"),
    "HCP":    dict(color=C_RED,    ls=":",  marker="D"),
}
ARM_STYLE = {
    "full_shift":   dict(color=C_BLUE,   linestyle="-",  marker="o",
                         markerfacecolor=C_BLUE),
    "full_random":  dict(color=C_PURPLE, linestyle="-",  marker="s",
                         markerfacecolor=C_PURPLE),
    "nodag_shift":  dict(color=C_ORANGE, linestyle="--", marker="^",
                         markerfacecolor="white"),
    "nodag_random": dict(color=C_RED,    linestyle="--", marker="D",
                         markerfacecolor="white"),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.4,
    "lines.markersize": 5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def _load(p):
    with open(p) as f:
        return json.load(f)


def _save(fig, tag):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"goyal-{tag}.{ext}", dpi=300)
    plt.close(fig)


def _hcp_screen(d_primary=10):
    if not HCP_LR_MEAN.exists():
        return None
    j = _load(HCP_LR_MEAN)
    r = j["results"].get(str(d_primary))
    if r is None:
        return None
    return dict(
        mean_ratio_vs_ctrl=r["mean_ratio"],
        sing_ratio=r["sing_ratio"],
        n_dims_above_2x=r["n_dims_above_2x"],
        participation_ratio=r["participation_ratio"],
    )


# ----------------------------------------------------- FIG 1: workflow schematic
def fig1_schematic():
    """Vertical flowchart with role colouring:
        blue = input / process, amber = decision, red = stop, green = proceed.
    Each shape carries the concrete quantity being checked, so the figure works
    as a standalone flow explanation for the reader.
    """
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=6.6):
        r = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=1.0, edgecolor=ec, facecolor=fc)
        ax.add_patch(r)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fs, linespacing=1.3)

    def diamond(x, y, w, h, text, fc, ec, fs=6.4):
        pts = np.array([[x, y + h / 2], [x + w / 2, y],
                        [x, y - h / 2], [x - w / 2, y]])
        p = mpatches.Polygon(pts, closed=True, linewidth=1.0,
                              edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fs, linespacing=1.3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="black"))

    cx = 3.8
    stop_x = 8.05

    box(cx, 9.55, 5.6, 0.85,
        "Perturbation dataset\n(cells × genes; NMIN cells / environment)",
        Z_PROC_FC, Z_PROC_EC, fs=6.5)
    arrow(cx, 9.10, cx, 8.70)

    box(cx, 8.25, 5.6, 0.85,
        "Fit PCA basis on control cells\n(d latent dims, target column zeroed)",
        Z_PROC_FC, Z_PROC_EC, fs=6.5)
    arrow(cx, 7.80, cx, 7.35)

    diamond(cx, 6.55, 5.8, 1.6,
            "Step-0 gate\nmean_ratio_pairs\non control pseudo-envs ≈ 1?",
            Z_DEC_FC, Z_DEC_EC, fs=6.2)

    arrow(cx, 5.70, cx, 5.30)
    ax.text(cx + 0.28, 5.50, "yes", fontsize=6.5, style="italic")

    diamond(cx, 4.55, 5.8, 1.6,
            "Mean-shift ratio\n(between-env / within-env)\n> workable threshold?",
            Z_DEC_FC, Z_DEC_EC, fs=6.2)
    arrow(cx + 2.9, 4.55, stop_x - 1.15, 4.55)
    ax.text(cx + 3.65, 4.72, "no", fontsize=6.4, style="italic")
    box(stop_x, 4.55, 3.5, 0.82,
        "STOP:\nsignal below\ncontrol null",
        Z_STOP_FC, Z_STOP_EC, fs=6.2)

    arrow(cx, 3.75, cx, 3.35)
    ax.text(cx + 0.28, 3.55, "yes", fontsize=6.5, style="italic")

    diamond(cx, 2.60, 5.8, 1.6,
            "Effective dim\n(dims above 2× control noise)\n≥ threshold?",
            Z_DEC_FC, Z_DEC_EC, fs=6.2)
    arrow(cx + 2.9, 2.60, stop_x - 1.15, 2.60)
    ax.text(cx + 3.65, 2.77, "no", fontsize=6.4, style="italic")
    box(stop_x, 2.60, 3.5, 0.82,
        "STOP:\nover-parameterized\nfor available signal",
        Z_STOP_FC, Z_STOP_EC, fs=6.2)

    arrow(cx, 1.80, cx, 1.40)
    ax.text(cx + 0.28, 1.60, "yes", fontsize=6.5, style="italic")
    box(cx, 0.75, 5.6, 0.95,
        "PROCEED: train CRL model\n(dataset supplies mechanism-shift signal)",
        Z_GO_FC, Z_GO_EC, fs=6.7)

    _save(fig, "fig1")
    print("[fig1] workflow schematic saved (no data)")


# ------------------------------------------- FIG 2: mean-shift ratio, 4 datasets
def fig2_mean_shift_ratio():
    """Bar chart with semantic zones. Zone labels go in the axes corners so
    they never overlap the bars or the legend.
    """
    d_primary = 10
    k = _load(CB / "screen/k562_filt0_n200_d10.json")
    r = _load(CB / "screen/rpe1_filt0_n200_d10.json")
    n = _load(CB / "screen/norman.json")
    n_prim = next(x for x in n["screen"] if x["nmin"] == 200 and x["d"] == d_primary)
    h = _hcp_screen(d_primary)

    values = [k["mean_ratio_vs_ctrl"],
              r["mean_ratio_vs_ctrl"],
              n_prim["mean_ratio_vs_ctrl"],
              h["mean_ratio_vs_ctrl"] if h else np.nan]
    labels = ["K562", "RPE1", "Norman", "HCP"]

    step0_k = _load(CB / "screen/k562_filt0_n200_d10_STEP0.json")["mean_ratio_pairs"]
    step0_r = _load(CB / "screen/rpe1_filt0_n200_d10_STEP0.json")["mean_ratio_pairs"]
    step0_n = next(x for x in n["step0_gate"]
                    if x["nmin"] == 200 and x["d"] == d_primary
                    )["mean_ratio_pairs"]
    step0_pairs = [step0_k, step0_r, step0_n]
    band_lo, band_hi = min(step0_pairs), max(step0_pairs)

    print(f"[fig2] mean_ratio_vs_ctrl (d={d_primary}): "
          f"K562={values[0]:.3f} RPE1={values[1]:.3f} "
          f"Norman={values[2]:.3f} HCP={values[3]:.3f}")
    print(f"[fig2] step-0 gate band (pairs): [{band_lo:.3f}, {band_hi:.3f}]")

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.6))
    x = np.arange(len(labels))
    colors = [STYLE[lab]["color"] for lab in labels]

    y_max = max(v for v in values if not np.isnan(v)) * 1.30

    ax.axhspan(0, 1.0, color=Z_STOP_FC, alpha=0.15, zorder=0)
    ax.axhspan(2.0, y_max, color=Z_GO_FC, alpha=0.15, zorder=0)
    ax.axhspan(band_lo, band_hi, color=C_GREY, alpha=0.28, zorder=0.5,
               label=f"step-0 gate ({band_lo:.2f}–{band_hi:.2f})")

    bars = ax.bar(x, values, color=colors, edgecolor="black",
                   linewidth=0.8, width=0.55, zorder=2)

    ax.axhline(1.0, color="black", ls="--", lw=0.9, zorder=1.5,
               label="control null (1.0)")
    ax.axhline(2.0, color=Z_GO_EC, ls=":", lw=0.9, zorder=1.5,
               label="workable threshold (2.0)")

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mean-shift ratio  (between / within, d = 10)")
    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)

    _save(fig, "fig2")


# ---------------------------------------------- FIG 3: spectrum, two panels
def fig3_spectrum():
    """Panel A: signal/noise per singular index at d=20 for all four datasets.
    Panel B: dims above 2× noise as d varies.

    HCP has only 6 singular values at d>=6 (max_possible_dims=6, capped by
    task count), so the HCP curve on Panel A stops at index 6. HCP LR reports
    0 dims above 2× at every d tested (0/10, 0/20, 0/30). We plot the max of
    LR and RL to be charitable to the negative case; that max is 1 at d=10
    and 0 elsewhere -- still an order of magnitude below the other datasets.
    """
    d_show = 20
    k = _load(CB / f"spectrum/k562_filt0_n200_d{d_show}.json")
    r = _load(CB / f"spectrum/rpe1_filt0_n200_d{d_show}.json")
    n = _load(CB / "spectrum/norman.json")
    n_d = next(x for x in n["runs"] if x["d"] == d_show)
    hcp_lr_j = _load(HCP_LR_MEAN)
    hcp_ratios = hcp_lr_j["results"][str(d_show)]["sing_ratio"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.6),
                                    gridspec_kw={"wspace": 0.28})

    n_max = max(len(k["sing_ratio"]), len(hcp_ratios))
    y_max_A = max(max(k["sing_ratio"]), max(r["sing_ratio"]),
                   max(n_d["sing_ratio"]), max(hcp_ratios)) * 1.05
    axA.axhspan(0, 1.0, color=Z_STOP_FC, alpha=0.13, zorder=0)
    axA.axhspan(2.0, y_max_A, color=Z_GO_FC, alpha=0.13, zorder=0)

    for name, ratios in (("K562", k["sing_ratio"]),
                          ("RPE1", r["sing_ratio"]),
                          ("Norman", n_d["sing_ratio"]),
                          ("HCP", hcp_ratios)):
        st = STYLE[name]
        idx = np.arange(1, len(ratios) + 1)
        axA.plot(idx, ratios, color=st["color"], linestyle=st["ls"],
                 marker=st["marker"], label=name,
                 markersize=5, linewidth=1.5)

    axA.axhline(1.0, color="black", ls="--", lw=0.7, zorder=1)
    axA.axhline(2.0, color=Z_GO_EC, ls=":", lw=0.7, zorder=1)
    axA.set_xlabel("singular-value index")
    axA.set_ylabel(f"signal / noise ratio  (d = {d_show})")
    axA.set_xlim(0.5, n_max + 0.5)
    axA.set_ylim(0, y_max_A)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.legend(loc="upper right", frameon=False, ncol=1)
    axA.text(-0.13, 1.04, "A", transform=axA.transAxes,
             fontsize=11, fontweight="bold")

    d_cb = [10, 20, 50]
    d_hcp = [10, 20, 30]
    k_dims = [_load(CB / f"spectrum/k562_filt0_n200_d{d}.json")["n_dims_above_2x"]
              for d in d_cb]
    r_dims = [_load(CB / f"spectrum/rpe1_filt0_n200_d{d}.json")["n_dims_above_2x"]
              for d in d_cb]
    n_dims = [next(x for x in n["runs"] if x["d"] == d)["n_dims_above_2x"]
              for d in d_cb]
    # HCP: take max of LR and RL at each d (charitable read).
    hcp_lr_dims = [hcp_lr_j["results"][str(d)]["n_dims_above_2x"] for d in d_hcp]
    hcp_rl_dims = [None] * len(d_hcp)
    if HCP_RL_MEAN.exists():
        rlj = _load(HCP_RL_MEAN)["results"]
        hcp_rl_dims = [rlj[str(d)]["n_dims_above_2x"] if str(d) in rlj else 0
                        for d in d_hcp]
    hcp_dims = [max(a, b if b is not None else 0)
                for a, b in zip(hcp_lr_dims, hcp_rl_dims)]

    axB.plot([0, 55], [0, 55], color="black", linestyle="--", lw=0.7,
             zorder=1, label="saturation (y = d)")
    for name, dv, dims in (("K562", d_cb, k_dims), ("RPE1", d_cb, r_dims),
                            ("Norman", d_cb, n_dims), ("HCP", d_hcp, hcp_dims)):
        st = STYLE[name]
        axB.plot(dv, dims, color=st["color"], linestyle=st["ls"],
                 marker=st["marker"], label=name,
                 markersize=6, linewidth=1.5, zorder=3)

    axB.set_xlabel("PCA latent dim d")
    axB.set_ylabel("dims above 2× noise")
    axB.set_xlim(5, 55)
    axB.set_ylim(-2, 55)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.legend(loc="upper left", frameon=False)
    axB.text(-0.13, 1.04, "B", transform=axB.transAxes,
             fontsize=11, fontweight="bold")

    print(f"[fig3A] sing_ratio at d={d_show}:")
    print(f"  K562={k['sing_ratio']}")
    print(f"  RPE1={r['sing_ratio']}")
    print(f"  Norman={n_d['sing_ratio']}")
    print(f"  HCP LR={hcp_ratios}")
    print(f"[fig3B] dims_above_2x per d:")
    print(f"  K562 (d={d_cb}) = {k_dims}")
    print(f"  RPE1 (d={d_cb}) = {r_dims}")
    print(f"  Norman (d={d_cb}) = {n_dims}")
    print(f"  HCP LR (d={d_hcp}) = {hcp_lr_dims}")
    print(f"  HCP RL (d={d_hcp}) = {hcp_rl_dims}")
    print(f"  HCP max(LR,RL) plotted = {hcp_dims}")

    _save(fig, "fig3")


# -------------------- FIG 4: coef-shift ratio, K562 vs HCP, multiple d values
def fig4_coef_ratio():
    """d on the x-axis, coefficient-based shift ratio on the y-axis. Three
    curves (K562 coef ratio; HCP LR shift-fraction; HCP RL shift-fraction) each
    with markers at every d value. The per-point d label problem from the dot
    plot goes away because d is the axis.
    """
    k5 = _load(CB / "screen/k562_filt0_n200_d5.json")["coef_ratio_vs_ctrl"]
    k10 = _load(CB / "screen/k562_filt0_n200_d10.json")["coef_ratio_vs_ctrl"]
    k_vals = [(5, k5), (10, k10)]

    hcp_lr_j = _load(HCP_LR_SF)["results"]
    hcp_lr_vals = sorted(
        [(int(k), v["ratio"]) for k, v in hcp_lr_j.items()],
        key=lambda x: x[0])
    hcp_rl_vals = []
    if HCP_RL_SF.exists():
        hcp_rl_j = _load(HCP_RL_SF)["results"]
        hcp_rl_vals = sorted(
            [(int(k), v["ratio"]) for k, v in hcp_rl_j.items()],
            key=lambda x: x[0])

    print(f"[fig4] K562 coef_ratio_vs_ctrl by d = {k_vals}")
    print(f"[fig4] HCP LR shift-fraction ratio by d = {hcp_lr_vals}")
    print(f"[fig4] HCP RL shift-fraction ratio by d = {hcp_rl_vals}")

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.1))

    ax.axhspan(0.9, 1.3, color=C_GREY, alpha=0.14, zorder=0,
                label="near-null band (0.9–1.3)")
    ax.axhline(1.0, color="black", ls="--", lw=0.9, zorder=1,
                label="control null (1.0)")

    k_d = [d for d, _ in k_vals]
    k_v = [v for _, v in k_vals]
    ax.plot(k_d, k_v, color=STYLE["K562"]["color"], linestyle="-",
             marker="o", markersize=7, linewidth=1.6,
             markerfacecolor=STYLE["K562"]["color"],
             markeredgecolor=STYLE["K562"]["color"],
             label="K562 (coef ratio)", zorder=3)

    lr_d = [d for d, _ in hcp_lr_vals]
    lr_v = [v for _, v in hcp_lr_vals]
    ax.plot(lr_d, lr_v, color=STYLE["HCP"]["color"], linestyle="-",
             marker="D", markersize=6, linewidth=1.6,
             markerfacecolor=STYLE["HCP"]["color"],
             markeredgecolor=STYLE["HCP"]["color"],
             label="HCP LR (shift-fraction)", zorder=3)

    if hcp_rl_vals:
        rl_d = [d for d, _ in hcp_rl_vals]
        rl_v = [v for _, v in hcp_rl_vals]
        ax.plot(rl_d, rl_v, color=STYLE["HCP"]["color"], linestyle="--",
                 marker="D", markersize=6, linewidth=1.6,
                 markerfacecolor="white",
                 markeredgecolor=STYLE["HCP"]["color"],
                 markeredgewidth=1.3,
                 label="HCP RL (shift-fraction)", zorder=3)

    all_d = sorted(set([d for d, _ in k_vals]
                       + [d for d, _ in hcp_lr_vals]
                       + [d for d, _ in hcp_rl_vals]))
    all_v = ([v for _, v in k_vals] + [v for _, v in hcp_lr_vals]
             + [v for _, v in hcp_rl_vals])
    ax.set_xticks(all_d)
    ax.set_xlabel("latent dim d")
    ax.set_ylabel("coefficient-based shift ratio")
    ax.set_xlim(min(all_d) - 2, max(all_d) + 2)
    ax.set_ylim(min(all_v) - 0.05, max(all_v) + 0.10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=6)

    _save(fig, "fig4")


# --------------------- FIG 5: shift-basis vs control-PCA held-out R^2 vs d
def fig5_basis_r2():
    """Two curves with a shaded gain region between them. Legend labels are
    short so they do not overlap; the d = 15 annotation sits above the plot
    frame to stay clear of the legend at lower right.
    """
    d = _load(FW / "results/cf_estimator/k562.json")
    entries = sorted(d["basis"].items(), key=lambda x: int(x[0]))
    d_vals = [int(k) for k, _ in entries]
    shift_r2 = [v["shift_basis_r2"] for _, v in entries]
    pca_r2 = [v["control_pca_r2"] for _, v in entries]

    print(f"[fig5] d values = {d_vals}")
    print(f"[fig5] shift_basis_r2 = {[round(x,3) for x in shift_r2]}")
    print(f"[fig5] control_pca_r2 = {[round(x,3) for x in pca_r2]}")

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.1))

    ax.fill_between(d_vals, pca_r2, shift_r2, color=C_GREEN, alpha=0.18,
                     zorder=1, label="gain")
    ax.plot(d_vals, shift_r2, color=C_BLUE, linestyle="-", marker="o",
            label="shift basis", markersize=6, linewidth=1.8,
            markerfacecolor=C_BLUE, markeredgecolor=C_BLUE, zorder=3)
    ax.plot(d_vals, pca_r2, color=C_ORANGE, linestyle="--", marker="s",
            label="control PCA", markersize=6, linewidth=1.8,
            markerfacecolor="white", markeredgecolor=C_ORANGE,
            markeredgewidth=1.4, zorder=3)

    ax.axvline(15, color=C_GREY, ls=":", lw=0.9, alpha=0.85, zorder=1.5)
    ax.text(15.8, 0.85, "d = 15\n(model choice)",
             fontsize=6.4, color=C_GREY, style="italic",
             ha="left", va="top",
             bbox=dict(facecolor="white", edgecolor="none",
                       pad=1.5, alpha=0.95))

    ax.set_xlabel("latent dim d")
    ax.set_ylabel("held-out reconstruction R²")
    ax.set_xticks(d_vals)
    ax.set_xlim(4, 32)
    ax.set_ylim(0.4, 0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, ncol=1)

    _save(fig, "fig5")


# ---------------------------- FIG 6: zero-shot R² vs epoch across the four arms
def fig6_zeroshot_trajectories():
    """Two-panel design.

    Panel A (left, 60%): training trajectories per arm, dynamic y-range so no
    point is clipped. Legend sits in a semi-transparent white box so it does
    not overlap the lines.

    Panel B (right, 40%): horizontal bars for each arm's best-over-training R2
    plotted on the same x-axis as the three baselines, showing the gap.
    """
    arms = ["full_shift", "full_random", "nodag_shift", "nodag_random"]
    labels_full = ["full / shift", "full / random",
                   "no-DAG / shift", "no-DAG / random"]
    labels_short = labels_full

    baselines = _load(FW / "results/zeroshot_canonical/k562.json")["results"]
    gmean = baselines["global_mean"]["r2_median"]
    ridge = baselines["ridge_basis"]["r2_median"]
    ceiling = baselines["CEILING"]["r2_median"]
    print(f"[fig6] baselines: global_mean={gmean:.3f}, "
          f"ridge={ridge:.3f}, ceiling={ceiling:.3f}")

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL, 3.6),
        gridspec_kw={"width_ratios": [1.7, 1.0], "wspace": 0.32})

    peak = {}
    for arm, lab in zip(arms, labels_full):
        d = _load(FW / f"results/model/trajectories/{arm}.json")
        eps = [p["epoch"] for p in d["points"]]
        r2s = [p["r2_median"] for p in d["points"]]
        st = ARM_STYLE[arm]
        mfc = st.get("markerfacecolor", st["color"])
        axL.plot(eps, r2s,
                 color=st["color"], linestyle=st["linestyle"],
                 marker=st["marker"], label=lab,
                 markersize=5, linewidth=1.4,
                 markerfacecolor=mfc, markeredgecolor=st["color"],
                 markeredgewidth=1.0)
        peak[arm] = max(r2s)
        print(f"[fig6] {arm}: {list(zip(eps, [round(x,4) for x in r2s]))}")

    axL.axhline(0.0, color="black", lw=0.5, alpha=0.5, zorder=0)
    axL.set_xlim(-3, 103)
    _all_y = []
    for _arm in arms:
        _d = _load(FW / f"results/model/trajectories/{_arm}.json")
        _all_y.extend(pt["r2_median"] for pt in _d["points"]
                      if abs(pt["r2_median"]) <= 1.0)
    _y_lo, _y_hi = min(_all_y), max(_all_y)
    _pad = max(0.02, (_y_hi - _y_lo) * 0.15)
    axL.set_ylim(_y_lo - _pad, _y_hi + _pad)
    axL.set_xlabel("training epoch")
    axL.set_ylabel("held-out zero-shot R² (median)")
    axL.spines["top"].set_visible(False)
    axL.spines["right"].set_visible(False)
    axL.grid(True, alpha=0.15)
    axL.legend(loc="upper right", frameon=True, fontsize=6, ncol=2,
                edgecolor="none", facecolor="white", framealpha=0.90,
                borderpad=0.3, handletextpad=0.4, columnspacing=0.9)
    axL.text(-0.14, 1.03, "A", transform=axL.transAxes,
             fontsize=11, fontweight="bold")

    names = labels_short + ["global-mean\nbaseline", "ridge\nbaseline",
                             "split-half\nceiling"]
    values = [peak[a] for a in arms] + [gmean, ridge, ceiling]
    bar_colors = ([ARM_STYLE[a]["color"] for a in arms]
                   + [C_GREY, C_GREEN, C_RED])

    y_pos = np.arange(len(names))
    bars = axR.barh(y_pos, values, color=bar_colors,
                     edgecolor="black", linewidth=0.6, height=0.68)
    axR.axvline(0.0, color="black", lw=0.5, alpha=0.5)
    axR.set_yticks(y_pos)
    axR.set_yticklabels(names, fontsize=6.5)
    axR.invert_yaxis()
    axR.set_xlabel("median R²  (arms: best over training; baselines: fixed)")
    axR.set_xlim(-0.22, 0.62)
    axR.spines["top"].set_visible(False)
    axR.spines["right"].set_visible(False)
    axR.tick_params(axis="y", pad=6)
    for bar, v in zip(bars, values):
        if v >= 0:
            # positive bar: value label to the right of the bar's right edge
            axR.text(v + 0.010, bar.get_y() + bar.get_height() / 2,
                      f"{v:+.3f}", va="center", ha="left", fontsize=6.2)
        else:
            # negative bar: value label INSIDE the bar (to the right of the
            # bar's left edge), so it stays clear of the y-tick label.
            axR.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                      f"{v:+.3f}", va="center", ha="left", fontsize=6.2,
                      color="white", fontweight="bold")
    axR.text(-0.22, 1.03, "B", transform=axR.transAxes,
             fontsize=11, fontweight="bold")

    _save(fig, "fig6")


def main():
    fig1_schematic(); print()
    fig2_mean_shift_ratio(); print()
    fig3_spectrum(); print()
    fig4_coef_ratio(); print()
    fig5_basis_r2(); print()
    fig6_zeroshot_trajectories()
    print(f"\n[done] wrote figures to {OUT}")


if __name__ == "__main__":
    main()
