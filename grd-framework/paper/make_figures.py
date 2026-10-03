"""Generate submission figures directly from committed GRD result JSONs.

The figures use only archived summaries under results/. They do not rerun an
experiment, read external data, or hard-code scientific result values.

Usage:
    python paper/make_figures.py --figure figure1
    python paper/make_figures.py --figure all
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUTPUT = REPO / "paper" / "figures"

DATASETS = [
    dict(
        key="K562",
        label="K562",
        modality="Perturb-seq",
        path=RESULTS / "e3" / "e3_K562_CRISPRi.json",
        color="#0072B2",
    ),
    dict(
        key="RPE1",
        label="RPE1",
        modality="Perturb-seq",
        path=RESULTS / "e3" / "e3_rpe1_gate_fixed.json",
        color="#D55E00",
    ),
    dict(
        key="Norman",
        label="Norman",
        modality="Perturb-seq",
        path=RESULTS / "e3" / "e3_Norman_CRISPRa_singlegene.json",
        color="#009E73",
    ),
    dict(
        key="HCP",
        label="HCP task",
        modality="fMRI",
        path=RESULTS / "e3" / "e3_fMRI_HCP_task.json",
        color="#CC79A7",
    ),
    dict(
        key="ABIDE",
        label="ABIDE site",
        modality="fMRI",
        path=RESULTS / "e3" / "e3_fMRI_ABIDE_site.json",
        color="#56B4E9",
    ),
]


def configure_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"missing committed result: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_real_data_summary():
    rows = []
    for spec in DATASETS:
        result = read_json(spec["path"])
        screen = result.get("perturbation_screen") or result.get(
            "environment_screen")
        corrected = screen["corrected_disjoint"]
        alignment = result["confound_alignment"]["corrected_disjoint_raw"]
        n_total = result.get("n_powered_perts", result.get("n_environments"))
        required = [
            corrected["raw_fraction"],
            corrected["bh_fraction"],
            alignment["top2_energy_median"],
            n_total,
        ]
        if any(value is None for value in required):
            raise ValueError(f"incomplete real-data summary in {spec['path']}")
        rows.append({
            **spec,
            "raw_fraction": float(corrected["raw_fraction"]),
            "bh_fraction": float(corrected["bh_fraction"]),
            "bh_count": int(corrected["bh_count"]),
            "n_total": int(n_total),
            "alignment": float(alignment["top2_energy_median"]),
        })
    return rows


def panel_label(ax, label):
    ax.text(
        -0.07,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_pipeline(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_label(ax, "a")

    boxes = [
        dict(
            x=0.02,
            color="#2F5597",
            title="GATE",
            lines=[
                "Test preconditions",
                "Power + shift + confound",
                "Proceed / cap / abstain",
            ],
        ),
        dict(
            x=0.36,
            color="#2A8F7B",
            title="RECOVER",
            lines=[
                "Use certified environments",
                "Estimate supported latents",
                "Preserve the gate's cap",
            ],
        ),
        dict(
            x=0.70,
            color="#B7791F",
            title="DISCOVER",
            lines=[
                "Learn only supportable edges",
                "Present / absent / undecided",
                "Report remaining ambiguity",
            ],
        ),
    ]
    width = 0.27
    height = 0.67
    y = 0.15
    for item in boxes:
        pale = mpl.colors.to_rgba(item["color"], 0.09)
        patch = FancyBboxPatch(
            (item["x"], y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=item["color"],
            facecolor=pale,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            item["x"] + 0.02,
            y + height - 0.13,
            item["title"],
            color=item["color"],
            fontsize=12,
            fontweight="bold",
            transform=ax.transAxes,
        )
        for line_index, line in enumerate(item["lines"]):
            ax.text(
                item["x"] + 0.02,
                y + height - 0.29 - 0.15 * line_index,
                line,
                color="#222222",
                fontsize=8.5,
                transform=ax.transAxes,
            )

    for x_start, x_end in [(0.295, 0.355), (0.635, 0.695)]:
        ax.add_patch(FancyArrowPatch(
            (x_start, 0.49),
            (x_end, 0.49),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#555555",
            transform=ax.transAxes,
        ))
    ax.text(
        0.5,
        0.96,
        "GRD makes identifiability preconditions an explicit decision layer",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )


def draw_detectability_alignment(ax, rows):
    panel_label(ax, "b")
    for row in rows:
        linewidth = 1.5 if row["key"] == "RPE1" else 0.8
        ax.scatter(
            row["raw_fraction"],
            row["alignment"],
            s=62,
            color=row["color"],
            edgecolor="white",
            linewidth=linewidth,
            zorder=3,
        )

    offsets = {
        "K562": (5, 8),
        "RPE1": (6, 8),
        "Norman": (5, -15),
        "HCP": (-58, 9),
        "ABIDE": (-64, -16),
    }
    for row in rows:
        ax.annotate(
            row["label"],
            (row["raw_fraction"], row["alignment"]),
            xytext=offsets[row["key"]],
            textcoords="offset points",
            fontsize=8,
            color=row["color"],
            fontweight="bold" if row["key"] == "RPE1" else "normal",
        )

    rpe1 = next(row for row in rows if row["key"] == "RPE1")
    ax.annotate(
        "20/20 structured\ncontrol splits fire",
        xy=(rpe1["raw_fraction"], rpe1["alignment"]),
        xytext=(0.64, 0.84),
        textcoords="data",
        fontsize=7.5,
        color=rpe1["color"],
        ha="left",
        arrowprops=dict(
            arrowstyle="->", color=rpe1["color"], linewidth=0.9),
    )
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0.20, 0.88)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.grid(True, color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.set_xlabel("Raw detectable fraction")
    ax.set_ylabel("Control-PC alignment\n(top-2 energy, median)")
    ax.set_title(
        "Raw detectability and control\nalignment diverge",
        loc="left",
        fontweight="bold",
        pad=8,
    )
    ax.text(
        0.01,
        0.215,
        "Corrected null; alignment is descriptive.",
        fontsize=7,
        color="#666666",
        va="bottom",
    )


def draw_bh_fractions(ax, rows):
    panel_label(ax, "c")
    ordered = [
        next(row for row in rows if row["key"] == key)
        for key in ["K562", "RPE1", "Norman", "HCP", "ABIDE"]
    ]
    y = list(range(len(ordered)))
    ax.barh(
        y,
        [row["bh_fraction"] for row in ordered],
        color=[row["color"] for row in ordered],
        height=0.58,
        alpha=0.92,
    )
    ax.set_yticks(y, [row["label"] for row in ordered])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("BH-FDR-certified fraction (q = 0.05)")
    ax.set_title(
        "BH-FDR defines the\nactionable set",
        loc="left",
        fontweight="bold",
        pad=8,
    )
    for index, row in enumerate(ordered):
        value = row["bh_fraction"]
        label_x = max(value + 0.025, 0.025)
        ax.text(
            label_x,
            index,
            f"{row['bh_count']}/{row['n_total']}",
            va="center",
            ha="left",
            fontsize=8,
            color="#222222",
        )
    ax.axhline(2.5, color="#BDBDBD", linewidth=0.8)
    ax.text(
        1.065,
        1.0,
        "Perturb-seq",
        rotation=90,
        ha="right",
        va="center",
        fontsize=7,
        color="#666666",
    )
    ax.text(
        1.065,
        3.5,
        "fMRI",
        rotation=90,
        ha="right",
        va="center",
        fontsize=7,
        color="#666666",
    )


def make_figure1():
    rows = load_real_data_summary()
    figure = plt.figure(figsize=(7.2, 6.0), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[0.82, 1.55],
        left=0.09,
        right=0.98,
        top=0.97,
        bottom=0.09,
        hspace=0.38,
        wspace=0.38,
    )
    pipeline_ax = figure.add_subplot(grid[0, :])
    scatter_ax = figure.add_subplot(grid[1, 0])
    bar_ax = figure.add_subplot(grid[1, 1])
    draw_pipeline(pipeline_ax)
    draw_detectability_alignment(scatter_ax, rows)
    draw_bh_fractions(bar_ax, rows)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "figure1_grd_overview"
    metadata = {
        "Title": "GRD framework and real-data gate summary",
        "Author": "Gaurav Goyal",
        "Subject": "Generated from committed GRD result JSONs",
    }
    figure.savefig(stem.with_suffix(".pdf"), metadata=metadata)
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(figure)
    print(stem.with_suffix(".pdf").relative_to(REPO))
    print(stem.with_suffix(".png").relative_to(REPO))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=["figure1", "all"],
        default="all",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_style()
    if args.figure in {"figure1", "all"}:
        make_figure1()


if __name__ == "__main__":
    main()
