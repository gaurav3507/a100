"""Fig 2: mean-shift ratio across datasets, PRIMARY METRIC = mean_ratio_pairs.

Why pairs and not vs_ctrl: in the step-0 gate the reference pool is empty
(03_screen.py line 63), so vs_ctrl compares an environment to ITSELF and returns
~0.67 -- an artifact, not a calibrated null. Only mean_ratio_pairs is calibrated
to 1.0 by step-0. pairs also measures the property CRL requires: environments must
differ FROM EACH OTHER, not merely from baseline.

Config: filt0 (headline), NMIN=200, d=10.
"""
import os, json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

R   = "/workspace/meridian-identifiability/causalbench/results/screen"
OUT = "/workspace/meridian-identifiability/framework/figures"
os.makedirs(OUT, exist_ok=True)

mpl.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9,
    "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

def get(path, *keys):
    d = json.load(open(os.path.join(R, path)))
    for k in keys:
        d = d[k]
    return float(d)

vals = {
    "K562":   get("k562_filt0_n200_d10.json", "mean_ratio_pairs"),
    "RPE1":   get("rpe1_filt0_n200_d10.json", "mean_ratio_pairs"),
    "Norman": get("norman.json", "summary", "primary_mean_ratio_pairs"),
    # NOTE: HCP is computed by hcp/scripts/mean_shift_v2.py (task/encode ratio),
    # NOT by 03_screen.py. Verify the constructions are comparable; if not, the
    # caption must state that HCP is an analogous but separately-computed quantity.
    "HCP":    get("norman.json", "reference", "hcp_mean_ratio"),
}
gate = [get("rpe1_filt0_n200_d10_STEP0.json", "mean_ratio_pairs"),
        get("k562_filt0_n200_d10_STEP0.json", "mean_ratio_pairs")]
gate = (min(gate), max(gate))

print("=== VALUES PLOTTED (verify against JSONs) ===")
for k, v in vals.items():
    print(f"  {k:<8} {v:.3f}")
print(f"  step-0 gate band: {gate[0]:.3f} - {gate[1]:.3f}")

# outcome annotation -- graded, not thresholded
note = {"K562": "signal present\n(ridge R²=0.23)",
        "Norman": "CRL succeeds\n(R²=0.99)",
        "RPE1": "intermediate\n(ceiling too low)",
        "HCP": "no signal"}
col  = {"K562": "#0072B2", "Norman": "#009E73",
        "RPE1": "#E69F00", "HCP": "#D55E00"}

names = ["K562", "Norman", "RPE1", "HCP"]          # ordered by ratio
fig, ax = plt.subplots(figsize=(7.16, 3.0))

bars = ax.bar(names, [vals[n] for n in names],
              color=[col[n] for n in names], edgecolor="black", linewidth=0.8, width=0.6)

ax.axhspan(gate[0], gate[1], color="0.75", alpha=0.55, zorder=0,
           label=f"step-0 calibration ({gate[0]:.2f}–{gate[1]:.2f})")
ax.axhline(1.0, color="black", ls="--", lw=1.0, label="within-condition null (1.0)")

for b, n in zip(bars, names):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.10,
            f"{vals[n]:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.text(b.get_x() + b.get_width()/2, 0.14, note[n], ha="center", va="bottom",
            fontsize=6.5, color="white", linespacing=1.25)

ax.set_ylabel("mean-shift ratio\n(between-environment / within-environment)")
ax.set_ylim(0, max(vals.values()) * 1.22)
ax.legend(loc="upper right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/goyal-fig2.{ext}", dpi=300, bbox_inches="tight")
print(f"\nwrote {OUT}/goyal-fig2.pdf and .png")
