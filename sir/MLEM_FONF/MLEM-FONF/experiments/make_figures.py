"""Publication-grade figures (IEEE TMI/JBHI conventions).

Conventions applied: double-column 7.16 in / single-column 3.5 in widths,
8-9 pt sans fonts, panel labels (a)(b)(c), top/right spines removed,
mean ± 95% CI bands (n = slices x realizations), consistent method identity
(proposed = bold red; supervised/generative = dashed warm-neutral; classical
= thin cool hues), data-driven annotations, vector PDF + 600-dpi PNG.

  python experiments/make_figures.py [--results DIR] [--trajectory --root R]

Figures: fig1_dose_psnr_ssim  (2x3: PSNR & SSIM vs dose, three settings)
         fig2_shift_slope     (Mayo -> LoDoPaB per-method slope chart)
         fig3_robustness_violin (per-realization PSNR at lowest dose)
         fig4_stability        (MLEM vs spectral step, both geometries)
"""
import argparse, csv, math, pathlib, sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mlem_fonf.version import banner

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
W2, W1 = 7.16, 3.5          # IEEE column widths (inches)

STYLE = {   # hue encodes the TIER (red = proposed, blue/violet = learned, green/ochre = classical priors, grey = baselines)
    "A-FONF":       dict(color="#C8102E", edge="#8A0B20", ls="-",  lw=2.2, marker="o", ms=4.2, z=10),
    "RED-CNN":      dict(color="#1B4F9C", edge="#123869", ls="--", lw=1.3, marker="s", ms=3.3, z=6),
    "LPD":          dict(color="#4C8BE0", edge="#2F62A8", ls="--", lw=1.3, marker="D", ms=3.0, z=6),
    "DPS":          dict(color="#8B5CF6", edge="#6D3FD1", ls="--", lw=1.3, marker="P", ms=3.7, z=6),
    "MLEM+AD":      dict(color="#1F9D55", edge="#15703C", ls="-",  lw=1.0, marker="^", ms=3.1, z=4),
    "MLEM+FuzzyAD": dict(color="#6CC7A0", edge="#3E9A73", ls="-",  lw=1.0, marker="v", ms=3.1, z=4),
    "MLEM+TV":      dict(color="#D98E04", edge="#A56A00", ls="-",  lw=1.0, marker="<", ms=3.1, z=4),
    "PnP-ADMM(TV)": dict(color="#F2C14E", edge="#C29500", ls="-",  lw=1.0, marker=">", ms=3.1, z=4),
    "MLEM":         dict(color="#8C8C8C", edge="#5E5E5E", ls="-",  lw=0.9, marker="x", ms=3.3, z=3),
    "FBP":          dict(color="#C4C4C4", edge="#8C8C8C", ls="-",  lw=0.8, marker=".", ms=3.0, z=2),
}
TITLES = {"mayo_parallel": "Mayo (in-distribution), parallel",
          "lodopab_parallel": "LoDoPaB (OOD), parallel",
          "mayo_fan": "Mayo, fan-beam"}
ORDER = ["benchmark_mayo_parallel.csv", "benchmark_lodopab_parallel.csv", "benchmark_mayo_fan.csv"]


def load(csvp):
    """{method: {incident: {'psnr': [...], 'ssim': [...]}}} (finite only)."""
    d = defaultdict(lambda: defaultdict(lambda: {"psnr": [], "ssim": []}))
    for r in csv.DictReader(open(csvp)):
        try:
            p, s = float(r["PSNR"]), float(r["MSSIM"])
        except ValueError:
            continue
        if not (math.isfinite(p) and math.isfinite(s)):
            continue
        d[r["method"]][float(r["incident"])]["psnr"].append(p)
        d[r["method"]][float(r["incident"])]["ssim"].append(s)
    return d


def ci(v):
    v = np.asarray(v, float)
    half = 1.96 * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), half


def panel_label(ax, s):
    ax.text(0.02, 0.97, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")


def fig1_dose(files, outdir):
    n = len(files)
    fig, axes = plt.subplots(2, n, figsize=(W2, 4.6), sharex=True)
    axes = np.array(axes).reshape(2, n)
    present = set()
    for j, f in enumerate(files):
        d = load(f); key = pathlib.Path(f).stem.replace("benchmark_", "")
        present |= set(d)
        for i, met in enumerate(("psnr", "ssim")):
            ax = axes[i, j]
            for m in STYLE:
                if m not in d:
                    continue
                st = STYLE[m]; incs = sorted(d[m])
                stats = [ci(d[m][k][met]) for k in incs]
                mu = np.array([s[0] for s in stats]); err = np.array([s[1] for s in stats])
                if len(incs) >= 2:
                    ax.fill_between(incs, mu - err, mu + err, color=st["color"], alpha=0.18,
                                    lw=0, zorder=st["z"] - 1)
                ax.plot(incs, mu, color=st["color"], ls=st["ls"] if len(incs) >= 2 else "none",
                        lw=st["lw"], marker=st["marker"], ms=st["ms"], mew=0.6, zorder=st["z"],
                        clip_on=False)
            ax.set_xscale("log"); ax.grid(alpha=0.2, lw=0.5)
            ax.set_xticks([60, 300, 1500, 3000]); ax.set_xticklabels(["60", "300", "1.5k", "3k"])
            ax.minorticks_off()
            if i == 0:
                ax.set_title(TITLES.get(key, key), pad=6)
            if i == 1:
                ax.set_xlabel("incident photons / ray")
            panel_label(ax, f"({'abcdef'[i * n + j]})")
        # data-driven annotation: proposed vs best supervised at the highest dose
        if "A-FONF" in d:
            top = max(d["A-FONF"])
            sup = [m for m in ("RED-CNN", "LPD", "DPS") if m in d and top in d[m]]
            if sup:
                best = max(sup, key=lambda m: np.mean(d[m][top]["psnr"]))
                gap = np.mean(d["A-FONF"][top]["psnr"]) - np.mean(d[best][top]["psnr"])
                axes[0, j].text(0.97, 0.05, f"A-FONF vs {best} @ {top:g}: {gap:+.1f} dB",
                                transform=axes[0, j].transAxes, fontsize=6.8, color="#b3202c",
                                ha="right", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    axes[0, 0].set_ylabel("PSNR (dB)"); axes[1, 0].set_ylabel("SSIM")
    lo = min(ax.get_ylim()[0] for ax in axes[0]); hi = max(ax.get_ylim()[1] for ax in axes[0])
    for ax in axes[0]:
        ax.set_ylim(lo, hi)                      # never clip a method's curve
    for ax in axes[1]:
        ax.set_ylim(0.0, 0.85)
    handles = [Line2D([0], [0], color=s["color"], ls=s["ls"], lw=s["lw"] + 0.4, marker=s["marker"],
                      ms=s["ms"] + 0.5, mew=0.6, label=m) for m, s in STYLE.items() if m in present]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=2.4, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=0.6, w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig1_dose_psnr_ssim.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig2_shift(files, outdir, inc=3000.0):
    """Slope chart: each method's mean PSNR in-distribution -> OOD at `inc`."""
    fa = [f for f in files if "mayo_parallel" in f.name]
    fb = [f for f in files if "lodopab" in f.name]
    if not (fa and fb):
        return
    A, B = load(fa[0]), load(fb[0])
    ms = [m for m in STYLE if m in A and m in B and inc in A[m] and inc in B[m] and m != "FBP"]
    fig, ax = plt.subplots(figsize=(W1, 3.6))
    ends = []
    for m in ms:
        a, b = np.mean(A[m][inc]["psnr"]), np.mean(B[m][inc]["psnr"])
        st = STYLE[m]
        ax.plot([0, 1], [a, b], color=st["color"], lw=st["lw"] + (0.8 if m == "A-FONF" else 0),
                ls=st["ls"], marker=st["marker"], ms=st["ms"] + 1, mew=0.6, zorder=st["z"], clip_on=False)
        ends.append([b, f"{m}  {b - a:+.1f}", st["color"], m == "A-FONF"])
    # spread right-hand labels so they never overlap; leader lines to the true point
    ends.sort(key=lambda e: e[0])
    gap = 0.30
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = (ys[-1] - ends[-1][0]) / 2
    ys = [y - shift for y in ys]
    for (b, lab, col, bold), y in zip(ends, ys):
        ax.plot([1.0, 1.035], [b, y], color=col, lw=0.5, alpha=0.7, clip_on=False)
        ax.text(1.045, y, lab, va="center", fontsize=6.8, color=col,
                fontweight="bold" if bold else "normal", clip_on=False)
    ax.set_xlim(-0.05, 1.05); ax.set_xticks([0, 1])
    ax.set_xticklabels(["Mayo\n(in-distribution)", "LoDoPaB\n(distribution shift)"])
    ax.set_ylabel(f"PSNR (dB) at {inc:g} photons / ray")
    ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.set_title("Ranking inverts under distribution shift", pad=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig2_shift_slope.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig3_robustness(files, outdir):
    """Per-realization PSNR at the lowest dose. Broken y-axis: violin bodies
    with median/IQR box on top, collapsed realizations on the bottom strip."""
    n = len(files)
    fig, axes = plt.subplots(2, n, figsize=(W2, 3.4), sharex="col",
                             gridspec_kw=dict(height_ratios=(4, 1.1), hspace=0.08, wspace=0.12))
    axes = np.array(axes).reshape(2, n)
    rng = np.random.default_rng(0)
    for k, f in enumerate(files):
        top, bot = axes[0, k], axes[1, k]
        d = load(f); key = pathlib.Path(f).stem.replace("benchmark_", "")
        low = min(kk for m in d for kk in d[m])
        ms = [m for m in STYLE if m in d and low in d[m] and m != "FBP"]
        data = [np.asarray(d[m][low]["psnr"]) for m in ms]
        floor = 8.0                                            # collapse boundary
        body = [v[v >= floor] for v in data]
        vp = top.violinplot([bv if len(bv) > 1 else np.array([floor, floor + 0.1]) for bv in body],
                            positions=range(len(ms)), widths=0.82, showextrema=False, points=160)
        for pc, m in zip(vp["bodies"], ms):
            pc.set_facecolor(STYLE[m]["color"]); pc.set_edgecolor(STYLE[m]["edge"])
            pc.set_alpha(0.75); pc.set_linewidth(0.8)
        ymax = max(v.max() for v in data)
        for i, (m, v, bv) in enumerate(zip(ms, data, body)):
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            top.add_patch(plt.Rectangle((i - 0.09, max(q1, floor)), 0.18, max(q3 - max(q1, floor), 0.05),
                                        fc="white", ec="#333333", lw=0.7, zorder=5))
            top.plot([i - 0.09, i + 0.09], [med, med], color="#111111", lw=1.3, zorder=6)
            top.scatter(np.full(len(bv), i) + rng.uniform(-0.14, 0.14, len(bv)), bv, s=2.2,
                        color=STYLE[m]["edge"], alpha=0.30, lw=0, zorder=3)
            cv = v[v < floor]
            if len(cv):
                bot.scatter(np.full(len(cv), i) + rng.uniform(-0.14, 0.14, len(cv)), cv, s=9,
                            color=STYLE[m]["color"], edgecolor=STYLE[m]["edge"], lw=0.4, zorder=3)
            pct = 100 * len(cv) / len(v)
            top.text(i, ymax + 1.0, f"{pct:.0f}%", ha="center", fontsize=6.6,
                     color="#C8102E" if pct >= 0.5 else "#7a7a7a", fontweight="bold" if pct >= 0.5 else "normal")
        ylo = min(v[v >= floor].min() for v in data if (v >= floor).any())
        top.set_ylim(max(floor, ylo - 1.0), ymax + 2.4)
        top.set_xlim(-0.6, len(ms) - 0.4)
        allc = (np.concatenate([v[v < floor] for v in data])
                if any((v < floor).any() for v in data) else np.array([floor - 1]))
        bot.set_ylim(allc.min() - 2, min(allc.max() + 2, floor - 0.2))
        top.spines["bottom"].set_visible(False); top.tick_params(axis="x", length=0)
        bot.spines["top"].set_visible(False)
        for ax in (top, bot):
            ax.grid(axis="y", alpha=0.18, lw=0.5)
        # axis-break glyphs
        kw = dict(transform=top.transAxes, color="k", clip_on=False, lw=0.7)
        top.plot((-0.012, 0.012), (-0.02, 0.02), **kw)
        kw = dict(transform=bot.transAxes, color="k", clip_on=False, lw=0.7)
        bot.plot((-0.012, 0.012), (0.93, 1.07), **kw)
        bot.set_xticks(range(len(ms))); bot.set_xticklabels(ms, rotation=38, ha="right", fontsize=6.6)
        top.set_title(TITLES.get(key, key), pad=16, fontsize=8)
        top.text(0.0, 1.20, f"({'abc'[k]})", transform=top.transAxes, fontsize=9,
                 fontweight="bold", va="bottom", ha="left")
        if k > 0:
            top.tick_params(labelleft=False); bot.tick_params(labelleft=False)
    axes[0, 0].set_ylabel("PSNR per realization (dB)"); axes[1, 0].set_ylabel("collapsed", fontsize=6.6)
    fig.suptitle(f"{low:g} photons / ray  |  violin: distribution (>= 8 dB)  |  white box: median, IQR  |  "
                 "strip: collapsed realizations  |  label: collapse share", fontsize=6.8, y=1.09)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig3_robustness_violin.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig4_stability(root, outdir, incident=60.0):
    from mlem_fonf import build_system_matrix, build_fanbeam_matrix, poisson_sinogram, mlem
    from mlem_fonf.algorithms import mlem_fonf_damped
    from mlem_fonf.metrics import psnr
    if root:
        from mlem_fonf import data as D
        _, _, ref = next(D.mayo_slices(root, "test", "full", every=4, limit=1))
    else:
        from mlem_fonf.phantoms import shepp_logan_mod
        ref = shepp_logan_mod(256)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.7), sharey=True)
    out = {}
    geoms = (("Parallel beam", build_system_matrix(256, 64, 90)),
             ("Fan beam", build_fanbeam_matrix(256, 64, 90, sod=600, odd=600)))
    RED, GREY = STYLE["A-FONF"]["color"], STYLE["MLEM"]["color"]
    for k, (ax, (name, A)) in enumerate(zip(axes, geoms)):
        g = poisson_sinogram(A, ref, incident, np.random.default_rng(0))
        ks, tm, tf = [], [], []
        mlem(g, A, 256, 800, callback=lambda it, f: (ks.append(it + 1), tm.append(psnr(ref, f)))
             if (it + 1) % 5 == 0 else None)
        mlem_fonf_damped(g, A, 256, 800, alpha=2.0, lam_dt=0.95, gamma=0.9,
                         callback=lambda it, f: tf.append(psnr(ref, f)) if (it + 1) % 5 == 0 else None)
        ks, tm, tf = np.array(ks), np.array(tm), np.array(tf)
        ax.fill_between(ks, tm, tf, where=tf >= tm, color=RED, alpha=0.10, lw=0, label="stability gap")
        ax.plot(ks, tm, color=GREY, lw=1.6, label="MLEM, unregularized")
        ax.plot(ks, tf, color=RED, lw=2.3, label="same engine + spectral step (A-FONF)")
        kpk = int(ks[np.argmax(tm)])
        ax.plot([kpk], [tm.max()], marker="o", ms=4.5, mfc="white", mec="#111111", mew=1.0, zorder=6, clip_on=False)
        ax.annotate(f"early-stop optimum (iter {kpk})", xy=(kpk, tm.max()),
                    xytext=(0.28, 0.78), textcoords="axes fraction", fontsize=6.6,
                    arrowprops=dict(arrowstyle="->", lw=0.6, color="#333333"))
        ax.text(0.98, 0.05, f"iteration 800:  {tm[-1]:.0f} dB  vs  {tf[-1]:.0f} dB",
                transform=ax.transAxes, ha="right", fontsize=7,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        if tm[-1] < 0:
            ax.text(ks[len(ks) // 2], (tm[-1] + tm.max()) / 2 - 2, "diverges", color=GREY, fontsize=7,
                    ha="left", style="italic")
        ax.axhline(0, color="k", lw=0.5, alpha=0.35)
        ax.set_xlabel("iteration"); ax.set_title(f"{name}, {incident:g} photons / ray", pad=10)
        ax.grid(alpha=0.18, lw=0.5)
        ax.text(-0.04, 1.08, f"({'ab'[k]})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
        out[name] = (float(tm[-1]), float(tf[-1]))
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].legend(frameon=False, loc="center", fontsize=7)
    fig.tight_layout(w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig4_stability.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--trajectory", action="store_true")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    banner("make_figures")
    rdir = pathlib.Path(a.results) if a.results else pathlib.Path(__file__).resolve().parents[1] / "results"
    outdir = rdir / "figures"; outdir.mkdir(parents=True, exist_ok=True)
    files = [rdir / f for f in ORDER if (rdir / f).exists()]
    if files:
        fig1_dose(files, outdir); print("fig1_dose_psnr_ssim ✓")
        fig2_shift(files, outdir); print("fig2_shift_slope ✓")
        fig3_robustness(files, outdir); print("fig3_robustness_violin ✓")
    if a.trajectory:
        res = fig4_stability(a.root, outdir)
        print("fig4_stability ✓", {k: f"{v[0]:.1f} vs {v[1]:.1f} dB" for k, v in res.items()})
    print("saved to", outdir)
