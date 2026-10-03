"""Eigenvalue-spectrum figure: the defence of the dimension claim.

WHY THIS FIGURE EXISTS. A reviewer will object, correctly, that the
participation ratio of an OBSERVED control covariance is not the LATENT
dimension. A low-dimensional latent process pushed through a noisy mixing map
can present with high observed dimension, so a large participation ratio is on
its own compatible with a small latent dimension plus noise.

The defence is not a bigger number, it is the SHAPE of the spectrum. Under
signal-plus-isotropic-noise the eigenvalues separate into a few large signal
eigenvalues and a noise bulk, and the boundary between them is visible as a
knee whose position is predicted by Marchenko-Pastur. If the observed spectrum
decays smoothly with no knee, and no eigenvalue bulk sits inside the MP
support, then "low-dimensional latent plus noise" does not describe the data.
The figure has to make that visible rather than assert it.

WHERE THE DATA COMES FROM. 85_dataset_descriptives.py persists the raw
eigenvalue array to a compressed sidecar beside each JSON, and the JSON
records the sidecar basename, the key, and an integrity triple
(lam_len / lam_sum / lam_sumsq). This script VERIFIES that triple against the
loaded array before plotting anything and exits non-zero on mismatch, naming
both values. It never recomputes, approximates or synthesises a spectrum: a
fabricated curve in a figure whose entire purpose is to show a real shape
would be worse than no figure. Artefacts written before spectrum persistence
carry no sidecar; the script says so and exits cleanly rather than crashing.

TWO FIGURES. The primary is at the MATCHED cap n=2000, because that is the
comparison where sample size is held constant across a 36x range of feature
counts (651 genes for RPE1 up to 23712 for Frangieh), so differences in shape
cannot be attributed to differing n. The 8000-cap version is emitted second as
a sample-size sensitivity check; there n varies across panels because several
control pools are smaller than the cap.

MARCHENKO-PASTUR, from first principles. For a p-variate sample covariance
built from n iid samples whose population covariance is sigma^2 * I, with
aspect ratio gamma = p / n, the sample eigenvalues converge to the
Marchenko-Pastur law supported on

    lambda_default = sigma^2 * (1 +/- sqrt(gamma))^2

so the upper edge of the pure-noise bulk is

    lambda_plus = sigma^2 * (1 + sqrt(p / n))^2

Any eigenvalue above lambda_plus is inconsistent with isotropic noise alone
and is evidence of signal; the bulk below it is what noise alone would
produce. When p > n the sample covariance is rank-deficient with at most
n - 1 non-zero eigenvalues, and the formula still gives the upper edge of the
non-zero bulk. sigma^2 is NOT taken as the mean of all eigenvalues, which is
inflated by the signal: it is estimated from the trailing fraction of the
spectrum (default: the median of the lower half), which is dominated by the
noise bulk. That choice is a judgement and is recorded on the figure.

Usage (once eigenvalues are persisted):
    python causalbench/scripts/90_spectrum_figure.py
    python causalbench/scripts/90_spectrum_figure.py --dir <artefact dir>
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# scripts/ sits at the repo root in this repository, so the root is
# one level up, not two. (It was two when these lived in
# causalbench/scripts/ in the source repo.)
REPO = HERE.parent
DEFAULT_DIR = REPO / "results/descriptives"
OUT_DIR = REPO / "paper/figures"

# only the control-cell spectra defend the dimension claim; the fMRI pooled
# spectra are a different quantity and are deliberately not panelled here
CTRL_KEY = "latent_dimension_from_controls"
# Eigenvalues live in a COMPRESSED SIDECAR beside the JSON, not inline: at cap
# 8000 Frangieh alone is 5707 float64 and there are 12 control-spectrum blocks.
# The JSON records the sidecar basename, the key inside it, and the integrity
# triple (lam_len / lam_sum / lam_sumsq) that the array must reproduce.
INTEGRITY = ("lam_len", "lam_sum", "lam_sumsq")
INTEGRITY_RTOL = 1e-12

# NUMERICAL RANK. Eigenvalues at or below NUM_RANK_TOL * lambda_max are
# floating-point residue from the SVD, not data. Before persistence this was
# invisible; with the real spectra it dominates four of the six panels, where
# the tail runs to ~1e-159 and a log axis then spans 160 orders of magnitude of
# noise. Worse, sigma^2 taken as the median of the trailing half collapses to
# ~0 and the Marchenko-Pastur edge becomes meaningless. So the spectrum is
# truncated at the numerical rank BEFORE sigma^2 and BEFORE plotting.
NUM_RANK_TOL = 1e-10


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def load_spectrum(dp, json_path):
    """Load the sidecar array and VERIFY it against the JSON before use.

    Returns (lam, note). lam is None when the sidecar is simply absent, which
    is not an error: it means the profiler predates spectrum persistence.
    A sidecar that is present but disagrees with its JSON IS an error and
    raises, because a silently-mismatched sidecar is precisely the stale
    artefact this project has been bitten by twice.
    """
    import numpy as np
    name, key = dp.get("spectra_sidecar"), dp.get("spectra_key")
    if not name or not key:
        return None, "no spectra_sidecar/spectra_key in the JSON"
    sc = Path(json_path).parent / name
    if not sc.exists():
        return None, f"sidecar named but missing on disk: {name}"
    with np.load(sc) as z:
        if key not in z.files:
            raise SystemExit(
                f"[fatal] sidecar {name} has no key {key!r}; "
                f"it holds {sorted(z.files)}")
        lam = np.asarray(z[key], dtype=np.float64)

    got = {"lam_len": int(lam.size), "lam_sum": float(lam.sum()),
           "lam_sumsq": float((lam ** 2).sum())}
    for f in INTEGRITY:
        exp = dp.get(f)
        if exp is None:
            raise SystemExit(f"[fatal] JSON is missing integrity field {f!r} "
                             f"for key {key!r}; refusing to plot unverified data")
        if f == "lam_len":
            ok = int(got[f]) == int(exp)
        else:
            ok = abs(got[f] - float(exp)) <= INTEGRITY_RTOL * max(
                abs(float(exp)), 1e-300)
        if not ok:
            raise SystemExit(
                f"[fatal] INTEGRITY MISMATCH for {key!r} in {name}\n"
                f"         {f}: JSON says {exp!r}, sidecar array gives "
                f"{got[f]!r}\n"
                f"         The sidecar is not the array the JSON summarises. "
                f"Refusing to plot.")
    return lam, "verified"


def mp_upper_edge(n, p, sigma2):
    """Marchenko-Pastur upper edge: sigma^2 * (1 + sqrt(p/n))^2."""
    if not n or not p or sigma2 is None:
        return None
    gamma = float(p) / float(n)
    return float(sigma2) * (1.0 + gamma ** 0.5) ** 2


def numerical_rank(lam, tol=NUM_RANK_TOL):
    """Count of eigenvalues above tol * lambda_max, and the truncated array.

    Everything past this index is floating-point residue from the SVD. It is
    dropped rather than plotted: it is not data, and leaving it in destroys
    both the y-axis scale and the sigma^2 estimate.
    """
    import numpy as np
    lam = np.sort(np.asarray(lam, dtype=np.float64))[::-1]
    keep = lam[lam > tol * lam[0]]
    return int(keep.size), keep


def sigma2_from_bulk(lam, frac=0.5):
    """Noise variance from the TRAILING fraction of the TRUNCATED spectrum.

    Median, not mean: the mean is inflated by the signal, which would push the
    MP edge up and hide exactly what the figure exists to show. Callers must
    pass an already-truncated array -- on a raw spectrum with a residue tail
    the median collapses to ~0 and the edge becomes meaningless.
    """
    lam = sorted((float(x) for x in lam), reverse=True)
    tail = lam[int(len(lam) * (1.0 - frac)):] or lam
    mid = len(tail) // 2
    return (tail[mid] if len(tail) % 2 else 0.5 * (tail[mid - 1] + tail[mid]))


def collect(dirpath):
    """Perturb-seq control spectra only, keyed (dataset, cap)."""
    panels, missing = [], []
    for p in sorted(Path(dirpath).glob("*.json")):
        try:
            doc = json.load(open(p))
        except ValueError:
            continue
        if (doc.get("meta") or {}).get("status") != "CURRENT":
            continue
        ds = doc.get("dataset")
        arm = ((doc.get("meta") or {}).get("config") or {}).get(
            "extra", {}).get("arm")
        label = f"{ds}_{arm}" if arm else str(ds)
        for b in doc.get("blocks") or []:
            if not isinstance(b, dict) or CTRL_KEY not in b:
                continue
            for cap, dp in (b[CTRL_KEY] or {}).items():
                if not isinstance(dp, dict):
                    continue
                lam, note = load_spectrum(dp, p)
                if lam is None:
                    missing.append((label, cap, p.name, note))
                else:
                    panels.append(dict(label=label, cap=str(cap), lam=lam,
                                       n=dp.get("n_control_used"),
                                       p=dp.get("n_genes"),
                                       pr=dp.get("participation_ratio")))
    return panels, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--outdir", default=str(OUT_DIR))
    ap.add_argument("--bulk-frac", type=float, default=0.5,
                    help="trailing fraction of the spectrum used to estimate "
                         "the noise variance for the MP edge")
    a = ap.parse_args()

    panels, missing = collect(a.dir)

    if not panels:
        print("=" * 74)
        print("SPECTRUM FIGURE NOT BUILT -- no eigenvalue array is stored.")
        print("=" * 74)
        print(f"  source dir : {rel(a.dir)}")
        print(f"  checked    : {len(missing)} control-spectrum block(s)")
        for label, cap, fn, note in missing[:12]:
            print(f"    {label:<22} cap={cap:<6} {note}")
        print("")
        print("  The artefacts persist only SUMMARIES of the spectrum")
        print("  (participation_ratio, effective_rank_exp_spectral_entropy,")
        print("  n_components_for_variance, top_eigenvalue_share). The full")
        print("  array cannot be reconstructed from those: infinitely many")
        print("  spectra share a given participation ratio and entropy.")
        print("")
        print("  NOTHING WAS APPROXIMATED OR SYNTHESISED. To enable this")
        print("  figure, persist the eigenvalue array in")
        print("  85_dataset_descriptives.spectrum_estimators (it already")
        print("  computes it as `lam` and discards it) and re-run the A100")
        print("  profiling.")
        return 2

    # ---- from here the data exists; build the figures
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    caps = sorted({d["cap"] for d in panels}, key=lambda c: int(c))
    primary = "2000" if "2000" in caps else caps[0]
    written = []

    for cap in caps:
        sel = sorted((d for d in panels if d["cap"] == cap),
                     key=lambda d: d["label"])
        if not sel:
            continue
        is_primary = (cap == primary)
        ncol = 3
        nrow = (len(sel) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.5 * nrow),
                                 squeeze=False)
        for ax, d in zip(axes.ravel(), sel):
            raw = np.sort(np.asarray(d["lam"], dtype=float))[::-1]
            nr, lam = numerical_rank(raw)          # TRUNCATE FIRST
            tot = raw.sum()                        # share of TOTAL variance
            share = lam / tot
            n, p = d["n"], d["p"]
            bound = min(int(n) - 1, int(p)) if n and p else None

            ax.semilogy(np.arange(1, nr + 1), share, lw=1.2, color="black",
                        label="observed")
            s2 = sigma2_from_bulk(lam, a.bulk_frac)   # from the TRUNCATED tail
            edge = mp_upper_edge(n, p, s2)
            cross = int((lam > edge).sum()) if edge else None
            if edge:
                ax.axhline(edge / tot, ls="--", lw=1.0, color="crimson",
                           label=f"MP edge, crossing at {cross}")
            if isinstance(d["pr"], (int, float)):
                ax.axvline(d["pr"], ls=":", lw=1.2, color="navy",
                           label=f"participation ratio = {d['pr']:.0f}")
            if cross:
                ax.axvline(cross, ls="-.", lw=1.0, color="crimson", alpha=0.6)

            # y-limits from the RETAINED range only; residue must not set scale
            ax.set_ylim(share.min() * 0.5, share.max() * 2.0)
            d["_report"] = dict(n=n, p=p, bound=bound, numrank=nr,
                                sigma2=s2, mp_cross=cross, pr=d["pr"],
                                span_oom=float(np.log10(lam[0] / lam[-1])))

            ax.set_title(f"{d['label']}   n={n}, p={p}   "
                         f"rank {nr}/{bound}", fontsize=9)
            ax.set_xlabel("eigenvalue index")
            ax.set_ylabel("eigenvalue / total variance")
            # rank-limited by sample size: p far exceeds what n can support
            if bound and p and bound < 0.5 * p:
                ax.text(0.02, 0.04,
                        f"rank-limited by sample size (n-1={bound} << p={p});\n"
                        f"MP edge computed on the retained portion",
                        transform=ax.transAxes, fontsize=6.5, va="bottom",
                        color="dimgray")
            elif bound and nr < 0.9 * bound:
                ax.text(0.02, 0.04,
                        f"numerical rank {nr} < bound {bound}: "
                        f"{bound - nr} dependent directions",
                        transform=ax.transAxes, fontsize=6.5, va="bottom",
                        color="dimgray")
            ax.legend(fontsize=7, loc="upper right")
        for ax in axes.ravel()[len(sel):]:
            ax.axis("off")
        kind = ("primary: matched sample size" if is_primary
                else "sensitivity check: n varies across panels")
        fig.suptitle(
            f"Control-covariance eigenvalue spectra, cap n={cap}  ({kind})\n"
            f"truncated at numerical rank, tol = {NUM_RANK_TOL:g} x lambda_max\n"
            f"MP edge: sigma^2 = median of the trailing {a.bulk_frac:.0%} of the "
            f"TRUNCATED spectrum, not the mean (which the signal inflates)\n"
            f"generated {utc_now()}", fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        stem = f"spectrum_cap{cap}" + ("_primary" if is_primary else
                                       "_sensitivity")
        fig.savefig(out / f"{stem}.pdf")
        fig.savefig(out / f"{stem}.png", dpi=300)
        plt.close(fig)
        written.append(f"{stem}.pdf/.png ({len(sel)} panels)")

    print(f"\nverified {len(panels)} spectra against their JSON integrity "
          f"triples\n")
    print(f"  {'panel':<24}{'cap':>6}{'n':>7}{'p':>7}{'bound':>7}{'numrank':>9}"
          f"{'sigma2':>12}{'MPcross':>9}{'PR':>8}{'span_oom':>10}")
    over = []
    for d in sorted(panels, key=lambda x: (x["cap"], x["label"])):
        r = d.get("_report")
        if not r:
            continue
        print(f"  {d['label']:<24}{d['cap']:>6}{r['n']:>7}{r['p']:>7}"
              f"{r['bound']:>7}{r['numrank']:>9}{r['sigma2']:>12.4g}"
              f"{r['mp_cross']:>9}{r['pr']:>8.0f}{r['span_oom']:>10.2f}")
        if r["span_oom"] > 6.0:
            over.append((d["label"], d["cap"], r["span_oom"]))
    if over:
        print("\n  !! SPAN CHECK FAILED -- retained y-range exceeds ~6 orders")
        for lab, cap, sp in over:
            print(f"     {lab} cap={cap}: {sp:.2f} orders")
        print("     NOT clipped. Reported as-is.")
    print("")
    for w in written:
        print(f"  wrote {rel(out)}/{w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
