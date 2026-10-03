"""Shared old-versus-corrected gate reporting for E3 real-data reruns.

The historical arm uses two size-matched bootstrap samples from the control
(`disjoint=False`). The corrected arm uses a disjoint pseudo-environment and
reference when the environment is at most half the control size
(`disjoint=True`). Both arms report raw alpha-level decisions and BH-FDR
decisions, and both retain per-environment values for auditability.

This module is an experiment helper, not a second gate implementation. All
statistics, null draws, empirical p-values, and BH decisions are delegated to
src/gate/precision_readout.py.
"""

import numpy as np


def _detect_family(pr, Y_envs, Y_obs, seed, alpha, B, q, disjoint,
                   readout="precision", Y_null=None):
    """Detect each environment on a paired deterministic RNG stream.

    readout and Y_null pass straight to pr.detect_with_pvalues (Tier 2). The
    defaults are the shared-reference precision gate every E3 script uses.
    """
    signals, thresholds, raw_detect, pvalues = [], [], [], []
    for i, Y in enumerate(Y_envs):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), i]))
        one = pr.detect_with_pvalues(
            [Y], Y_obs, alpha=alpha, B=B, rng=rng, q=q,
            disjoint=disjoint, readout=readout, Y_null=Y_null)
        signals.append(float(one["signals"][0]))
        thresholds.append(float(one["thresholds"][0]))
        raw_detect.append(bool(one["raw_detect"][0]))
        pvalues.append(float(one["pvalues"][0]))
    bh_detect = pr.bh_fdr(pvalues, q=q)
    return dict(
        signals=signals,
        thresholds=thresholds,
        raw_detect=raw_detect,
        pvalues=pvalues,
        bh_detect=bh_detect.tolist(),
        raw_count=int(sum(raw_detect)),
        bh_count=int(bh_detect.sum()),
        alpha=float(alpha),
        q=float(q),
        B=int(B),
        disjoint=bool(disjoint),
    )


def compare_geometries(pr, Y_envs, Y_obs, seed, alpha=0.05, B=500, q=0.05):
    """Run the same environment family under historical and corrected nulls."""
    Y_envs = [np.asarray(Y) for Y in Y_envs]
    Y_obs = np.asarray(Y_obs)
    old = _detect_family(
        pr, Y_envs, Y_obs, seed, alpha, B, q, disjoint=False)
    corrected = _detect_family(
        pr, Y_envs, Y_obs, seed, alpha, B, q, disjoint=True)
    old["disjoint_applied"] = [False] * len(Y_envs)
    corrected["disjoint_applied"] = [
        bool(pr._use_disjoint_null(len(Y_obs), len(Y), Y_obs.shape[1], True))
        for Y in Y_envs
    ]
    return dict(old_two_bootstrap=old, corrected_disjoint=corrected)


def _fraction(count, total):
    return round(float(count / total), 6) if total else None


def summarize(result):
    """Compact JSON-safe summary of one detect_with_pvalues result."""
    signals = np.asarray(result["signals"], dtype=float)
    thresholds = np.asarray(result["thresholds"], dtype=float)
    pvalues = np.asarray(result["pvalues"], dtype=float)
    n = len(signals)
    ratios = np.divide(
        signals, thresholds, out=np.full(signals.shape, np.nan),
        where=thresholds != 0.0)
    finite = ratios[np.isfinite(ratios)]
    return dict(
        n_environments=n,
        n_disjoint_applied=int(sum(result["disjoint_applied"])),
        raw_count=int(result["raw_count"]),
        raw_fraction=_fraction(result["raw_count"], n),
        bh_count=int(result["bh_count"]),
        bh_fraction=_fraction(result["bh_count"], n),
        ratio_median=None if not finite.size else round(float(np.median(finite)), 6),
        ratio_p90=None if not finite.size else round(float(np.quantile(finite, 0.9)), 6),
        pvalue_median=None if not pvalues.size else round(float(np.median(pvalues)), 6),
        threshold_median=(None if not thresholds.size
                          else round(float(np.median(thresholds)), 6)),
        min_attainable_pvalue=round(1.0 / (int(result["B"]) + 1), 9),
        alpha=float(result["alpha"]),
        q=float(result["q"]),
        B=int(result["B"]),
    )


def comparison_summary(comparison):
    """Side-by-side summaries plus decision-transition counts."""
    old = comparison["old_two_bootstrap"]
    corrected = comparison["corrected_disjoint"]
    old_raw = np.asarray(old["raw_detect"], dtype=bool)
    new_raw = np.asarray(corrected["raw_detect"], dtype=bool)
    old_bh = np.asarray(old["bh_detect"], dtype=bool)
    new_bh = np.asarray(corrected["bh_detect"], dtype=bool)
    return dict(
        old_two_bootstrap=summarize(old),
        corrected_disjoint=summarize(corrected),
        transitions=dict(
            raw_old_only=int(np.sum(old_raw & ~new_raw)),
            raw_corrected_only=int(np.sum(~old_raw & new_raw)),
            raw_both=int(np.sum(old_raw & new_raw)),
            bh_old_only=int(np.sum(old_bh & ~new_bh)),
            bh_corrected_only=int(np.sum(~old_bh & new_bh)),
            bh_both=int(np.sum(old_bh & new_bh)),
        ),
    )


def decision_records(labels, Y_envs, comparison):
    """Return one fully auditable JSON record per environment."""
    labels = [str(label) for label in labels]
    Y_envs = [np.asarray(Y) for Y in Y_envs]
    if len(labels) != len(Y_envs):
        raise ValueError("labels and Y_envs must have the same length")
    old = comparison["old_two_bootstrap"]
    corrected = comparison["corrected_disjoint"]
    if len(labels) != len(old["signals"]) or len(labels) != len(corrected["signals"]):
        raise ValueError("comparison length does not match labels")

    rows = []
    for i, (label, Y) in enumerate(zip(labels, Y_envs)):
        old_signal = float(old["signals"][i])
        new_signal = float(corrected["signals"][i])
        if not np.isclose(old_signal, new_signal, rtol=0.0, atol=1e-12):
            raise AssertionError("observed signal changed across null geometries")
        rows.append(dict(
            environment=label,
            n_samples=int(len(Y)),
            signal=old_signal,
            old_two_bootstrap=dict(
                threshold=float(old["thresholds"][i]),
                pvalue=float(old["pvalues"][i]),
                raw_detect=bool(old["raw_detect"][i]),
                bh_detect=bool(old["bh_detect"][i]),
                disjoint_applied=False,
            ),
            corrected_disjoint=dict(
                threshold=float(corrected["thresholds"][i]),
                pvalue=float(corrected["pvalues"][i]),
                raw_detect=bool(corrected["raw_detect"][i]),
                bh_detect=bool(corrected["bh_detect"][i]),
                disjoint_applied=bool(corrected["disjoint_applied"][i]),
            ),
        ))
    return rows


def shift_alignment(Y_envs, Y_obs, detected, top_k=2):
    """Energy of detected mean-shift directions in the first control PCs."""
    energies = []
    for Y, keep in zip(Y_envs, detected):
        if not keep:
            continue
        shift = np.asarray(Y).mean(0) - np.asarray(Y_obs).mean(0)
        norm = float(np.linalg.norm(shift))
        direction = shift / (norm + 1e-12)
        energies.append(float(np.sum(direction[:top_k] ** 2)))
    if not energies:
        return dict(n_detected=0, top2_energy_mean=None, top2_energy_median=None)
    return dict(
        n_detected=len(energies),
        top2_energy_mean=round(float(np.mean(energies)), 6),
        top2_energy_median=round(float(np.median(energies)), 6),
    )
