"""Faithful E3 positive control under both null geometries and BH-FDR."""

import importlib.util
import json
from pathlib import Path

import anndata as ad
import numpy as np

from e3_gate_compare import (compare_geometries, comparison_summary,
                             decision_records)
from data_paths import perturbseq_path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pr = load("pr", "src/gate/precision_readout.py")
SIM = load("sim", "sim/simulator.py")
BK = load("bk", "src/recover/backbone.py")

DLAT = 10
DPROJ = 10
R_PLANT = 3
NPER = 200
IV_SCALE = 0.1
EDGE_PROB = 0.4
SEED = 0
ALPHA = 0.05
Q = 0.05
B_BOOT = 500


def assert_equal_size_paths_identical(comparison):
    old = comparison["old_two_bootstrap"]
    corrected = comparison["corrected_disjoint"]
    for key in ("signals", "thresholds", "pvalues", "raw_detect", "bh_detect"):
        if old[key] != corrected[key]:
            raise AssertionError(
                f"equal-size positive-control path changed across geometries: {key}")


def main():
    rng = np.random.default_rng(SEED)

    A = ad.read_h5ad(perturbseq_path("causalbench_k562.h5ad"))
    X = (A.X.toarray().astype(np.float64) if hasattr(A.X, "toarray")
         else np.asarray(A.X, np.float64))
    g = A.obs["guide_ids"].astype(str).values
    Xc = X[g == ""]
    del A, X
    DGENE = Xc.shape[1]
    Xc_centered = Xc - Xc.mean(0)
    bg_total = float(Xc_centered.var(0).sum())
    print(f"real noise pool: {Xc_centered.shape}, total var {bg_total:.3f}")

    env_specs = [SIM.EnvSpec("basis", None, ()), SIM.EnvSpec("obs", None, ())]
    for i in range(R_PLANT):
        env_specs.append(SIM.EnvSpec(f"iv{i}", "hard", (i,), IV_SCALE))
    ds = SIM.simulate(
        DLAT, DGENE, NPER, env_specs, SEED, edge_prob=EDGE_PROB)
    environments = ds.environments
    print("envs:", list(environments.keys()))
    for i in range(R_PLANT):
        rank = SIM.constructed_rank(
            ds.B, ds.noise_var, "hard", (i,), iv_scale=IV_SCALE)
        print(f"  iv{i} constructed rank: {rank}")

    mixing = rng.standard_normal((DGENE, DLAT))
    mixing /= np.linalg.norm(mixing, axis=0, keepdims=True)
    signal_total = float((environments["obs"].Z @ mixing.T).var(0).sum()) + 1e-12

    def make_env(Z, snr, seed):
        local_rng = np.random.default_rng(seed)
        if snr > 0:
            signal = ((Z @ mixing.T)
                      * np.sqrt(snr * bg_total / signal_total))
        else:
            signal = np.zeros((len(Z), DGENE))
        noise_idx = local_rng.integers(0, len(Xc_centered), len(Z))
        return signal + Xc_centered[noise_idx]

    def run_gate(snr, seeds, gate_seed):
        Xbasis = make_env(environments["basis"].Z, snr, seeds["basis"])
        Xobs = make_env(environments["obs"].Z, snr, seeds["obs"])
        Xint = [make_env(environments[f"iv{k}"].Z, snr, seeds["iv"] + k)
                for k in range(R_PLANT)]
        mu, W = BK.fit_pca(Xbasis, DPROJ)
        Yobs = BK.project(Xobs, mu, W)
        Yint = [BK.project(values, mu, W) for values in Xint]
        labels = [f"iv{k}" for k in range(R_PLANT)]
        planted_cmp = compare_geometries(
            pr, Yint, Yobs, seed=gate_seed, alpha=ALPHA, B=B_BOOT, q=Q)
        assert_equal_size_paths_identical(planted_cmp)

        nfake = 30
        Yfake = []
        for j in range(nfake):
            Xfake = make_env(environments["basis"].Z, snr, 700_000 + j)
            Yfake.append(BK.project(Xfake, mu, W))
        fake_cmp = compare_geometries(
            pr, Yfake, Yobs, seed=10_000 + gate_seed,
            alpha=ALPHA, B=B_BOOT, q=Q)
        assert_equal_size_paths_identical(fake_cmp)

        fake_labels = [f"null_basis_draw_{i:02d}" for i in range(nfake)]
        return dict(
            snr=float(snr),
            planted_screen=comparison_summary(planted_cmp),
            negative_control_screen=comparison_summary(fake_cmp),
            planted_records=decision_records(labels, Yint, planted_cmp),
            negative_control_records=decision_records(
                fake_labels, Yfake, fake_cmp),
        )

    smoke = run_gate(
        8.0, dict(basis=9000, obs=9001, iv=9100), gate_seed=80_000)
    smoke_count = smoke["planted_screen"]["corrected_disjoint"]["raw_count"]
    print(f"smoke: corrected raw {smoke_count}/{R_PLANT} at snr=8")
    if smoke_count < R_PLANT:
        raise SystemExit(
            f"[stop] construction broken: strong planted intervention not detected "
            f"({smoke_count}/{R_PLANT})")

    dose_response = []
    for index, snr in enumerate([0.0, 0.5, 1.0, 2.0, 4.0]):
        result = run_gate(
            snr, dict(basis=5000, obs=5001, iv=5100),
            gate_seed=30_000 + index)
        dose_response.append(result)
        print(
            f"snr {snr}: old raw/BH "
            f"{result['planted_screen']['old_two_bootstrap']['raw_count']}/"
            f"{result['planted_screen']['old_two_bootstrap']['bh_count']} | "
            f"corrected raw/BH "
            f"{result['planted_screen']['corrected_disjoint']['raw_count']}/"
            f"{result['planted_screen']['corrected_disjoint']['bh_count']}")

    report = dict(
        experiment=(
            "Faithful positive control: shared-SCM reduced-variance interventions "
            "mixed into gene space with real K562 control noise"),
        d_latent=DLAT,
        d_gene=DGENE,
        d_proj=DPROJ,
        planted_rank=R_PLANT,
        cells_per_env=NPER,
        iv_scale=IV_SCALE,
        alpha=ALPHA,
        q=Q,
        B=B_BOOT,
        primary_decision="corrected_disjoint BH-FDR at q=0.05",
        bh_families=(
            "Within each SNR arm, BH applied separately to planted interventions "
            "and null basis draws"),
        equal_size_geometry_expected_identical=True,
        scale="signal total variance = SNR times real background total variance",
        null="independent un-intervened basis draws versus observational draw",
        note="snr=0 is a null sanity check, not a real-data E3 verdict",
        real_bg_var_total=round(bg_total, 6),
        smoke_snr8=smoke,
        dose_response=dose_response,
    )
    out = Path("results/e3_poscontrol/poscontrol_final.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in (
        "experiment", "planted_rank", "cells_per_env", "primary_decision",
        "equal_size_geometry_expected_identical", "smoke_snr8")}, indent=2))


if __name__ == "__main__":
    main()
