"""E5 (Tier 2, Experiment B): the split-control gate.

Goal: remove the shared-reference dependence that prevents a clean BH statement.
In the E3 (shared) design every environment's observed statistic and its null
both use the same control cells, so the per-environment p-values share one
reference sample.

Split-control design, for each dataset and gate seed: split the control cells
once, 50/50 at random (SeedSequence([SPLIT_SALT, seed])). Half A builds every
null: each environment's null draws and thresholds come from Half A only, with the
unchanged size matching (disjoint pseudo-environment/reference split inside Half A
when n_e <= n_A/2, two-resample otherwise). Half B is the reference: the observed
statistic is T_e = lambda_max(Omega_e - Omega_B), and Half B is never resampled.
BH within family at q=0.05 is unchanged. The control-fit projection is unchanged
(fitted on all control cells, as in E3), and no recovery is run here.

Every family is screened under both designs on identical inputs and identical
per-environment null RNG streams, so the file carries its own shared-design
reproduction of E3 and a shared_vs_split block per dataset or family:
raw_count and bh_count under both designs, and the Jaccard overlap of the BH sets
(None when both are empty).

Parts:
    real           K562 / RPE1 / Norman, five gate seeds, perturbations plus the E3
                   random pure-control family (e3_stability_perturbseq.py layout).
    rpe1-confound  RPE1 perturbations, 20 random and 20 structured control splits,
                   seed 0 (e3_rpe1_confound_check.py layout).
    poscontrol     the faithful positive control at the same signal-to-background
                   ratios (e3_poscontrol_faithful.py layout).

PRE-REGISTERED EXPECTATION (from the Tier 2 brief): verdicts barely move. K562 and
Norman stay at 0 after BH, RPE1 stays near 48, structured splits still fire,
random splits still do not, the positive control stays 0/3 through ratio 2 and
3/3 at ratio 8. A large drop in RPE1's BH count is a result, not a bug; nothing is
tuned.

Positive-control geometry, two recorded variants. With the E3 observational draw
of 200 cells each half has 100, every planted environment (200 cells) is larger
than Half A, and the split null falls back to 200 draws with replacement from 100
rows while the observed statistic uses a 100-cell reference: not like-for-like.
The 400-cell variant appends 200 more observational latents from the same SCM so
each half has 200 cells, matching every environment. Both are run in one
invocation (e5_poscontrol.json and e5_poscontrol_obs400.json); the geometry of
each is written into its artifact.

Usage:
    python experiments/e5_split_control.py real --dataset rpe1
    python experiments/e5_split_control.py rpe1-confound
    python experiments/e5_split_control.py poscontrol
    python experiments/e5_split_control.py real --dataset rpe1 --selftest --out-dir /tmp/x
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

import tier2_common as T2

warnings.filterwarnings("ignore", message=r".*encountered in matmul",
                        category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results" / "e5_split_control"
DESIGNS = ("corrected_disjoint", "split_control")
DESIGN_NOTE = dict(
    corrected_disjoint=("shared-reference design: observed statistic and null "
                        "both use all control cells (reproduces E3)"),
    split_control=("Half A builds every null, Half B is the fixed reference and "
                   "is never resampled"),
)


def _both(Y_envs, Y_obs, Y_null, Y_ref, seed, B):
    shared = T2.screen(Y_envs, Y_obs, seed, B=B)
    split = T2.screen(Y_envs, Y_ref, seed, Y_null=Y_null, B=B)
    return shared, split


def _reproduces(labels, shared, reference):
    """Does the shared design reproduce the committed E3 decisions?"""
    if reference is None:
        return None
    if reference["labels"] != list(labels):
        return dict(labels_match=False, raw_match=None, bh_match=None)
    return dict(
        labels_match=True,
        raw_match=T2.selected(labels, shared["raw_detect"]) == reference["raw"],
        bh_match=T2.selected(labels, shared["bh_detect"]) == reference["bh"],
    )


def _aggregate(rows, screen_name):
    out = {}
    for design in DESIGNS:
        for decision in ("raw", "bh"):
            vals = [row[screen_name][design][f"{decision}_fraction"] for row in rows]
            stats = T2.seed_sd(vals)
            prefix = f"{design}_{decision}_fraction"
            out[f"{prefix}_mean"] = stats["mean"]
            out[f"{prefix}_sd"] = stats["sd"]
            out[f"{prefix}_per_seed"] = stats["per_seed"]
    return out


def _svs_summary(entries):
    """Seed means for a list of shared_vs_split entries."""
    def mean(path):
        vals = [e[path[0]][path[1]] for e in entries]
        return round(float(np.mean(vals)), 4)
    jac = [e["jaccard_bh"] for e in entries if e["jaccard_bh"] is not None]
    return dict(
        shared_raw_count_mean=mean(("shared", "raw_count")),
        shared_bh_count_mean=mean(("shared", "bh_count")),
        split_raw_count_mean=mean(("split", "raw_count")),
        split_bh_count_mean=mean(("split", "bh_count")),
        jaccard_bh_mean=(round(float(np.mean(jac)), 4) if jac else None),
        n_seeds_both_bh_empty=int(sum(e["both_bh_empty"] for e in entries)),
    )


# ------------------------------------------------------------------ real panel
def run_real(key, out_dir, seeds, B, selftest=False):
    started = time.time()
    panel = T2.make_selftest_panel(key) if selftest else T2.load_panel(key)
    name = panel["name"]
    Xc, perts = panel["Xc"], panel["perts"]
    proj = T2.control_projection(Xc)
    Yobs = proj(Xc)
    Yperts = [proj(X) for X in panel["Xperts"]]
    sizes = [int(len(Y)) for Y in Yperts]
    print(f"{name}: control {len(Xc)} | powered perts {len(perts)}", flush=True)
    stab = None if selftest else T2.e3_stability_sets(name)

    rows, svs = [], []
    for seed in seeds:
        Y_null, Y_ref, split_info = T2.split_halves(Yobs, seed)
        shared, split = _both(Yperts, Yobs, Y_null, Y_ref, seed, B)
        fake_labels, Yfake = T2.random_controls(Xc, proj, sizes, seed, T2.N_FAKE)
        fshared, fsplit = _both(Yfake, Yobs, Y_null, Y_ref,
                                T2.FAKE_SEED_OFFSET + seed, B)
        designs = dict(corrected_disjoint=shared, split_control=split)
        entry = T2.shared_vs_split_entry(perts, shared, split)
        entry["seed"] = seed
        entry["negative_controls"] = T2.shared_vs_split_entry(
            fake_labels, fshared, fsplit)
        svs.append(entry)
        rows.append(dict(
            seed=seed,
            n_perts=len(perts),
            split=split_info,
            perturbation_screen=T2.design_comparison(shared, split),
            negative_control_screen=T2.design_comparison(fshared, fsplit),
            confound_alignment=T2.alignment(Yperts, Yobs, designs),
            bh_selected_ids=dict(
                corrected_disjoint=T2.selected(perts, shared["bh_detect"]),
                split_control=T2.selected(perts, split["bh_detect"])),
            shared_reproduces_e3_stability=_reproduces(
                perts, shared, stab["per_seed"].get(seed) if stab else None),
            per_perturbation=T2.design_records(perts, Yperts, designs),
            negative_control_records=T2.design_records(
                fake_labels, Yfake,
                dict(corrected_disjoint=fshared, split_control=fsplit)),
        ))
        print(f"  seed {seed}: shared raw/BH {shared['raw_count']}/{shared['bh_count']}"
              f" | split raw/BH {split['raw_count']}/{split['bh_count']} | Jaccard BH "
              f"{entry['jaccard_bh']} | controls shared {fshared['raw_count']}/"
              f"{fshared['bh_count']} split {fsplit['raw_count']}/{fsplit['bh_count']}"
              f" ({time.time() - started:.0f}s)", flush=True)

    block = dict(
        dataset=name,
        source=panel["source"],
        n_control=int(len(Xc)),
        n_perts=len(perts),
        seeds=list(seeds),
        alpha=T2.ALPHA, q=T2.Q, B=B,
        primary_decision="split_control BH-FDR at q=0.05",
        bh_families=("Within each seed and design, BH applied separately to powered "
                     "perturbations and random controls"),
        perturbation_stability=_aggregate(rows, "perturbation_screen"),
        negative_control_stability=_aggregate(rows, "negative_control_screen"),
        shared_vs_split=dict(per_seed=svs, summary=_svs_summary(svs)),
        e3_reference=({k: stab[k] for k in ("source", "sha256")} if stab else None),
        selftest=bool(selftest),
        per_seed=rows,
        provenance=T2.provenance(__file__),
        wall_seconds=round(time.time() - started, 1),
    )
    top = dict(
        experiment="e5_split_control_real_panel",
        designs=DESIGN_NOTE,
        config=dict(readout="precision", d_proj=T2.D_PROJ, nmin=T2.NMIN,
                    alpha=T2.ALPHA, q=T2.Q, B=B, seeds=list(seeds),
                    n_random_controls=T2.N_FAKE, split_salt=T2.SPLIT_SALT,
                    projection="control-fit PCA on all control cells (unchanged)",
                    jaccard_convention="None when both BH sets are empty"),
    )
    out = Path(out_dir) / "e5_real_panel.json"
    T2.update_dataset_block(out, top, name, block)
    print(f"written {out} [{name}] ({block['wall_seconds']}s)", flush=True)


# ------------------------------------------------------------------ RPE1 confound
def run_rpe1_confound(out_dir, B, selftest=False):
    started = time.time()
    seed = 0
    n_random = 20
    panel = T2.make_selftest_panel("rpe1") if selftest else T2.load_panel("rpe1")
    Xc, perts = panel["Xc"], panel["perts"]
    proj = T2.control_projection(Xc)
    Yobs = proj(Xc)
    Yperts = [proj(X) for X in panel["Xperts"]]
    sizes = [len(Y) for Y in Yperts]
    Y_null, Y_ref, split_info = T2.split_halves(Yobs, seed)

    struct_labels, Ystruct = T2.structured_controls(Xc, proj, Yobs, sizes)
    random_labels, Yrandom = T2.random_controls(Xc, proj, sizes, seed, n_random)
    families = dict(
        perturbations=(perts, Yperts, seed),
        random_controls=(random_labels, Yrandom, T2.FAKE_SEED_OFFSET + seed),
        structured_controls=(struct_labels, Ystruct, T2.STRUCT_SEED_OFFSET + seed),
    )
    results = {}
    for fam, (labels, Ys, fam_seed) in families.items():
        results[fam] = _both(Ys, Yobs, Y_null, Y_ref, fam_seed, B)
        s, p = results[fam]
        print(f"  {fam}: shared raw/BH {s['raw_count']}/{s['bh_count']} | split "
              f"raw/BH {p['raw_count']}/{p['bh_count']} of {len(labels)}", flush=True)

    ref = None
    ref_path = REPO / "results" / "e3" / "e3_rpe1_confound_check.json"
    if not selftest and ref_path.exists():
        doc = json.loads(ref_path.read_text())
        ref = dict(source=str(ref_path.relative_to(REPO)),
                   sha256=T2.sha256_file(ref_path), checks={})
        for fam, rec_key, screen_key in (
                ("perturbations", "per_perturbation", "perturbation_screen"),
                ("random_controls", "random_control_records", "random_control_screen"),
                ("structured_controls", "structured_control_records",
                 "structured_control_screen")):
            recs = doc[rec_key]
            reference = dict(
                labels=[r["environment"] for r in recs],
                raw=[r["environment"] for r in recs if r["corrected_disjoint"]["raw_detect"]],
                bh=[r["environment"] for r in recs if r["corrected_disjoint"]["bh_detect"]])
            ref["checks"][fam] = _reproduces(families[fam][0], results[fam][0], reference)

    def designs(fam):
        s, p = results[fam]
        return dict(corrected_disjoint=s, split_control=p)

    s_pert, p_pert = results["perturbations"]
    report = dict(
        experiment="e5_split_control_rpe1_confound",
        dataset="causalbench_rpe1" if not selftest else "selftest_rpe1",
        n_control=int(len(Xc)),
        n_powered_perts=len(perts),
        d_proj=T2.D_PROJ, nmin=T2.NMIN, alpha=T2.ALPHA, q=T2.Q, B=B,
        designs=DESIGN_NOTE,
        split=split_info,
        primary_decision="split_control BH-FDR at q=0.05",
        bh_families=("Within each design, BH applied separately to powered "
                     "perturbations, random controls, and structured controls"),
        n_detected_perts=int(p_pert["bh_count"]),
        perturbation_screen=T2.design_comparison(*results["perturbations"]),
        random_control_screen=T2.design_comparison(*results["random_controls"]),
        structured_control_screen=T2.design_comparison(
            *results["structured_controls"]),
        perturbation_shift_alignment=T2.alignment(
            Yperts, Yobs, designs("perturbations")),
        interpretation=(
            "Structured-control firing and leading-PC alignment diagnose association "
            "with control heterogeneity; they do not prove that heterogeneity caused "
            "the perturbation detections. Structured and random splits are drawn from "
            "all control cells, as in E3, so they overlap both halves."),
        shared_vs_split={fam: T2.shared_vs_split_entry(families[fam][0], *results[fam])
                         for fam in families},
        shared_reproduces_e3=ref,
        per_perturbation=T2.design_records(perts, Yperts, designs("perturbations")),
        random_control_records=T2.design_records(
            random_labels, Yrandom, designs("random_controls")),
        structured_control_records=T2.design_records(
            struct_labels, Ystruct, designs("structured_controls")),
        selftest=bool(selftest),
        provenance=T2.provenance(__file__),
        wall_seconds=round(time.time() - started, 1),
    )
    out = Path(out_dir) / "e5_rpe1_confound.json"
    T2.write_json(out, report)
    print(f"written {out} ({report['wall_seconds']}s)", flush=True)


# ------------------------------------------------------------------ positive control
# Construction constants. Must equal experiments/e3_poscontrol_faithful.py; checked
# at run time against that module on real (non-selftest) runs.
PC = dict(DLAT=10, DPROJ=10, R_PLANT=3, NPER=200, IV_SCALE=0.1, EDGE_PROB=0.4,
          SEED=0, ALPHA=0.05, Q=0.05, B_BOOT=500)
PC_SMOKE = dict(snr=8.0, seeds=dict(basis=9000, obs=9001, iv=9100), gate_seed=80_000)
PC_DOSE = [0.0, 0.5, 1.0, 2.0, 4.0]
PC_NFAKE = 30


def _check_poscontrol_constants():
    module = T2.load_module("grd_e3_poscontrol_for_e5",
                            "experiments/e3_poscontrol_faithful.py")
    mismatched = {k: (v, getattr(module, k)) for k, v in PC.items()
                  if getattr(module, k) != v}
    if mismatched:
        raise SystemExit(f"[stop] constants differ from e3_poscontrol_faithful: "
                         f"{mismatched}")


def _poscontrol_noise_pool(selftest):
    if selftest:
        rng = np.random.default_rng(91_000)
        L = rng.standard_normal((15, 120))
        Xc = rng.standard_normal((3000, 15)) @ L + rng.standard_normal((3000, 120))
        return Xc
    import anndata as ad
    A = ad.read_h5ad(T2.perturbseq_path("causalbench_k562.h5ad"))
    X = A.X
    g = A.obs["guide_ids"].astype(str).values
    del A
    return T2._dense_rows(X, g == "")


# Observational draw sizes for the positive control. 200 is the E3 construction; with
# 200 cells the split halves (100) are smaller than every 200-cell environment. 400
# draws 200 more observational latents from the same SCM so each half has 200 cells.
PC_N_OBS = (200, 400)
PC_OBS_EXTRA_SEED = 40_400
PC_GEOMETRY_NOTE = {
    200: ("The observational draw has 200 cells, so each half has 100 and every "
          "200-cell planted or null environment is larger than Half A. The "
          "split-control null therefore uses the two-resample path (200 draws with "
          "replacement from 100 rows) while the observed statistic compares 200 cells "
          "with a 100-cell reference. Run as specified; not a like-for-like null for "
          "this construction. The shared design here is the E3 equal-size path."),
    400: ("The observational draw has 400 cells: the E3 200 plus 200 more "
          "observational latents from the same SCM (seed 40400), the first 200 "
          "identical to the 200-cell variant. Each half has 200 cells, equal to every "
          "planted and null environment, so the split null is the equal-size "
          "two-resample path within Half A and the observed statistic compares 200 "
          "cells with a 200-cell reference. The shared design now uses the disjoint "
          "null (n_e = 200 <= 400/2), so it is not the E3 construction and is not "
          "checked against the committed E3 result. The signal scale is defined on "
          "the original 200 observational cells in both variants, so the ratios are "
          "the same."),
}


def run_poscontrol(out_dir, B, selftest=False, n_obs_list=PC_N_OBS):
    started = time.time()
    SIM = T2.load_module("grd_sim_for_e5", "sim/simulator.py")
    BK = T2.load_module("grd_backbone_for_e5", "src/recover/backbone.py")
    if not selftest:
        _check_poscontrol_constants()
    rng = np.random.default_rng(PC["SEED"])

    Xc = _poscontrol_noise_pool(selftest)
    DGENE = Xc.shape[1]
    Xc_centered = Xc - Xc.mean(0)
    bg_total = float(Xc_centered.var(0).sum())
    print(f"noise pool {Xc_centered.shape}, total var {bg_total:.3f}", flush=True)

    env_specs = [SIM.EnvSpec("basis", None, ()), SIM.EnvSpec("obs", None, ())]
    for i in range(PC["R_PLANT"]):
        env_specs.append(SIM.EnvSpec(f"iv{i}", "hard", (i,), PC["IV_SCALE"]))
    ds = SIM.simulate(PC["DLAT"], DGENE, PC["NPER"], env_specs, PC["SEED"],
                      edge_prob=PC["EDGE_PROB"])
    environments = ds.environments

    mixing = rng.standard_normal((DGENE, PC["DLAT"]))
    mixing /= np.linalg.norm(mixing, axis=0, keepdims=True)
    signal_total = float((environments["obs"].Z @ mixing.T).var(0).sum()) + 1e-12

    def make_env(Z, snr, seed):
        local_rng = np.random.default_rng(seed)
        if snr > 0:
            signal = (Z @ mixing.T) * np.sqrt(snr * bg_total / signal_total)
        else:
            signal = np.zeros((len(Z), DGENE))
        noise_idx = local_rng.integers(0, len(Xc_centered), len(Z))
        return signal + Xc_centered[noise_idx]

    def obs_latents(n_obs):
        Z = environments["obs"].Z
        if n_obs == len(Z):
            return Z
        extra = SIM.sample_latent(ds.B, ds.noise_var, n_obs - len(Z),
                                  np.random.default_rng(PC_OBS_EXTRA_SEED))
        return np.vstack([Z, extra])

    def run_gate(snr, seeds, gate_seed, Z_obs):
        Xbasis = make_env(environments["basis"].Z, snr, seeds["basis"])
        Xobs = make_env(Z_obs, snr, seeds["obs"])
        Xint = [make_env(environments[f"iv{k}"].Z, snr, seeds["iv"] + k)
                for k in range(PC["R_PLANT"])]
        mu, W = BK.fit_pca(Xbasis, PC["DPROJ"])
        Yobs = BK.project(Xobs, mu, W)
        Yint = [BK.project(values, mu, W) for values in Xint]
        labels = [f"iv{k}" for k in range(PC["R_PLANT"])]
        Y_null, Y_ref, split_info = T2.split_halves(Yobs, gate_seed)
        planted = _both(Yint, Yobs, Y_null, Y_ref, gate_seed, B)

        Yfake = []
        for j in range(PC_NFAKE):
            Xfake = make_env(environments["basis"].Z, snr, 700_000 + j)
            Yfake.append(BK.project(Xfake, mu, W))
        fake_labels = [f"null_basis_draw_{i:02d}" for i in range(PC_NFAKE)]
        fakes = _both(Yfake, Yobs, Y_null, Y_ref, 10_000 + gate_seed, B)
        return dict(
            snr=float(snr),
            split=split_info,
            planted_screen=T2.design_comparison(*planted),
            negative_control_screen=T2.design_comparison(*fakes),
            shared_vs_split=dict(
                planted=T2.shared_vs_split_entry(labels, *planted),
                null_draws=T2.shared_vs_split_entry(fake_labels, *fakes)),
            planted_records=T2.design_records(
                labels, Yint, dict(corrected_disjoint=planted[0],
                                   split_control=planted[1])),
            negative_control_records=T2.design_records(
                fake_labels, Yfake, dict(corrected_disjoint=fakes[0],
                                         split_control=fakes[1])),
        )

    for n_obs in n_obs_list:
        variant_started = time.time()
        Z_obs = obs_latents(n_obs)
        print(f"--- observational draw {n_obs} cells", flush=True)
        smoke = run_gate(PC_SMOKE["snr"], PC_SMOKE["seeds"], PC_SMOKE["gate_seed"],
                         Z_obs)
        smoke_count = smoke["planted_screen"]["corrected_disjoint"]["raw_count"]
        print(f"smoke: shared raw {smoke_count}/{PC['R_PLANT']} at snr=8", flush=True)
        if smoke_count < PC["R_PLANT"] and not selftest:
            raise SystemExit(
                f"[stop] construction broken: strong planted intervention not "
                f"detected by the shared design ({smoke_count}/{PC['R_PLANT']}, "
                f"{n_obs} observational cells)")

        dose = []
        for index, snr in enumerate(PC_DOSE):
            result = run_gate(snr, dict(basis=5000, obs=5001, iv=5100),
                              30_000 + index, Z_obs)
            dose.append(result)
            ps = result["planted_screen"]
            print(f"snr {snr}: shared raw/BH {ps['corrected_disjoint']['raw_count']}/"
                  f"{ps['corrected_disjoint']['bh_count']} | split raw/BH "
                  f"{ps['split_control']['raw_count']}/"
                  f"{ps['split_control']['bh_count']}", flush=True)

        ref = None
        ref_path = REPO / "results" / "e3_poscontrol" / "poscontrol_final.json"
        if n_obs == PC["NPER"] and not selftest and ref_path.exists():
            doc = json.loads(ref_path.read_text())
            committed = {r["snr"]: r for r in [doc["smoke_snr8"]] + doc["dose_response"]}
            checks = {}
            for r in [smoke] + dose:
                c = committed.get(r["snr"])
                if c is None:
                    continue
                checks[str(r["snr"])] = {
                    screen: all(
                        r[screen]["corrected_disjoint"][k]
                        == c[screen]["corrected_disjoint"][k]
                        for k in ("raw_count", "bh_count", "n_environments"))
                    for screen in ("planted_screen", "negative_control_screen")}
            ref = dict(source=str(ref_path.relative_to(REPO)),
                       sha256=T2.sha256_file(ref_path), counts_match=checks)

        report = dict(
            experiment=("E5 split-control positive control: shared-SCM "
                        "reduced-variance interventions mixed into gene space with "
                        "real K562 control noise, screened under the shared and "
                        "split-control designs"),
            n_obs=int(n_obs),
            d_latent=PC["DLAT"], d_gene=DGENE, d_proj=PC["DPROJ"],
            planted_rank=PC["R_PLANT"], cells_per_env=PC["NPER"],
            iv_scale=PC["IV_SCALE"], alpha=PC["ALPHA"], q=PC["Q"], B=B,
            designs=DESIGN_NOTE,
            primary_decision="split_control BH-FDR at q=0.05",
            bh_families=("Within each SNR arm and design, BH applied separately to "
                         "planted interventions and null basis draws"),
            scale=("signal total variance = SNR times real background total "
                   "variance, with the signal scale fixed on the original 200 "
                   "observational cells"),
            null="independent un-intervened basis draws versus observational draw",
            note="snr=0 is a null sanity check, not a real-data E3 verdict",
            split_geometry_note=PC_GEOMETRY_NOTE[n_obs],
            real_bg_var_total=round(bg_total, 6),
            shared_reproduces_e3=(ref if n_obs == PC["NPER"] else
                                  "not applicable: different observational draw"),
            smoke_snr8=smoke,
            dose_response=dose,
            shared_vs_split={str(r["snr"]): r["shared_vs_split"]
                             for r in [smoke] + dose},
            selftest=bool(selftest),
            provenance=T2.provenance(
                __file__, extra_files=["experiments/e3_poscontrol_faithful.py",
                                       "sim/simulator.py"]),
            wall_seconds=round(time.time() - variant_started, 1),
        )
        name = ("e5_poscontrol.json" if n_obs == PC["NPER"]
                else f"e5_poscontrol_obs{n_obs}.json")
        out = Path(out_dir) / name
        T2.write_json(out, report)
        print(f"written {out} ({report['wall_seconds']}s)", flush=True)
    print(f"positive control done ({time.time() - started:.0f}s)", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="part", required=True)
    for part in ("real", "rpe1-confound", "poscontrol"):
        sp = sub.add_parser(part)
        sp.add_argument("--out-dir", default=str(RESULTS))
        sp.add_argument("--selftest", action="store_true",
                        help="synthetic inputs, code path only; needs --out-dir")
        sp.add_argument("--B", type=int, default=T2.B_BOOT)
        if part == "real":
            sp.add_argument("--dataset", choices=sorted(T2.DATASETS), required=True)
            sp.add_argument("--seeds", type=int, nargs="+", default=T2.SEEDS)
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest and Path(args.out_dir).resolve() == RESULTS:
        raise SystemExit("--selftest must not write under results/; pass --out-dir")
    os.chdir(REPO)
    if args.part == "real":
        run_real(args.dataset, args.out_dir, args.seeds, args.B,
                 selftest=args.selftest)
    elif args.part == "rpe1-confound":
        run_rpe1_confound(args.out_dir, args.B, selftest=args.selftest)
    else:
        run_poscontrol(args.out_dir, args.B, selftest=args.selftest)


if __name__ == "__main__":
    main()
