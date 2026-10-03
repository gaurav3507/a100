"""One-shot migration of ranktest results into <statistic>/<gate>/<timestamp>.

PROVENANCE IS TAKEN FROM COMMIT ORDER, NOT FILENAMES. Filenames in this lane
are unreliable: gate0.json held four different statistics at four different
times, and Gate 0's per-run keys are identical under every statistic, so
content sniffing cannot tell them apart either. The MIGRATION table below
pins each artefact to the statistic that was in force in the code when that
artefact was written, read off the commit graph:

    6caeacd  core, marginal exceedance count      -> diy_retired
    a71b49c  reject_rank2 = lam[2] > band[2]      -> diy_retired (zeroband)
    9380d3a  Chen-Fang CF-A, kappa-tuned r_hat    -> cfa_kappa
    cc2cab8  Chen-Fang CF-T, tuning-free r_hat    -> cft
    4f818f8  LFC, r_hat === r0, no tuning         -> lfc

NOTHING IS DELETED AND NOTHING IS EDITED IN PLACE. Historical artefacts are
recovered from git and written fresh; on-disk originals are MOVED to
_migrated_originals/ and their old path recorded in meta.migrated_from.
"""
import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
import importlib.util
_spec = importlib.util.spec_from_file_location("_rio", HERE / "84_results_io.py")
RIO = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(RIO)

RESULTS = RIO.RESULTS
ORIG = RESULTS / "_migrated_originals"
REPO = HERE.parents[1]

LFC_SOFT = "lfc/power_soft"          # what the superseded curve points at

# (source, statistic, gate, timestamp, status, suffix, superseded_by, note, config)
# source is ("git", sha, filename) or ("disk", filename) or ("text", filename)
MIGRATION = [
    # ---------------- diy_retired -------------------------------------------
    (("git", "a9ae78d", "gate0.json"), "diy_retired", "gate0",
     "2026-08-07T13:20:56", "HISTORICAL-FOR-COMPARISON-TABLE", "", None,
     "marginal exceedance count; null distribution ~Binomial(d, alpha), "
     "rejected on pure control data at up to 0.09 at d=20",
     dict(alpha=0.05, B=500, n_e=None, d=[5, 10, 20], d_latent=20, D=200,
          n_env=300, seeds=list(range(10)), draws_per_point=200)),
    (("git", "a71b49c", "gate0.json"), "diy_retired", "gate0",
     "2026-08-07T16:37:44", "HISTORICAL-FOR-COMPARISON-TABLE", "", None,
     "zero-signal band, reject_rank2 = lam[2] > band[2]; passed Gate 0 and "
     "then failed Gate 1 by rejecting on intervention strength",
     dict(alpha=0.05, B=500, n_e=None, d=[5, 10, 20], d_latent=20, D=200,
          n_env=300, seeds=list(range(10)), draws_per_point=200)),
    (("git", "bd37b2c", "gate1.json"), "diy_retired", "gate1",
     "2026-08-08T13:59:52", "HISTORICAL-FOR-COMPARISON-TABLE", "", None,
     "zero-signal band; 1a FAIL at 0.438/0.225, 1b FAIL on soft",
     dict(alpha=0.05, B=500, n_e=[500, 2000], d=10, d_latent=[10, 20], D=[200, 1000],
          n_env=None, seeds=list(range(10)), draws_per_point=1)),
    # ---------------- cfa_kappa ---------------------------------------------
    (("git", "4bd658a", "gate0.json"), "cfa_kappa", "gate0",
     "2026-08-09T16:29:04", "HISTORICAL-FOR-COMPARISON-TABLE", "", None,
     "Chen-Fang CF-A with kappa-tuned r_hat", None),
    (("git", "4bd658a", "gate1.json"), "cfa_kappa", "gate1",
     "2026-08-09T16:29:04", "HISTORICAL-FOR-COMPARISON-TABLE", "", None,
     "Chen-Fang CF-A; 1a 0.113/0.013, 1b soft FAIL", None),
    (("git", "4bd658a", "acceptance_and_power.json"), "cfa_kappa", "acceptance",
     "2026-08-09T16:29:04", "HISTORICAL-FOR-COMPARISON-TABLE", "", None,
     "magnitude sweep that accepted the CF swap: FLAT-AT-ALPHA, "
     "0.000/0.000/0.012/0.040/0.032/0.025 vs the retired rule's "
     "0.060/0.058/0.115/0.182/0.295/0.308",
     dict(alpha=0.05, B=500, n_e=2000, d=10, d_latent=10, D=200, n_env=None,
          seeds=list(range(10)), draws_per_point=600)),
    (("git", "4bd658a", "acceptance_and_power.json"), "cfa_kappa", "power_soft",
     "2026-08-09T16:29:04", "SUPERSEDED", "", LFC_SOFT,
     "SUPERSEDED, DO NOT QUOTE. Soft k=3 power curve measured under the "
     "kappa-tuned CF-A statistic, which is less conservative than LFC and "
     "therefore OVERSTATES power: 0.500 vs the LFC 0.300 at n_e=500 and "
     "0.600 vs 0.400 standardised. Superseded by the LFC curve. "
     "CORRECTION to the handoff description: this curve is CF-A, not CF-T, "
     "its values are 0.500/0.700/1.000/1.000 raw (not 0.475/0.438, which are "
     "the Gate 1 k=3 soft point rates under CF-T), and its floor by the "
     "standard rule is 8000, not ~2000.",
     dict(alpha=0.05, B=500, n_e=[500, 2000, 8000, 20000], d=10, d_latent=10,
          D=200, n_env=None, seeds=list(range(10)), draws_per_point=1)),
    (("git", "1215e56", "gate0_CFA_kappa.json"), "cfa_kappa", "gate0",
     "2026-08-10T09:43:05", "HISTORICAL-FOR-COMPARISON-TABLE", "", None,
     "CF-A rerun under the one-sided Gate 0 criterion; PASS, all cells below "
     "the band", None),
    # ---------------- cft ----------------------------------------------------
    (("git", "1215e56", "gate0_CFT.json"), "cft", "gate0",
     "2026-08-10T09:43:05", "HISTORICAL-FOR-COMPARISON-TABLE", "", None,
     "tuning-free sequential r_hat; Gate 0 PASS, cells in band", None),
    (("git", "1215e56", "gate1_CFT.json"), "cft", "gate1",
     "2026-08-10T09:43:05", "HISTORICAL-FOR-COMPARISON-TABLE", "", None,
     "tuning-free sequential r_hat; 1a REGRESSED to 0.225/0.225 because "
     "r_hat under-selected (r_hat=0 in 108 of 160 k=1 soft runs)", None),
    # ---------------- lfc, current ------------------------------------------
    (("disk", "gate0.json"), "lfc", "gate0", None, "CURRENT", "", None,
     "LFC calibration, r_hat === 2; one-sided upper check", None),
    (("disk", "gate1.json"), "lfc", "gate1", None, "CURRENT", "", None,
     "LFC; 1a PASS on the interior, boundary disclosed at 1.94x/1.67x", None),
    (("disk", "gate2.json"), "lfc", "gate2", None, "CURRENT", "", None,
     "LFC; kill criterion NOT triggered", None),
    (("disk", "gate2_s0_reduction.json"), "lfc", "gate2", None, "CURRENT",
     "__s0_reduction", None,
     "simulator self-check: s=0 MLP mixing reproduces linear A2@A1 exactly",
     dict(alpha=0.05, B=500, n_e=2000, d=10, d_latent=10, D=200, n_env=None,
          seeds=list(range(10)), draws_per_point=1)),
    (("disk", "soft_power_curve_lfc.json"), "lfc", "power_soft", None,
     "CURRENT", "", None, "soft k=3 power curve under LFC; floor 8000",
     dict(alpha=0.05, B=500, n_e=[500, 2000, 8000, 20000], d=10, d_latent=10,
          D=200, n_env=None, seeds=list(range(10)), draws_per_point=1)),
    (("disk", "hard_power_curve_lfc.json"), "lfc", "power_hard", None,
     "CURRENT", "", None,
     "hard k=3 power curve under LFC; floor 125, the smallest point tested",
     dict(alpha=0.05, B=500, n_e=[125, 250, 500, 1000, 2000, 8000], d=10,
          d_latent=10, D=200, n_env=None, seeds=list(range(10)),
          draws_per_point=1)),
    (("disk", "taskC_boundary_nscaling.json"), "lfc", "gate1", None, "CURRENT",
     "__boundary_nscaling", None,
     "rank-2 boundary excess vs n_e; flat across 40x, hence structural",
     dict(alpha=0.05, B=500, n_e=[500, 2000, 8000, 20000], d=10, d_latent=10,
          D=200, n_env=None, seeds=list(range(10)), draws_per_point=200)),
    (("disk", "taskD_alpha_sensitivity.json"), "lfc", "alpha_sensitivity", None,
     "CURRENT", "", None, "realised rejection at the boundary vs nominal alpha",
     dict(alpha=[0.01, 0.025, 0.05], B=500, n_e=2000, d=10, d_latent=10, D=200,
          n_env=None, seeds=list(range(10)), draws_per_point=200)),
    (("disk", "taskA_sstar_decay.json"), "lfc", "envelope", None, "CURRENT", "",
     None,
     "s* vs d_latent with the s=0 linear-mixing control; no decay law is "
     "fittable, and d_latent=80 fails under LINEAR mixing",
     dict(alpha=0.05, B=500, n_e=2000, d=10, d_latent=[5, 10, 20, 40, 80],
          D=200, n_env=None, seeds=list(range(10)), draws_per_point=400)),
    (("disk", "taskA_fine_s_envelope.json"), "lfc", "envelope", None,
     "SUPERSEDED", "__fine_s_dl10", "lfc/envelope",
     "single-d_latent fine grid, subsumed by the full s*-vs-d sweep",
     dict(alpha=0.05, B=500, n_e=2000, d=10, d_latent=10, D=200, n_env=None,
          seeds=list(range(10)), draws_per_point=400)),
    (("disk", "taskA_fine_s_envelope_dl20.json"), "lfc", "envelope", None,
     "SUPERSEDED", "__fine_s_dl20", "lfc/envelope",
     "single-d_latent fine grid, subsumed by the full s*-vs-d sweep",
     dict(alpha=0.05, B=500, n_e=2000, d=10, d_latent=20, D=200, n_env=None,
          seeds=list(range(10)), draws_per_point=400)),
    # ---------------- lfc, text logs wrapped as artefacts --------------------
    (("text", "taskA_s0_linear_control.txt"), "lfc", "envelope", None,
     "CURRENT", "__s0_linear_control", None,
     "s=0 control: at d_latent=80 the test rejects at 0.150 raw under exactly "
     "LINEAR mixing, so the envelope collapse there is not a nonlinearity effect",
     dict(alpha=0.05, B=500, n_e=2000, d=10, d_latent=[5, 10, 20, 40, 80],
          D=200, n_env=None, seeds=list(range(10)), draws_per_point=400)),
    (("text", "taskA_decay_raw.txt"), "lfc", "envelope", None, "CURRENT",
     "__decay_raw_log", None, "raw stdout of the s*-vs-d sweep",
     dict(alpha=0.05, B=500, n_e=2000, d=10, d_latent=[5, 10, 20, 40, 80],
          D=200, n_env=None, seeds=list(range(10)), draws_per_point=400)),
    (("text", "gap_ratio_separation.txt"), "lfc", "battery", None, "CURRENT",
     "__gap_ratio_separation", None,
     "measured separation of soft from hard-on-non-source: AUC 0.602 at "
     "n_e=500 rising to 0.834 at 20000",
     dict(alpha=None, B=None, n_e=[500, 2000, 8000, 20000], d=10, d_latent=10,
          D=200, n_env=None, seeds=list(range(30)), draws_per_point=1)),
]


def file_mtime_iso(p):
    return dt.datetime.fromtimestamp(Path(p).stat().st_mtime).strftime(
        "%Y-%m-%dT%H:%M:%S")


def commit_of_path(rel):
    r = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%h", "--", rel],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def cfg_from_payload(payload):
    """Best-effort config lift from a gate JSON's own config block."""
    c = payload.get("config") or {}
    cfgs = c.get("configs") or []
    return dict(
        alpha=c.get("alpha"),
        B=c.get("B_null", c.get("B")),
        n_e=sorted({x.get("n") for x in cfgs if isinstance(x, dict)}) or None,
        d=c.get("d_set", c.get("d")),
        d_latent=(c.get("d_latent")
                  or sorted({x.get("d_latent") for x in cfgs if isinstance(x, dict)})
                  or None),
        D=(c.get("D") or sorted({x.get("D") for x in cfgs if isinstance(x, dict)}) or None),
        n_env=c.get("n_env"),
        seeds=c.get("seeds"),
        draws_per_point=c.get("n_splits"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="without this, prints the plan and writes nothing")
    a = ap.parse_args()

    written, moved, plan = [], [], []
    for src, stat, gate, ts, status, suffix, sup, note, cfg in MIGRATION:
        kind = src[0]
        if kind == "git":
            _, sha, name = src
            rel = f"causalbench/results/ranktest/{name}"
            r = subprocess.run(["git", "-C", str(REPO), "show", f"{sha}:{rel}"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  MISS git {sha}:{name}"); continue
            payload = json.loads(r.stdout)
            old = f"git:{sha}:{rel}"
            commit = sha
        else:
            _, name = src
            p = RESULTS / name
            if not p.exists():
                print(f"  MISS disk {name}"); continue
            if kind == "text":
                payload = {"raw_text": p.read_text()}
            else:
                payload = json.load(open(p))
            ts = ts or file_mtime_iso(p)
            old = f"causalbench/results/ranktest/{name}"
            commit = commit_of_path(old)

        # a split source keeps only its own slice
        if gate == "power_soft" and "soft_power_curve" in payload:
            payload = {"soft_power_curve": payload["soft_power_curve"]}
        elif gate == "acceptance" and "acceptance" in payload:
            payload = {"acceptance": payload["acceptance"]}

        meta = RIO.make_meta(stat, gate, ts, cfg or cfg_from_payload(payload),
                             status=status, superseded_by=sup,
                             migrated_from=old, note=note, commit=commit)
        dest = RIO.results_path(stat, gate, ts, suffix=suffix)
        plan.append((old, str(dest.relative_to(REPO)), status))
        if a.apply:
            RIO.write_results(payload, meta, path=dest)
            written.append(dest)

    # move on-disk originals aside; never delete
    if a.apply:
        ORIG.mkdir(parents=True, exist_ok=True)
        for src, *_ in MIGRATION:
            if src[0] in ("disk", "text"):
                p = RESULTS / src[1]
                if p.exists():
                    shutil.move(str(p), str(ORIG / src[1]))
                    moved.append(src[1])
        man = RESULTS / "manifest.json"
        if man.exists():
            shutil.move(str(man), str(ORIG / "manifest.json"))
            moved.append("manifest.json")

    print(f"{'APPLIED' if a.apply else 'PLAN (dry run)'} -- {len(plan)} artefacts")
    for old, new, status in plan:
        print(f"  {status:<32} {old}\n  {'':<32} -> {new}")
    if a.apply:
        print(f"\nwrote {len(written)} files; moved {len(moved)} originals to "
              f"{ORIG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
