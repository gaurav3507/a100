"""Screen one Perturb-seq dataset at one gate seed with the Tier 2 switches.

    python experiments/tier2_screen.py --dataset rpe1 --seed 0
    python experiments/tier2_screen.py --dataset rpe1 --seed 0 --readout covariance
    python experiments/tier2_screen.py --dataset rpe1 --seed 0 --split-control

Defaults (readout=precision, --split-control off) are the E3 corrected-disjoint
gate. In that configuration this is a reproduction check: the raw and BH sets are
compared with results/e3_stability for the same seed (and results/e3_decisions for
seed 0), and the exit code is 1 on any mismatch. The Tier 2 launcher runs it on
RPE1 seed 0 before anything else. With either switch set it prints the decisions
and, with --out, writes a compact decision export. The paper artifacts come from
e4_second_backbone.py and e5_split_control.py, which call the same functions.
"""

import argparse
import sys
import time
import warnings

import tier2_common as T2

warnings.filterwarnings("ignore", message=r".*encountered in matmul",
                        category=RuntimeWarning)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(T2.DATASETS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--readout", choices=list(T2.PR.READOUTS), default="precision")
    p.add_argument("--split-control", action="store_true",
                   help="split-control design (default off: shared reference)")
    p.add_argument("--B", type=int, default=T2.B_BOOT)
    p.add_argument("--out", default=None, help="optional JSON decision export")
    p.add_argument("--selftest", action="store_true",
                   help="synthetic panel, code path only")
    return p.parse_args()


def main():
    args = parse_args()
    started = time.time()
    panel = (T2.make_selftest_panel(args.dataset) if args.selftest
             else T2.load_panel(args.dataset))
    name, Xc, perts = panel["name"], panel["Xc"], panel["perts"]
    proj = T2.control_projection(Xc)
    Yobs = proj(Xc)
    Yperts = [proj(X) for X in panel["Xperts"]]

    Y_ref, Y_null, split_info = Yobs, None, None
    if args.split_control:
        Y_null, Y_ref, split_info = T2.split_halves(Yobs, args.seed)
    res = T2.screen(Yperts, Y_ref, args.seed, readout=args.readout,
                    Y_null=Y_null, B=args.B)
    raw_ids = T2.selected(perts, res["raw_detect"])
    bh_ids = T2.selected(perts, res["bh_detect"])
    design = "split_control" if args.split_control else "shared_reference"
    print(f"{name} seed {args.seed} readout={args.readout} design={design}: "
          f"raw {res['raw_count']}/{len(perts)}  BH {res['bh_count']}/{len(perts)}  "
          f"({time.time() - started:.0f}s)", flush=True)

    status = 0
    if args.readout == "precision" and not args.split_control and not args.selftest:
        checks = {}
        stab = T2.e3_stability_sets(name)
        if stab and args.seed in stab["per_seed"]:
            ref = stab["per_seed"][args.seed]
            checks["e3_stability"] = dict(
                labels=ref["labels"] == perts, raw=ref["raw"] == raw_ids,
                bh=ref["bh"] == bh_ids)
        dec = T2.e3_decision_set(name)
        if dec and args.seed == 0:
            checks["e3_decisions"] = dict(labels=dec["labels"] == perts,
                                          bh=dec["bh"] == bh_ids)
        ok = bool(checks) and all(all(v.values()) for v in checks.values())
        print(f"REPRODUCTION {'PASS' if ok else 'FAIL'}: {checks or 'no reference'}",
              flush=True)
        status = 0 if ok else 1

    if args.out:
        T2.write_json(args.out, dict(
            dataset=name, seed=args.seed, readout=args.readout, design=design,
            split=split_info, n_environments=len(perts),
            raw_count=int(res["raw_count"]), bh_count=int(res["bh_count"]),
            raw_selected_ids=raw_ids, bh_selected_ids=bh_ids,
            decisions=T2.design_records(perts, Yperts, {design: res}),
            selftest=bool(args.selftest),
            provenance=T2.provenance(__file__)))
    sys.exit(status)


if __name__ == "__main__":
    main()
