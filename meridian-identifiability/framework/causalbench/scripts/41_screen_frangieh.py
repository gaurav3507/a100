"""Environment-validity screen on Frangieh 2021 (SCP1064, Perturb-CITE-seq melanoma).

Derived from 40_screen_norman.py. The metric helpers (fit_pca, project, coefs,
offdiag) are imported from 03_screen.py via importlib.util exactly as that script
does it, so the arithmetic stays byte-identical to the CausalBench screen. Only
the loader and the outer loop differ.

WHAT IS DIFFERENT FROM 40_screen_norman.py
------------------------------------------
1. PER-ARM. Frangieh has three experimental conditions (Co-culture, Control,
   IFNg). Each arm gets its OWN control-cell PCA basis and its OWN reference
   pool, and is screened separately. The arms are never pooled: a pooled screen
   would put the arm effect in the numerator and inflate the ratio.

2. CHUNKED float32 LOADER. The Norman loader densifies the whole matrix to
   float64 (40_screen_norman.py:67-70). Frangieh is ~218k cells of dense CSV;
   that path would need ~40 GB. Here the expression CSV is read in chunks,
   restricted to the MOI==1 cells before anything is materialised, and held as
   float32. Orientation (genes-as-rows vs cells-as-rows) is detected from the
   header and reported.

3. mean_ratio_pairs IS PRIMARY EVERYWHERE. 40_screen_norman.py printed
   mean_ratio_vs_ctrl in its summary while comparing it against a pairs-based
   reference band -- two different metrics on one line. Here the summary and
   every print carry mean_ratio_pairs only. vs_ctrl is retained inside the
   per-run records as secondary supplementary data and never surfaces in the
   summary.

4. STEP-0 GATE IS BUILT IN AND PAIRS-BASED. For Norman the gate had to be
   repaired afterwards by 40b_fix_step0_gate.py, which patched the output JSON
   rather than the script. Here the gate is computed from mean_ratio_pairs
   directly, per arm, and step0_pass is decided in this script.
   (Why pairs: under step-0 the reference pool is empty by construction --
   03_screen.py:63 -- so the vs_ctrl draw falls back to the pseudo-environment's
   own rows and compares an environment against itself. Only pairs is
   calibrated to ~1.0.)

FRANGIEH-SPECIFIC DATA HANDLING
-------------------------------
* RNA_metadata.csv row 2 is SCP's TYPE convention row, not a cell. Read with
  skiprows=[1] or it becomes a phantom cell.
* Keep MOI==1 only. MOI==0 has an empty sgRNA field; MOI>=2 is multi-guide and
  has no single well-defined target.
* Target is parsed off the sgRNA by stripping the TRAILING guide index with
  ^(.*)_\\d+$ . Splitting on the first underscore would mangle the control
  names (NO_SITE_1 -> "NO", ONE_NON-GENE_SITE_4 -> "ONE").
* Controls are the pooled NO_SITE_* and ONE_NON-GENE_SITE_* guides.
* Whether the target genes appear as expression columns is recorded in the
  output. Norman had 0/105, which made the target-column drop in project() a
  no-op; for Frangieh it is expected to be live, so the drop actually fires.

Output:
    causalbench/results/screen/frangieh.json

Usage (A100):
    python causalbench/scripts/41_screen_frangieh.py
    python causalbench/scripts/41_screen_frangieh.py --hvg 5000   # cap genes
"""
import argparse
import json
import os
import re
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

FRANGIEH = Path("/workspace/external/frangieh")
META_CSV = FRANGIEH / "RNA_metadata.csv"
EXPR_CSV = FRANGIEH / "RNA_expression.csv.gz"

CAUSALBENCH = Path("/workspace/meridian-identifiability/causalbench")
SCREEN_DIR = CAUSALBENCH / "results/screen"

SEED = 0
CTRL_LABEL = "non-targeting"

NMIN = 100
D_SET = (10,)

ARMS = ("Co-culture", "Control", "IFNγ")   # U+03B3 (Greek small letter gamma), verified from RNA_metadata.csv

CONTROL_PREFIXES = ("NO_SITE", "ONE_NON-GENE_SITE")
GUIDE_INDEX_RE = re.compile(r"^(.*)_\d+$")

CHUNK_ROWS = 2000


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCREEN_MOD = _load_module(CAUSALBENCH / "scripts/03_screen.py", "_screen03")
fit_pca = SCREEN_MOD.fit_pca
project = SCREEN_MOD.project
coefs = SCREEN_MOD.coefs
offdiag = SCREEN_MOD.offdiag
N_PAIRS = SCREEN_MOD.N_PAIRS


# --------------------------------------------------------------------- metadata
def sgrna_to_target(s):
    """Strip the TRAILING guide index only. NO_SITE_1 -> NO_SITE."""
    if not isinstance(s, str) or s == "":
        return None
    m = GUIDE_INDEX_RE.match(s)
    return m.group(1) if m else s


def is_control_target(t):
    return t is not None and any(t.startswith(p) for p in CONTROL_PREFIXES)


def load_metadata():
    print(f"[meta] {META_CSV}", flush=True)
    # skiprows=[1] drops SCP's TYPE convention row.
    md = pd.read_csv(META_CSV, skiprows=[1])
    print(f"[meta] {len(md)} rows, columns: {list(md.columns)}", flush=True)

    for col in ("NAME", "condition", "MOI", "sgRNA"):
        if col not in md.columns:
            sys.exit(f"[meta] FATAL: expected column {col!r} not found")

    moi = pd.to_numeric(md["MOI"], errors="coerce")
    n_moi0 = int((moi == 0).sum())
    n_moi1 = int((moi == 1).sum())
    n_moi2p = int((moi >= 2).sum())
    print(f"[meta] MOI==0 {n_moi0}   MOI==1 {n_moi1}   MOI>=2 {n_moi2p}",
          flush=True)

    keep = (moi == 1).to_numpy()
    md = md.loc[keep].copy()

    md["target"] = md["sgRNA"].map(sgrna_to_target)
    n_unparsed = int(md["target"].isna().sum())
    if n_unparsed:
        print(f"[meta] WARNING: {n_unparsed} MOI==1 rows have unparseable sgRNA",
              flush=True)
        md = md.loc[md["target"].notna()].copy()

    ctrl_mask = md["target"].map(is_control_target).to_numpy()
    md["iv"] = np.where(ctrl_mask, CTRL_LABEL, md["target"].to_numpy())

    n_ctrl = int(ctrl_mask.sum())
    ctrl_names = sorted(set(md.loc[ctrl_mask, "target"]))
    print(f"[meta] kept {len(md)} MOI==1 cells; {n_ctrl} pooled control cells "
          f"from {ctrl_names}", flush=True)
    print(f"[meta] arm sizes (MOI==1): "
          f"{md['condition'].value_counts().to_dict()}", flush=True)

    meta = dict(
        n_rows_metadata_after_skiprows=int(len(pd.read_csv(META_CSV,
                                                            skiprows=[1],
                                                            usecols=["NAME"]))),
        n_moi0=n_moi0, n_moi1=n_moi1, n_moi2plus=n_moi2p,
        n_cells_kept=int(len(md)),
        n_control_cells=n_ctrl,
        control_guide_families=ctrl_names,
        n_unparseable_sgrna=n_unparsed,
        arm_sizes_moi1={str(k): int(v)
                        for k, v in md["condition"].value_counts().items()},
    )
    return md, meta


# ------------------------------------------------------------------- expression
def detect_orientation(path, n_meta_cells):
    hdr = pd.read_csv(path, nrows=0)
    cols = list(hdr.columns)
    ncols = len(cols)
    # SCP convention is genes-as-rows: header is [<gene col>, cell1, cell2, ...]
    genes_as_rows = abs((ncols - 1) - n_meta_cells) < abs(ncols - n_meta_cells)
    print(f"[expr] header has {ncols} columns; metadata lists {n_meta_cells} "
          f"cells -> orientation = "
          f"{'genes-as-rows' if genes_as_rows else 'cells-as-rows'}", flush=True)
    return genes_as_rows, cols


def load_expression(path, wanted_cells, hvg=None):
    """Return (X float32 [n_wanted x n_genes], gene_names, cell_order)."""
    n_meta_cells = len(wanted_cells)
    genes_as_rows, cols = detect_orientation(path, n_meta_cells)
    wanted_set = set(wanted_cells)

    if genes_as_rows:
        # columns are cells; rows are genes. Take the wanted columns per chunk.
        cell_cols = cols[1:]
        keep_cols = [c for c in cell_cols if c in wanted_set]
        missing = len(wanted_set) - len(keep_cols)
        if missing:
            print(f"[expr] WARNING: {missing} metadata cells absent from the "
                  f"expression header", flush=True)
        print(f"[expr] reading {len(keep_cols)} cell columns in chunks of "
              f"{CHUNK_ROWS} genes", flush=True)
        blocks, gene_names = [], []
        for i, chunk in enumerate(pd.read_csv(path, index_col=0,
                                               usecols=[cols[0]] + keep_cols,
                                               chunksize=CHUNK_ROWS)):
            gene_names.extend(map(str, chunk.index))
            blocks.append(chunk.to_numpy(dtype=np.float32))
            if i % 5 == 0:
                print(f"[expr]   chunk {i}: {len(gene_names)} genes so far",
                      flush=True)
        G = np.vstack(blocks)          # genes x cells
        del blocks
        X = np.ascontiguousarray(G.T)  # cells x genes
        del G
        cell_order = keep_cols
    else:
        # rows are cells; columns are genes. Filter rows per chunk.
        gene_names = [str(c) for c in cols[1:]]
        print(f"[expr] reading {len(gene_names)} gene columns, filtering rows "
              f"in chunks of {CHUNK_ROWS} cells", flush=True)
        blocks, cell_order = [], []
        for i, chunk in enumerate(pd.read_csv(path, index_col=0,
                                               chunksize=CHUNK_ROWS)):
            sel = chunk.index.astype(str).isin(wanted_set)
            if sel.any():
                sub = chunk.loc[sel]
                cell_order.extend(map(str, sub.index))
                blocks.append(sub.to_numpy(dtype=np.float32))
            if i % 20 == 0:
                print(f"[expr]   chunk {i}: {len(cell_order)} cells kept so far",
                      flush=True)
        X = np.vstack(blocks)
        del blocks

    print(f"[expr] matrix {X.shape} float32 "
          f"({X.nbytes / 1e9:.2f} GB), {len(gene_names)} genes", flush=True)

    if hvg is not None and hvg < X.shape[1]:
        v = X.var(axis=0)
        idx = np.argsort(v)[::-1][:hvg]
        idx.sort()
        X = np.ascontiguousarray(X[:, idx])
        gene_names = [gene_names[i] for i in idx]
        print(f"[expr] HVG subset to {hvg} genes -> {X.shape} "
              f"({X.nbytes / 1e9:.2f} GB)", flush=True)

    return X, gene_names, cell_order


# ------------------------------------------------------------------ screen core
def screen_run(X, iv, vn, nmin, d, step0, seed=SEED, no_drop=False):
    """Mirror of 03_screen.py run(): identical arithmetic, Frangieh as input.

    seed: RNG seed for subsampling and pair selection. Default SEED=0
        preserves the pre-seed-sweep numbers byte-for-byte.
    no_drop: when True, the target-column drop in project() is disabled
        (drops=set() everywhere). Used by --no-drop in the drop-sensitivity
        rerun.

    Returns a dict. On too-few-environments it returns the dict with an
    'aborted' key rather than raising, matching 40_screen_norman.py:120-122.
    """
    rng = np.random.default_rng(seed)
    half = nmin // 2
    gidx = {g: i for i, g in enumerate(vn)}
    ctrl_rows = np.where(iv == CTRL_LABEL)[0]

    if len(ctrl_rows) < nmin:
        return dict(step0=step0, nmin=nmin, d=d, n_envs=0,
                    n_control_cells=int(len(ctrl_rows)),
                    seed=int(seed), no_drop=bool(no_drop),
                    aborted="fewer_control_cells_than_nmin")

    mu, W = fit_pca(X[ctrl_rows], d)

    if step0:
        rng.shuffle(ctrl_rows)
        k = len(ctrl_rows) // nmin
        envs = [(f"ctrl{i}", ctrl_rows[i * nmin:(i + 1) * nmin], None)
                for i in range(k)]
        ref_pool = ctrl_rows[:0]
    else:
        us, cs = np.unique(iv, return_counts=True)
        keep = [(g, n) for g, n in zip(us, cs)
                if g not in (CTRL_LABEL, "excluded") and n >= nmin]
        envs = [(g, np.where(iv == g)[0], gidx.get(g)) for g, _ in keep]
        ref_pool = ctrl_rows

    if len(envs) < 4:
        return dict(step0=step0, nmin=nmin, d=d, n_envs=len(envs),
                    seed=int(seed), no_drop=bool(no_drop),
                    aborted="fewer_than_4_envs")

    within_c, within_m, between_c, between_m = [], [], [], []
    for g, rows, drop in envs:
        rows = rng.choice(rows, nmin, replace=False)
        drops = (set() if no_drop
                 else ({drop} if drop is not None else set()))
        Z = project(X[rows], mu, W, drops)
        Za, Zb = Z[:half], Z[half:nmin]
        within_c.append(np.abs(offdiag(coefs(Za)) - offdiag(coefs(Zb))))
        within_m.append(np.linalg.norm(Za.mean(0) - Zb.mean(0)))

        ref = rng.choice(ref_pool if len(ref_pool) else rows, half,
                          replace=False)
        Zr = project(X[ref], mu, W, drops)
        between_c.append(np.abs(offdiag(coefs(Za)) - offdiag(coefs(Zr))))
        between_m.append(np.linalg.norm(Za.mean(0) - Zr.mean(0)))

    pair_c, pair_m = [], []
    for _ in range(min(N_PAIRS, len(envs) * (len(envs) - 1) // 2)):
        i, j = rng.choice(len(envs), 2, replace=False)
        (ga, ra, da), (gb, rb, db) = envs[i], envs[j]
        drops = (set() if no_drop
                 else {x for x in (da, db) if x is not None})
        Za = project(X[rng.choice(ra, half, replace=False)], mu, W, drops)
        Zb = project(X[rng.choice(rb, half, replace=False)], mu, W, drops)
        pair_c.append(np.abs(offdiag(coefs(Za)) - offdiag(coefs(Zb))))
        pair_m.append(np.linalg.norm(Za.mean(0) - Zb.mean(0)))

    wc = np.concatenate(within_c)
    bc = np.concatenate(between_c)
    pc = np.concatenate(pair_c) if pair_c else np.array([np.nan])
    thr = np.percentile(wc, 95)

    return dict(
        step0=step0, nmin=nmin, d=d, n_envs=len(envs), n_fit=half,
        n_control_cells=int(len(ctrl_rows)),
        seed=int(seed), no_drop=bool(no_drop),
        # PRIMARY
        mean_ratio_pairs=float(np.median(pair_m) / np.median(within_m)),
        coef_ratio_pairs=float(np.median(pc) / np.median(wc)),
        shift_frac_pairs=float((pc > thr).mean()),
        # secondary, supplement only -- never surfaced in the summary
        mean_ratio_vs_ctrl=float(np.median(between_m) / np.median(within_m)),
        coef_ratio_vs_ctrl=float(np.median(bc) / np.median(wc)),
        shift_frac_vs_ctrl=float((bc > thr).mean()),
        shift_frac_within=float((wc > thr).mean()),
    )


def atomic_write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.rename(tmp, path)


# ------------------------------------------------------------------------- main
def _run_one_seed(X, iv_all, cond_all, vn, meta, nmin, seed, no_drop):
    """Run the per-arm screen at a single seed. Returns arms_out dict.
    Loading is done ONCE in main; this reuses (X, iv_all, cond_all, vn).
    """
    arms_out = {}
    for arm in ARMS:
        sel = cond_all == arm
        n_arm = int(sel.sum())
        print(f"\n{'=' * 60}\n  seed={seed} no_drop={no_drop} ARM {arm}  "
              f"({n_arm} MOI==1 cells)\n{'=' * 60}", flush=True)
        if n_arm == 0:
            arms_out[arm] = dict(n_cells=0, aborted="no_cells_for_arm")
            continue

        X_arm = np.ascontiguousarray(X[sel])
        iv_arm = iv_all[sel]

        us, cs = np.unique(iv_arm, return_counts=True)
        n_ctrl_arm = int(cs[us == CTRL_LABEL].sum()) if CTRL_LABEL in us else 0
        surviving = [(g, int(n)) for g, n in zip(us, cs)
                     if g != CTRL_LABEL and n >= nmin]
        print(f"    control cells in arm : {n_ctrl_arm}", flush=True)
        print(f"    targets total        : {int((us != CTRL_LABEL).sum())}",
              flush=True)
        print(f"    targets >= NMIN={nmin}   : {len(surviving)}", flush=True)

        step0_runs, screen_runs = [], []
        for d in D_SET:
            r0 = screen_run(X_arm, iv_arm, vn, nmin, d, step0=True,
                             seed=seed, no_drop=no_drop)
            step0_runs.append(r0)
            if "aborted" in r0:
                print(f"    [step0 d={d}] ABORTED: {r0['aborted']}",
                      flush=True)
            else:
                print(f"    [step0 d={d}] mean_ratio_pairs="
                      f"{r0['mean_ratio_pairs']:.3f}  n_envs={r0['n_envs']}",
                      flush=True)

            r1 = screen_run(X_arm, iv_arm, vn, nmin, d, step0=False,
                             seed=seed, no_drop=no_drop)
            screen_runs.append(r1)
            if "aborted" in r1:
                print(f"    [screen d={d}] ABORTED: {r1['aborted']}",
                      flush=True)
            else:
                print(f"    [screen d={d}] mean_ratio_pairs="
                      f"{r1['mean_ratio_pairs']:.3f}  n_envs={r1['n_envs']}",
                      flush=True)

        gate_vals = [r["mean_ratio_pairs"] for r in step0_runs
                     if "mean_ratio_pairs" in r]
        gate_pass = (all(0.9 <= x <= 1.1 for x in gate_vals)
                     if gate_vals else False)
        prim = next((r for r in screen_runs
                     if r.get("d") == D_SET[0] and "aborted" not in r), None)

        arms_out[arm] = dict(
            n_cells=n_arm,
            n_control_cells=n_ctrl_arm,
            n_targets_total=int((us != CTRL_LABEL).sum()),
            n_targets_surviving_nmin=len(surviving),
            surviving_targets=[g for g, _ in surviving],
            step0_gate=step0_runs,
            screen=screen_runs,
            summary=dict(
                nmin=nmin, d=D_SET[0], seed=int(seed), no_drop=bool(no_drop),
                step0_gate_metric="mean_ratio_pairs",
                step0_pairs=gate_vals,
                step0_pass=gate_pass,
                n_envs=prim.get("n_envs") if prim else None,
                mean_ratio_pairs=prim.get("mean_ratio_pairs") if prim else None,
                coef_ratio_pairs=prim.get("coef_ratio_pairs") if prim else None,
                aborted=None if prim else "no_valid_screen_run",
            ),
        )
    return arms_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hvg", type=int, default=None,
                    help="cap to the N most variable genes (default: all)")
    ap.add_argument("--nmin", type=int, default=NMIN)
    ap.add_argument("--seeds", default="0",
                    help="comma-separated list of RNG seeds (default: 0)")
    ap.add_argument("--no-drop", action="store_true",
                    help="disable the target-column drop in project()")
    ap.add_argument("--tag", default="",
                    help="suffix appended to output filename")
    ap.add_argument("--out", default=None,
                    help="explicit output path (overrides --tag)")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    tag = a.tag or ("_seeds" + "".join(str(s) for s in seeds)
                    + ("_nodrop" if a.no_drop else ""))
    out_path = Path(a.out) if a.out else (
        SCREEN_DIR / f"frangieh{tag}.json")
    if out_path.exists():
        print(f"[skip] {out_path} already exists", flush=True)
        return

    md, meta = load_metadata()
    X, vn, cell_order = load_expression(EXPR_CSV, list(md["NAME"].astype(str)),
                                         hvg=a.hvg)

    # Align metadata rows to the expression cell order.
    md = md.set_index(md["NAME"].astype(str)).reindex(cell_order)
    n_dropped = int(md["iv"].isna().sum())
    if n_dropped:
        print(f"[align] {n_dropped} expression cells had no metadata row; "
              f"dropping", flush=True)
        ok = md["iv"].notna().to_numpy()
        X, md = X[ok], md.loc[ok]
    iv_all = md["iv"].to_numpy().astype(object)
    cond_all = md["condition"].to_numpy().astype(str)
    print(f"[align] aligned {X.shape[0]} cells", flush=True)

    gene_set = set(vn)
    all_targets = sorted({t for t in iv_all if t != CTRL_LABEL})
    in_cols = sum(1 for t in all_targets if t in gene_set)
    meta.update(
        n_genes=int(X.shape[1]),
        hvg_cap=a.hvg,
        n_unique_targets=len(all_targets),
        n_targets_in_expression_columns=in_cols,
        target_column_drop_is_noop=(in_cols == 0),
    )
    print(f"[align] {len(all_targets)} unique targets, {in_cols} present as "
          f"expression columns "
          f"(target-column drop is "
          f"{'a NO-OP' if in_cols == 0 else 'LIVE'})", flush=True)

    per_seed = {}
    for seed in seeds:
        arms_out = _run_one_seed(X, iv_all, cond_all, vn, meta, a.nmin,
                                  seed, a.no_drop)
        per_seed[str(seed)] = arms_out

    # Preserve prior single-seed shape when only seed 0 is run without
    # --no-drop, so old consumers keep working.
    single_shape = (len(seeds) == 1 and seeds[0] == 0 and not a.no_drop)

    out = dict(
        dataset="frangieh",
        source_metadata=str(META_CSV),
        source_expression=str(EXPR_CSV),
        seeds=seeds,
        no_drop=bool(a.no_drop),
        nmin=a.nmin,
        d_set=list(D_SET),
        primary_metric="mean_ratio_pairs",
        per_arm=True,
        **meta,
        per_seed=per_seed,
    )
    if single_shape:
        out["seed"] = seeds[0]
        out["arms"] = per_seed[str(seeds[0])]
    atomic_write_json(out_path, out)

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 78, flush=True)
    print(f"FRANGIEH SEED SWEEP -- primary metric: mean_ratio_pairs   "
          f"(no_drop={a.no_drop})", flush=True)
    print("=" * 78, flush=True)
    header = (f"  {'arm':<12}"
              + "".join(f"{'seed ' + str(s):>12}" for s in seeds)
              + f"{'mean':>10}{'sd':>8}")
    print(header)
    for arm in ARMS:
        vals_pair = []
        vals_gate = []
        for s in seeds:
            A = per_seed[str(s)].get(arm, {})
            sm = A.get("summary", {})
            v = sm.get("mean_ratio_pairs")
            g = sm.get("step0_pairs")
            vals_pair.append(v)
            vals_gate.append(g)
        mean_val = float(np.mean([v for v in vals_pair if v is not None]))
        sd_val = float(np.std([v for v in vals_pair if v is not None]))
        row = f"  {arm:<12}"
        for v in vals_pair:
            row += f"{(f'{v:.3f}' if v is not None else 'ABORT'):>12}"
        row += f"{mean_val:>10.3f}{sd_val:>8.3f}"
        print(row)
    print()
    print(f"  step-0 gate (per seed, per arm):")
    for arm in ARMS:
        for s in seeds:
            A = per_seed[str(s)].get(arm, {})
            gate = A.get("summary", {}).get("step0_pairs", [])
            print(f"    seed {s}  {arm:<12}  step0_pairs={gate}")
    print()
    print(f"  NMIN={a.nmin}, d={D_SET[0]}, seeds={seeds}, "
          f"no_drop={a.no_drop}, per-arm bases")
    print(f"  targets present as expression columns: "
          f"{meta['n_targets_in_expression_columns']}/"
          f"{meta['n_unique_targets']}  "
          f"(drop is {'NO-OP' if meta['target_column_drop_is_noop'] else 'LIVE'})")
    print(f"\n[write] {out_path}", flush=True)


if __name__ == "__main__":
    main()
