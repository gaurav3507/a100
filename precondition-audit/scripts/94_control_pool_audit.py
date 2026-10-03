"""Is the n = 8000 panel of Fig. 2(b) honestly labelled, and where did each
number come from?

TWO QUESTIONS, ONE PASS.

(1) MATCHING. The panel is captioned as a matched sample at n = 8000 control
cells. That is only true for a condition that HAS 8000 controls.
85_dataset_descriptives.py subsamples with `n_use = int(min(cap, n_all))`, so a
condition with a smaller pool silently contributes all of it at a smaller n and
the panel is then a mix of matched and unmatched points. This reports
available_controls, n_effective and a matched boolean per condition.

(2) PROVENANCE. This is now the more valuable half. One published row currently
has no traceable source: three Norman files sit on the A100, all shaped
[108497, 5000], and which one produced the published participation ratio is not
recorded anywhere. Every record here therefore carries the resolved ABSOLUTE
input path and a content fingerprint of the value array, so no future row can
be un-sourceable in the same way.

NO PR LITERAL LIVES IN THIS FILE. The reproduction gate reads its expected
value out of the committed descriptives artefact for that dataset and cap. A
literal would be unsourceable in exactly the way this script exists to fix, and
would fail a correct computation the moment the artefact moved.

READ AND RECOMPUTE, NOT A NEW MEASUREMENT. The control matrix comes from 85's
own loader, captured rather than re-derived, and the estimators are 85's own
`spectrum_estimators`.

NO VERDICT. This reports availability, matching and provenance. It expresses no
assumption verdict and Phase B remains not authorised.

Usage (A100). Long-running batch job; launch detached.

    nohup python -u scripts/94_control_pool_audit.py \
        > logs/control_pool_audit.log 2>&1 &

    python scripts/94_control_pool_audit.py --dataset k562
    python scripts/94_control_pool_audit.py --norman-candidates

Env:
    PRECOND_DATA   raw data root; required. Every input is existence-checked
                   against a candidate list and a miss names every path tried.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# scripts/ sits at the repo root in this repository, so the root is one level
# up, not two. (It was two when these lived in causalbench/scripts/.)
REPO = HERE.parent
DESCRIPTIVES_DIR = REPO / "results" / "descriptives"
OUT_DIR = REPO / "results" / "regen" / "control_pool"
LOCKFILE = REPO / "logs" / "control_pool_audit.lock"

CAP = 8000                      # the cap the Fig. 2(b) panel claims
FALLBACK_CAP = 2000             # every condition has an artefact at this cap
DATASETS = ("k562", "rpe1", "norman",
            "frangieh_coculture", "frangieh_control", "frangieh_ifng")

# The reproduction gate compares against the COMMITTED artefact, never a
# literal. Tolerance only; no participation-ratio value appears in this file.
GATE_DATASET = "k562"
GATE_TOL = 1e-10

# Norman: three files on the A100, all shaped [108497, 5000], and it is not
# recorded which produced the published number. All three are tried and all
# three are reported; this script does not pick one.
#
# Established before writing this list, and not re-derived here:
#   Norman2019_raw.h5ad and datasets/Norman2019.h5ad have byte-identical value
#   arrays (sha prefix 55ef3d07db20, max 8.1284); Norman2019_prep_new.h5ad
#   differs (5f33ad755e66, max 7.7348). None holds integral values, so the file
#   named "raw" is already log-transformed.
# The doubled datasets/datasets/ on the second and third is the real tree on
# the A100, not a typo. These are RELATIVE to $PRECOND_EXTERNAL so no absolute
# path is hardcoded; the only literal is that variable's default, below.
NORMAN_RELATIVE = (
    "discrepancy_vae/datasets/Norman2019_raw.h5ad",
    "discrepancy_vae/datasets/datasets/Norman2019.h5ad",
    "discrepancy_vae/datasets/datasets/Norman2019_prep_new.h5ad",
)
# Norman also stays resolvable as a bare filename against the $PRECOND_DATA
# roots, so a machine that keeps it alongside the other inputs still works.
NORMAN_FILENAME = "Norman2019_raw.h5ad"
NORMAN_CANDIDATE_CAP = 2000     # the cap the published Norman number is quoted at
DEFAULT_EXTERNAL = "/workspace/external"   # $PRECOND_EXTERNAL default

# Third-party trees whose layout does not match the $PRECOND_DATA convention,
# relative to $PRECOND_EXTERNAL. Norman is irregular (the doubled
# datasets/datasets/); Frangieh is regular but simply lives outside the data
# root. Both are handled by the SAME resolver: an absolute candidate is used
# directly, and $PRECOND_EXTERNAL is joined ahead of the $PRECOND_DATA roots.
# There is deliberately no second resolver. Two that can drift is how Frangieh
# came to fail here after Norman had already been fixed.
EXTERNAL_RELATIVE = {
    "norman": NORMAN_RELATIVE,
    "frangieh": ("frangieh/RNA_metadata.csv",
                 "frangieh/RNA_expression.csv.gz"),
}

# Files whose absence makes this script meaningless. The simulator is asserted
# even though this script does not import it: scripts 93 and 94 are a pair, and
# a repo missing the simulator cannot produce the paper's numbers at all. If it
# is absent this stops. It is never vendored from elsewhere, never reimplemented.
SIMULATOR = HERE / "81_ranktest_oracle.py"
REQUIRED_SCRIPTS = (SIMULATOR,
                    HERE / "85_dataset_descriptives.py",
                    HERE / "80_ranktest_core.py",
                    HERE / "84_results_io.py")

# Bare filenames, joined against the $PRECOND_DATA root list. Norman is NOT
# here: its files live outside those roots and are handled by
# norman_candidate_paths(), which searches absolute candidates first.
INPUTS = {
    "k562": ("dataset_k562.npz",),
    "rpe1": ("dataset_rpe1.npz",),
    "norman": (),
    "frangieh_coculture": (), "frangieh_control": (), "frangieh_ifng": (),
}

FP_NONZERO_N = 50_000           # entries hashed for the content fingerprint


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    real = os.makedirs
    os.makedirs = lambda *a, **k: None       # loaders mkdir at import time
    try:
        spec.loader.exec_module(mod)
    finally:
        os.makedirs = real
    return mod


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def value_fingerprint(X, label):
    """Content fingerprint of a value array. Discriminates near-identical files.

    shape / dtype / max / mean, plus a sha256 over the first FP_NONZERO_N
    non-zero entries in row-major order. Non-zero because these matrices are
    mostly exact zeros: a hash over the leading raw entries would agree across
    files that differ everywhere it matters.
    """
    if X is None:
        return None
    A = np.asarray(X)
    flat = A.reshape(-1)
    nz = flat[flat != 0]
    take = nz[:FP_NONZERO_N]
    h = hashlib.sha256(np.ascontiguousarray(take, dtype=np.float64).tobytes())
    return dict(
        label=label,
        shape=[int(x) for x in A.shape],
        dtype=str(A.dtype),
        max=float(A.max()) if A.size else None,
        mean=float(A.mean()) if A.size else None,
        n_nonzero_hashed=int(take.size),
        nonzero_available=int(nz.size),
        sha256_first_nonzero=h.hexdigest(),
        hash_spec=(f"sha256 of the first {FP_NONZERO_N} non-zero entries, "
                   f"row-major, cast to float64"),
    )


def data_roots():
    """Ordered candidate roots. $PRECOND_DATA first, then the usual places."""
    roots = []
    if os.environ.get("PRECOND_DATA"):
        roots.append(("$PRECOND_DATA", Path(os.environ["PRECOND_DATA"])))
    roots += [("repo data/", REPO / "data"),
              ("A100 project", Path("/workspace/precondition-audit/data")),
              ("A100 legacy", Path("/workspace/ranktest-diagnostics/data"))]
    return roots


def external_root():
    """Where third-party checkouts live. $PRECOND_EXTERNAL, else the default."""
    return Path(os.environ.get("PRECOND_EXTERNAL") or DEFAULT_EXTERNAL)


def resolve_candidate(cand, roots=None):
    """One candidate -> (hit or None, [every path tried, in order]).

    THE FIX. An ABSOLUTE candidate is used directly; it names a file, not a
    filename to look for. Joining it against a search-directory list is what
    broke Norman: its files sit under $PRECOND_EXTERNAL, that directory is not
    on the $PRECOND_DATA root list, and so the only directory that could ever
    have held them was never searched.

    A RELATIVE candidate keeps the original behaviour and is joined against
    each root in order. The other five conditions rely on that and are
    unchanged.
    """
    roots = data_roots() if roots is None else roots
    c = Path(cand)
    if c.is_absolute():
        return (c if c.exists() else None), [("absolute", c)]
    tried = [(lbl, r / c) for lbl, r in roots]
    hit = next((q for _, q in tried if q.exists()), None)
    return hit, tried


def external_candidates(*names):
    """$PRECOND_EXTERNAL joined with the name parts a loader asked for.

    Loaders call resolve_data("frangieh", "RNA_metadata.csv"); this turns that
    into $PRECOND_EXTERNAL/frangieh/RNA_metadata.csv without the loader knowing
    the variable exists. The explicit EXTERNAL_RELATIVE lists cover trees whose
    layout the name parts do not describe.
    """
    ext = external_root()
    out = [("$PRECOND_EXTERNAL", ext.joinpath(*[str(n) for n in names]))]
    tail = str(names[-1]) if names else ""
    for rels in EXTERNAL_RELATIVE.values():
        for rel in rels:
            q = ext / rel
            if Path(rel).name == tail and q not in [x for _, x in out]:
                out.append(("$PRECOND_EXTERNAL", q))
    return out


def norman_candidate_paths():
    """The three A100 files, then the bare-filename fallback. Order is search
    order, and every entry is reported on failure whether absolute or not."""
    ext = external_root()
    out = [("$PRECOND_EXTERNAL", ext / rel) for rel in NORMAN_RELATIVE]
    out += [(lbl, r / NORMAN_FILENAME) for lbl, r in data_roots()]
    return out


def first_existing_norman():
    """The path a default Norman run should use, or None. Never guesses."""
    for _, p in norman_candidate_paths():
        if p.exists():
            return p
    return None


# ===================================================================== step 0
def step0_gate(datasets):
    """Everything that must be true before any measurement. Fails loudly."""
    print("=" * 70, flush=True)
    print(" STEP-0 GATE", flush=True)
    print("=" * 70, flush=True)
    fatal = []

    root = os.environ.get("PRECOND_DATA")
    if not root:
        fatal.append("$PRECOND_DATA is not set.")
    else:
        print(f"  [env]  PRECOND_DATA = {root}", flush=True)
        if not Path(root).is_dir():
            fatal.append(f"$PRECOND_DATA is set but is not a directory: {root}")

    roots = data_roots()
    print(f"  [env]  PRECOND_EXTERNAL = {external_root()}"
          f"{'' if os.environ.get('PRECOND_EXTERNAL') else '   (default)'}",
          flush=True)
    for ds in datasets:
        for rel in INPUTS.get(ds, ()):
            hit, tried = resolve_candidate(rel, roots)
            if hit is None:
                fatal.append(
                    f"input for {ds} not found: {rel}\n"
                    + "".join(f"        tried {lbl:<18} {q}\n"
                              for lbl, q in tried).rstrip())
            else:
                print(f"  [data] {ds:<20} {rel:<24} -> {hit.resolve()}",
                      flush=True)

    # Norman searches absolute candidates first, then the bare filename. At
    # least one must exist; --norman-candidates needs all three and asserts
    # that separately.
    if "norman" in datasets:
        cands = norman_candidate_paths()
        found = [(lbl, q) for lbl, q in cands if q.exists()]
        if not found:
            fatal.append(
                "no Norman input found. Tried, in order:\n"
                + "".join(f"        tried {lbl:<18} {q}\n"
                          for lbl, q in cands).rstrip()
                + "\n        Set $PRECOND_EXTERNAL (currently "
                  f"{external_root()}) or $PRECOND_DATA.")
        else:
            for lbl, q in found:
                print(f"  [data] {'norman':<20} {lbl:<24} -> {q.resolve()}",
                      flush=True)
            print(f"  [data] {'norman':<20} {'default run uses':<24} "
                  f"-> {found[0][1].resolve()}", flush=True)
    # Frangieh: check the ACTUAL CSVs, not merely that some root is a directory.
    # The old directory check passed on any machine with a data root and then
    # let 41_screen_frangieh die inside the loader, which is where three
    # completed conditions were lost.
    if any(d.startswith("frangieh") for d in datasets):
        for rel in EXTERNAL_RELATIVE["frangieh"]:
            parts = Path(rel).parts
            cands = external_candidates(*parts) + [
                (lbl, r.joinpath(*parts)) for lbl, r in roots]
            hit = next((q for _, q in cands if q.exists()), None)
            if hit is None:
                fatal.append(
                    f"Frangieh input not found: {rel}\n"
                    + "".join(f"        tried {lbl:<18} {q}\n"
                              for lbl, q in cands).rstrip()
                    + "\n        Set $PRECOND_EXTERNAL (currently "
                      f"{external_root()}) or $PRECOND_DATA.")
            else:
                print(f"  [data] {'frangieh_*':<20} {Path(rel).name:<24} "
                      f"-> {hit.resolve()}", flush=True)

    for p in REQUIRED_SCRIPTS:
        if not p.exists():
            fatal.append(
                f"{p.name} is not in this repository (looked in {p.parent}).\n"
                "        STOPPING. Do not vendor a copy from another repo and\n"
                "        do not reimplement it: every number downstream depends\n"
                "        on that file being present and unmodified.")
        else:
            print(f"  [code] {p.name:<28} {sha256(p)}", flush=True)
    print(f"  [code] simulator resolved path : {SIMULATOR.resolve()}", flush=True)

    if fatal:
        print("\n[fatal] step-0 gate FAILED:", file=sys.stderr)
        for f in fatal:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(2)

    report = resolution_report(datasets)
    print(f"\n  {'condition':<22}{'branch':<26}{'resolved via':<20}path")
    print("  " + "-" * 96)
    for r in report:
        print(f"  {r['dataset']:<22}{r['branch']:<26}{str(r['root_label']):<20}"
              f"{r['path'] or '-'}")
    print("  " + "-" * 96)
    print("  [gate] PASS\n", flush=True)
    return report


def resolution_report(datasets):
    """Which branch of the ONE resolver each condition takes, and where it lands.

    Three branches exist and no condition invents a fourth:
      absolute            an absolute candidate, used directly
      external-relative   joined under $PRECOND_EXTERNAL
      data-root-relative  joined against the $PRECOND_DATA root list
    """
    roots = data_roots()
    out = []
    for ds in datasets:
        base = ds.split("_")[0]
        if base == "frangieh":
            rels = EXTERNAL_RELATIVE["frangieh"]
        elif base == "norman":
            rels = None
        else:
            rels = INPUTS.get(ds, ())
        if base == "norman":
            cands = norman_candidate_paths()
            hit = next(((lbl, q) for lbl, q in cands if q.exists()),
                       (None, None))
            branch = ("external-relative" if hit[0] == "$PRECOND_EXTERNAL"
                      else ("data-root-relative" if hit[0] else "UNRESOLVED"))
            out.append(dict(dataset=ds, branch=branch, root_label=hit[0],
                            path=(str(hit[1].resolve()) if hit[1] else None),
                            n_candidates=len(cands)))
            continue
        for rel in rels:
            parts = Path(rel).parts
            cands = external_candidates(*parts) + [
                (lbl, r.joinpath(*parts)) for lbl, r in roots]
            hit = next(((lbl, q) for lbl, q in cands if q.exists()),
                       (None, None))
            branch = ("external-relative" if hit[0] == "$PRECOND_EXTERNAL"
                      else ("data-root-relative" if hit[0] else "UNRESOLVED"))
            out.append(dict(dataset=ds, input=rel, branch=branch,
                            root_label=hit[0],
                            path=(str(hit[1].resolve()) if hit[1] else None),
                            n_candidates=len(cands)))
    return out


# ============================================================ expected values
def expected_pr(dataset, cap):
    """The committed participation ratio for (dataset, cap), read off disk.

    Returns (value, artefact_name, cap_used). Falls back to FALLBACK_CAP when
    the requested cap has no artefact, because the gate is a reproduction check
    and a check at a different cap still proves the pipeline has not moved.
    Never falls back to a literal: there is no literal to fall back to.
    """
    for want in (cap, FALLBACK_CAP):
        for p in sorted(DESCRIPTIVES_DIR.glob("*.json")):
            try:
                doc = json.load(open(p))
            except ValueError:
                continue
            if (doc.get("meta") or {}).get("status") != "CURRENT":
                continue
            # The arm can live in three places and Frangieh uses the third.
            # doc["dataset"] is bare "frangieh" for all three arms, doc["arm"]
            # and meta.config.arm are null, and the only record of which arm
            # this is sits in blocks[i]["label"] as "frangieh:coculture". Keying
            # on the first two alone silently returned None for every Frangieh
            # row, which is how three raw-arm participation ratios went missing
            # from the intrinsic-dimension comparison.
            base = doc.get("dataset")
            arm = doc.get("arm") or (doc.get("meta") or {}
                                     ).get("config", {}).get("arm")
            for blk in doc.get("blocks") or []:
                label = base
                if arm and not str(label).endswith(str(arm)):
                    label = f"{base}_{arm}"
                blk_label = blk.get("label")
                if blk_label:
                    label = str(blk_label).replace(":", "_")
                if label != dataset:
                    continue
                ld = blk.get("latent_dimension_from_controls") or {}
                rec = ld.get(str(want))
                if rec and rec.get("participation_ratio") is not None:
                    return float(rec["participation_ratio"]), p.name, int(want)
    return None, None, None


# ============================================================== control pools
RESOLVER_NAMES = ("resolve_data", "_resolve", "norman_h5ad",
                  "frangieh_meta_csv", "frangieh_expr_csv")


def _wrap_resolvers(mod, seen, restore):
    """Wrap a loader's path resolvers so the resolved ABSOLUTE path is kept.

    Resolvers, not loaders: the resolver is the single point where a path
    becomes a file. Every wrap is recorded in `restore` and undone afterwards.
    """
    if mod is None:
        return
    for fname in RESOLVER_NAMES:
        real = getattr(mod, fname, None)
        if real is None or getattr(real, "_audit_wrapped", False):
            continue

        generic = fname in ("resolve_data", "_resolve")

        def wrapper(*a, _real=real, _generic=generic, _fn=fname, **kw):
            # $PRECOND_EXTERNAL FIRST, for the resolvers that take name parts.
            # The named wrappers (norman_h5ad, frangieh_*_csv) delegate to these,
            # so covering the generic pair covers every loader.
            ext_tried = []
            if _generic and a and not kw.get("arg"):
                for lbl, q in external_candidates(*a):
                    ext_tried.append((lbl, q))
                    if q.exists():
                        seen.append(str(q.resolve()))
                        return q
            try:
                got = _real(*a, **kw)
            except SystemExit as e:
                if not ext_tried:
                    raise
                raise SystemExit(
                    str(e).rstrip()
                    + "\n  Also tried, before the roots above:\n  "
                    + "\n  ".join(f"{lbl}: {q}" for lbl, q in ext_tried)
                    + f"\n  Set $PRECOND_EXTERNAL (currently "
                      f"{external_root()}) if the data lives outside "
                      f"$PRECOND_DATA.")
            try:
                seen.append(str(Path(got).resolve()))
            except TypeError:
                pass
            return got

        wrapper._audit_wrapped = True
        setattr(mod, fname, wrapper)
        restore.append((mod, fname, real))


def control_matrix(DESC, dataset, args, norman_path=None):
    """The EXACT control matrix 85_dataset_descriptives feeds its estimators.

    Same capture 91_preprocessing_sweep.py uses: 85's own loader runs with
    latent_dim_block swapped for a grab, so the matrix and the rng arrive in
    precisely the state the estimators would have seen. Re-deriving the
    selection here would drift from 85 the first time either changed.

    Also grabs the FULL value array where 85 materialises one, and the absolute
    path of every file the loaders resolved. The resolvers are reached by
    intercepting 85's OWN `_load_module`: do_norman and do_frangieh each load a
    fresh module instance inside the call, so patching a separately-loaded copy
    would patch an object 85 never touches and would silently record nothing.
    03_screen is already bound as DESC.SCREEN at 85's import, so it is wrapped
    directly.

    Frangieh never materialises a full matrix by design: it reads only this
    arm's capped control cells, because RNA_expression.csv.gz is about 24.8 GB
    uncompressed. Its full-array fingerprint is null with a reason, not a
    fabricated number.
    """
    grabbed, paths, restore = {}, [], []

    def capture_ld(Xc, rng, spectra=None, key_prefix=""):
        grabbed["Xc"] = np.asarray(Xc)
        grabbed["rng"] = rng
        return {}

    real_ld = DESC.latent_dim_block
    real_pp = DESC.profile_perturb
    real_loadmod = DESC._load_module

    def capture_pp(X, iv, label, rng, spectra=None):
        grabbed["X_full"] = np.asarray(X)
        return real_pp(X, iv, label, rng, spectra)

    def loadmod(path, name):
        mod = real_loadmod(path, name)
        _wrap_resolvers(mod, paths, restore)
        if norman_path is not None and hasattr(mod, "norman_h5ad"):
            restore.append((mod, "norman_h5ad", mod.norman_h5ad))
            mod.norman_h5ad = lambda arg=None, _p=Path(norman_path): _p
            paths.append(str(Path(norman_path).resolve()))
        return mod

    base = dataset.split("_")[0]
    if base == "norman" and norman_path is None:
        # 40_screen_norman resolves a bare filename against the $PRECOND_DATA
        # roots only, so it cannot see $PRECOND_EXTERNAL. Hand it the resolved
        # path rather than let it fail in a place this script cannot report.
        norman_path = first_existing_norman()
    _wrap_resolvers(getattr(DESC, "SCREEN", None), paths, restore)
    DESC.latent_dim_block = capture_ld
    DESC.profile_perturb = capture_pp
    DESC._load_module = loadmod
    try:
        ns = argparse.Namespace(
            dataset=base, hvg=args.hvg, seed=args.seed,
            arm=(dataset.split("_", 1)[1] if dataset.startswith("frangieh_")
                 else None),
            abide_npz=None, n_spec_cap=None, overlap_check=None, selftest=False)
        rng = np.random.default_rng(args.seed)
        DESC.LOADERS[base](rng, ns, {})
    finally:
        DESC.latent_dim_block = real_ld
        DESC.profile_perturb = real_pp
        DESC._load_module = real_loadmod
        for mod, fname, real in reversed(restore):
            setattr(mod, fname, real)
    if "Xc" not in grabbed:
        sys.exit(f"[fatal] could not capture the control matrix for {dataset}")
    # dict.fromkeys keeps first-seen order and drops repeats
    return (grabbed["Xc"], grabbed["rng"], grabbed.get("X_full"),
            list(dict.fromkeys(paths)))


def measure(DESC, Xc, cap, rng):
    return DESC.spectrum_estimators(Xc, cap, rng)


def audit_one(DESC, dataset, args, cap=CAP, norman_path=None):
    Xc, rng, X_full, paths = control_matrix(DESC, dataset, args, norman_path)
    n_all, p = Xc.shape
    est = measure(DESC, Xc, cap, rng)

    full_fp = value_fingerprint(X_full, "full value array")
    if full_fp is None:
        full_fp = dict(
            label="full value array",
            unavailable=("85 never materialises a full matrix for this "
                         "condition: it reads only this arm's capped control "
                         "cells, because RNA_expression.csv.gz is about "
                         "24.8 GB uncompressed. Not fabricated."))
    prov = dict(
        resolved_input_paths=paths,
        input_path=(paths[0] if paths else None),
        n_cells=(int(X_full.shape[0]) if X_full is not None else None),
        n_genes=int(p),
        n_control_cells=int(n_all),
        full_value_array=full_fp,
        control_value_array=value_fingerprint(Xc, "control matrix"),
        legacy_fingerprint=DESC.data_fingerprint(Xc),
    )
    if est.get("error"):
        return dict(dataset=dataset, status="error", reason=est["error"],
                    available_controls=int(n_all), cap_requested=cap,
                    provenance=prov)

    n_eff = int(est["n_control_used"])
    exp, src, exp_cap = expected_pr(dataset, cap)
    got = float(est["participation_ratio"])
    return dict(
        dataset=dataset,
        status="ok",
        available_controls=int(n_all),
        n_effective=n_eff,
        cap_requested=cap,
        matched=bool(n_eff == cap),
        shortfall=int(max(0, cap - n_eff)),
        n_genes=int(p),
        participation_ratio=got,
        effective_rank_exp_spectral_entropy=float(
            est["effective_rank_exp_spectral_entropy"]),
        n_components_for_variance_95pct=int(
            est["n_components_for_variance"]["95pct"]),
        top_eigenvalue_share=float(est["top_eigenvalue_share"]),
        rank_bound=int(est["rank_bound"]),
        expected_participation_ratio=exp,
        expected_source=src,
        expected_at_cap=exp_cap,
        reproduces_expected=(None if exp is None or exp_cap != cap
                             else bool(abs(got - exp) <= GATE_TOL)),
        provenance=prov,
    )


# ========================================================= norman candidates
def norman_candidates(DESC, args):
    """All three Norman files at the published cap. Reports; never picks one."""
    cap = NORMAN_CANDIDATE_CAP
    exp, src, exp_cap = expected_pr("norman", cap)
    print("=" * 70, flush=True)
    print(f" NORMAN CANDIDATES at cap {cap}", flush=True)
    if exp is None:
        print("  no committed norman artefact at this cap; nothing to compare "
              "against", flush=True)
    else:
        print(f"  expected participation ratio {exp!r}", flush=True)
        print(f"  read from results/descriptives/{src} at cap {exp_cap}",
              flush=True)
    print("=" * 70, flush=True)

    out = []
    for _lbl, cand in [(l, q) for l, q in norman_candidate_paths()
                       if l == "$PRECOND_EXTERNAL"]:
        p = Path(cand)
        print(f"\n--- [{p.name}] ---\n    {p}", flush=True)
        if not p.exists():
            print("    ABSENT on this machine", flush=True)
            out.append(dict(path=str(p), status="absent"))
            continue
        try:
            rec = audit_one(DESC, "norman", args, cap=cap, norman_path=p)
        except Exception as e:                           # noqa: BLE001
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            out.append(dict(path=str(p), status="error",
                            reason=f"{type(e).__name__}: {e}"))
            continue
        got = rec.get("participation_ratio")
        rec["candidate_path"] = str(p.resolve())
        rec["reproduces_published"] = (
            None if exp is None or got is None
            else bool(abs(got - exp) <= GATE_TOL))
        out.append(rec)
        fp = (rec["provenance"]["full_value_array"] or {})
        print(f"    PR={got!r}", flush=True)
        print(f"    reproduces published: {rec['reproduces_published']}",
              flush=True)
        print(f"    shape={fp.get('shape')} dtype={fp.get('dtype')} "
              f"max={fp.get('max')} mean={fp.get('mean')}", flush=True)
        print(f"    sha256(first {FP_NONZERO_N} nonzero)="
              f"{str(fp.get('sha256_first_nonzero'))[:12]}", flush=True)

    print("\n" + "=" * 70)
    print(f" {'file':<34}{'PR':>22}{'reproduces':>13}")
    print("-" * 70)
    for r in out:
        name = Path(r.get("candidate_path") or r["path"]).name
        if r.get("status") in ("absent", "error"):
            print(f" {name:<34}{r['status']:>22}{'-':>13}")
            continue
        print(f" {name:<34}{r['participation_ratio']:>22.10f}"
              f"{str(r['reproduces_published']):>13}")
    print("-" * 70)
    print(" All three are reported. This script does not pick one.")
    print("=" * 70)
    return dict(cap=cap, expected_participation_ratio=exp,
                expected_source=src, candidates=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS,
                    help="run one condition only (default: all six)")
    ap.add_argument("--norman-candidates", action="store_true",
                    help="try all three Norman files and report all three")
    ap.add_argument("--hvg", type=int, default=None,
                    help="Frangieh gene cap; default unset, all genes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(OUT_DIR))
    a = ap.parse_args()
    todo = [a.dataset] if a.dataset else list(DATASETS)
    if a.norman_candidates:
        todo = ["norman"]

    step0_gate(todo)

    LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    if LOCKFILE.exists():
        other = LOCKFILE.read_text().strip()
        alive = False
        if other.isdigit():
            try:
                os.kill(int(other), 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            sys.exit(f"[fatal] another audit is live (pid {other}, "
                     f"lock {LOCKFILE}). Two concurrent batches into one output "
                     f"directory is how the 11 Aug artefacts had to be "
                     f"discarded. Refusing.")
        print(f"[lock] stale lock from pid {other or 'unknown'}; reclaiming",
              flush=True)
        LOCKFILE.unlink()
    LOCKFILE.write_text(str(os.getpid()))

    try:
        out = Path(a.outdir)
        # A stale directory must never let an exists-check report old numbers
        # as fresh. Only a full run clears it, so a scoped run appends.
        if not a.dataset and not a.norman_candidates and out.exists():
            print(f"[clean] rm -rf {out}", flush=True)
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        DESC = _load(HERE / "85_dataset_descriptives.py", "_descriptives")

        if a.norman_candidates:
            block = norman_candidates(DESC, a)
            ts = DESC.utc_stamp()
            path = out / f"{ts.replace(':', '-')}__norman_candidates.json"
            path.write_text(json.dumps(
                dict(norman_candidates=block,
                     contains_no_test=True,
                     disclaimer=("DESCRIPTIVE ONLY. No rank test was run and "
                                 "no assumption verdict is expressed or "
                                 "implied."),
                     simulator_sha256=sha256(SIMULATOR),
                     peak_rss_mb=DESC.peak_rss_mb()), indent=2) + "\n")
            print(f"\n[write] {path}", flush=True)
            n_ok = sum(1 for c in block["candidates"]
                       if c.get("status") == "ok")
            if n_ok != len(NORMAN_RELATIVE):
                print(f"\n[fatal] SHORTFALL: {n_ok} candidate(s) measured for "
                      f"{len(NORMAN_RELATIVE)} requested. This run is "
                      f"INCOMPLETE.", file=sys.stderr)
                sys.exit(1)
            return

        records, failed = [], []
        for ds in todo:
            print(f"\n--- [{ds}] ---", flush=True)
            try:
                rec = audit_one(DESC, ds, a)
            except Exception as e:                       # noqa: BLE001
                print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
                failed.append((ds, f"{type(e).__name__}: {e}"))
                continue
            records.append(rec)
            if rec["status"] != "ok":
                print(f"    {rec['status']}: {rec.get('reason')}", flush=True)
                continue
            print(f"    input     : {rec['provenance']['input_path']}",
                  flush=True)
            print(f"    available={rec['available_controls']}  "
                  f"n_effective={rec['n_effective']}  "
                  f"matched={rec['matched']}  "
                  f"PR={rec['participation_ratio']:.6f}", flush=True)

            if ds == GATE_DATASET:
                exp, src, exp_cap = (rec["expected_participation_ratio"],
                                     rec["expected_source"],
                                     rec["expected_at_cap"])
                if exp is None:
                    sys.exit(
                        f"\n[fatal] STEP-0 REPRODUCTION GATE CANNOT RUN for "
                        f"{ds}\n"
                        f"        No CURRENT descriptives artefact carries a "
                        f"participation ratio at\n"
                        f"        cap {CAP} or {FALLBACK_CAP}. There is no "
                        f"literal to fall back on, by design.\n"
                        f"        Looked in {DESCRIPTIVES_DIR}")
                if exp_cap == CAP:
                    got = rec["participation_ratio"]
                else:
                    # The artefact has no entry at CAP, so the gate falls back
                    # to the cap it does have. Recompute at that cap rather
                    # than compare across caps, which would always disagree.
                    Xc_g, rng_g, _, _ = control_matrix(DESC, ds, a)
                    got = float(measure(DESC, Xc_g, exp_cap,
                                        rng_g)["participation_ratio"])
                if abs(got - exp) > GATE_TOL:
                    sys.exit(
                        f"\n[fatal] STEP-0 REPRODUCTION GATE FAILED for {ds}\n"
                        f"        expected {exp!r}\n"
                        f"        got      {got!r}\n"
                        f"        |diff| = {abs(got - exp):.3e} > tol "
                        f"{GATE_TOL:g}\n"
                        f"        compared at cap {exp_cap}, read from "
                        f"results/descriptives/{src}\n"
                        f"        This is a recompute of a committed number, "
                        f"not a new measurement.\n"
                        f"        A disagreement means the control selection "
                        f"or the estimators have\n"
                        f"        moved, so every other row here is measuring "
                        f"the pipeline. Refusing.")
                print(f"    [gate] reproduces {src} at cap {exp_cap}: "
                      f"{exp!r}  PASS", flush=True)

        if failed or len(records) != len(todo):
            names = {r["dataset"] for r in records}
            missing = [d for d in todo if d not in names]
            print(f"\n[fatal] SHORTFALL: {len(records)} record(s) for "
                  f"{len(todo)} requested.", file=sys.stderr)
            for d in missing:
                print(f"  - {d}: {dict(failed).get(d, 'no record produced')}",
                      file=sys.stderr)
            print("        This run is INCOMPLETE. Do not read the artefact as "
                  "a full audit.", file=sys.stderr)
            sys.exit(1)

        ts = DESC.utc_stamp()
        payload = dict(
            cap_requested=CAP,
            fallback_cap=FALLBACK_CAP,
            datasets=list(todo),
            n_matched=sum(1 for r in records if r.get("matched")),
            n_unmatched=sum(1 for r in records
                            if r.get("status") == "ok" and not r["matched"]),
            records=records,
            norman_candidates_note=(
                "Three Norman files exist on the A100 and which produced the "
                "published number is not recorded. Run "
                "--norman-candidates to measure all three."),
            norman_candidate_paths=[str(q) for l, q in
                                    norman_candidate_paths()
                                    if l == "$PRECOND_EXTERNAL"],
            external_root=str(external_root()),
            contains_no_test=True,
            disclaimer=("DESCRIPTIVE ONLY. No rank test was run and no "
                        "assumption verdict is expressed or implied."),
            simulator_sha256=sha256(SIMULATOR),
            simulator_path=str(SIMULATOR.relative_to(REPO)),
            peak_rss_mb=DESC.peak_rss_mb(),
        )
        path = out / f"{ts.replace(':', '-')}__control_pool_audit.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\n[write] {path}", flush=True)

        print("\n" + "=" * 70)
        print(f" {'dataset':<22}{'available':>11}{'n_eff':>8}{'matched':>9}"
              f"{'PR':>14}")
        print("-" * 70)
        for r in records:
            if r["status"] != "ok":
                print(f" {r['dataset']:<22}{r['available_controls']:>11}"
                      f"{'-':>8}{'-':>9}{r['status']:>14}")
                continue
            print(f" {r['dataset']:<22}{r['available_controls']:>11}"
                  f"{r['n_effective']:>8}{str(r['matched']):>9}"
                  f"{r['participation_ratio']:>14.4f}")
        print("-" * 70)
        print(f" matched at n={CAP}: {payload['n_matched']} of {len(records)}")
        print("=" * 70)
    finally:
        if LOCKFILE.exists():
            LOCKFILE.unlink()


if __name__ == "__main__":
    main()
