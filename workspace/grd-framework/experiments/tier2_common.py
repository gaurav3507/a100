"""Shared real-data plumbing for the Tier 2 experiments (E4 real panel, E5).

Data loading, candidate selection, the control-fit projection, and the random and
structured control constructions follow e3_perturbseq_panel.py,
e3_stability_perturbseq.py and e3_rpe1_confound_check.py exactly. The environment
order, and so each environment's SeedSequence([seed, i]) null stream, is the same
as in E3. Every statistic, null draw, p-value and BH decision goes through
src/gate/precision_readout.py via e3_gate_compare._detect_family. Nothing here is a
second gate implementation.

One loading difference, with no numerical effect: E3 densified the whole matrix and
then selected rows; this module selects rows and then densifies them, which gives
the same float64 values with a much smaller peak memory.

Selftest panels (make_selftest_panel) are synthetic arrays that exercise the code
path only. They carry no result and are never written under results/.
"""

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data_paths import perturbseq_path  # noqa: E402
from e3_gate_compare import _detect_family, shift_alignment, summarize  # noqa: E402


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, str(REPO / relative_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PR = load_module("grd_tier2_precision_readout", "src/gate/precision_readout.py")

# Same settings as the E3 real-data scripts.
D_PROJ = 10
NMIN = 200
ALPHA = 0.05
Q = 0.05
B_BOOT = 500
SEEDS = list(range(5))
N_FAKE = 50
FAKE_SEED_OFFSET = 10_000
STRUCT_SEED_OFFSET = 20_000
# Fixed salt for the split-control RNG stream: split for gate seed s comes from
# SeedSequence([SPLIT_SALT, s]), independent of every null stream.
SPLIT_SALT = 5_050

DATASETS = {
    "k562": dict(name="K562", file="causalbench_k562.h5ad", ctrl="", single=False),
    "rpe1": dict(name="RPE1", file="causalbench_rpe1.h5ad", ctrl="", single=False),
    "norman": dict(name="Norman", file="Norman2019_raw.h5ad", ctrl="", single=True),
}
SOURCE_FILES = [
    "src/gate/precision_readout.py",
    "src/gate/rank_readout.py",
    "src/recover/backbone.py",
    "experiments/e3_gate_compare.py",
    "experiments/tier2_common.py",
]


# ------------------------------------------------------------------ provenance
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_commit():
    """git HEAD with a -dirty suffix for tracked edits; None outside a git checkout.

    Untracked files are ignored here (unlike e0_oracle._code_commit) because the
    A100 checkout may hold untracked data or logs; the per-file SHA-256 values in
    provenance() pin the code regardless.
    """
    try:
        head = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain",
             "--untracked-files=no"],
            stderr=subprocess.DEVNULL, text=True).strip() != ""
        return head + ("-dirty" if dirty else "")
    except Exception:
        return None


def provenance(script_path, extra_files=()):
    files = list(SOURCE_FILES) + [str(Path(script_path).resolve().relative_to(REPO))]
    files += list(extra_files)
    return dict(
        code_commit=code_commit(),
        source_sha256={f: sha256_file(REPO / f) for f in dict.fromkeys(files)},
        numpy_version=np.__version__,
        python_version=platform.python_version(),
        host=platform.node(),
    )


# ------------------------------------------------------------------ JSON output
def write_json(path, document):
    """Atomic write so a crash never leaves a half-written artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(tmp, path)


def update_dataset_block(path, top_level, dataset_name, block):
    """Insert one dataset block into a multi-dataset artifact, keeping the others.

    Each dataset is written as soon as it finishes, so a crash on a later dataset
    keeps the earlier ones. Each block carries its own provenance.
    """
    path = Path(path)
    document = json.loads(path.read_text()) if path.exists() else {}
    document.update(top_level)
    datasets = document.setdefault("datasets", {})
    datasets[dataset_name] = block
    document["datasets"] = {k: datasets[k] for k in sorted(datasets)}
    write_json(path, document)
    return document


# ------------------------------------------------------------------ data loading
def _is_single(label, ctrl):
    return (label != ctrl and label != "" and "," not in label
            and "+" not in label and "_" not in label)


def candidates(g, ctrl, single):
    """Candidate perturbation labels, in the order E3 uses."""
    if single:
        return sorted({label for label in np.unique(g) if _is_single(label, ctrl)})
    return [label for label in np.unique(g) if label != ctrl]


def _dense_rows(X, mask):
    rows = X[np.flatnonzero(mask)]
    if hasattr(rows, "toarray"):
        return rows.toarray().astype(np.float64)
    return np.asarray(rows, np.float64)


def load_panel(key):
    """Controls and powered perturbations of one Perturb-seq dataset."""
    import anndata as ad

    spec = DATASETS[key]
    path = perturbseq_path(spec["file"])
    A = ad.read_h5ad(path)
    X = A.X
    g = A.obs["guide_ids"].astype(str).values
    n_cells, n_genes = (int(s) for s in A.shape)
    del A
    ctrl = spec["ctrl"]
    cand = candidates(g, ctrl, spec["single"])
    perts = [p for p in cand if (g == p).sum() >= NMIN]
    return dict(
        name=spec["name"],
        source=str(path),
        n_cells=n_cells,
        n_genes=n_genes,
        control_label=repr(ctrl),
        Xc=_dense_rows(X, g == ctrl),
        perts=perts,
        Xperts=[_dense_rows(X, g == p) for p in perts],
    )


def make_selftest_panel(key, seed=0):
    """Synthetic stand-in for load_panel: code-path check only, never a result.

    Controls are correlated Gaussian cells. Half the perturbations shrink one
    latent direction (a detectable variance reduction), half are pure control-like
    draws. Sizes span the disjoint and non-disjoint null paths.
    """
    rng = np.random.default_rng(90_000 + seed)
    n_genes, d_true = 60, 12
    mixing = rng.standard_normal((d_true, n_genes))

    def draw(n, shrink=None):
        Z = rng.standard_normal((n, d_true))
        if shrink is not None:
            Z[:, shrink] *= 0.25
        return Z @ mixing + 0.5 * rng.standard_normal((n, n_genes))

    Xc = draw(2400)
    sizes = [200, 230, 260, 300, 350, 420, 500, 640, 800, 1300]
    perts, Xperts = [], []
    for i, n in enumerate(sizes):
        perts.append(f"SELFTEST_{key.upper()}_{i:02d}")
        Xperts.append(draw(n, shrink=(i % d_true) if i % 2 == 0 else None))
    return dict(
        name=DATASETS[key]["name"], source="synthetic selftest",
        n_cells=int(len(Xc) + sum(sizes)), n_genes=n_genes,
        control_label=repr(""), Xc=Xc, perts=perts, Xperts=Xperts)


def control_projection(Xc, d=D_PROJ):
    """Control-fit PCA projection, exactly as the E3 scripts build it."""
    mu = Xc.mean(0)
    _, _, Vt = np.linalg.svd(Xc - mu, full_matrices=False)
    Bp = Vt[:d].T
    return lambda M: (M - mu) @ Bp


def random_controls(Xc, proj, sizes, seed, n_fake):
    """E3 random pure-control environments (same selection_rng sequence)."""
    selection_rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_fake):
        n = int(selection_rng.choice(sizes))
        idx = selection_rng.choice(len(Xc), n, replace=False)
        out.append(proj(Xc[idx]))
    labels = [f"random_control_{i:02d}" for i in range(n_fake)]
    return labels, out


def structured_controls(Xc, proj, Yobs, sizes, d=D_PROJ):
    """E3 structured control splits: lowest and highest m controls on each PC."""
    m = int(np.median(sizes))
    labels, Ys = [], []
    for j in range(d):
        order = np.argsort(Yobs[:, j])
        labels.extend([f"pc{j}_low", f"pc{j}_high"])
        Ys.extend([proj(Xc[order[:m]]), proj(Xc[order[-m:]])])
    return labels, Ys


# ------------------------------------------------------------------ gate screens
def split_halves(Yobs, gate_seed):
    """The split-control design's single 50/50 control split for one gate seed."""
    rng = np.random.default_rng(np.random.SeedSequence([SPLIT_SALT, int(gate_seed)]))
    null_idx, ref_idx = PR.split_control_indices(len(Yobs), rng)
    info = dict(
        n_control=int(len(Yobs)),
        n_null_half=int(len(null_idx)),
        n_reference_half=int(len(ref_idx)),
        split_rng=f"SeedSequence([{SPLIT_SALT}, {int(gate_seed)}])",
        null_idx_sha256=hashlib.sha256(
            np.ascontiguousarray(null_idx, dtype=np.int64).tobytes()).hexdigest(),
    )
    return Yobs[null_idx], Yobs[ref_idx], info


def screen(Y_envs, Y_ref, seed, readout="precision", Y_null=None, B=B_BOOT):
    """One BH family through the shared E3 detection helper (corrected disjoint).

    Y_null=None: shared-reference design, nulls drawn from Y_ref.
    Y_null given: split-control design, nulls drawn from Y_null only.
    """
    Y_envs = [np.asarray(Y) for Y in Y_envs]
    Y_ref = np.asarray(Y_ref)
    result = _detect_family(PR, Y_envs, Y_ref, seed, ALPHA, B, Q, disjoint=True,
                            readout=readout, Y_null=Y_null)
    pool = Y_ref if Y_null is None else np.asarray(Y_null)
    result["disjoint_applied"] = [
        bool(PR._use_disjoint_null(len(pool), len(Y), pool.shape[1], True))
        for Y in Y_envs
    ]
    return result


def selected(labels, flags):
    return [str(label) for label, keep in zip(labels, flags) if keep]


def jaccard(a, b):
    """|A n B| / |A u B|; None when both sets are empty (undefined, not 1)."""
    a, b = set(a), set(b)
    union = a | b
    if not union:
        return None
    return round(len(a & b) / len(union), 6)


def design_records(labels, Y_envs, designs):
    """One auditable record per environment, one sub-dict per design."""
    rows = []
    for i, (label, Y) in enumerate(zip(labels, Y_envs)):
        row = dict(environment=str(label), n_samples=int(len(Y)))
        for name, result in designs.items():
            row[name] = dict(
                signal=float(result["signals"][i]),
                threshold=float(result["thresholds"][i]),
                pvalue=float(result["pvalues"][i]),
                raw_detect=bool(result["raw_detect"][i]),
                bh_detect=bool(result["bh_detect"][i]),
                disjoint_applied=bool(result["disjoint_applied"][i]),
            )
        rows.append(row)
    return rows


def design_comparison(shared, split):
    """Side-by-side summaries plus decision transitions (mirrors E3's layout)."""
    s_raw = np.asarray(shared["raw_detect"], dtype=bool)
    p_raw = np.asarray(split["raw_detect"], dtype=bool)
    s_bh = np.asarray(shared["bh_detect"], dtype=bool)
    p_bh = np.asarray(split["bh_detect"], dtype=bool)
    return dict(
        corrected_disjoint=summarize(shared),
        split_control=summarize(split),
        transitions=dict(
            raw_shared_only=int(np.sum(s_raw & ~p_raw)),
            raw_split_only=int(np.sum(~s_raw & p_raw)),
            raw_both=int(np.sum(s_raw & p_raw)),
            bh_shared_only=int(np.sum(s_bh & ~p_bh)),
            bh_split_only=int(np.sum(~s_bh & p_bh)),
            bh_both=int(np.sum(s_bh & p_bh)),
        ),
    )


def shared_vs_split_entry(labels, shared, split):
    s_bh = selected(labels, shared["bh_detect"])
    p_bh = selected(labels, split["bh_detect"])
    return dict(
        n_environments=len(labels),
        shared=dict(raw_count=int(shared["raw_count"]),
                    bh_count=int(shared["bh_count"])),
        split=dict(raw_count=int(split["raw_count"]),
                   bh_count=int(split["bh_count"])),
        jaccard_bh=jaccard(s_bh, p_bh),
        jaccard_raw=jaccard(selected(labels, shared["raw_detect"]),
                            selected(labels, split["raw_detect"])),
        both_bh_empty=bool(not s_bh and not p_bh),
    )


def alignment(Y_envs, Y_obs, designs):
    """E3 shift alignment (mean shift vs the full control) for each design."""
    out = {}
    for name, result in designs.items():
        out[f"{name}_raw"] = shift_alignment(Y_envs, Y_obs, result["raw_detect"])
        out[f"{name}_bh"] = shift_alignment(Y_envs, Y_obs, result["bh_detect"])
    return out


# ------------------------------------------------------------------ E3 references
def e3_stability_sets(name):
    """Per-seed corrected-disjoint raw and BH sets from results/e3_stability."""
    path = REPO / "results" / "e3_stability" / f"{name}.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    out = {}
    for row in doc["per_seed"]:
        recs = row["per_perturbation"]
        out[int(row["seed"])] = dict(
            raw=[r["environment"] for r in recs if r["corrected_disjoint"]["raw_detect"]],
            bh=[r["environment"] for r in recs if r["corrected_disjoint"]["bh_detect"]],
            labels=[r["environment"] for r in recs],
        )
    return dict(source=str(path.relative_to(REPO)), sha256=sha256_file(path),
                per_seed=out)


def e3_decision_set(name):
    """Seed-0 corrected-disjoint BH set from results/e3_decisions."""
    path = REPO / "results" / "e3_decisions" / f"{name}.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    return dict(source=str(path.relative_to(REPO)), sha256=sha256_file(path),
                bh=list(doc["detected_ids"]),
                labels=[r["environment"] for r in doc["decisions"]])


def seed_sd(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=round(float(values.mean()), 6),
                sd=round(float(values.std()), 6),
                per_seed=[round(float(v), 6) for v in values])
