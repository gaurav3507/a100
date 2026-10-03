"""E1-capped: exercise Gate -> Recover -> Discover when the cap binds.

E1-full validates the clean end-to-end pipeline only when all five intervention
environments pass the gate. This experiment supplies only three interventions,
so the gate can certify at most k=3<d=5 and the selective path must execute.

Construction:

* The latent DAG is the fixed chain 0->1->2->3->4.
* Only interventions 0, 1, and 2 are supplied. They form an ancestral-closed
  recovered subset, preventing omitted descendants from changing the graph among
  the recovered nodes.
* Noise variances, the 200-by-5 mixing matrix, and every sample vary across seeds.
* Hard interventions use the same reducing variance (0.1), sample size (4000),
  PCA projection, precision gate, backbone row estimator, and Discover call as
  E1-full.

The wrapper is deliberately minimal. Gate decisions choose the certified nodes.
Recovery invokes the same ``backbone._unmixing_row`` used inside
``backbone.recover``, but only for certified interventions. Discover receives the
resulting k-dimensional representation and its k interventions. Its local edge
statuses are mapped back to the original five node labels. Every ordered pair
touching an uncovered node receives ``EDGE_UNDECIDED_UNRECOVERED``; no method is
allowed to commit an edge outside the recovered subset.

PRE-REGISTERED PASS CONDITIONS, for every seed:

1. gate n_recoverable is exactly 3 and strictly below d=5;
2. Hungarian MCC on the three certified latents is greater than 0.90;
3. SHD over Discover's decided edges among certified nodes is zero;
4. all 14 ordered pairs touching nodes 3 or 4 are explicitly undecided;
5. no edge touching an uncovered node is committed.

``--smoke`` uses three seeds and B=100. The default uses ten seeds and B=500.
Existing E1 code and results are never modified.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np


warnings.filterwarnings(
    "ignore", message=r".*encountered in matmul", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results" / "e1_capped"

FULL_SEEDS = list(range(10))
SMOKE_SEEDS = list(range(3))
D_LATENT = 5
D_OBS = 200
D_PROJ = 5
N_PER_ENV = 4000
OBS_NOISE_FRAC = 0.1
ALPHA = 0.05
FULL_BOOT = 500
SMOKE_BOOT = 100
IV_REDUCING = 0.1
RECOVERY_MCC = 0.90
SUPPLIED_NODES = (0, 1, 2)
UNCOVERED_NODES = (3, 4)
UNDECIDED_UNRECOVERED = "EDGE_UNDECIDED_UNRECOVERED"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SIM = _load(REPO / "sim" / "simulator.py", "grd_sim_e1_capped")
PR = _load(
    REPO / "src" / "gate" / "precision_readout.py",
    "grd_precision_e1_capped",
)
BK = _load(REPO / "src" / "recover" / "backbone.py", "grd_backbone_e1_capped")
DISC = _load(REPO / "src" / "discover" / "discover.py", "grd_discover_e1_capped")
E0 = _load(REPO / "experiments" / "e0_oracle.py", "grd_e0_e1_capped")


def fixed_chain():
    """B[j,k] is k->j: 0->1->2->3->4."""
    B = np.zeros((D_LATENT, D_LATENT))
    B[1, 0] = 0.8
    B[2, 1] = -0.9
    B[3, 2] = 0.7
    B[4, 3] = -0.8
    return B


def true_edges(B):
    return {
        (int(parent), int(child))
        for child in range(D_LATENT)
        for parent in range(D_LATENT)
        if abs(B[child, parent]) > 1e-9
    }


def build_data(seed):
    """Sample the fixed SCM with the committed simulator's primitive functions."""
    rng = np.random.default_rng(seed)
    B = fixed_chain()
    noise_var = rng.uniform(0.5, 1.5, D_LATENT)
    mixing = rng.standard_normal((D_OBS, D_LATENT))
    specs = [
        SIM.EnvSpec("basis", None, ()),
        SIM.EnvSpec("obs", None, ()),
    ] + [
        SIM.EnvSpec(f"iv{node}", "hard", (node,), IV_REDUCING)
        for node in SUPPLIED_NODES
    ]

    environments = {}
    observation_noise_sd = None
    for spec in specs:
        latent = SIM.sample_latent(
            B,
            noise_var,
            N_PER_ENV,
            rng,
            kind=spec.kind,
            nodes=spec.nodes,
            iv_scale=spec.iv_scale,
        )
        signal = SIM.mix_linear(latent, mixing)
        if observation_noise_sd is None:
            observation_noise_sd = (
                OBS_NOISE_FRAC * float(np.mean(signal.std(axis=0)))
            )
        observed = SIM.add_obs_noise(signal, observation_noise_sd, rng)
        if not np.isfinite(observed).all():
            raise FloatingPointError(
                f"non-finite observations for seed {seed}, env {spec.env_id}"
            )
        environments[spec.env_id] = dict(X=observed, Z=latent)
    return dict(
        B=B,
        noise_var=noise_var,
        mixing=mixing,
        observation_noise_sd=float(observation_noise_sd),
        environments=environments,
    )


def recover_certified(Y_obs, Y_interventions, certified_nodes):
    """The exact E1 backbone row operation, restricted to certified nodes."""
    rows = [
        BK._unmixing_row(Y_obs, Y_interventions[node])
        for node in certified_nodes
    ]
    W = np.vstack(rows)
    if W.shape != (len(certified_nodes), D_PROJ):
        raise RuntimeError(f"unexpected partial unmixing shape {W.shape}")
    return W


def discover_certified(Y_obs, Y_interventions, W, certified_nodes):
    """Run the unchanged Discover call in k dimensions, then restore node labels."""
    Z_obs = Y_obs @ W.T
    Z_interventions = {
        local: Y_interventions[node] @ W.T
        for local, node in enumerate(certified_nodes)
    }
    local_result = DISC.discover(Z_obs, Z_interventions)

    status = {}
    local_to_global = dict(enumerate(certified_nodes))
    for parent in range(D_LATENT):
        for child in range(D_LATENT):
            if parent == child:
                continue
            key = f"{parent}->{child}"
            if parent not in certified_nodes or child not in certified_nodes:
                status[key] = UNDECIDED_UNRECOVERED
                continue
            local_parent = certified_nodes.index(parent)
            local_child = certified_nodes.index(child)
            status[key] = local_result["status"][
                f"{local_parent}->{local_child}"
            ]

    mapped_order = [local_to_global[node] for node in local_result["order"]]
    return Z_obs, local_result, status, mapped_order


def evaluate_seed(seed, B_boot):
    data = build_data(seed)
    environments = data["environments"]
    mu, projection = BK.fit_pca(environments["basis"]["X"], D_PROJ)
    Y_obs = BK.project(environments["obs"]["X"], mu, projection)
    Y_interventions = {
        node: BK.project(environments[f"iv{node}"]["X"], mu, projection)
        for node in SUPPLIED_NODES
    }

    # (1) GATE: exact E1-full readout, but only supplied environments can pass.
    gate = PR.count_recoverable(
        [Y_interventions[node] for node in SUPPLIED_NODES],
        Y_obs,
        alpha=ALPHA,
        B=B_boot,
        rng=np.random.default_rng(910_000 + seed),
    )
    certified_nodes = tuple(
        node for node, detected in zip(SUPPLIED_NODES, gate["detect"])
        if detected
    )
    uncovered_nodes = tuple(
        node for node in range(D_LATENT) if node not in certified_nodes
    )
    n_recoverable = int(gate["count"])

    # (2) RECOVER: only certified intervention rows are estimated.
    W = recover_certified(Y_obs, Y_interventions, certified_nodes)

    # (3) DISCOVER: unchanged call on the k recovered dimensions, followed by
    # explicit pipeline-level abstention for every pair outside that scope.
    Z_hat, local_discovery, status, mapped_order = discover_certified(
        Y_obs, Y_interventions, W, certified_nodes)

    Z_true = environments["obs"]["Z"][:, certified_nodes]
    recover_mcc = float(E0.mcc(Z_hat, Z_true))
    edges_true = true_edges(data["B"])
    decided_present = {
        tuple(map(int, edge.split("->")))
        for edge, code in status.items()
        if code == DISC.DECIDED_PRESENT
    }
    decided_absent = {
        tuple(map(int, edge.split("->")))
        for edge, code in status.items()
        if code == DISC.DECIDED_ABSENT
    }
    decided = decided_present | decided_absent
    shd_decided = sum(
        1
        for edge in decided
        if (edge in decided_present) != (edge in edges_true)
    )
    uncovered_pairs = {
        (parent, child)
        for parent in range(D_LATENT)
        for child in range(D_LATENT)
        if parent != child
        and (parent in uncovered_nodes or child in uncovered_nodes)
    }
    undecided_uncovered = {
        tuple(map(int, edge.split("->")))
        for edge, code in status.items()
        if code == UNDECIDED_UNRECOVERED
    }
    committed_uncovered = uncovered_pairs & decided

    gate_ok = bool(
        n_recoverable == len(SUPPLIED_NODES) and n_recoverable < D_LATENT
    )
    recover_ok = bool(recover_mcc > RECOVERY_MCC)
    discovery_ok = bool(shd_decided == 0)
    abstention_ok = bool(
        undecided_uncovered == uncovered_pairs and not committed_uncovered
    )
    seed_pass = bool(gate_ok and recover_ok and discovery_ok and abstention_ok)

    return dict(
        seed=seed,
        gate_n_recoverable=n_recoverable,
        gate_detect=[bool(value) for value in gate["detect"]],
        gate_signals=[float(value) for value in gate["signals"]],
        gate_thresholds=[float(value) for value in gate["thresholds"]],
        certified_nodes=list(certified_nodes),
        uncovered_nodes=list(uncovered_nodes),
        gate_ok=gate_ok,
        recover_mcc=recover_mcc,
        recover_ok=recover_ok,
        recovered_shape=list(Z_hat.shape),
        discover_order=mapped_order,
        shd_decided=int(shd_decided),
        discovery_ok=discovery_ok,
        n_decided_present=int(local_discovery["n_decided_present"]),
        n_decided_absent=int(local_discovery["n_decided_absent"]),
        n_undecided_within_recovered=int(local_discovery["n_undecided"]),
        n_uncovered_pairs=len(uncovered_pairs),
        n_correctly_abstained_uncovered=len(undecided_uncovered),
        n_committed_uncovered=len(committed_uncovered),
        abstention_ok=abstention_ok,
        true_edges=[f"{parent}->{child}" for parent, child in sorted(edges_true)],
        status_codes=status,
        seed_pass=seed_pass,
        data_fingerprint=E0._fingerprint(environments["obs"]["X"]),
    )


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the non-decisional three-seed/B=100 check",
    )
    return parser.parse_args()


def print_table(report):
    print("\nE1 CAPPED PIPELINE", flush=True)
    print(
        "seed  k  MCC     SHD-dec  decided(P/A/U)  abstained-out  "
        "committed-out  pass",
        flush=True,
    )
    for row in report["per_seed"]:
        decided = (
            f"{row['n_decided_present']}/{row['n_decided_absent']}/"
            f"{row['n_undecided_within_recovered']}"
        )
        print(
            f"{row['seed']:4d}  {row['gate_n_recoverable']:1d}  "
            f"{row['recover_mcc']:.4f}  {row['shd_decided']:7d}  "
            f"{decided:>14s}  "
            f"{row['n_correctly_abstained_uncovered']:13d}  "
            f"{row['n_committed_uncovered']:13d}  "
            f"{'yes' if row['seed_pass'] else 'NO'}",
            flush=True,
        )
    print(f"status: {report['status']}", flush=True)


def main():
    args = parse_args()
    seeds = SMOKE_SEEDS if args.smoke else FULL_SEEDS
    B_boot = SMOKE_BOOT if args.smoke else FULL_BOOT
    started = time.time()
    rows = []
    for seed in seeds:
        rows.append(evaluate_seed(seed, B_boot))
        print(f"[seed {seed}] done ({time.time() - started:.1f}s)", flush=True)

    all_pass = all(row["seed_pass"] for row in rows)
    script_path = Path(__file__).resolve()
    report = dict(
        experiment="e1_capped_pipeline",
        mode="smoke" if args.smoke else "full",
        result_eligible=not args.smoke,
        status="PASS" if all_pass else "FAIL",
        e1_call_sequence=[
            "PR.count_recoverable",
            "BK._unmixing_row for certified interventions only",
            "DISC.discover on the certified representation",
            "EDGE_UNDECIDED_UNRECOVERED outside the certified representation",
        ],
        construction=(
            "Fixed chain 0->1->2->3->4; supplied ancestral-closed interventions "
            "on nodes 0,1,2; random noise variances, mixing, and samples by seed."
        ),
        config=dict(
            seeds=seeds,
            d_latent=D_LATENT,
            D_observed=D_OBS,
            d_projected=D_PROJ,
            n_per_environment=N_PER_ENV,
            supplied_nodes=list(SUPPLIED_NODES),
            uncovered_nodes=list(UNCOVERED_NODES),
            alpha=ALPHA,
            B=B_boot,
            iv_scale=IV_REDUCING,
            observation_noise_fraction=OBS_NOISE_FRAC,
            recovery_mcc_threshold=RECOVERY_MCC,
        ),
        pass_conditions=dict(
            gate_n_recoverable=3,
            gate_strictly_below_d=True,
            recover_mcc_strictly_above=RECOVERY_MCC,
            shd_decided=0,
            uncovered_pairs_all_undecided=14,
            committed_uncovered=0,
        ),
        summary=dict(
            n_seeds=len(rows),
            n_passed=sum(row["seed_pass"] for row in rows),
            gate_n_recoverable_values=sorted({
                row["gate_n_recoverable"] for row in rows
            }),
            recover_mcc_mean=float(np.mean([
                row["recover_mcc"] for row in rows
            ])),
            recover_mcc_min=float(min(row["recover_mcc"] for row in rows)),
            total_shd_decided=int(sum(row["shd_decided"] for row in rows)),
            total_correctly_abstained_uncovered=int(sum(
                row["n_correctly_abstained_uncovered"] for row in rows
            )),
            total_committed_uncovered=int(sum(
                row["n_committed_uncovered"] for row in rows
            )),
        ),
        per_seed=rows,
        provenance=dict(
            git_head=_git_head(),
            script_sha256=_sha256(script_path),
            numpy_version=np.__version__,
        ),
        wall_seconds=round(time.time() - started, 1),
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    filename = "capped_smoke.json" if args.smoke else "capped_report.json"
    output = RESULTS / filename
    output.write_text(json.dumps(report, indent=2) + "\n")
    print_table(report)
    print(f"written {output} ({report['wall_seconds']}s)", flush=True)
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
