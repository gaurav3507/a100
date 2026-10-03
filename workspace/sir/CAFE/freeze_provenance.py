#!/usr/bin/env python3
"""
CAFE Phase 0 - Repository and Environment Freeze (Gate 0 artifact).

Records everything needed to make any later result byte-traceable to one
commit, one environment, and one set of checkpoint hashes. Run this ONCE
after the environment is built and checkpoints are in place, BEFORE any
local code modification.

Outputs (written to --out, default ./paper_artifacts/provenance):
  provenance.json      full machine-readable record
  requirements.lock    exact pip freeze at freeze time
  provenance.txt       human-readable summary

Design notes:
  - Degrades gracefully with no GPU / no torch (records "unavailable"),
    so it can be dry-run on a login node or laptop.
  - No third-party imports required; standard library only.
  - --selftest runs a CPU-only self-check and exits non-zero on failure.
"""

import argparse, hashlib, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path

# Checkpoint files this project depends on. Paths are relative to repo root.
EXPECTED_CHECKPOINTS = [
    "checkpoints",            # C-MET connector checkpoint(s)
    "pretrained_weights",     # EDTalk.pt, EDTalk Audio2Lip.pt
    "data_preprocess/shape_predictor_68_face_landmarks.dat",
]

# Python packages whose versions materially affect reproduction.
KEY_PACKAGES = [
    "torch", "torchvision", "torchaudio", "transformers", "funasr",
    "audiocraft", "basicsr", "gfpgan", "timm", "face_alignment",
    "moviepy", "librosa", "numba", "numpy", "opencv-python", "spacy",
    "torchdiffeq", "deepfilternet", "gradio", "tensorboard",
]


def _run(cmd):
    """Run a shell command, return stripped stdout or None on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hash_target(root, rel):
    """Hash a file, or every file under a dir, relative to root. Returns list."""
    base = Path(root) / rel
    records = []
    if base.is_file():
        records.append({"path": rel, "sha256": sha256_file(base),
                        "bytes": base.stat().st_size})
    elif base.is_dir():
        for p in sorted(base.rglob("*")):
            if p.is_file():
                records.append({"path": str(p.relative_to(root)),
                                "sha256": sha256_file(p),
                                "bytes": p.stat().st_size})
    else:
        records.append({"path": rel, "sha256": None, "bytes": None,
                        "note": "MISSING at freeze time"})
    return records


def git_info(root):
    def g(*a):
        return _run(["git", "-C", str(root), *a])
    return {
        "commit": g("rev-parse", "HEAD"),
        "short": g("rev-parse", "--short", "HEAD"),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(g("status", "--porcelain")),
        "remote": g("config", "--get", "remote.origin.url"),
        "last_commit_date": g("log", "-1", "--format=%cI"),
    }


def package_versions():
    versions = {}
    try:
        from importlib import metadata as im
        for name in KEY_PACKAGES:
            try:
                versions[name] = im.version(name)
            except Exception:
                versions[name] = None
    except Exception:
        for name in KEY_PACKAGES:
            versions[name] = None
    return versions


def torch_gpu_info():
    info = {"torch_available": False}
    try:
        import torch
        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        info["torch_cuda_build"] = getattr(torch.version, "cuda", None)
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["cuda_runtime_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["devices"] = []
            for i in range(torch.cuda.device_count()):
                cap = torch.cuda.get_device_capability(i)
                info["devices"].append({
                    "name": torch.cuda.get_device_name(i),
                    "capability": f"sm_{cap[0]}{cap[1]}",
                    "total_mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1),
                })
    except Exception as e:
        info["error"] = repr(e)
    return info


def nvidia_smi():
    raw = _run(["nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader"])
    smi_cuda = None
    full = _run(["nvidia-smi"])
    if full:
        for line in full.splitlines():
            if "CUDA Version" in line:
                smi_cuda = line.split("CUDA Version:")[-1].strip().strip("|").strip()
                break
    return {"query": raw, "driver_cuda": smi_cuda}


def pip_freeze():
    out = _run([sys.executable, "-m", "pip", "freeze"])
    return out.splitlines() if out else []


def build_record(root):
    return {
        "schema": "cafe.phase0.provenance/1",
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(Path(root).resolve()),
        "host": {
            "hostname": platform.node(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "python_exe": sys.executable,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "git": git_info(root),
        "gpu": {"torch": torch_gpu_info(), "nvidia_smi": nvidia_smi()},
        "packages": package_versions(),
        "checkpoints": {rel: hash_target(root, rel) for rel in EXPECTED_CHECKPOINTS},
    }


def write_outputs(record, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").write_text(json.dumps(record, indent=2))
    (out / "requirements.lock").write_text("\n".join(pip_freeze()) + "\n")

    lines = []
    g = record["git"]
    lines.append("CAFE Phase 0 provenance")
    lines.append("=" * 60)
    lines.append(f"frozen_at : {record['frozen_at_utc']}")
    lines.append(f"host      : {record['host']['hostname']}  env={record['host']['conda_env']}")
    lines.append(f"python    : {record['host']['python']}")
    lines.append(f"git       : {g.get('short')} ({g.get('branch')})  dirty={g.get('dirty')}")
    lines.append(f"remote    : {g.get('remote')}")
    tg = record["gpu"]["torch"]
    if tg.get("torch_available"):
        lines.append(f"torch     : {tg.get('torch_version')}  cuda_build={tg.get('torch_cuda_build')}")
        for d in tg.get("devices", []) or []:
            lines.append(f"  gpu     : {d['name']} {d['capability']} {d['total_mem_gb']}GB")
    else:
        lines.append("torch     : unavailable at freeze time")
    lines.append(f"smi       : {record['gpu']['nvidia_smi'].get('query')}")
    lines.append("-" * 60)
    missing = []
    for rel, recs in record["checkpoints"].items():
        for r in recs:
            tag = "MISSING" if r.get("sha256") is None else r["sha256"][:12]
            if r.get("sha256") is None:
                missing.append(r["path"])
            lines.append(f"ckpt {tag}  {r['path']}")
    (out / "provenance.txt").write_text("\n".join(lines) + "\n")
    return missing


def selftest():
    """CPU-only sanity check. Creates a temp git repo + fake checkpoint."""
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # fake checkpoint files
        (td / "checkpoints").mkdir()
        (td / "checkpoints" / "model.pth").write_bytes(b"weights-bytes")
        (td / "pretrained_weights").mkdir()
        (td / "pretrained_weights" / "EDTalk.pt").write_bytes(b"edtalk")
        (td / "data_preprocess").mkdir()
        (td / "data_preprocess" / "shape_predictor_68_face_landmarks.dat").write_bytes(b"dat")
        # init a git repo so git_info returns a commit
        if shutil.which("git"):
            for cmd in (["init", "-q"],
                        ["config", "user.email", "t@t"],
                        ["config", "user.name", "t"],
                        ["add", "-A"],
                        ["commit", "-qm", "init"]):
                subprocess.run(["git", "-C", str(td), *cmd], capture_output=True)
        rec = build_record(td)
        # assertions
        h = rec["checkpoints"]["checkpoints"][0]["sha256"]
        assert h == hashlib.sha256(b"weights-bytes").hexdigest(), "hash mismatch"
        assert rec["schema"].startswith("cafe.phase0"), "schema tag wrong"
        assert "packages" in rec and isinstance(rec["packages"], dict)
        missing = write_outputs(rec, td / "prov")
        assert (td / "prov" / "provenance.json").exists(), "json not written"
        assert (td / "prov" / "provenance.txt").exists(), "txt not written"
        assert missing == [], f"unexpected missing: {missing}"
        if shutil.which("git"):
            assert rec["git"]["commit"], "git commit not captured"
        print("selftest OK: hashing, git capture, output writing, missing-detection all pass")
    return ok


def main():
    ap = argparse.ArgumentParser(description="CAFE Phase 0 environment/repo freeze")
    ap.add_argument("--root", default=".", help="C-MET repo root (default: cwd)")
    ap.add_argument("--out", default="./paper_artifacts/provenance", help="output dir")
    ap.add_argument("--selftest", action="store_true", help="run CPU self-test and exit")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    record = build_record(args.root)
    missing = write_outputs(record, args.out)
    print(f"[freeze] wrote provenance to {args.out}")
    if missing:
        print(f"[freeze] WARNING: {len(missing)} expected checkpoint path(s) missing:")
        for m in missing:
            print(f"           - {m}")
        print("[freeze] Gate 0 is NOT satisfiable until these are present.")
        sys.exit(2)
    print("[freeze] all expected checkpoints present and hashed.")


if __name__ == "__main__":
    main()
