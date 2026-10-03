"""Write SHA-256 provenance for every external input used by E3/iLCS."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from data_paths import real_data_inputs


REPO = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return dict(
        kind="file",
        resolved_path=str(path),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def directory_record(path):
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(item for item in path.rglob("*.npy") if item.is_file())
    if not files:
        raise FileNotFoundError(f"no .npy inputs under {path}")
    tree = hashlib.sha256()
    total_bytes = 0
    for item in files:
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        digest = sha256_file(item)
        tree.update(f"{relative}\t{size}\t{digest}\n".encode())
        total_bytes += size
    return dict(
        kind="npy_tree",
        resolved_path=str(path),
        file_count=len(files),
        size_bytes=total_bytes,
        manifest_sha256=tree.hexdigest(),
        manifest_definition=(
            "SHA-256 over sorted lines: relative_path<TAB>size<TAB>file_sha256"
        ),
    )


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results" / "repro" / "data_manifest.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = {}
    for name, path in real_data_inputs().items():
        records[name] = directory_record(path) if path.is_dir() else file_record(path)
        print(f"hashed {name}: {records[name]['resolved_path']}", flush=True)
    report = dict(
        schema="grd_external_data_manifest_v1",
        code_commit=git_head(),
        inputs=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"written {args.output}", flush=True)


if __name__ == "__main__":
    main()
