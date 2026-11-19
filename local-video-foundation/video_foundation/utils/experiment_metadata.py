"""Helpers for attaching reproducible provenance to CPU model evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> dict:
    """Hash file names and contents in a directory in stable path order."""
    root = Path(path).resolve()
    aggregate = hashlib.sha256()
    file_count = 0
    for file_path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        relative = file_path.relative_to(root).as_posix()
        file_digest = sha256_file(file_path)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(file_digest.encode("ascii"))
        aggregate.update(b"\n")
        file_count += 1
    return {
        "path": str(root),
        "file_count": file_count,
        "sha256": aggregate.hexdigest(),
    }


def _json_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_state(start_path: Path) -> dict:
    current = start_path.resolve()
    if current.is_file():
        current = current.parent
    while current != current.parent and not (current / ".git").exists():
        current = current.parent
    if not (current / ".git").exists():
        return {"revision": None, "tracked_worktree_dirty": None}

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=current, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=current,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "tracked_worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def build_evaluation_manifest(
    *,
    task: str,
    args: dict,
    checkpoint_paths: dict,
    checkpoint_payloads: dict,
    data_dir: str | Path,
    source_paths: list[str | Path],
    elapsed_seconds: float,
    torch_version: str,
    cpu_threads: int,
) -> dict:
    """Build an evaluation manifest; elapsed time excludes provenance hashing."""
    checkpoints = {}
    for label, path in checkpoint_paths.items():
        resolved = Path(path).resolve()
        payload = checkpoint_payloads[label]
        config = payload.get("config", {})
        checkpoints[label] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            "step": payload.get("step"),
            "config": config,
            "config_sha256": _json_sha256(config),
        }

    source_files = {}
    for path in source_paths:
        resolved = Path(path).resolve()
        source_files[str(resolved)] = sha256_file(resolved)

    return {
        "schema_version": 1,
        "task": task,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_elapsed_seconds_excluding_manifest_hashing": round(float(elapsed_seconds), 6),
        "device": "cpu",
        "cpu_threads": int(cpu_threads),
        "logical_cpu_count": os.cpu_count(),
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": str(torch_version),
        },
        "git": _git_state(Path(source_paths[0]) if source_paths else Path.cwd()),
        "arguments": args,
        "checkpoints": checkpoints,
        "dataset": sha256_directory(data_dir),
        "source_files_sha256": source_files,
    }


def build_generation_manifest(
    *,
    task: str,
    generation_mode: str,
    args: dict,
    checkpoint_path: str | Path,
    checkpoint_payload: dict,
    source_paths: list[str | Path],
    elapsed_seconds: float,
    torch_version: str,
    cpu_threads: int,
) -> dict:
    """Build provenance for a generated artifact that has no dataset input."""
    resolved_checkpoint = Path(checkpoint_path).resolve()
    config = checkpoint_payload.get("config", {})
    source_files = {}
    for path in source_paths:
        resolved = Path(path).resolve()
        source_files[str(resolved)] = sha256_file(resolved)
    return {
        "schema_version": 1,
        "task": task,
        "generation_mode": generation_mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds_excluding_manifest_hashing": round(float(elapsed_seconds), 6),
        "device": "cpu",
        "cpu_threads": int(cpu_threads),
        "logical_cpu_count": os.cpu_count(),
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": str(torch_version),
        },
        "git": _git_state(Path(source_paths[0]) if source_paths else Path.cwd()),
        "arguments": args,
        "input_dataset": None,
        "checkpoint": {
            "path": str(resolved_checkpoint),
            "sha256": sha256_file(resolved_checkpoint),
            "step": checkpoint_payload.get("step"),
            "config": config,
            "config_sha256": _json_sha256(config),
        },
        "source_files_sha256": source_files,
    }

