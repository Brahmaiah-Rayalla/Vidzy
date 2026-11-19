"""Verify the downloaded bundle hash and every file digest in its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("release/vidzy_cpu_raw_triangle_bundle.zip"),
        help="Path to the model bundle ZIP",
    )
    args = parser.parse_args()
    archive_path = args.archive
    if not archive_path.is_file():
        parser.error(f"bundle not found: {archive_path}")

    sidecar = archive_path.with_suffix(".sha256")
    if sidecar.exists():
        expected_archive_hash = sidecar.read_text(encoding="ascii").split()[0].lower()
        with archive_path.open("rb") as stream:
            actual_archive_hash = sha256_stream(stream)
        if actual_archive_hash != expected_archive_hash:
            raise SystemExit(
                f"Archive SHA-256 mismatch: expected {expected_archive_hash}, got {actual_archive_hash}"
            )

    with zipfile.ZipFile(archive_path, "r") as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise SystemExit(f"Corrupt ZIP member: {corrupt_member}")
        names = set(archive.namelist())
        manifest_name = "model_bundle_manifest.json"
        if manifest_name not in names:
            raise SystemExit(f"Bundle is missing {manifest_name}")
        manifest = json.loads(archive.read(manifest_name))
        errors = []
        for record in manifest.get("files", []):
            member = record["path"]
            if member not in names:
                errors.append(f"missing: {member}")
                continue
            with archive.open(member) as stream:
                actual = sha256_stream(stream)
            if actual != record["sha256"]:
                errors.append(f"hash mismatch: {member}")
        if errors:
            raise SystemExit("Bundle verification failed:\n" + "\n".join(errors))

    print(f"Verified {len(manifest['files'])} bundled files: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
