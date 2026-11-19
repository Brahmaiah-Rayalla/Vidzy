"""Build a compact, checksummed CPU inference bundle for the current model route."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "release"
BUNDLE_PATH = RELEASE_DIR / "vidzy_cpu_raw_triangle_bundle.zip"

SOURCE_CHECKPOINTS = {
    "checkpoints/cpu_triangle_raw_rgb_v90_native_centered_lr5e5/checkpoint_final.pt": (
        "checkpoints/cpu_triangle_raw_rgb_v90_native_centered_lr5e5/checkpoint_final.pt",
    ),
    "checkpoints/cpu_triangle_small_raw_rgb_v91_area_edge_refine/checkpoint_final.pt": (
        "checkpoints/cpu_triangle_small_raw_rgb_v91_area_edge_refine/checkpoint_final.pt",
    ),
    "checkpoints/cpu_triangle_small_raw_rgb_v92_area_calibration/checkpoint_final.pt": (
        "checkpoints/cpu_triangle_small_raw_rgb_v92_area_calibration/checkpoint_final.pt",
    ),
    "checkpoints/cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold/checkpoint_final.pt": (
        "checkpoints/cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold/checkpoint_final.pt",
    ),
    "checkpoints/cpu_triangle_large_raw_rgb_v86_128_dilated_refine/checkpoint_final.pt": (
        "checkpoints/cpu_triangle_large_raw_rgb_v86_128_dilated_refine/checkpoint_final.pt",
    ),
    "outputs/cpu_triangle_raw_rgb_v90_native_centered_lr5e5/video_v37_motion_v90_spatial_composite.pt": (
        "checkpoints/cpu_video_v37_motion_v90_spatial_composite.pt",
    ),
}

CONFIGS = (
    "configs/cpu_triangle_raw_rgb_v90_native_antialias_lr5e5.yaml",
    "configs/cpu_triangle_small_raw_rgb_v91_area_edge_refine.yaml",
    "configs/cpu_triangle_small_raw_rgb_v92_area_calibration.yaml",
    "configs/cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold.yaml",
)

EXAMPLES = {
    "outputs/cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold/routed_small_raw_samples/grid.png": (
        "examples/small_triangle_raw_diffusion_grid.png",
    ),
    "outputs/cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold/raw_triangle_small_6s_384.mp4": (
        "examples/small_triangle_raw_anchor_6s.mp4",
    ),
    "outputs/cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold/raw_triangle_small_6s_384_preview.png": (
        "examples/small_triangle_raw_anchor_preview.png",
    ),
    "outputs/cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold/raw_triangle_small_6s_384.metrics.json": (
        "examples/small_triangle_raw_anchor_metrics.json",
    ),
    "outputs/cpu_triangle_raw_rgb_v90_native_centered_lr5e5/raw_triangle_large_6s_384.mp4": (
        "examples/large_triangle_raw_anchor_6s.mp4",
    ),
    "outputs/cpu_triangle_raw_rgb_v90_native_centered_lr5e5/raw_triangle_large_6s_384_preview.png": (
        "examples/large_triangle_raw_anchor_preview.png",
    ),
    "outputs/cpu_triangle_raw_rgb_v90_native_centered_lr5e5/raw_triangle_large_6s_384.metrics.json": (
        "examples/large_triangle_raw_anchor_metrics.json",
    ),
}

V90_EVAL = ROOT / "outputs/cpu_triangle_raw_rgb_v90_native_centered_lr5e5/raw_eval_12seed/paired_generation_metrics.json"
V91_EVAL = ROOT / "outputs/cpu_triangle_small_raw_rgb_v91_area_edge_refine/paired_eval_v90_to_v91_12seed/paired_generation_metrics.json"
V92_EVAL = ROOT / "outputs/cpu_triangle_small_raw_rgb_v92_area_calibration/paired_eval_v91_to_v92_12seed/paired_generation_metrics.json"
V93_EVAL = ROOT / "outputs/cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold/paired_eval_v92_to_v93_12seed/paired_generation_metrics.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(relative_path: str) -> Path:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required bundle input is missing: {path}")
    return path


def compact_checkpoint(source: Path, destination: Path) -> dict:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload or "config" not in payload:
        raise ValueError(f"Not a supported model checkpoint: {source}")
    compact = {
        key: payload[key]
        for key in ("model_state_dict", "raw_model_state_dict", "step", "config", "seed")
        if key in payload
    }
    if not compact["model_state_dict"]:
        raise ValueError(f"Checkpoint has an empty model state: {source}")
    torch.save(compact, destination)
    return {
        "source_sha256": sha256_file(source),
        "packaged_sha256": sha256_file(destination),
        "step": payload.get("step"),
        "config": payload["config"],
        "state_tensor_count": len(compact["model_state_dict"]),
        "optimizer_state_included": False,
    }


def read_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload["summary_by_checkpoint_and_size"]["size_conditioned"]


def build_routed_metrics(checkpoint_info: dict) -> dict:
    v90 = read_summary(V90_EVAL)
    v91 = read_summary(V91_EVAL)
    v92 = read_summary(V92_EVAL)
    v93 = read_summary(V93_EVAL)
    route = {
        "small": ("v93_small_specialist", v93["small"]),
        "medium": ("v90_generalist", v90["medium"]),
        "large": ("v90_generalist", v90["large"]),
    }
    metric_names = (
        "centered_triangle_iou",
        "raw_all_components_iou_unaligned",
        "boundary_f1_1px",
        "foreground_area_ratio",
    )
    sizes = {}
    for size, (checkpoint_name, summary) in route.items():
        sizes[size] = {
            "checkpoint_route": checkpoint_name,
            "paired_seed_count": 12,
            "metrics": {
                name: summary[name]
                for name in metric_names
            },
        }
    with V93_EVAL.open("r", encoding="utf-8") as stream:
        paired_v93 = json.load(stream)
    paired_rows = paired_v93["metrics_by_sample"]
    v92_small = {
        row["seed"]: row for row in paired_rows
        if row["checkpoint"] == "baseline" and row["size"] == "small"
    }
    v93_small = {
        row["seed"]: row for row in paired_rows
        if row["checkpoint"] == "size_conditioned" and row["size"] == "small"
    }
    comparison_metrics = metric_names
    win_counts = {}
    for metric in comparison_metrics:
        if metric == "foreground_area_ratio":
            win_counts[metric] = sum(
                abs(v93_small[seed][metric] - 1.0) < abs(v92_small[seed][metric] - 1.0)
                for seed in v92_small
            )
        else:
            win_counts[metric] = sum(
                v93_small[seed][metric] > v92_small[seed][metric]
                for seed in v92_small
            )
    return {
        "task": "raw_rgb_triangle_diffusion",
        "evaluation_resolution": 128,
        "evaluation_condition": "blue triangle, 12 fixed paired generation seeds",
        "route": {
            "small": "v93 small specialist",
            "medium": "v90 generalist",
            "large": "v90 generalist",
        },
        "checkpoints": checkpoint_info,
        "size_metrics": sizes,
        "small_triangle_checkpoint_history": {
            name: {metric: summary[metric] for metric in metric_names}
            for name, summary in (
                ("v90", v90["small"]), ("v91", v91["small"]),
                ("v92", v92["small"]), ("v93", v93["small"]),
            )
        },
        "v93_paired_comparison_to_v92_small": {
            "seed_count": len(v92_small),
            "v92_metrics": {name: v92["small"][name] for name in metric_names},
            "v93_metrics": {name: v93["small"][name] for name in metric_names},
            "seeds_won_by_v93": win_counts,
        },
        "notes": [
            "Centered IoU is center-aligned; raw IoU retains all components and does not align position.",
            "Boundary F1 is measured at a one-pixel tolerance.",
            "The v93 specialist was evaluated as a small-triangle route; do not use it for medium or large sizes.",
            "v93 was paired against v92 on the same 12 seeds: area closeness improved on 11/12 seeds, centered IoU on 10/12, boundary F1 on 11/12, and raw unaligned IoU on 5/12.",
            "Small-triangle area is about 1.00 times target on average, but raw position can still vary by seed.",
            "This spatial diffusion model supplies one anchor image; six-second video motion is rendered by the separate v37 CPU motion checkpoint.",
        ],
    }


def main() -> None:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_info = {}
    with tempfile.TemporaryDirectory(prefix="vidzy_bundle_", dir=RELEASE_DIR) as temp_name:
        staging = Path(temp_name)

        for source_relative, (bundle_relative,) in SOURCE_CHECKPOINTS.items():
            source = require_file(source_relative)
            destination = staging / bundle_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_info[bundle_relative] = compact_checkpoint(source, destination)

        for relative in CONFIGS:
            source = require_file(relative)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        for source_relative, (bundle_relative,) in EXAMPLES.items():
            source = require_file(source_relative)
            destination = staging / bundle_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        (staging / "README_BUNDLE.md").write_text(
            "# Vidzy CPU model bundle quick start\n\n"
            "This archive contains model weights and sample outputs, not the Python source code. "
            "Download the matching Vidzy source repository, put this ZIP in its `release/` "
            "directory, and run commands from the source repository root.\n\n"
            "Verify and extract the archive:\n\n"
            "```powershell\n"
            "python tools\\verify_cpu_model_bundle.py\n"
            "Expand-Archive -LiteralPath .\\release\\vidzy_cpu_raw_triangle_bundle.zip -DestinationPath . -Force\n"
            "```\n\n"
            "Install dependencies as described in the repository `README.md`, then generate "
            "a raw small-triangle image:\n\n"
            "```powershell\n"
            "python -m video_foundation.inference.sample_image `\n"
            "  --checkpoint checkpoints\\cpu_triangle_raw_rgb_v90_native_centered_lr5e5\\checkpoint_final.pt `\n"
            "  --small_triangle_checkpoint checkpoints\\cpu_triangle_small_raw_rgb_v93_eval_aligned_threshold\\checkpoint_final.pt `\n"
            "  --medium_large_checkpoint checkpoints\\cpu_triangle_raw_rgb_v90_native_centered_lr5e5\\checkpoint_final.pt `\n"
            "  --output_dir outputs\\bundle_quickstart `\n"
            "  --num_samples 4 --sampling_steps 50 --seed 5678 `\n"
            "  --class_label 2 --size small --vary_colors --raw_generation --device cpu\n"
            "```\n\n"
            "The six-second example uses `checkpoints\\cpu_video_v37_motion_v90_spatial_composite.pt` "
            "for motion, v93 for small spatial anchors, and v90 for medium/large anchors. See the "
            "source repository README for the full video command, training recipes, evaluation "
            "commands, and current limitations.\n",
            encoding="utf-8",
        )

        provenance = staging / "provenance"
        provenance.mkdir(parents=True, exist_ok=True)
        (provenance / "routed_metrics.json").write_text(
            json.dumps(build_routed_metrics(checkpoint_info), indent=2) + "\n",
            encoding="utf-8",
        )
        (provenance / "README.md").write_text(
            "# Model bundle provenance\n\n"
            "The routed metrics use the recorded fixed-seed evaluations for v90, v91, v92, and v93. "
            "Checkpoint hashes below distinguish the original training files from the compact "
            "inference files in this archive. The compact files preserve the model weights, "
            "configuration, step, and seed while omitting optimizer state.\n\n"
            "See `routed_metrics.json` for per-size means and standard deviations. Re-run the "
            "evaluation commands in the repository README to create a fresh evaluation on "
            "another machine.\n",
            encoding="utf-8",
        )

        file_records = []
        for path in sorted(p for p in staging.rglob("*") if p.is_file()):
            file_records.append({
                "path": path.relative_to(staging).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        bundle_manifest = {
            "bundle": "vidzy_cpu_raw_triangle_bundle",
            "created_from_workspace": True,
            "weights_format": "PyTorch checkpoint; inference and fine-tuning state only",
            "files": file_records,
        }
        (staging / "model_bundle_manifest.json").write_text(
            json.dumps(bundle_manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        temporary_zip = RELEASE_DIR / "vidzy_cpu_raw_triangle_bundle.tmp.zip"
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(p for p in staging.rglob("*") if p.is_file()):
                archive.write(path, path.relative_to(staging).as_posix())
        temporary_zip.replace(BUNDLE_PATH)

    archive_manifest = {
        "bundle": BUNDLE_PATH.name,
        "size_bytes": BUNDLE_PATH.stat().st_size,
        "sha256": sha256_file(BUNDLE_PATH),
    }
    (RELEASE_DIR / "vidzy_cpu_raw_triangle_bundle.sha256").write_text(
        f"{archive_manifest['sha256']}  {BUNDLE_PATH.name}\n",
        encoding="ascii",
    )
    (RELEASE_DIR / "vidzy_cpu_raw_triangle_bundle_manifest.json").write_text(
        json.dumps(archive_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(archive_manifest, indent=2))


if __name__ == "__main__":
    main()
