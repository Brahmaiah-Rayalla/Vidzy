"""Create native-resolution, antialiased triangle data for raw RGB diffusion."""

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw


PALETTE = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
EXTENTS_64 = (15, 29, 42)
SIZE_NAMES = ("small", "medium", "large")


def render_triangle(image_size, extent, color_id, rng, supersample, center_jitter, angle_jitter):
    """Rasterize a subpixel-shifted triangle at high resolution, then downsample once."""
    scale = int(supersample)
    high_size = int(image_size) * scale
    center_x = (image_size - 1) / 2.0 + rng.uniform(-center_jitter, center_jitter)
    center_y = (image_size - 1) / 2.0 + rng.uniform(-center_jitter, center_jitter)
    angle = math.radians(rng.uniform(-angle_jitter, angle_jitter)) - math.pi / 2.0

    # Equilateral vertices, expressed around their centroid. Re-normalize the
    # rotated bounding box so each size label retains the same maximum extent.
    radius = float(extent) / math.sqrt(3.0)
    vertices = [
        (radius * math.cos(angle + 2.0 * math.pi * index / 3.0),
         radius * math.sin(angle + 2.0 * math.pi * index / 3.0))
        for index in range(3)
    ]
    max_extent = max(
        max(x for x, _ in vertices) - min(x for x, _ in vertices),
        max(y for _, y in vertices) - min(y for _, y in vertices),
    )
    correction = float(extent) / max_extent
    high_vertices = [
        (((center_x + x * correction) + 0.5) * scale - 0.5,
         ((center_y + y * correction) + 0.5) * scale - 0.5)
        for x, y in vertices
    ]

    high_image = Image.new("RGB", (high_size, high_size), (0, 0, 0))
    ImageDraw.Draw(high_image).polygon(high_vertices, fill=PALETTE[color_id])
    return high_image.resize((image_size, image_size), Image.Resampling.LANCZOS)


def write_split(output_dir, split, image_size, train_per_combo, val_per_combo,
                seed, supersample, center_jitter, angle_jitter):
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    count_per_combo = train_per_combo if split == "train" else val_per_combo
    rng = random.Random(seed + (0 if split == "train" else 1_000_003))
    shape_labels, color_labels, size_labels = [], [], []
    preview = Image.new("RGB", (image_size * 4, image_size * 3), (32, 32, 32))
    preview_written = set()
    index = 0

    for size_id, extent_64 in enumerate(EXTENTS_64):
        extent = max(1, round(extent_64 * image_size / 64.0))
        for color_id in range(len(PALETTE)):
            for local_index in range(count_per_combo):
                image = render_triangle(
                    image_size=image_size,
                    extent=extent,
                    color_id=color_id,
                    rng=rng,
                    supersample=supersample,
                    center_jitter=center_jitter,
                    angle_jitter=angle_jitter,
                )
                image.save(split_dir / f"image_{index:06d}.png")
                shape_labels.append(2)  # triangle
                color_labels.append(color_id)
                size_labels.append(size_id)
                if local_index == 0:
                    preview.paste(image, (color_id * image_size, size_id * image_size))
                    preview_written.add((size_id, color_id))
                index += 1

    for name, labels in (("class_labels.json", shape_labels),
                         ("color_labels.json", color_labels),
                         ("size_labels.json", size_labels)):
        (split_dir / name).write_text(json.dumps(labels), encoding="utf-8")
    preview.save(split_dir / "preview_grid.png")
    if len(preview_written) != 12:
        raise RuntimeError("Native triangle preview is missing a condition combination")
    return len(shape_labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--train_per_combo", type=int, default=256)
    parser.add_argument("--val_per_combo", type=int, default=64)
    parser.add_argument("--seed", type=int, default=88001)
    parser.add_argument("--supersample", type=int, default=8)
    parser.add_argument(
        "--center_jitter", type=float, default=0.0,
        help="Pixel position variation; keep zero for centered video appearance anchors",
    )
    parser.add_argument(
        "--angle_jitter", type=float, default=0.0,
        help="Degree variation; keep zero when the class label does not encode orientation",
    )
    args = parser.parse_args()
    if args.image_size < 64 or args.train_per_combo < 1 or args.val_per_combo < 1:
        parser.error("image_size must be >=64 and per-combination counts must be positive")
    if args.supersample < 2 or args.center_jitter < 0 or args.angle_jitter < 0:
        parser.error("supersample must be >=2 and jitter values must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in ("train", "val"):
        counts[split] = write_split(
            args.output_dir, split, args.image_size, args.train_per_combo,
            args.val_per_combo, args.seed, args.supersample,
            args.center_jitter, args.angle_jitter,
        )
    manifest = {
        "schema_version": 1,
        "dataset": "native_resolution_antialiased_triangles",
        "image_size": args.image_size,
        "split_counts": counts,
        "per_condition_counts": {
            "train": args.train_per_combo,
            "val": args.val_per_combo,
        },
        "condition_combinations": 12,
        "shape_id": 2,
        "colors_rgb": [list(color) for color in PALETTE],
        "size_extents_pixels": [
            max(1, round(extent * args.image_size / 64.0)) for extent in EXTENTS_64
        ],
        "seed": args.seed,
        "supersample": args.supersample,
        "center_jitter_pixels": args.center_jitter,
        "angle_jitter_degrees": args.angle_jitter,
        "background_rgb": [0, 0, 0],
        "rasterization": "high-resolution polygon coverage, single Lanczos downsample",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
