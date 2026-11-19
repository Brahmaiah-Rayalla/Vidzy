"""Canonicalize one shape/size condition in a prepared image dataset.

The size labels in the source curriculum cover broad intervals, while the
inference curriculum uses one fixed extent per label. This utility creates a
matched training split by resizing the foreground silhouette to the same
canonical extent used by inference and centering it on the image canvas.
"""

import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image


PALETTE = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
)
CANONICAL_EXTENTS_64 = (15, 29, 42)


def canonicalize_split(input_dir, output_dir, shape_class, size_class, image_size, all_sizes=False):
    with open(os.path.join(input_dir, "class_labels.json"), encoding="utf-8") as handle:
        shape_labels = json.load(handle)
    with open(os.path.join(input_dir, "color_labels.json"), encoding="utf-8") as handle:
        color_labels = json.load(handle)
    with open(os.path.join(input_dir, "size_labels.json"), encoding="utf-8") as handle:
        size_labels = json.load(handle)

    image_names = sorted(
        name for name in os.listdir(input_dir)
        if name.lower().endswith(".png") and name != "preview_grid.png"
    )
    if not (len(image_names) == len(shape_labels) == len(color_labels) == len(size_labels)):
        raise ValueError("Prepared image and label counts do not match")
    if not 0 <= size_class < len(CANONICAL_EXTENTS_64):
        raise ValueError(f"size_class must be in [0, {len(CANONICAL_EXTENTS_64) - 1}]")

    os.makedirs(output_dir, exist_ok=True)
    changed = 0
    for index, name in enumerate(image_names):
        source_path = os.path.join(input_dir, name)
        target_path = os.path.join(output_dir, name)
        selected_size = int(size_labels[index]) if all_sizes else size_class
        if shape_labels[index] != shape_class or size_labels[index] != selected_size:
            shutil.copy2(source_path, target_path)
            continue

        target_extent = max(1, round(CANONICAL_EXTENTS_64[selected_size] * image_size / 64))

        with Image.open(source_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        foreground = np.any(rgb != 0, axis=-1)
        ys, xs = np.where(foreground)
        if not len(xs):
            raise ValueError(f"No foreground found in {source_path}")
        crop = foreground[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        crop_h, crop_w = crop.shape
        scale = target_extent / max(crop_h, crop_w)
        new_h = max(1, min(target_extent, round(crop_h * scale)))
        new_w = max(1, min(target_extent, round(crop_w * scale)))
        resized = Image.fromarray(crop.astype(np.uint8) * 255).resize(
            (new_w, new_h), Image.Resampling.NEAREST
        )
        resized_mask = np.asarray(resized) > 0

        canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        # Center the bounding box using integer placement so every image with
        # this label has the same target extent and no interpolation fringe.
        x0 = (image_size - new_w) // 2
        y0 = (image_size - new_h) // 2
        color_id = int(color_labels[index])
        if not 0 <= color_id < len(PALETTE):
            raise ValueError(f"Invalid color class {color_id} in {source_path}")
        region = canvas[y0:y0 + new_h, x0:x0 + new_w]
        region[resized_mask] = PALETTE[color_id]
        Image.fromarray(canvas).save(target_path)
        changed += 1

    for filename in ("class_labels.json", "color_labels.json", "size_labels.json"):
        shutil.copy2(os.path.join(input_dir, filename), os.path.join(output_dir, filename))
    preview = os.path.join(input_dir, "preview_grid.png")
    if os.path.isfile(preview):
        shutil.copy2(preview, os.path.join(output_dir, "preview_grid.png"))
    print(
        f"Canonicalized {changed} shape={shape_class}, "
        f"sizes={'all' if all_sizes else size_class} images in {output_dir}"
    )
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shape_class", type=int, default=2)
    parser.add_argument("--size_class", type=int, default=2)
    parser.add_argument("--all_sizes", action="store_true",
                        help="Canonicalize the selected shape at every size label")
    parser.add_argument("--image_size", type=int, default=64)
    args = parser.parse_args()
    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("input_dir and output_dir must be different")
    canonicalize_split(
        args.input_dir, args.output_dir, args.shape_class, args.size_class,
        args.image_size, args.all_sizes,
    )


if __name__ == "__main__":
    main()
