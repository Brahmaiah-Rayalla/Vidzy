"""Extract centered, palette-normalized stills with shape/color/size labels."""

import argparse
import json
import os

import numpy as np
from PIL import Image


SHAPES = {"circle": 0, "square": 1, "triangle": 2}
COLORS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0)}
COLOR_IDS = {name: index for index, name in enumerate(COLORS)}
SOURCE_COLORS = {
    "red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0),
    "yellow": (255, 255, 0), "white": (255, 255, 255), "black": (0, 0, 0),
    "orange": (255, 165, 0), "purple": (128, 0, 128), "cyan": (0, 255, 255),
    "pink": (255, 192, 203),
}


def prepare_split(source_dir, output_dir, image_size):
    with open(os.path.join(source_dir, "all_metadata.json"), encoding="utf-8") as handle:
        metadata = json.load(handle)
    videos = sorted(
        os.path.join(source_dir, name) for name in os.listdir(source_dir)
        if name.startswith("video_") and os.path.isdir(os.path.join(source_dir, name))
    )
    if len(videos) != len(metadata):
        raise ValueError(f"Found {len(videos)} video folders for {len(metadata)} metadata rows")
    os.makedirs(output_dir, exist_ok=True)
    shape_labels, color_labels, size_labels = [], [], []
    for index, (directory, item) in enumerate(zip(videos, metadata)):
        paths = sorted(name for name in os.listdir(directory) if name.lower().endswith(".png"))
        if not paths:
            raise ValueError(f"No frame images in {directory}")
        with Image.open(os.path.join(directory, paths[0])) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape[:2] != (image_size, image_size):
            image = Image.fromarray(rgb).resize((image_size, image_size), Image.Resampling.NEAREST)
            rgb = np.asarray(image)
        background = np.asarray(SOURCE_COLORS[item["background_color"]], dtype=np.uint8)
        mask = np.any(rgb != background[None, None, :], axis=-1)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            raise ValueError(f"Could not detect foreground in {directory}")
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        crop_mask = mask[y0:y1, x0:x1]
        crop_h, crop_w = crop_mask.shape
        canvas_mask = np.zeros((image_size, image_size), dtype=bool)
        center_x = float(xs.mean())
        center_y = float(ys.mean())
        target_x = int(round(center_x - crop_w / 2.0))
        target_y = int(round(center_y - crop_h / 2.0))
        # Shift the object center to the canvas center while preserving its
        # actual raster size. This removes random translation as a nuisance.
        target_x += int(round((image_size - 1) / 2 - center_x))
        target_y += int(round((image_size - 1) / 2 - center_y))
        dst_x0, dst_y0 = max(0, target_x), max(0, target_y)
        dst_x1, dst_y1 = min(image_size, target_x + crop_w), min(image_size, target_y + crop_h)
        src_x0, src_y0 = dst_x0 - target_x, dst_y0 - target_y
        canvas_mask[dst_y0:dst_y1, dst_x0:dst_x1] = crop_mask[
            src_y0:src_y0 + (dst_y1 - dst_y0), src_x0:src_x0 + (dst_x1 - dst_x0)
        ]
        canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        canvas[canvas_mask] = COLORS[item["color"]]
        Image.fromarray(canvas).save(os.path.join(output_dir, f"image_{index:05d}.png"))
        shape_labels.append(SHAPES[item["shape"]])
        color_labels.append(COLOR_IDS[item["color"]])
        actual_size = int(item["object_size"])
        size_labels.append(0 if actual_size <= 20 else (1 if actual_size <= 35 else 2))
    for filename, labels in (("class_labels.json", shape_labels),
                             ("color_labels.json", color_labels),
                             ("size_labels.json", size_labels)):
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as handle:
            json.dump(labels, handle)
    print(f"Prepared {len(metadata)} size-conditioned images in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Root containing train/ and val/ video splits")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=64)
    args = parser.parse_args()
    for split in ("train", "val"):
        prepare_split(os.path.join(args.data_dir, split), os.path.join(args.output_dir, split), args.image_size)


if __name__ == "__main__":
    main()
