"""Render a compact paired contact sheet from a completed L31 evaluation."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DEFAULT_EVAL = HERE / "eval_checkpoint_2500_seed42_n30"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--evaluation_dir", type=Path, default=DEFAULT_EVAL)
args = parser.parse_args()

report = json.loads((args.evaluation_dir / "per_sample_metrics.json").read_text(encoding="utf-8"))
baseline = report["stage3_l27_migrated_baseline"]
candidate = report["l31_step_2500"]
colors = ["red", "blue", "yellow"]
shapes = ["circle", "triangle"]
tile = 192
label_width = 100
header_height = 68
row_height = tile + 32
width = label_width + 6 * tile
height = header_height + 6 * row_height
canvas = Image.new("RGB", (width, height), (250, 250, 250))
draw = ImageDraw.Draw(canvas)

draw.text((label_width, 8), "L27 migrated baseline", fill=(20, 20, 20))
draw.text((label_width + 3 * tile, 8), "L31 step 2,500", fill=(20, 20, 20))
for column in range(6):
    within_group = column % 3
    draw.text((label_width + column * tile + 5, 35), f"sample {within_group}", fill=(70, 70, 70))

def by_class(rows, class_id):
    return sorted((row for row in rows if row["class_id"] == class_id), key=lambda row: row["sample_id"])

for class_id in range(6):
    y = header_height + class_id * row_height
    class_name = f"{colors[class_id // 2]}\n{shapes[class_id % 2]}"
    draw.multiline_text((5, y + tile // 2 - 14), class_name, fill=(20, 20, 20), spacing=2)
    for group_index, rows in enumerate((baseline, candidate)):
        samples = by_class(rows, class_id)[:3]
        for column, sample in enumerate(samples):
            x = label_width + group_index * 3 * tile + column * tile
            image = Image.open(sample["image_path"]).convert("RGB").resize((tile, tile), Image.Resampling.NEAREST)
            canvas.paste(image, (x, y))
            outline = (40, 140, 55) if sample["joint_correct"] else (180, 55, 55)
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=outline, width=2)
    draw.line((0, y + row_height - 1, width, y + row_height - 1), fill=(210, 210, 210), width=1)

output_path = args.evaluation_dir / "matched_generation_grid_compact.png"
canvas.save(output_path)
print(output_path.resolve())
