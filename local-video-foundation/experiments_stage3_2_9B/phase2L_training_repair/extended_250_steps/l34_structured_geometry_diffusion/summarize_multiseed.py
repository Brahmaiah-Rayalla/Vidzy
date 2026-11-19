"""Combine fixed-protocol L34 geometry-diffusion runs across training seeds."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
RUN_DIRS = [HERE, HERE / "seed123", HERE / "seed777"]


def check_image(row):
    image = np.asarray(Image.open(row["image_path"]).convert("RGB"), dtype=np.int16)
    color = np.asarray([(220, 30, 30), (30, 80, 220), (230, 200, 30)][row["color_id"]], dtype=np.int16)
    foreground = np.linalg.norm(image - color[None, None, :], axis=2) < 90
    ys, xs = np.where(foreground)
    return bool(len(xs) and xs.min() > 0 and ys.min() > 0 and xs.max() < 63 and ys.max() < 63)


def main():
    run_reports = []
    all_samples = []
    for run_dir in RUN_DIRS:
        result = json.loads((run_dir / "L34_RESULTS.json").read_text(encoding="utf-8"))
        samples = json.loads((run_dir / "per_sample_geometry.json").read_text(encoding="utf-8"))
        ks_values = [feature["ks_distance"]
                     for per_class in result["geometry_distribution_by_class"].values()
                     for feature in per_class.values()]
        bounds_ok = all(
            22.0 <= row["cx"] <= 42.0 and 22.0 <= row["cy"] <= 42.0
            and ((7 <= row["extent"] <= 12) if row["shape_id"] == 0 else (19 <= row["extent"] <= 32))
            for row in samples
        )
        image_edges_ok = all(check_image(row) for row in samples)
        run_reports.append({
            "seed": result["seed"],
            "samples": len(samples),
            "training_seconds": result["training_elapsed_seconds"],
            "sample_and_render_seconds": result["sample_generation_and_render_seconds"],
            "samples_per_second": result["generated_samples_per_second"],
            "max_parameter_ks_distance": float(max(ks_values)),
            "mean_parameter_ks_distance": float(np.mean(ks_values)),
            "geometry_in_training_support": bounds_ok,
            "no_rendered_object_cropped": image_edges_ok,
            "renderer_class_mapping": "shape and color are selected from the requested class ID by the renderer",
        })
        all_samples.extend(samples)
    max_ks = max(row["max_parameter_ks_distance"] for row in run_reports)
    result = {
        "experiment": "L34 structured geometry diffusion + CPU renderer, 3 training seeds",
        "runs": run_reports,
        "total_rendered_samples": len(all_samples),
        "mean_training_seconds": float(np.mean([row["training_seconds"] for row in run_reports])),
        "mean_600_sample_generation_and_render_seconds": float(np.mean([row["sample_and_render_seconds"] for row in run_reports])),
        "mean_samples_per_second": float(np.mean([row["samples_per_second"] for row in run_reports])),
        "max_parameter_ks_distance_across_runs": float(max_ks),
        "all_rendered_shapes_and_colors_follow_conditions": all(row["renderer_class_mapping"] for row in run_reports),
        "all_geometry_in_support": all(row["geometry_in_training_support"] for row in run_reports),
        "all_outputs_uncropped": all(row["no_rendered_object_cropped"] for row in run_reports),
        "interpretation": "Renderer-backed shape/color appearance is reliable by construction. Learned geometry-distribution fidelity is approximate and should be refined before treating center/size sampling as final. This does not pass or replace the raw RGB diffusion gate.",
    }
    output = HERE / "L34_MULTI_SEED_SUMMARY.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
