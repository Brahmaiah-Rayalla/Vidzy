"""Recompute the fixed-noise paired L32-to-L32b shape/color pilot comparison."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "l32_cpu_factorized_salience_pilot" / "per_sample_metrics.json"
CANDIDATE = HERE / "per_sample_metrics.json"
OUTPUT = HERE / "paired_pilot_analysis.json"


def read_samples(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(row["class_id"], row["sample_id"]): row for row in rows}


def bootstrap_pair(old, new, rng):
    delta = np.asarray(new, dtype=np.float64) - np.asarray(old, dtype=np.float64)
    indices = rng.integers(0, len(delta), size=(20000, len(delta)))
    means = delta[indices].mean(axis=1)
    return {
        "baseline": float(np.mean(old)),
        "candidate": float(np.mean(new)),
        "delta": float(delta.mean()),
        "improved_pairs": int(np.sum((np.asarray(old) == 0) & (np.asarray(new) == 1))),
        "regressed_pairs": int(np.sum((np.asarray(old) == 1) & (np.asarray(new) == 0))),
        "bootstrap_95pct_interval": [float(x) for x in np.quantile(means, [0.025, 0.975])],
    }


def main():
    old, new = read_samples(BASELINE), read_samples(CANDIDATE)
    if old.keys() != new.keys():
        raise RuntimeError("Baseline and candidate sample keys do not match")
    keys = sorted(old)
    if any(old[key]["seed"] != new[key]["seed"] for key in keys):
        raise RuntimeError("Baseline and candidate noise seeds do not match")
    rng = np.random.default_rng(9137)
    metrics = {}
    for metric in ("joint_correct", "shape_correct", "color_correct"):
        metrics[metric] = bootstrap_pair(
            [old[key][metric] for key in keys],
            [new[key][metric] for key in keys], rng,
        )
    per_class = {}
    for class_id in range(6):
        class_keys = [key for key in keys if key[0] == class_id]
        per_class[str(class_id)] = {
            "class_name": ["red circle", "red triangle", "blue circle", "blue triangle", "yellow circle", "yellow triangle"][class_id],
            "baseline_joint_accuracy": float(np.mean([old[key]["joint_correct"] for key in class_keys])),
            "candidate_joint_accuracy": float(np.mean([new[key]["joint_correct"] for key in class_keys])),
            "improved_pairs": int(sum(not old[key]["joint_correct"] and new[key]["joint_correct"] for key in class_keys)),
            "regressed_pairs": int(sum(old[key]["joint_correct"] and not new[key]["joint_correct"] for key in class_keys)),
        }
    result = {
        "comparison": "L32 1200-step EMA vs L32b 400-step conditioned-timestep EMA",
        "samples": len(keys),
        "same_noise_seeds": True,
        "metrics": metrics,
        "per_class_joint": per_class,
        "note": "The pilot runner initially omitted the return from its paired_stats helper; these paired metrics were recomputed from the saved per-sample records.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result_path = HERE / "L32B_RESULTS.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
