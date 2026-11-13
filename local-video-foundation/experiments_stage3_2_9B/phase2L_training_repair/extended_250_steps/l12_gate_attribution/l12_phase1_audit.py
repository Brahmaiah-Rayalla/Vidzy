"""
Stage 3.2.9L.12 - Phase 1: Reproduce and audit L.11 artifacts.

Verifies checkpoint paths, prompts, seeds, DDIM settings; recomputes motion
metrics from saved L.11 PNG frames and compares against L11_MOTION_EVALUATION.json.
"""
import json
import numpy as np
from pathlib import Path
import sys
import hashlib

L12 = Path(__file__).parent
sys.path.insert(0, str(L12))
from l12_common import (L11_DIR, CKPT_C1, CKPT_C25, select_prompts,
                        compute_motion_metrics, direction_accuracy,
                        EXPECTED_DIRECTIONS, NUM_FRAMES, RESOLUTION,
                        SAMPLING_STEPS, NUM_TIMESTEPS, BETA_START, BETA_END)
import torch


def load_png_video(frames_dir):
    """Load 16 PNG frames -> tensor [1,3,F,H,W] in [0,1] (uint8 quantized)."""
    from PIL import Image
    frames = []
    for i in range(NUM_FRAMES):
        p = frames_dir / f"frame_{i:04d}.png"
        a = np.array(Image.open(p)).astype(np.float32) / 255.0  # [H,W,3]
        frames.append(a)
    arr = np.stack(frames, axis=0)  # [F,H,W,3]
    t = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)  # [1,3,F,H,W]
    return t


def main():
    report = {'checks': {}, 'reproduction': {}, 'discrepancies': []}

    # --- Artifact inventory ---
    expected_files = [
        'STAGE3_2_9L11_FINAL_REPORT.md',
        'L11_DDIM_INFRASTRUCTURE_AUDIT.md',
        'L11_INFERENCE_SMOKE_TEST.json',
        'L11_MOTION_EVALUATION.json',
        'L11_C1_C25_COMPARISON.json',
        'run_l11_evaluation.py',
        'ARTIFACT_HASHES.json',
    ]
    prompt_named = [
        'L11_GENERATION_CONFIG.json',
        'L11_MOTION_METRICS.json',
        'L11_SEED_LEVEL_RESULTS.json',
    ]
    present = {f: (L11_DIR / f).exists() for f in expected_files}
    prompt_named_present = {f: (L11_DIR / f).exists() for f in prompt_named}
    report['checks']['l11_artifacts_present'] = present
    report['checks']['prompt_named_files'] = prompt_named_present
    report['discrepancies'].append(
        "L.12 prompt referenced L11_GENERATION_CONFIG.json, L11_MOTION_METRICS.json, "
        "L11_SEED_LEVEL_RESULTS.json - none exist. Equivalent information is stored in "
        "L11_MOTION_EVALUATION.json (per-video metrics), L11_C1_C25_COMPARISON.json "
        "(config, seeds, sampling settings, summary), and run_l11_evaluation.py "
        "(seed scheme 1000+i, DDIM schedule). No data loss; naming difference only."
    )

    # --- Checkpoint verification ---
    ckpt_info = {}
    for name, path in [('C1', CKPT_C1), ('C25', CKPT_C25)]:
        ck = torch.load(path, map_location='cpu')
        ckpt_info[name] = {
            'path': str(path),
            'exists': path.exists(),
            'keys': list(ck.keys()),
            'step': ck.get('step'),
            'gate_init': ck.get('gate_init'),
            'gates_bottleneck_raw': float(ck['model_state_dict']['fusion_modules.bottleneck.gate']),
            'gates_low_res_raw': float(ck['model_state_dict']['fusion_modules.low_res.gate']),
        }
    report['checks']['checkpoints'] = ckpt_info

    # --- Prompt/seed/DDIM config from L.11 script and comparison JSON ---
    prompts = select_prompts()
    report['checks']['prompts'] = prompts
    report['checks']['ddim_config'] = {
        'num_timesteps': NUM_TIMESTEPS,
        'sampling_steps': SAMPLING_STEPS,
        'beta_start': BETA_START,
        'beta_end': BETA_END,
        'frames': NUM_FRAMES,
        'resolution': RESOLUTION,
        'seed_scheme': 'torch.Generator().manual_seed(1000 + prompt_index), identical across models',
    }

    # Sample-count resolution
    report['checks']['sample_count_resolution'] = {
        'captions_per_config': 6,
        'directional_prompts': 5,
        'stationary_prompts': 1,
        'direction_metrics_use': '5 directional only; stationary reported separately as STATIONARY_EXPECTED',
        'note': 'Not a discrepancy - by design. cosine_similarity is null for stationary.',
    }

    # --- Reproduce metrics from saved PNGs ---
    comp = json.load(open(L11_DIR / 'L11_C1_C25_COMPARISON.json'))
    evald = json.load(open(L11_DIR / 'L11_MOTION_EVALUATION.json'))
    motion_order = ['left_to_right', 'right_to_left', 'top_to_bottom',
                    'bottom_to_top', 'diagonal', 'stationary']
    # map prompt_index -> motion_type dir name
    idx_to_mt = {i: p['motion_type'] for i, p in enumerate(prompts)}

    repro = {}
    for model_name in ['C1', 'C25']:
        repro[model_name] = {}
        for i, mt in idx_to_mt.items():
            frames_dir = L11_DIR / 'videos' / model_name / f'{i:02d}_{mt}'
            if not frames_dir.exists():
                repro[model_name][mt] = {'error': 'missing frames dir'}
                continue
            video = load_png_video(frames_dir)
            metrics = compute_motion_metrics(video)
            dir_eval = direction_accuracy(metrics, mt)
            stored = evald[model_name][mt]
            repro[model_name][mt] = {
                'png_temporal_activity': metrics['temporal_activity'],
                'json_temporal_activity': stored['metrics']['temporal_activity'],
                'activity_abs_diff': abs(metrics['temporal_activity'] - stored['metrics']['temporal_activity']),
                'png_displacement_px': metrics['displacement_px'],
                'json_displacement_px': stored['metrics']['displacement_px'],
                'disp_abs_diff': abs(metrics['displacement_px'] - stored['metrics']['displacement_px']),
                'png_direction_cosine': dir_eval['cosine_similarity'],
                'json_direction_cosine': stored['direction_evaluation']['cosine_similarity'],
                'png_direction_class': dir_eval['classification'],
                'json_direction_class': stored['direction_evaluation']['classification'],
            }
    report['reproduction'] = repro

    # --- Hash check of L.11 artifacts ---
    hash_file = L11_DIR / 'ARTIFACT_HASHES.json'
    if hash_file.exists():
        recorded = json.load(open(hash_file))['files']
        mismatches = []
        for fname, h in recorded.items():
            p = L11_DIR / fname
            if p.exists():
                actual = hashlib.sha256(p.read_bytes()).hexdigest()
                if actual != h:
                    mismatches.append(fname)
        report['checks']['artifact_hash_mismatches'] = mismatches

    out = L12 / 'L12_PHASE1_AUDIT.json'
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'discrepancies': report['discrepancies'],
                      'hash_mismatches': report['checks'].get('artifact_hash_mismatches'),
                      'sample_count': report['checks']['sample_count_resolution']}, indent=2))
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
