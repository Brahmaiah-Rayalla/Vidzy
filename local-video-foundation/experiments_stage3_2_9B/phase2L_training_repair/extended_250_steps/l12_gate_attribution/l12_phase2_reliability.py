"""
Stage 3.2.9L.12 - Phase 2: Motion metric reliability audit on L.11 outputs.

For each of the 12 L.11 videos (C1/C25 x 6 prompts):
- color/shape-based object detection rate per frame
- detection confidence (mask mass plausibility)
- color-mask centroid vs generic diff centroid agreement
- contact sheets for representative videos
"""
import json
import numpy as np
from pathlib import Path
import sys
from PIL import Image

L12 = Path(__file__).parent
sys.path.insert(0, str(L12))
from l12_common import (L11_DIR, NUM_FRAMES, compute_motion_metrics,
                        compute_detection_metrics, make_contact_sheet,
                        video_to_numpy, PROMPT_COLORS)
import torch


def load_png_video(frames_dir):
    frames = []
    for i in range(NUM_FRAMES):
        p = frames_dir / f"frame_{i:04d}.png"
        a = np.array(Image.open(p)).astype(np.float32) / 255.0
        frames.append(a)
    arr = np.stack(frames, axis=0)
    return torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)


def centroid_from_mask(video, motion_type):
    """Return per-frame color-mask centroids (or None)."""
    det = compute_detection_metrics(video, motion_type)
    if not det['available']:
        return det, None
    v = video_to_numpy(video)
    F, H, W, C = v.shape
    r, g, b = v[..., 0], v[..., 1], v[..., 2]
    mask = PROMPT_COLORS[motion_type]['rule'](r, g, b)
    col_idx = np.arange(W)[None, :]
    row_idx = np.arange(H)[:, None]
    cents = []
    for f in range(F):
        m = mask[f]
        mass = int(m.sum())
        if 15 <= mass <= int(0.35 * H * W):
            w = m.astype(np.float32)
            cents.append([float((w * col_idx).sum() / w.sum()),
                          float((w * row_idx).sum() / w.sum())])
        else:
            cents.append(None)
    return det, cents


def main():
    out_dir = L12 / 'reliability'
    out_dir.mkdir(exist_ok=True)
    (out_dir / 'contact_sheets').mkdir(exist_ok=True)

    prompts = [
        ('left_to_right', 0), ('right_to_left', 1), ('top_to_bottom', 2),
        ('bottom_to_top', 3), ('diagonal', 4), ('stationary', 5),
    ]

    results = {}
    for model in ['C1', 'C25']:
        results[model] = {}
        for mt, idx in prompts:
            frames_dir = L11_DIR / 'videos' / model / f'{idx:02d}_{mt}'
            video = load_png_video(frames_dir)
            metrics = compute_motion_metrics(video)
            det, color_cents = centroid_from_mask(video, mt)

            # Agreement: does color centroid exist and roughly track diff centroid?
            diff_traj = np.array(metrics['centroid_trajectory'])
            agreement = None
            if color_cents is not None:
                valid = [(i, c) for i, c in enumerate(color_cents) if c is not None]
                if len(valid) >= 4:
                    # compare displacement vectors
                    first_i, first_c = valid[0]
                    last_i, last_c = valid[-1]
                    cd = np.array(last_c) - np.array(first_c)
                    dd = diff_traj[-1] - diff_traj[0]
                    if np.linalg.norm(cd) > 1e-6 and np.linalg.norm(dd) > 1e-6:
                        agreement = float(np.dot(cd, dd) / (np.linalg.norm(cd) * np.linalg.norm(dd)))

            results[model][mt] = {
                'motion_metrics_summary': {
                    'temporal_activity': metrics['temporal_activity'],
                    'displacement_px': metrics['displacement_px'],
                    'direction_vector': metrics['displacement_vector'],
                },
                'detection': {k: v for k, v in det.items() if k != 'per_frame'} if det.get('available') else det,
                'detection_per_frame': det.get('per_frame') if det.get('available') else None,
                'color_vs_diff_centroid_cosine': agreement,
            }

            # Contact sheet for representative videos: all C25 + C1 stationary & one directional
            if (model == 'C25') or (model == 'C1' and mt in ('stationary', 'right_to_left', 'diagonal')):
                make_contact_sheet(video, out_dir / 'contact_sheets' / f'{model}_{idx:02d}_{mt}.png')

    # Reliability classification per video
    reliability = {}
    for model in ['C1', 'C25']:
        reliability[model] = {}
        for mt, _ in prompts:
            det = results[model][mt]['detection']
            if not det.get('available'):
                reliability[model][mt] = 'NO_COLOR_PRIOR'
                continue
            rate = det['detection_rate']
            mass_mean = det['mask_mass_mean']
            # Reliable: >=75% frames detected with plausible mass
            # Low-confidence: 25-75%
            # Unreliable: <25%
            if rate >= 0.75:
                tier = 'RELIABLE_OBJECT_TRACKING'
            elif rate >= 0.25:
                tier = 'LOW_CONFIDENCE_TRACKING'
            else:
                tier = 'UNRELIABLE_TRACKING'
            reliability[model][mt] = {
                'tier': tier,
                'detection_rate': rate,
                'mask_mass_mean': mass_mean,
                'color_direction_cosine': det.get('color_direction_cosine'),
                'color_displacement_px': det.get('color_centroid_displacement_px'),
            }

    summary = {
        'results': results,
        'reliability_tiers': reliability,
        'methodology_notes': [
            'Generic diff-centroid tracks the largest temporal-change region vs the temporal median - it does NOT verify an object is present; it can lock onto background flicker or texture drift.',
            'Color-mask detection uses prompt color priors (blue/pink/white/purple/cyan) with plausible mass range 15px..35% of frame. Generated videos are abstract; detection rates measure color-presence, not semantic objecthood.',
            'color_direction_cosine compares color-mask displacement vs expected direction - supporting evidence only.',
            'Contact sheets saved under reliability/contact_sheets/.',
        ],
    }
    out = L12 / 'L12_METRIC_RELIABILITY_AUDIT.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(reliability, indent=2))
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
