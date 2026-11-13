"""
Stage 3.2.9L.12 - Phase 6: Finalist multi-seed evaluation.

Runs the two selected finalist gate configurations with two additional
fixed seeds (seed_base 2000 and 3000, same per-prompt scheme  base+i)
across the same six captions.

Usage: python l12_pass2_multiseed.py <finalist1> <finalist2>
Config names must be keys of GATE_CONFIGS (e.g. C25_25_10 C25_10_25).
"""
import json
import time
import sys
from pathlib import Path
import torch

L12 = Path(__file__).parent
sys.path.insert(0, str(L12))
from l12_common import (CKPT_C1, CKPT_C25, load_model, get_gate_raw_and_activated,
                        set_gate_override, select_prompts, build_diffusion_schedule,
                        ddim_generate, compute_motion_metrics,
                        compute_detection_metrics, compute_global_motion_metrics,
                        direction_accuracy, direction_accuracy_from_translation,
                        make_contact_sheet, video_to_numpy,
                        GATE_CONFIGS, NUM_FRAMES, RESOLUTION,
                        NUM_TIMESTEPS, BETA_START, BETA_END)
from models.text_encoder import SimpleTokenizer
import l12_common
import numpy as np

base_dir = l12_common.base_dir
OUT_JSON = L12 / 'L12_FINALIST_MULTI_SEED_RESULTS.json'
VIDEOS_DIR = L12 / 'videos_multiseed'
EXTRA_SEED_BASES = [2000, 3000]


def noise_for(prompt_index, seed_base):
    g = torch.Generator().manual_seed(seed_base + prompt_index)
    return torch.randn(1, 3, NUM_FRAMES, RESOLUTION, RESOLUTION, generator=g)


def evaluate_video(video, motion_type):
    metrics = compute_motion_metrics(video)
    det = compute_detection_metrics(video, motion_type)
    gm = compute_global_motion_metrics(video)
    trans_cos = direction_accuracy_from_translation(gm, motion_type)
    return {
        'motion_metrics': {k: v for k, v in metrics.items() if k != 'centroid_trajectory'},
        'centroid_trajectory': metrics['centroid_trajectory'],
        'direction_evaluation': direction_accuracy(metrics, motion_type),
        'detection': {k: v for k, v in det.items() if k != 'per_frame'},
        'detection_per_frame': det.get('per_frame'),
        'global_motion': {k: v for k, v in gm.items() if k != 'per_pair_shifts'},
        'global_motion_per_pair': gm['per_pair_shifts'],
        'global_direction_cosine': trans_cos,
    }


def save_video_outputs(video, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / 'video.npz', video=video_to_numpy(video))
    v = video_to_numpy(video)
    v8 = (np.clip(v, 0, 1) * 255).astype(np.uint8)
    from PIL import Image
    for fi in range(NUM_FRAMES):
        Image.fromarray(v8[fi]).save(out_dir / f'frame_{fi:04d}.png')
    make_contact_sheet(torch.from_numpy(v).permute(3, 0, 1, 2).unsqueeze(0),
                       out_dir / 'contact_sheet.png')


def load_existing():
    if OUT_JSON.exists():
        return json.load(open(OUT_JSON))
    return {'finalists': {}, 'seed_bases': EXTRA_SEED_BASES,
            'seed_scheme': 'torch.Generator().manual_seed(seed_base + prompt_index)',
            'sampling': {'num_timesteps': NUM_TIMESTEPS, 'sampling_steps': 25,
                         'beta_start': BETA_START, 'beta_end': BETA_END,
                         'frames': NUM_FRAMES, 'resolution': RESOLUTION}}


def save(data):
    tmp = OUT_JSON.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f)
    tmp.replace(OUT_JSON)


def main():
    if len(sys.argv) < 3:
        print('Usage: l12_pass2_multiseed.py <config1> <config2>')
        return 1
    finalists = sys.argv[1:3]
    for f in finalists:
        if f not in GATE_CONFIGS:
            print(f'Unknown config {f}')
            return 1

    t0 = time.time()
    tokenizer_path = base_dir / 'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json'
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    prompts = select_prompts()
    alphas_cumprod, timesteps = build_diffusion_schedule()

    data = load_existing()
    data.setdefault('finalists', {})

    print('Loading C25 model...')
    model = load_model(CKPT_C25)
    native = {n: float(model.fusion_modules[n].gate.item()) for n in ['bottleneck', 'low_res']}

    for cfg_name in finalists:
        cfg = GATE_CONFIGS[cfg_name]
        set_gate_override(model, cfg['bottleneck'], cfg['low_res'])
        actual = get_gate_raw_and_activated(model)
        print(f'[{cfg_name}] gates={actual}')
        data['finalists'].setdefault(cfg_name, {
            'requested': {'bottleneck': cfg['bottleneck'], 'low_res': cfg['low_res']},
            'actual_gates': actual,
            'seeds': {},
        })
        for seed_base in EXTRA_SEED_BASES:
            seed_key = f'base_{seed_base}'
            data['finalists'][cfg_name]['seeds'].setdefault(seed_key, {'prompts': {}})
            prompts_done = data['finalists'][cfg_name]['seeds'][seed_key]['prompts']
            for i, p in enumerate(prompts):
                mt = p['motion_type']
                if mt in prompts_done:
                    continue
                noise = noise_for(i, seed_base)
                print(f'  [{cfg_name}] seed={seed_base} {i+1}/6 [{mt}]', flush=True)
                t_start = time.time()
                video = ddim_generate(model, tokenizer, p['caption'], noise,
                                      alphas_cumprod, timesteps)
                elapsed = time.time() - t_start
                res = evaluate_video(video, mt)
                res['generation_time_s'] = elapsed
                res['caption'] = p['caption']
                res['prompt_index'] = i
                res['seed_base'] = seed_base
                prompts_done[mt] = res
                save_video_outputs(video, VIDEOS_DIR / cfg_name / seed_key / f'{i:02d}_{mt}')
                save(data)
                print(f'    done {elapsed:.0f}s act={res["motion_metrics"]["temporal_activity"]:.3f} '
                      f'disp={res["motion_metrics"]["displacement_px"]:.1f}', flush=True)
        # restore
        model.fusion_modules['bottleneck'].gate.data.fill_(native['bottleneck'])
        model.fusion_modules['low_res'].gate.data.fill_(native['low_res'])
        save(data)

    print(f'\nMulti-seed pass complete in {(time.time()-t0)/60:.1f} min')
    print(f'Saved {OUT_JSON}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
