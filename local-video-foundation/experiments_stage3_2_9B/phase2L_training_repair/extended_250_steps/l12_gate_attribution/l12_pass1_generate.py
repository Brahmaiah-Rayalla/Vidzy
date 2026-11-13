"""
Stage 3.2.9L.12 - Phase 4/5: Matched first-pass gate ablation.

7 gate configurations x 6 prompts = 42 videos.
C1 and C25 models loaded once; C25 gate overrides applied in-place per
configuration (inference only; native logits restored after each config).
Seeds identical to L.11 (torch.Generator seed 1000 + prompt_index).

Saves per video: float tensor .npz, 16 PNG frames, contact sheet,
and appends metrics to L12_FIRST_PASS_RESULTS.json incrementally.
"""
import json
import time
import numpy as np
from pathlib import Path
import sys
import torch

L12 = Path(__file__).parent
sys.path.insert(0, str(L12))
from l12_common import (CKPT_C1, CKPT_C25, load_model, get_gate_raw_and_activated,
                        set_gate_override, select_prompts, build_diffusion_schedule,
                        fixed_noise_for_prompt, ddim_generate, compute_motion_metrics,
                        compute_detection_metrics, compute_global_motion_metrics,
                        direction_accuracy, direction_accuracy_from_translation,
                        make_contact_sheet, video_to_numpy,
                        GATE_CONFIGS, NUM_FRAMES, RESOLUTION, SAMPLING_STEPS,
                        NUM_TIMESTEPS, BETA_START, BETA_END, L12_DIR)
from models.text_encoder import SimpleTokenizer
import l12_common

base_dir = l12_common.base_dir
OUT_JSON = L12 / 'L12_FIRST_PASS_RESULTS.json'
VIDEOS_DIR = L12 / 'videos_pass1'


def save_video_outputs(video, out_dir, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f'{tag}.npz', video=video_to_numpy(video))
    v = video_to_numpy(video)
    v8 = (np.clip(v, 0, 1) * 255).astype(np.uint8)
    from PIL import Image
    for fi in range(NUM_FRAMES):
        Image.fromarray(v8[fi]).save(out_dir / f'frame_{fi:04d}.png')
    make_contact_sheet(torch.from_numpy(v).permute(3, 0, 1, 2).unsqueeze(0),
                       out_dir / 'contact_sheet.png')


def evaluate_video(video, motion_type):
    metrics = compute_motion_metrics(video)
    det = compute_detection_metrics(video, motion_type)
    gm = compute_global_motion_metrics(video)
    trans_cos = direction_accuracy_from_translation(gm, motion_type)
    return {
        'motion_metrics': {k: v for k, v in metrics.items() if k != 'centroid_trajectory'},
        'centroid_trajectory': metrics['centroid_trajectory'],
        'direction_evaluation': direction_accuracy(metrics, motion_type),
        'detection': {k: v for k, v in det.items() if k not in ('per_frame',)},
        'detection_per_frame': det.get('per_frame'),
        'global_motion': {k: v for k, v in gm.items() if k != 'per_pair_shifts'},
        'global_motion_per_pair': gm['per_pair_shifts'],
        'global_direction_cosine': trans_cos,
    }


def load_existing():
    if OUT_JSON.exists():
        return json.load(open(OUT_JSON))
    return {'pass': 1, 'seed_scheme': 'torch.Generator().manual_seed(1000 + prompt_index)',
            'configs': {}, 'sampling': {
                'num_timesteps': NUM_TIMESTEPS, 'sampling_steps': SAMPLING_STEPS,
                'beta_start': BETA_START, 'beta_end': BETA_END,
                'frames': NUM_FRAMES, 'resolution': RESOLUTION}}


def save(data):
    tmp = OUT_JSON.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f)
    tmp.replace(OUT_JSON)


def main():
    t0 = time.time()
    tokenizer_path = base_dir / 'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json'
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    prompts = select_prompts()
    alphas_cumprod, timesteps = build_diffusion_schedule()

    data = load_existing()
    data.setdefault('configs', {})

    # Load both models once
    print('Loading C1 model...')
    model_c1 = load_model(CKPT_C1)
    native_c1 = {n: float(model_c1.fusion_modules[n].gate.item()) for n in ['bottleneck', 'low_res']}
    print('Loading C25 model...')
    model_c25 = load_model(CKPT_C25)
    native_c25 = {n: float(model_c25.fusion_modules[n].gate.item()) for n in ['bottleneck', 'low_res']}

    for cfg_name, cfg in GATE_CONFIGS.items():
        if cfg_name in data['configs'] and all(
                p['motion_type'] in data['configs'][cfg_name].get('prompts', {})
                for p in prompts):
            print(f'[{cfg_name}] already complete, skipping')
            continue

        ckpt_tag = cfg['checkpoint']
        model = model_c1 if ckpt_tag == 'C1' else model_c25
        native = native_c1 if ckpt_tag == 'C1' else native_c25

        # Apply gate override (inference only)
        if cfg['bottleneck'] is None:
            model.fusion_modules['bottleneck'].gate.data.fill_(native['bottleneck'])
            model.fusion_modules['low_res'].gate.data.fill_(native['low_res'])
        else:
            set_gate_override(model, cfg['bottleneck'], cfg['low_res'])
        actual_gates = get_gate_raw_and_activated(model)
        print(f'[{cfg_name}] gates={actual_gates}')

        data['configs'].setdefault(cfg_name, {
            'checkpoint': ckpt_tag,
            'requested': {'bottleneck': cfg['bottleneck'], 'low_res': cfg['low_res']},
            'actual_gates': actual_gates,
            'prompts': {},
        })

        for i, p in enumerate(prompts):
            mt = p['motion_type']
            if mt in data['configs'][cfg_name]['prompts']:
                continue
            noise = fixed_noise_for_prompt(i)
            print(f'  [{cfg_name}] {i+1}/6 [{mt}]', flush=True)
            t_start = time.time()
            video = ddim_generate(model, tokenizer, p['caption'], noise,
                                  alphas_cumprod, timesteps)
            elapsed = time.time() - t_start
            eval_res = evaluate_video(video, mt)
            eval_res['generation_time_s'] = elapsed
            eval_res['caption'] = p['caption']
            eval_res['prompt_index'] = i
            data['configs'][cfg_name]['prompts'][mt] = eval_res

            tag = f'{cfg_name}_{i:02d}_{mt}'
            save_video_outputs(video, VIDEOS_DIR / cfg_name / f'{i:02d}_{mt}', tag)
            save(data)
            print(f'    done {elapsed:.0f}s act={eval_res["motion_metrics"]["temporal_activity"]:.3f} '
                  f'disp={eval_res["motion_metrics"]["displacement_px"]:.1f} '
                  f'cos={eval_res["direction_evaluation"]["cosine_similarity"]}', flush=True)

        # Restore native gates after config
        model.fusion_modules['bottleneck'].gate.data.fill_(native['bottleneck'])
        model.fusion_modules['low_res'].gate.data.fill_(native['low_res'])
        save(data)

    # Final gate state check
    data['final_gate_state'] = {
        'C1': get_gate_raw_and_activated(model_c1),
        'C25': get_gate_raw_and_activated(model_c25),
    }
    save(data)
    print(f'\nPass 1 complete in {(time.time()-t0)/60:.1f} min')
    print(f'Saved {OUT_JSON}')


if __name__ == '__main__':
    sys.exit(main())
