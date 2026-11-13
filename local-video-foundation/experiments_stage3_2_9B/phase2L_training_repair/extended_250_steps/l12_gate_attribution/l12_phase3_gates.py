"""
Stage 3.2.9L.12 - Phase 3: Fusion gate implementation inspection and
runtime activation/residual measurement.

Records native trained gate values (raw logits + sigmoid) for C1 and C25
step-250 checkpoints, then measures - via forward hooks on the temporal
modules during real DDIM forward passes - the magnitude of temporal
residuals and the effective gated contribution injected into the spatial
decoder, for native and overridden gate configurations.
"""
import json
import torch
import numpy as np
from pathlib import Path
import sys

L12 = Path(__file__).parent
sys.path.insert(0, str(L12))
from l12_common import (CKPT_C1, CKPT_C25, load_model, get_gate_raw_and_activated,
                        set_gate_override, encode_prompt, select_prompts,
                        build_diffusion_schedule, GATE_LOGIT, GATE_ZERO_LOGIT,
                        NUM_FRAMES, RESOLUTION, torch as _torch)
import l12_common
from models.text_encoder import SimpleTokenizer

base_dir = l12_common.base_dir
OUT = L12 / 'L12_GATE_ACTIVATION_ANALYSIS.json'


def measure_forward_stats(model, tokenizer, prompt, timesteps_wanted):
    """
    Run forward passes at selected timesteps with hooks on temporal modules.
    Returns per-call gate activations and residual statistics.
    """
    text_emb, text_mask = encode_prompt(model, tokenizer, prompt)
    g = torch.Generator().manual_seed(1000 + 5)  # stationary prompt index
    x = torch.randn(1, 3, NUM_FRAMES, RESOLUTION, RESOLUTION, generator=g)

    captures = {}

    def make_hook(name):
        def hook(module, inp, out):
            captures[name] = {
                'input': inp[0].detach().clone(),
                'output': out.detach().clone(),
            }
        return hook

    handles = []
    for name in ['bottleneck', 'low_res']:
        handles.append(model.temporal_modules[name].register_forward_hook(make_hook(name)))

    alphas_cumprod, ts = build_diffusion_schedule()
    ts_list = list(ts)
    # pick timesteps spread across schedule
    wanted = [ts_list[int(i * (len(ts_list) - 1) / (timesteps_wanted - 1))] for i in range(timesteps_wanted)]

    results = []
    for t_val in wanted:
        captures.clear()
        x_t = x + 0.1 * torch.randn_like(x) * float(t_val) / 1000.0  # vary input slightly per t
        with torch.no_grad():
            eps = model(x_t, torch.tensor([int(t_val)]).long(), text_emb, text_mask)
        gates = get_gate_raw_and_activated(model)
        call = {'t': int(t_val), 'gates': gates, 'eps_std': float(eps.std())}
        for name in ['bottleneck', 'low_res']:
            if name in captures:
                tin = captures[name]['input']
                tout = captures[name]['output']
                residual = tout - tin
                gate_act = gates[name]['activated']
                call[name] = {
                    'temporal_feature_norm': float(tin.norm()),
                    'residual_norm': float(residual.norm()),
                    'residual_rel_to_feature': float(residual.norm() / (tin.norm() + 1e-9)),
                    'gated_delta_norm': float((gate_act * residual).norm()),
                    'gated_delta_rel': float((gate_act * residual).norm() / (tin.norm() + 1e-9)),
                }
        results.append(call)

    for h in handles:
        h.remove()
    return results


def main():
    tokenizer_path = base_dir / 'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json'
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    prompts = select_prompts()
    stationary_caption = [p['caption'] for p in prompts if p['motion_type'] == 'stationary'][0]
    ltr_caption = [p['caption'] for p in prompts if p['motion_type'] == 'left_to_right'][0]

    report = {
        'implementation': {
            'file': 'experiments_stage3_2_7/video_foundation/models/video_unet_fixed.py',
            'gate_mechanism': 'FeatureLevelTemporalFusion stores raw logit; forward applies sigmoid. In _forward_with_temporal: fused = temporal + sigmoid(gate) * (temporal_processed - temporal); fused features injected into spatial decoder via injected_features.',
            'gate_placement': {
                'bottleneck': 'spatial UNet bottleneck features, 128 channels',
                'low_res': 'spatial UNet low-res features, 64 channels',
            },
            'independence': 'Two independent scalar gates; each scales its own temporal residual before decoder injection.',
            'near_zero_bypass': 'If BOTH activated gates < 0.001, forward() routes to _forward_frame_independent (true identity path).',
        },
        'native_gates': {},
        'runtime_measurements': {},
    }

    # Native gate values from both checkpoints
    for name, ckpt in [('C1', CKPT_C1), ('C25', CKPT_C25)]:
        model = load_model(ckpt)
        report['native_gates'][name] = get_gate_raw_and_activated(model)
        del model

    # Runtime measurement on C25 native + overrides, stationary prompt
    print('Loading C25 model for runtime measurement...')
    model = load_model(CKPT_C25)

    configs = [
        ('C25_native', None, None),
        ('C25_10_10', 0.10, 0.10),
        ('C25_25_10', 0.25, 0.10),
        ('C25_10_25', 0.10, 0.25),
        ('C25_25_0', 0.25, 0.0),
        ('C25_0_25', 0.0, 0.25),
    ]
    # reload native gate logits for restore
    native = {n: float(model.fusion_modules[n].gate.item()) for n in ['bottleneck', 'low_res']}

    for cfg_name, bn, lr in configs:
        print(f'Measuring {cfg_name}...')
        if bn is None:
            model.fusion_modules['bottleneck'].gate.data.fill_(native['bottleneck'])
            model.fusion_modules['low_res'].gate.data.fill_(native['low_res'])
        else:
            set_gate_override(model, bn, lr)
        stats = measure_forward_stats(model, tokenizer, stationary_caption, timesteps_wanted=3)
        report['runtime_measurements'][cfg_name] = {
            'requested': {'bottleneck': bn, 'low_res': lr},
            'actual_gates': get_gate_raw_and_activated(model),
            'prompt': 'stationary',
            'forward_passes': stats,
        }

    # Also one directional prompt for C25 native vs C25_0_25 to see gate effect on residuals
    for cfg_name, bn, lr in [('C25_native', None, None), ('C25_0_25', 0.0, 0.25)]:
        if bn is None:
            model.fusion_modules['bottleneck'].gate.data.fill_(native['bottleneck'])
            model.fusion_modules['low_res'].gate.data.fill_(native['low_res'])
        else:
            set_gate_override(model, bn, lr)
        stats = measure_forward_stats(model, tokenizer, ltr_caption, timesteps_wanted=3)
        report['runtime_measurements'][cfg_name + '_left_to_right'] = {
            'requested': {'bottleneck': bn, 'low_res': lr},
            'actual_gates': get_gate_raw_and_activated(model),
            'prompt': 'left_to_right',
            'forward_passes': stats,
        }

    # restore native
    model.fusion_modules['bottleneck'].gate.data.fill_(native['bottleneck'])
    model.fusion_modules['low_res'].gate.data.fill_(native['low_res'])
    report['post_restore_gates'] = get_gate_raw_and_activated(model)

    with open(OUT, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Saved {OUT}')
    print(json.dumps(report['native_gates'], indent=2))


if __name__ == '__main__':
    main()
