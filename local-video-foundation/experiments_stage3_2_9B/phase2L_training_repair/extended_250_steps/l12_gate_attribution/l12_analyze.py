"""
Stage 3.2.9L.12 - Phase 7: Aggregate pass-1 and multi-seed results into the
comparison table and supporting JSONs.
"""
import json
import numpy as np
from pathlib import Path

L12 = Path(__file__).parent
PASS1 = L12 / 'L12_FIRST_PASS_RESULTS.json'
PASS2 = L12 / 'L12_FINALIST_MULTI_SEED_RESULTS.json'
OUT = L12 / 'L12_C1_C25_GATE_COMPARISON.json'

DIRECTIONAL = ['left_to_right', 'right_to_left', 'top_to_bottom',
               'bottom_to_top', 'diagonal']
ALL_MT = DIRECTIONAL + ['stationary']


def config_summary(prompts):
    """Aggregate metrics for one config's prompt dict."""
    dir_cos = []
    dir_match = 0
    dir_partial = 0
    acts = []
    disps = []
    stat_disp = []
    stat_jitter = []
    cos_gm = []
    stat_gm_net = []
    for mt in DIRECTIONAL:
        p = prompts[mt]
        de = p['direction_evaluation']
        if de['cosine_similarity'] is not None:
            dir_cos.append(de['cosine_similarity'])
            if de['classification'] == 'DIRECTION_MATCH':
                dir_match += 1
            elif de['classification'] == 'PARTIAL':
                dir_partial += 1
        if p.get('global_direction_cosine') is not None:
            cos_gm.append(p['global_direction_cosine'])
        acts.append(p['motion_metrics']['temporal_activity'])
        disps.append(p['motion_metrics']['displacement_px'])
    sp = prompts['stationary']
    stat_disp.append(sp['motion_metrics']['displacement_px'])
    stat_jitter.append(sp['motion_metrics']['centroid_jitter_px'])
    stat_gm = sp['global_motion']['net_translation_px']
    stat_gm_net.append(float(np.linalg.norm(stat_gm)))
    # trajectory consistency across directional prompts
    straight = [prompts[mt]['motion_metrics']['straightness'] for mt in DIRECTIONAL]
    abrupt = [prompts[mt]['motion_metrics']['abrupt_transition_ratio'] for mt in DIRECTIONAL]
    det_rates = [prompts[mt]['detection']['detection_rate'] for mt in ALL_MT]
    return {
        'direction_matches': dir_match,
        'direction_partial': dir_partial,
        'directional_prompts': len(DIRECTIONAL),
        'mean_direction_cosine': float(np.mean(dir_cos)) if dir_cos else None,
        'mean_global_translation_cosine': float(np.mean(cos_gm)) if cos_gm else None,
        'global_translation_matches_0.5': int(sum(1 for c in cos_gm if c is not None and c > 0.5)),
        'mean_temporal_activity': float(np.mean(acts)),
        'mean_displacement_px': float(np.mean(disps)),
        'stationary_displacement_px': float(np.mean(stat_disp)),
        'stationary_centroid_jitter_px': float(np.mean(stat_jitter)),
        'stationary_global_net_translation_px': float(np.mean(stat_gm_net)),
        'mean_straightness': float(np.mean(straight)),
        'mean_abrupt_ratio': float(np.mean(abrupt)),
        'mean_detection_rate': float(np.mean(det_rates)),
        'min_detection_rate': float(np.min(det_rates)),
    }


def per_prompt_table(configs):
    """Per-prompt rows for the comparison table."""
    rows = {}
    for cfg_name, cfg in configs.items():
        rows[cfg_name] = {}
        for mt in ALL_MT:
            p = cfg['prompts'][mt]
            rows[cfg_name][mt] = {
                'temporal_activity': p['motion_metrics']['temporal_activity'],
                'displacement_px': p['motion_metrics']['displacement_px'],
                'direction_cosine': p['direction_evaluation']['cosine_similarity'],
                'direction_class': p['direction_evaluation']['classification'],
                'global_direction_cosine': p.get('global_direction_cosine'),
                'global_net_translation': p['global_motion']['net_translation_px'],
                'detection_rate': p['detection']['detection_rate'],
                'straightness': p['motion_metrics']['straightness'],
                'abrupt_ratio': p['motion_metrics']['abrupt_transition_ratio'],
            }
    return rows


def main():
    pass1 = json.load(open(PASS1))
    result = {
        'pass1_seed_scheme': pass1['seed_scheme'],
        'sampling': pass1['sampling'],
        'configs': {},
        'per_prompt': {},
    }
    for cfg_name, cfg in pass1['configs'].items():
        prompts = cfg['prompts']
        if len(prompts) < 6:
            continue
        result['configs'][cfg_name] = {
            'checkpoint': cfg['checkpoint'],
            'requested_gates': cfg['requested'],
            'actual_gates': cfg['actual_gates'],
            'summary': config_summary(prompts),
        }
    result['per_prompt'] = per_prompt_table(
        {k: v for k, v in pass1['configs'].items() if len(v.get('prompts', {})) == 6})

    # Multi-seed aggregation
    if PASS2.exists():
        pass2 = json.load(open(PASS2))
        ms = {}
        for cfg_name, cfg in pass2.get('finalists', {}).items():
            per_seed = {}
            for seed_key, seed_data in cfg.get('seeds', {}).items():
                prompts = seed_data['prompts']
                if len(prompts) == 6:
                    per_seed[seed_key] = config_summary(prompts)
            if per_seed:
                # aggregate across seeds
                keys = ['direction_matches', 'mean_direction_cosine',
                        'mean_global_translation_cosine', 'mean_temporal_activity',
                        'stationary_displacement_px', 'stationary_global_net_translation_px',
                        'mean_detection_rate']
                agg = {}
                for k in keys:
                    vals = [s[k] for s in per_seed.values() if s.get(k) is not None]
                    agg[k] = float(np.mean(vals)) if vals else None
                agg['per_seed'] = per_seed
                ms[cfg_name] = agg
        result['multiseed'] = ms

    with open(OUT, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v['summary'] for k, v in result['configs'].items()}, indent=2))
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
