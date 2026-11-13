"""
Stage 3.2.9L.11 - DDIM Verification and C1 vs C25 Motion Evaluation

Generates matched videos with the step-250 C1 (1% gate init) and C25 (25% gate
init) checkpoints using DDIM sampling, then computes quantitative motion
metrics (temporal activity, object trajectory, direction accuracy).

Inference path mirrors the L.7/L.10 training forward exactly:
    tokenizer -> token_ids/mask -> video_unet.text_encoder -> (text_emb, text_mask)
    -> video_unet(x_t, t, text_emb, text_mask)

DDIM math matches phase2H_temporal_instrumentation/video_generation_runner.py
DDIMSampler (linear beta schedule 0.0001->0.02, 1000 timesteps, 25 sampling
steps, deterministic update).
"""

import torch
import json
import numpy as np
from pathlib import Path
import sys
import time
import hashlib

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
stage3_2_7_dir = base_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(stage3_2_7_dir))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation" / "models"))

from models.conditional_unet import ConditionalUNet
from models.video_unet_fixed import VideoUNetFixed
from models.text_encoder_split import TextEncoderSplit, load_from_standard_encoder
from models.text_encoder import TextEncoder, SimpleTokenizer

torch.set_num_threads(8)

OUT_DIR = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l11_ddim_motion_eval"
CKPT_C1 = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/C1/checkpoint_step_000250.pt"
CKPT_C25 = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/C25/checkpoint_step_000250.pt"

NUM_FRAMES = 16
RESOLUTION = 64
SAMPLING_STEPS = 25
NUM_TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02

EXPECTED_DIRECTIONS = {
    'left_to_right': (1.0, 0.0),
    'right_to_left': (-1.0, 0.0),
    'top_to_bottom': (0.0, 1.0),
    'bottom_to_top': (0.0, -1.0),
    'diagonal': (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
    'stationary': (0.0, 0.0),
}


def build_diffusion_schedule():
    betas = torch.linspace(BETA_START, BETA_END, NUM_TIMESTEPS)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    timesteps = torch.linspace(NUM_TIMESTEPS - 1, 0, SAMPLING_STEPS).long()
    return alphas_cumprod, timesteps


def load_model(checkpoint_path):
    """Load VideoUNetFixed exactly as the L.7/L.10 training scripts do."""
    device = torch.device('cpu')
    stage2_checkpoint = base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt"
    checkpoint = torch.load(stage2_checkpoint, map_location=device)
    actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

    spatial_unet = ConditionalUNet(
        in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0,
        num_timesteps=1000, text_dim=256, n_heads=4, use_cross_attn=True
    )
    spatial_unet.load_state_dict(checkpoint['model_state_dict'])

    text_encoder_standard = TextEncoder(
        vocab_size=actual_vocab_size, d_model=256, n_heads=4,
        n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0
    )
    text_encoder_standard.load_state_dict(checkpoint['text_encoder_state_dict'])

    split_encoder = load_from_standard_encoder(
        text_encoder_standard, motion_vocab_size=40,
        freeze_transformer=True, original_vocab_size=42
    )

    video_unet = VideoUNetFixed(
        spatial_unet=spatial_unet, text_encoder=split_encoder,
        num_frames=NUM_FRAMES, image_size=RESOLUTION, temporal_kernel=3,
        temporal_layers=2, freeze_spatial=True, use_temporal=True,
        enable_temporal_processing=True
    )

    exp_checkpoint = torch.load(checkpoint_path, map_location=device)
    video_unet.load_state_dict(exp_checkpoint['model_state_dict'])
    video_unet.eval()
    video_unet = video_unet.to(device)

    gate_values = video_unet.get_gate_values()
    return video_unet, gate_values


def encode_prompt(model, tokenizer, prompt):
    token_ids, mask = tokenizer.encode(prompt, max_length=64)
    token_ids = torch.tensor([token_ids])
    mask = torch.tensor([mask], dtype=torch.float)
    with torch.no_grad():
        text_emb, text_mask = model.text_encoder(token_ids, mask)
    return text_emb, text_mask


def ddim_generate(model, tokenizer, prompt, fixed_noise, alphas_cumprod, timesteps):
    """Run deterministic DDIM sampling. Returns video tensor [1,3,F,H,W] in [0,1]."""
    text_emb, text_mask = encode_prompt(model, tokenizer, prompt)
    x = fixed_noise.clone()
    diagnostics = []

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        t_tensor = torch.tensor([t]).long()

        with torch.no_grad():
            predicted_noise = model(x, t_tensor, text_emb, text_mask)

        alpha_t = alphas_cumprod[t]
        alpha_t_prev = alphas_cumprod[t_prev]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)

        pred_x_0 = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
        direction = sqrt_one_minus_alpha_t_prev * predicted_noise
        x = sqrt_alpha_t_prev * pred_x_0 + direction

        diagnostics.append({
            'step': i,
            't': int(t),
            'has_nan': bool(torch.isnan(x).any().item()),
            'has_inf': bool(torch.isinf(x).any().item()),
            'x_mean': float(x.mean().item()),
            'x_std': float(x.std().item()),
        })
        if diagnostics[-1]['has_nan'] or diagnostics[-1]['has_inf']:
            break

    video = (x.clamp(-1, 1) + 1) / 2
    return video, diagnostics


def compute_motion_metrics(video):
    """
    video: torch tensor [1,3,F,H,W] in [0,1].
    Returns temporal activity, trajectory, and direction metrics.
    """
    v = video[0].permute(1, 2, 3, 0).numpy()  # [F,H,W,3]
    F, H, W, C = v.shape
    gray = v.mean(axis=3)  # [F,H,W]

    # Temporal activity: mean abs diff between consecutive frames
    frame_diffs = np.abs(np.diff(gray, axis=0))
    temporal_activity = float(frame_diffs.mean())
    temporal_activity_max = float(frame_diffs.max())

    # Per-frame "changed region" weighted centroid vs temporal median.
    # Index convention: displacement_vector = [x (column), y (row)].
    median_frame = np.median(gray, axis=0)
    weights_per_frame = []
    centroids_xy = []
    col_idx = np.arange(W)[None, :]
    row_idx = np.arange(H)[:, None]
    for f in range(F):
        diff = np.abs(gray[f] - median_frame)
        # Robust threshold: keep top ~15% of changed pixels
        thresh = np.percentile(diff, 85)
        w = np.clip(diff - thresh, 0, None)
        total = w.sum()
        weights_per_frame.append(float(total))
        if total > 1e-8:
            cx = float((w * col_idx).sum() / total)
            cy = float((w * row_idx).sum() / total)
        else:
            cx, cy = W / 2, H / 2
        centroids_xy.append((cx, cy))
    centroids_xy = np.array(centroids_xy)

    displacement = centroids_xy[-1] - centroids_xy[0]
    disp_mag = float(np.linalg.norm(displacement))
    path_length = float(np.sum(np.linalg.norm(np.diff(centroids_xy, axis=0), axis=1)))
    straightness = disp_mag / path_length if path_length > 1e-8 else 0.0

    return {
        'temporal_activity': temporal_activity,
        'temporal_activity_max': temporal_activity_max,
        'frame_mean_std': float(gray.std()),
        'video_min': float(v.min()),
        'video_max': float(v.max()),
        'displacement_px': disp_mag,
        'displacement_vector': [float(displacement[0]), float(displacement[1])],
        'path_length_px': path_length,
        'straightness': float(straightness),
        'centroid_start': [float(centroids_xy[0, 0]), float(centroids_xy[0, 1])],
        'centroid_end': [float(centroids_xy[-1, 0]), float(centroids_xy[-1, 1])],
        'has_nan': bool(np.isnan(v).any()),
        'has_inf': bool(np.isinf(v).any()),
    }


def direction_accuracy(metrics, motion_type):
    """Cosine similarity between measured displacement and expected direction."""
    expected = np.array(EXPECTED_DIRECTIONS[motion_type])
    measured = np.array(metrics['displacement_vector'])
    exp_norm = np.linalg.norm(expected)
    mea_norm = np.linalg.norm(measured)
    if exp_norm == 0 or mea_norm < 1e-6:
        return {
            'expected': EXPECTED_DIRECTIONS[motion_type],
            'measured': metrics['displacement_vector'],
            'cosine_similarity': None,
            'sign_match_x': None,
            'sign_match_y': None,
            'classification': 'STATIONARY_EXPECTED' if exp_norm == 0 else 'NO_MEASURED_MOTION',
        }
    cos = float(np.dot(expected, measured) / (exp_norm * mea_norm))
    return {
        'expected': EXPECTED_DIRECTIONS[motion_type],
        'measured': metrics['displacement_vector'],
        'cosine_similarity': cos,
        'sign_match_x': bool(np.sign(measured[0]) == np.sign(expected[0])) if expected[0] != 0 else None,
        'sign_match_y': bool(np.sign(measured[1]) == np.sign(expected[1])) if expected[1] != 0 else None,
        'classification': 'DIRECTION_MATCH' if cos > 0.5 else ('PARTIAL' if cos > 0 else 'DIRECTION_MISMATCH'),
    }


def select_prompts():
    """Pick first training caption per motion type from the synthetic dataset."""
    metadata_path = base_dir / "experiments_stage3_2_7/data_synthetic/train_metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    seen = {}
    for item in metadata:
        mt = item['motion_type']
        if mt not in seen:
            seen[mt] = item['caption']
    ordered = ['left_to_right', 'right_to_left', 'top_to_bottom',
               'bottom_to_top', 'diagonal', 'stationary']
    prompts = []
    for mt in ordered:
        if mt in seen:
            prompts.append({'motion_type': mt, 'caption': seen[mt]})
    return prompts


def smoke_test(model, tokenizer, alphas_cumprod, timesteps):
    """Phase 3: single denoise + 5 DDIM steps sanity check."""
    results = {'tests': []}
    prompt = "A medium blue circle slowly moving from left to right on a white background."

    # Test 1: single denoise
    text_emb, text_mask = encode_prompt(model, tokenizer, prompt)
    noise = torch.randn(1, 3, NUM_FRAMES, RESOLUTION, RESOLUTION)
    t = torch.tensor([999]).long()
    with torch.no_grad():
        eps = model(noise, t, text_emb, text_mask)
    results['tests'].append({
        'test': 'single_denoise',
        'output_shape': list(eps.shape),
        'has_nan': bool(torch.isnan(eps).any().item()),
        'has_inf': bool(torch.isinf(eps).any().item()),
        'mean': float(eps.mean().item()),
        'std': float(eps.std().item()),
    })

    # Test 2: 5 DDIM steps
    torch.manual_seed(42)
    fixed_noise = torch.randn(1, 3, NUM_FRAMES, RESOLUTION, RESOLUTION)
    sub_timesteps = timesteps[:6]  # 5 update pairs
    video, diag = ddim_generate(model, tokenizer, prompt, fixed_noise,
                                alphas_cumprod, sub_timesteps)
    results['tests'].append({
        'test': 'five_ddim_steps',
        'steps_completed': len(diag),
        'any_nan': any(d['has_nan'] for d in diag),
        'any_inf': any(d['has_inf'] for d in diag),
        'output_min': float(video.min()),
        'output_max': float(video.max()),
    })
    results['passed'] = all(
        (not t.get('has_nan', t.get('any_nan', False))) and
        (not t.get('has_inf', t.get('any_inf', False)))
        for t in results['tests']
    )
    return results


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    alphas_cumprod, timesteps = build_diffusion_schedule()

    tokenizer_path = base_dir / "experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json"
    tokenizer = SimpleTokenizer.load(tokenizer_path)

    prompts = select_prompts()
    print(f"Selected {len(prompts)} prompts:")
    for p in prompts:
        print(f"  [{p['motion_type']}] {p['caption']}")

    # ---- Phase 3: smoke test on C25 ----
    print("\n=== PHASE 3: SMOKE TEST (C25 step-250) ===")
    model_c25, gates_c25 = load_model(CKPT_C25)
    print(f"C25 gates: {gates_c25}")
    smoke = smoke_test(model_c25, tokenizer, alphas_cumprod, timesteps)
    smoke_path = OUT_DIR / "L11_INFERENCE_SMOKE_TEST.json"
    with open(smoke_path, 'w') as f:
        json.dump(smoke, f, indent=2)
    print(f"Smoke test passed={smoke['passed']}; saved to {smoke_path}")
    if not smoke['passed']:
        print("SMOKE TEST FAILED - aborting")
        return 1

    # ---- Phase 4: matched generation ----
    print("\n=== PHASE 4: MATCHED DDIM GENERATION ===")
    model_c1, gates_c1 = load_model(CKPT_C1)
    print(f"C1 gates: {gates_c1}")

    all_results = {}
    for model_name, model in [('C1', model_c1), ('C25', model_c25)]:
        all_results[model_name] = {}
        for i, p in enumerate(prompts):
            # Fixed noise per prompt index - identical across models
            g = torch.Generator().manual_seed(1000 + i)
            fixed_noise = torch.randn(1, 3, NUM_FRAMES, RESOLUTION, RESOLUTION, generator=g)

            print(f"[{model_name}] prompt {i+1}/{len(prompts)} [{p['motion_type']}]")
            t_start = time.time()
            video, diag = ddim_generate(model, tokenizer, p['caption'],
                                        fixed_noise, alphas_cumprod, timesteps)
            elapsed = time.time() - t_start
            print(f"  done in {elapsed:.1f}s, steps={len(diag)}, "
                  f"nan={any(d['has_nan'] for d in diag)}")

            metrics = compute_motion_metrics(video)
            dir_eval = direction_accuracy(metrics, p['motion_type'])

            # Save raw video frames for audit (PNG)
            frames_dir = OUT_DIR / "videos" / model_name / f"{i:02d}_{p['motion_type']}"
            frames_dir.mkdir(parents=True, exist_ok=True)
            v_np = video[0].permute(1, 2, 3, 0).numpy()  # [F,H,W,3]
            v_np = (np.clip(v_np, 0, 1) * 255).astype(np.uint8)
            try:
                from PIL import Image
                for fi in range(NUM_FRAMES):
                    Image.fromarray(v_np[fi]).save(frames_dir / f"frame_{fi:04d}.png")
            except Exception as e:
                print(f"  frame save failed: {e}")

            all_results[model_name][p['motion_type']] = {
                'caption': p['caption'],
                'prompt_index': i,
                'generation_time_s': elapsed,
                'ddim_steps_completed': len(diag),
                'metrics': metrics,
                'direction_evaluation': dir_eval,
            }

    # ---- Phase 5+6: comparison ----
    comparison = {
        'checkpoints': {
            'C1': str(CKPT_C1),
            'C25': str(CKPT_C25),
        },
        'gates': {'C1': gates_c1, 'C25': gates_c25},
        'sampling': {
            'num_timesteps': NUM_TIMESTEPS,
            'sampling_steps': SAMPLING_STEPS,
            'beta_start': BETA_START,
            'beta_end': BETA_END,
            'frames': NUM_FRAMES,
            'resolution': RESOLUTION,
            'noise': 'fixed per prompt index (seed 1000+i), identical across models',
        },
        'per_prompt': {},
        'summary': {},
    }

    motion_types = [p['motion_type'] for p in prompts]
    for mt in motion_types:
        c1 = all_results['C1'][mt]
        c25 = all_results['C25'][mt]
        comparison['per_prompt'][mt] = {
            'caption': c1['caption'],
            'C1': {
                'temporal_activity': c1['metrics']['temporal_activity'],
                'displacement_px': c1['metrics']['displacement_px'],
                'path_length_px': c1['metrics']['path_length_px'],
                'straightness': c1['metrics']['straightness'],
                'direction_cosine': c1['direction_evaluation']['cosine_similarity'],
                'direction_class': c1['direction_evaluation']['classification'],
            },
            'C25': {
                'temporal_activity': c25['metrics']['temporal_activity'],
                'displacement_px': c25['metrics']['displacement_px'],
                'path_length_px': c25['metrics']['path_length_px'],
                'straightness': c25['metrics']['straightness'],
                'direction_cosine': c25['direction_evaluation']['cosine_similarity'],
                'direction_class': c25['direction_evaluation']['classification'],
            },
        }

    # Aggregate
    def agg(model):
        acts = [all_results[model][mt]['metrics']['temporal_activity'] for mt in motion_types]
        disps = [all_results[model][mt]['metrics']['displacement_px'] for mt in motion_types]
        cosines = [all_results[model][mt]['direction_evaluation']['cosine_similarity']
                   for mt in motion_types
                   if all_results[model][mt]['direction_evaluation']['cosine_similarity'] is not None]
        classes = [all_results[model][mt]['direction_evaluation']['classification']
                   for mt in motion_types]
        return {
            'mean_temporal_activity': float(np.mean(acts)),
            'mean_displacement_px': float(np.mean(disps)),
            'mean_direction_cosine': float(np.mean(cosines)) if cosines else None,
            'direction_class_counts': {c: classes.count(c) for c in set(classes)},
        }

    comparison['summary'] = {'C1': agg('C1'), 'C25': agg('C25')}

    eval_path = OUT_DIR / "L11_MOTION_EVALUATION.json"
    with open(eval_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    comp_path = OUT_DIR / "L11_C1_C25_COMPARISON.json"
    with open(comp_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(comparison['summary'], indent=2))
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print(f"Saved: {eval_path}")
    print(f"Saved: {comp_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
