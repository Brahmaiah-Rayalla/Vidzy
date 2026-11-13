"""
Stage 3.2.9L.12 - shared utilities for gate attribution experiments.

Reuses the verified L.11 inference path:
    tokenizer -> video_unet.text_encoder -> VideoUNetFixed(x_t, t, text_emb, text_mask)
DDIM math identical to L.11 / Stage 3.2.9E DDIMSampler.
"""

import torch
import json
import numpy as np
from pathlib import Path
import sys
import math

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

L11_DIR = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l11_ddim_motion_eval"
L12_DIR = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/l12_gate_attribution"
CKPT_C1 = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/C1/checkpoint_step_000250.pt"
CKPT_C25 = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/extended_250_steps/recovery_l10/C25/checkpoint_step_000250.pt"

NUM_FRAMES = 16
RESOLUTION = 64
SAMPLING_STEPS = 25
NUM_TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02

# Gate override table: activated sigmoid value -> raw logit stored in .gate
GATE_LOGIT = {
    0.50: 0.0,
    0.25: -1.0986122886681098,
    0.10: -2.1972245773362196,
    0.01: -4.59511985013459,
    0.001: -6.906754778648553,
}
GATE_ZERO_LOGIT = -15.0  # sigmoid(-15) ~= 3e-7, effectively zero but keeps temporal path

GATE_CONFIGS = {
    'C1_native':          {'checkpoint': 'C1',  'bottleneck': None, 'low_res': None},
    'C25_native':         {'checkpoint': 'C25', 'bottleneck': None, 'low_res': None},
    'C25_10_10':          {'checkpoint': 'C25', 'bottleneck': 0.10, 'low_res': 0.10},
    'C25_25_10':          {'checkpoint': 'C25', 'bottleneck': 0.25, 'low_res': 0.10},
    'C25_10_25':          {'checkpoint': 'C25', 'bottleneck': 0.10, 'low_res': 0.25},
    'C25_25_0':           {'checkpoint': 'C25', 'bottleneck': 0.25, 'low_res': 0.0},
    'C25_0_25':           {'checkpoint': 'C25', 'bottleneck': 0.0,  'low_res': 0.25},
}

EXPECTED_DIRECTIONS = {
    'left_to_right': (1.0, 0.0),
    'right_to_left': (-1.0, 0.0),
    'top_to_bottom': (0.0, 1.0),
    'bottom_to_top': (0.0, -1.0),
    'diagonal': (1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
    'stationary': (0.0, 0.0),
}

# Prompt color priors for object detection (RGB thresholds, loose)
PROMPT_COLORS = {
    'left_to_right':   {'name': 'blue',   'rule': lambda r, g, b: (b > 0.45) & (b > r * 1.3) & (b > g * 1.1)},
    'right_to_left':   {'name': 'pink',   'rule': lambda r, g, b: (r > 0.45) & (b > 0.30) & (r > g * 1.4) & (b > g * 1.2)},
    'top_to_bottom':   {'name': 'pink',   'rule': lambda r, g, b: (r > 0.45) & (b > 0.30) & (r > g * 1.4) & (b > g * 1.2)},
    'bottom_to_top':   {'name': 'white',  'rule': lambda r, g, b: (r > 0.75) & (g > 0.75) & (b > 0.75)},
    'diagonal':        {'name': 'purple', 'rule': lambda r, g, b: (r > 0.30) & (b > 0.30) & (g < r * 0.7) & (g < b * 0.7) & (np.abs(r - b) < 0.35)},
    'stationary':      {'name': 'cyan',   'rule': lambda r, g, b: (g > 0.40) & (b > 0.40) & (r < g * 0.6) & (r < b * 0.6)},
}


def build_diffusion_schedule():
    betas = torch.linspace(BETA_START, BETA_END, NUM_TIMESTEPS)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    timesteps = torch.linspace(NUM_TIMESTEPS - 1, 0, SAMPLING_STEPS).long()
    return alphas_cumprod, timesteps


def load_model(checkpoint_path):
    """Load VideoUNetFixed exactly as L.7/L.10/L.11 do."""
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
    return video_unet


def get_gate_raw_and_activated(model):
    out = {}
    for name in ['bottleneck', 'low_res']:
        mod = model.fusion_modules[name]
        raw = float(mod.gate.item())
        act = float(torch.sigmoid(mod.gate).item())
        out[name] = {'raw_logit': raw, 'activated': act}
    return out


def set_gate_override(model, bottleneck_activated, low_res_activated):
    """Inference-only gate override. Pass None to keep native value."""
    if bottleneck_activated is not None:
        raw = GATE_ZERO_LOGIT if bottleneck_activated <= 0.0 else GATE_LOGIT[bottleneck_activated]
        model.fusion_modules['bottleneck'].gate.data.fill_(raw)
    if low_res_activated is not None:
        raw = GATE_ZERO_LOGIT if low_res_activated <= 0.0 else GATE_LOGIT[low_res_activated]
        model.fusion_modules['low_res'].gate.data.fill_(raw)


def encode_prompt(model, tokenizer, prompt):
    token_ids, mask = tokenizer.encode(prompt, max_length=64)
    token_ids = torch.tensor([token_ids])
    mask = torch.tensor([mask], dtype=torch.float)
    with torch.no_grad():
        text_emb, text_mask = model.text_encoder(token_ids, mask)
    return text_emb, text_mask


def ddim_generate(model, tokenizer, prompt, fixed_noise, alphas_cumprod, timesteps,
                  return_diagnostics=False):
    """Deterministic DDIM sampling identical to L.11."""
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

        diag = {
            'step': i, 't': int(t),
            'has_nan': bool(torch.isnan(x).any().item()),
            'has_inf': bool(torch.isinf(x).any().item()),
            'x_mean': float(x.mean().item()),
            'x_std': float(x.std().item()),
        }
        diagnostics.append(diag)
        if diag['has_nan'] or diag['has_inf']:
            break

    video = (x.clamp(-1, 1) + 1) / 2
    if return_diagnostics:
        return video, diagnostics
    return video


def fixed_noise_for_prompt(prompt_index):
    """L.11 seed scheme: torch.Generator seeded 1000 + prompt_index."""
    g = torch.Generator().manual_seed(1000 + prompt_index)
    return torch.randn(1, 3, NUM_FRAMES, RESOLUTION, RESOLUTION, generator=g)


def select_prompts():
    """First training caption per motion type - identical to L.11."""
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
    return [{'motion_type': mt, 'caption': seen[mt]} for mt in ordered if mt in seen]


def video_to_numpy(video):
    """[1,3,F,H,W] in [0,1] -> [F,H,W,3] float32."""
    return video[0].permute(1, 2, 3, 0).numpy().astype(np.float32)


def compute_motion_metrics(video):
    """Identical metric definitions to L.11 run_l11_evaluation.py."""
    v = video_to_numpy(video)
    F, H, W, C = v.shape
    gray = v.mean(axis=3)

    frame_diffs = np.abs(np.diff(gray, axis=0))
    temporal_activity = float(frame_diffs.mean())
    temporal_activity_max = float(frame_diffs.max())

    median_frame = np.median(gray, axis=0)
    col_idx = np.arange(W)[None, :]
    row_idx = np.arange(H)[:, None]
    centroids_xy = []
    for f in range(F):
        diff = np.abs(gray[f] - median_frame)
        thresh = np.percentile(diff, 85)
        w = np.clip(diff - thresh, 0, None)
        total = w.sum()
        if total > 1e-8:
            cx = float((w * col_idx).sum() / total)
            cy = float((w * row_idx).sum() / total)
        else:
            cx, cy = W / 2, H / 2
        centroids_xy.append((cx, cy))
    centroids_xy = np.array(centroids_xy)

    displacement = centroids_xy[-1] - centroids_xy[0]
    disp_mag = float(np.linalg.norm(displacement))
    diffs = np.diff(centroids_xy, axis=0)
    step_lengths = np.linalg.norm(diffs, axis=1)
    path_length = float(step_lengths.sum())
    straightness = disp_mag / path_length if path_length > 1e-8 else 0.0
    # frame-to-frame position variation (jitter)
    centroid_jitter = float(step_lengths.mean())
    # abrupt transitions: max single-frame centroid jump relative to mean
    abrupt_ratio = float(step_lengths.max() / (step_lengths.mean() + 1e-9))

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
        'centroid_jitter_px': centroid_jitter,
        'abrupt_transition_ratio': abrupt_ratio,
        'centroid_trajectory': centroids_xy.tolist(),
        'has_nan': bool(np.isnan(v).any()),
        'has_inf': bool(np.isinf(v).any()),
    }


def direction_accuracy(metrics, motion_type):
    """Identical to L.11 direction_accuracy."""
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


def compute_detection_metrics(video, motion_type):
    """
    Color/shape-based object detection reliability metrics.
    Returns per-frame detection flags, mask mass, color-mask centroid,
    and agreement with the generic diff centroid.
    """
    v = video_to_numpy(video)  # [F,H,W,3] float in [0,1]
    F, H, W, C = v.shape
    r, g, b = v[..., 0], v[..., 1], v[..., 2]

    if motion_type not in PROMPT_COLORS:
        return {'available': False, 'reason': 'no color prior'}
    rule = PROMPT_COLORS[motion_type]['rule']
    mask = rule(r, g, b)  # [F,H,W] bool

    # Expected object size ~ (2*radius+1)^2 px, radius 4-8 -> ~81-289 px
    # Detection if mask mass within plausible range
    plausible_min, plausible_max = 15, int(0.35 * H * W)

    frames = []
    det_centroids = []
    col_idx = np.arange(W)[None, :]
    row_idx = np.arange(H)[:, None]
    for f in range(F):
        m = mask[f]
        mass = int(m.sum())
        detected = plausible_min <= mass <= plausible_max
        if detected:
            w = m.astype(np.float32)
            tot = w.sum()
            cx = float((w * col_idx).sum() / tot)
            cy = float((w * row_idx).sum() / tot)
            det_centroids.append((cx, cy))
        else:
            det_centroids.append(None)
        frames.append({'frame': f, 'mask_mass': mass, 'detected': bool(detected)})

    n_det = sum(1 for x in det_centroids if x is not None)
    # Color-mask centroid trajectory over detected frames only
    det_traj = [c for c in det_centroids if c is not None]
    color_disp = None
    color_cos = None
    if len(det_traj) >= 4:
        arr = np.array(det_traj)
        d = arr[-1] - arr[0]
        color_disp = float(np.linalg.norm(d))
        exp = np.array(EXPECTED_DIRECTIONS[motion_type])
        if np.linalg.norm(exp) > 0 and np.linalg.norm(d) > 1e-6:
            color_cos = float(np.dot(exp, d) / (np.linalg.norm(exp) * np.linalg.norm(d)))

    # Mask mass stats
    masses = np.array([f_['mask_mass'] for f_ in frames], dtype=float)

    return {
        'available': True,
        'color_prior': PROMPT_COLORS[motion_type]['name'],
        'frames_detected': n_det,
        'detection_rate': n_det / F,
        'mask_mass_mean': float(masses.mean()),
        'mask_mass_std': float(masses.std()),
        'color_centroid_displacement_px': color_disp,
        'color_direction_cosine': color_cos,
        'per_frame': frames,
    }


def compute_global_motion_metrics(video, max_shift=10):
    """
    Supporting motion measurement independent of object detection.
    For each consecutive frame pair, find the integer (dx, dy) shift in
    [-max_shift, max_shift] that minimizes MSE between grayscale frames
    (frame t+1 shifted back onto frame t). Reports the translation sequence,
    net displacement, directional coherence vs expected motion, and the
    fraction of frame pairs whose shift agrees with the expected direction.
    """
    v = video_to_numpy(video)
    F, H, W, C = v.shape
    gray = v.mean(axis=3)  # [F,H,W]

    shifts = []
    for f in range(F - 1):
        a = gray[f]
        b = gray[f + 1]
        best = (None, np.inf)
        m = max_shift
        a_c = a[m:H - m, m:W - m]
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                b_c = b[m + dy:H - m + dy, m + dx:W - m + dx]
                mse = float(np.mean((a_c - b_c) ** 2))
                if mse < best[1]:
                    best = ((dx, dy), mse)
        shifts.append({'frame_pair': f, 'dx': best[0][0], 'dy': best[0][1],
                       'mse': best[1]})

    arr = np.array([[s['dx'], s['dy']] for s in shifts], dtype=float)
    net = arr.sum(axis=0) if len(arr) else np.zeros(2)
    mean_shift = arr.mean(axis=0) if len(arr) else np.zeros(2)
    if np.linalg.norm(net) > 1e-6:
        dots = arr @ net
        agree = float(np.mean(dots > 0))
    else:
        agree = 0.0
    return {
        'num_pairs': len(shifts),
        'net_translation_px': [float(net[0]), float(net[1])],
        'mean_translation_px': [float(mean_shift[0]), float(mean_shift[1])],
        'direction_coherence': agree,
        'per_pair_shifts': shifts,
    }


def direction_accuracy_from_translation(global_motion, motion_type):
    """Cosine between expected direction and measured global translation."""
    expected = np.array(EXPECTED_DIRECTIONS[motion_type])
    measured = np.array(global_motion['net_translation_px'])
    exp_norm = np.linalg.norm(expected)
    mea_norm = np.linalg.norm(measured)
    if exp_norm == 0 or mea_norm < 1e-6:
        return None
    return float(np.dot(expected, measured) / (exp_norm * mea_norm))


def make_contact_sheet(video, out_path, cols=8):
    """Save a contact sheet PNG of all frames."""
    from PIL import Image
    v = video_to_numpy(video)  # [F,H,W,3]
    F, H, W, _ = v.shape
    rows = int(np.ceil(F / cols))
    sheet = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
    for i in range(F):
        r, c = divmod(i, cols)
        sheet[r * H:(r + 1) * H, c * W:(c + 1) * W] = (np.clip(v[i], 0, 1) * 255).astype(np.uint8)
    Image.fromarray(sheet).save(out_path)
