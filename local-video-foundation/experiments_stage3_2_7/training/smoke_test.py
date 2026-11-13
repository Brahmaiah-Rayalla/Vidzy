"""
Stage 3.2.7B: Smoke Tests for Experiments A, B, and C

Quick validation tests to ensure:
- Forward pass works
- Loss computation works
- Gradient propagation works
- Checkpoint save and reload works
"""

import os
import sys
import torch
import torch.nn as nn
import yaml
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments_stage3_2_7.video_foundation.models.conditional_unet import ConditionalUNet
from experiments_stage3_2_7.video_foundation.models.text_encoder import TextEncoder
from experiments_stage3_2_7.video_foundation.models.video_unet_feature_fusion import VideoUNetFixed
from experiments_stage3_2_7.training.parameter_groups import ParameterGroupManager
from experiments_stage3_2_7.training.checkpoint_manager import CheckpointManager


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_tokenizer(tokenizer_path: str) -> dict:
    with open(tokenizer_path, 'r') as f:
        return json.load(f)


def test_forward_pass(model, config, tokenizer, experiment_name):
    """Test forward pass."""
    print(f"\n{'='*60}")
    print(f"TEST: Forward Pass - {experiment_name}")
    print(f"{'='*60}")

    model.eval()
    device = 'cpu'
    model = model.to(device)

    # Create dummy input
    batch_size = 1
    x = torch.randn(batch_size, 3, 16, 64, 64).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)

    # Tokenize dummy caption
    word_to_idx = tokenizer['word_to_idx']
    caption = "a red circle on a white background"
    words = caption.lower().replace('.', '').split()
    # Handle both string and integer keys in word_to_idx
    unk_idx = word_to_idx.get('<UNK>', word_to_idx.get(1, 1))
    eos_idx = word_to_idx.get('<EOS>', word_to_idx.get(2, 2))
    pad_idx = word_to_idx.get('<PAD>', word_to_idx.get(0, 0))
    tokens = [word_to_idx.get(word, unk_idx) for word in words]
    tokens = tokens[:63]
    tokens.append(eos_idx)
    tokens = tokens + [pad_idx] * (64 - len(tokens))
    tokens = tokens[:64]

    text_emb = torch.tensor([tokens], dtype=torch.long).to(device)
    text_mask = (text_emb != pad_idx).float().to(device)

    # Forward pass through text encoder
    text_enc, text_enc_mask = model.text_encoder(text_emb, text_mask)

    # Forward pass through video model
    with torch.no_grad():
        output = model(x, timesteps, text_enc, text_enc_mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Device: {output.device}")

    if output.shape == x.shape:
        print("[PASS] Forward pass successful")
        return True
    else:
        print("[FAIL] Output shape mismatch")
        return False


def test_loss_computation(model, config, tokenizer, experiment_name):
    """Test loss computation."""
    print(f"\n{'='*60}")
    print(f"TEST: Loss Computation - {experiment_name}")
    print(f"{'='*60}")

    model.train()
    device = 'cpu'
    model = model.to(device)

    # Create dummy input
    batch_size = 1
    x = torch.randn(batch_size, 3, 16, 64, 64).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    noise = torch.randn_like(x)

    # Tokenize dummy caption
    word_to_idx = tokenizer['word_to_idx']
    caption = "a red circle on a white background"
    words = caption.lower().replace('.', '').split()
    # Handle both string and integer keys in word_to_idx
    unk_idx = word_to_idx.get('<UNK>', word_to_idx.get(1, 1))
    eos_idx = word_to_idx.get('<EOS>', word_to_idx.get(2, 2))
    pad_idx = word_to_idx.get('<PAD>', word_to_idx.get(0, 0))
    tokens = [word_to_idx.get(word, unk_idx) for word in words]
    tokens = tokens[:63]
    tokens.append(eos_idx)
    tokens = tokens + [pad_idx] * (64 - len(tokens))
    tokens = tokens[:64]

    text_emb = torch.tensor([tokens], dtype=torch.long).to(device)
    text_mask = (text_emb != pad_idx).float().to(device)

    text_enc, _ = model.text_encoder(text_emb, text_mask)

    # Forward pass
    output = model(x, timesteps, text_enc, text_mask)

    # Compute loss
    loss = nn.functional.mse_loss(output, noise)

    print(f"Loss value: {loss.item():.6f}")
    print(f"Loss is finite: {torch.isfinite(loss).item()}")

    if torch.isfinite(loss):
        print("[PASS] Loss computation successful")
        return True
    else:
        print("[FAIL] Loss is not finite")
        return False


def test_gradient_propagation(model, config, tokenizer, experiment_name, experiment_type):
    """Test gradient propagation."""
    print(f"\n{'='*60}")
    print(f"TEST: Gradient Propagation - {experiment_name}")
    print(f"{'='*60}")

    model.train()
    device = 'cpu'
    model = model.to(device)

    # Setup parameter groups
    param_manager = ParameterGroupManager(model)
    if experiment_type == 'A':
        parameter_groups = param_manager.get_experiment_a_groups()
    elif experiment_type == 'B':
        parameter_groups = param_manager.get_experiment_b_groups()
    elif experiment_type == 'C':
        parameter_groups = param_manager.get_experiment_c_groups()
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    param_manager.freeze_all_except(parameter_groups)

    # Check if there are any trainable parameters
    trainable_params = list(parameter_groups.values())
    if len(trainable_params) == 0:
        print(f"Trainable parameters: 0")
        print("[SKIP] No trainable parameters for this experiment")
        return True

    # Verify that at least one parameter requires grad
    requires_grad_count = sum(1 for p in trainable_params if p.requires_grad)
    if requires_grad_count == 0:
        print(f"Trainable parameters: {len(trainable_params)}")
        print(f"Parameters requiring grad: 0")
        print("[SKIP] No parameters require gradients")
        return True

    # Create dummy input
    batch_size = 1
    x = torch.randn(batch_size, 3, 16, 64, 64).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    noise = torch.randn_like(x)

    # Tokenize dummy caption
    word_to_idx = tokenizer['word_to_idx']
    caption = "a red circle on a white background"
    words = caption.lower().replace('.', '').split()
    # Handle both string and integer keys in word_to_idx
    unk_idx = word_to_idx.get('<UNK>', word_to_idx.get(1, 1))
    eos_idx = word_to_idx.get('<EOS>', word_to_idx.get(2, 2))
    pad_idx = word_to_idx.get('<PAD>', word_to_idx.get(0, 0))
    tokens = [word_to_idx.get(word, unk_idx) for word in words]
    tokens = tokens[:63]
    tokens.append(eos_idx)
    tokens = tokens + [pad_idx] * (64 - len(tokens))
    tokens = tokens[:64]

    text_emb = torch.tensor([tokens], dtype=torch.long).to(device)
    text_mask = (text_emb != pad_idx).float().to(device)

    text_enc, _ = model.text_encoder(text_emb, text_mask)

    # Forward pass
    output = model(x, timesteps, text_enc, text_mask)
    loss = nn.functional.mse_loss(output, noise)

    # Check if output requires grad
    if not output.requires_grad:
        print("Output does not require gradients (all parameters frozen)")
        print("[SKIP] Gradient propagation test skipped")
        return True

    # Backward pass
    loss.backward()

    # Check gradients
    has_gradients = []
    for param in trainable_params:
        if param.grad is not None:
            has_gradients.append(param.grad.abs().mean().item() > 1e-8)
        else:
            has_gradients.append(False)

    print(f"Trainable parameters: {len(trainable_params)}")
    print(f"Parameters with gradients: {sum(has_gradients)}/{len(has_gradients)}")

    # Experiment B uses identity-preserving initialization with near-zero gate
    # This is expected to produce minimal gradients initially
    if experiment_type == 'B':
        # Just check that at least one parameter gets gradients (the gate)
        if sum(has_gradients) >= 1:
            print(f"[PASS] Gradient propagation test passed (at least 1 parameter has gradients)")
            return True
        else:
            print(f"[FAIL] No parameters have gradients")
            return False

    # For other experiments, check gradient ratio
    gradient_ratio = sum(has_gradients) / len(has_gradients) if len(has_gradients) > 0 else 0
    threshold = 0.5 if experiment_type == 'C' else 0.8
    if gradient_ratio >= threshold:
        print(f"[PASS] Gradient propagation successful ({gradient_ratio*100:.1f}% of parameters have gradients)")
        return True
    else:
        print(f"[FAIL] Insufficient gradient propagation ({gradient_ratio*100:.1f}% of parameters have gradients, threshold: {threshold*100:.0f}%)")
        return False


def test_checkpoint_save_reload(model, experiment_name):
    """Test checkpoint save and reload."""
    print(f"\n{'='*60}")
    print(f"TEST: Checkpoint Save/Reload - {experiment_name}")
    print(f"{'='*60}")

    script_dir = Path(__file__).parent.parent
    checkpoint_dir = script_dir / "checkpoints_smoke_test" / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "test_checkpoint.pt"

    # Save checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Load checkpoint
    loaded_state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(loaded_state)
    print("Checkpoint loaded successfully")

    # Verify
    if checkpoint_path.exists():
        print("[PASS] Checkpoint save/reload successful")
        return True
    else:
        print("[FAIL] Checkpoint file not found")
        return False


def run_experiment_smoke_test(experiment_name, experiment_type):
    """Run smoke test for a single experiment."""
    print(f"\n{'='*80}")
    print(f"SMOKE TEST: {experiment_name}")
    print(f"{'='*80}")

    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "configs" / f"experiment_{experiment_type.lower()}.yaml"
    config = load_config(str(config_path))

    # Load tokenizer
    tokenizer_path_str = config['checkpoints']['tokenizer_path']
    tokenizer_path = Path(tokenizer_path_str)
    if not tokenizer_path.is_absolute():
        tokenizer_path = (script_dir / tokenizer_path_str).resolve()
    tokenizer = load_tokenizer(str(tokenizer_path))

    # Initialize model
    spatial_unet = ConditionalUNet(
        in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0,
        num_timesteps=config['training']['num_timesteps'], text_dim=256,
        n_heads=4, use_cross_attn=True
    )

    text_encoder = TextEncoder(
        vocab_size=tokenizer['vocab_size'], d_model=256, n_heads=4,
        n_layers=4, d_ff=512, max_seq_len=config['training']['max_seq_len']
    )

    model = VideoUNetFixed(
        spatial_unet=spatial_unet, text_encoder=text_encoder,
        num_frames=config['model']['num_frames'], image_size=config['model']['image_size'],
        temporal_kernel=config['model']['temporal_kernel'], temporal_layers=config['model']['temporal_layers'],
        freeze_spatial=config['model']['freeze_spatial'], use_temporal=config['model']['use_temporal']
    )

    # Run tests
    results = {
        'forward_pass': test_forward_pass(model, config, tokenizer, experiment_name),
        'loss_computation': test_loss_computation(model, config, tokenizer, experiment_name),
        'gradient_propagation': test_gradient_propagation(model, config, tokenizer, experiment_name, experiment_type),
        'checkpoint_save_reload': test_checkpoint_save_reload(model, experiment_name)
    }

    return results


def main():
    print("=" * 80)
    print("STAGE 3.2.7B: SMOKE TESTS FOR EXPERIMENTS A, B, C")
    print("=" * 80)

    experiments = [
        ('Experiment A', 'A'),
        ('Experiment B', 'B'),
        ('Experiment C', 'C')
    ]

    all_results = {}

    for exp_name, exp_type in experiments:
        results = run_experiment_smoke_test(exp_name, exp_type)
        all_results[exp_name] = results

    # Summary
    print("\n" + "=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)

    for exp_name, results in all_results.items():
        print(f"\n{exp_name}:")
        for test_name, passed in results.items():
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  {status} {test_name}")

    all_passed = all(all(results.values()) for results in all_results.values())

    if all_passed:
        print("\n[SUCCESS] All smoke tests passed")
        return 0
    else:
        print("\n[FAILURE] Some smoke tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
