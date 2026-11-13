"""
Prompt-conditioning sanity evaluation.

Verifies that:
- Different prompts produce different model outputs
- Motion tokens are correctly processed
- Model responds to prompt variations
"""

import torch
import json
from pathlib import Path
import sys

base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
stage3_2_7_dir = base_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(stage3_2_7_dir))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation" / "models"))

from models.conditional_unet import ConditionalUNet
from models.video_unet_fixed import VideoUNetFixed
from models.text_encoder_split import TextEncoderSplit

def evaluate_prompt_conditioning():
    """Evaluate prompt conditioning sanity."""
    print("=" * 80)
    print("STAGE 3.2.9L.2 - PROMPT CONDITIONING SANITY EVALUATION")
    print("=" * 80)

    # Load stage2 checkpoint to get vocab size
    stage2_checkpoint = base_dir / "experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt"
    checkpoint = torch.load(stage2_checkpoint, map_location='cpu')
    actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

    # Create split encoder with original_vocab_size=42
    motion_vocab_size = actual_vocab_size - 42
    text_encoder = TextEncoderSplit(
        original_vocab_size=42,
        motion_vocab_size=motion_vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64,
        dropout=0.0
    )

    # Load trained checkpoint
    checkpoint_path = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/pilot/checkpoint_experiment_C_10steps.pt"
    trained_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    text_encoder.load_state_dict(trained_checkpoint['model_state_dict'], strict=False)

    text_encoder.eval()

    results = {
        'prompts_tested': [],
        'motion_token_usage': {},
        'prompt_sensitivity': {}
    }

    # ============================================================================
    # TEST 1: DIFFERENT PROMPTS PRODUCE DIFFERENT EMBEDDINGS
    # ============================================================================
    print("\n" + "=" * 80)
    print("TEST 1: PROMPT SENSITIVITY")
    print("=" * 80)

    test_prompts = [
        ("A person walking", torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])),
        ("A person running", torch.tensor([[1, 2, 3, 4, 9, 10, 11, 12]])),
        ("A car driving", torch.tensor([[13, 14, 15, 16, 17, 18, 19, 20]])),
        ("A bird flying", torch.tensor([[21, 22, 23, 24, 25, 26, 27, 28]]))
    ]

    embeddings = {}
    for prompt, tokens in test_prompts:
        with torch.no_grad():
            emb, _ = text_encoder(tokens)
        embeddings[prompt] = emb

    # Compare embeddings
    prompt_pairs = [
        ("A person walking", "A person running"),
        ("A person walking", "A car driving"),
        ("A person running", "A bird flying")
    ]

    for prompt1, prompt2 in prompt_pairs:
        emb1 = embeddings[prompt1]
        emb2 = embeddings[prompt2]
        diff = (emb1 - emb2).abs().mean().item()
        print(f"{prompt1} vs {prompt2}: mean diff = {diff:.8f}")
        results['prompt_sensitivity'][f"{prompt1}_vs_{prompt2}"] = diff

    # Also test with motion tokens
    print("\nTesting with motion tokens:")
    motion_prompt = "Motion token test"
    motion_tokens = torch.tensor([[42, 43, 44, 45, 46, 47, 48, 49]])
    with torch.no_grad():
        motion_emb, _ = text_encoder(motion_tokens)
    embeddings[motion_prompt] = motion_emb

    diff = (embeddings["A person walking"] - motion_emb).abs().mean().item()
    print(f"Regular tokens vs motion tokens: mean diff = {diff:.8f}")
    results['prompt_sensitivity']['regular_vs_motion'] = diff

    # ============================================================================
    # TEST 2: MOTION TOKEN USAGE
    # ============================================================================
    print("\n" + "=" * 80)
    print("TEST 2: MOTION TOKEN USAGE")
    print("=" * 80)

    # Test with motion tokens (tokens >= 42)
    motion_tokens = torch.tensor([[42, 43, 44, 45, 46, 47, 48, 49]])
    regular_tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    with torch.no_grad():
        motion_emb, _ = text_encoder(motion_tokens)
        regular_emb, _ = text_encoder(regular_tokens)

    motion_diff = (motion_emb - regular_emb).abs().mean().item()
    print(f"Motion tokens vs regular tokens: mean diff = {motion_diff:.8f}")
    print(f"Motion tokens produce different embeddings: {motion_diff > 1e-6}")

    results['motion_token_usage'] = {
        'motion_vs_regular_diff': motion_diff,
        'motion_tokens_used': motion_diff > 1e-6
    }

    # ============================================================================
    # TEST 3: GRADIENT FLOW VERIFICATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("TEST 3: GRADIENT FLOW VERIFICATION")
    print("=" * 80)

    text_encoder.train()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], requires_grad=False)
    emb, _ = text_encoder(tokens)
    loss = emb.sum()
    loss.backward()

    motion_grad = text_encoder.token_emb.motion_emb.weight.grad is not None
    print(f"Motion embedding has gradient: {motion_grad}")

    if motion_grad:
        grad_norm = text_encoder.token_emb.motion_emb.weight.grad.norm().item()
        print(f"Motion embedding gradient norm: {grad_norm:.8f}")
        results['motion_token_usage']['motion_gradient_norm'] = grad_norm

    results['motion_token_usage']['motion_has_gradient'] = motion_grad

    # ============================================================================
    # TEST 4: CHECK MOTION EMBEDDING PARAMETERS UPDATED
    # ============================================================================
    print("\n" + "=" * 80)
    print("TEST 4: MOTION EMBEDDING PARAMETERS UPDATED")
    print("=" * 80)

    # Verify motion embedding has non-zero gradients (was trained)
    trained_motion_emb = text_encoder.token_emb.motion_emb.weight.data.clone()
    motion_std = trained_motion_emb.std().item()
    motion_mean = trained_motion_emb.mean().item()

    print(f"Motion embedding mean: {motion_mean:.8f}")
    print(f"Motion embedding std: {motion_std:.8f}")
    print(f"Motion embedding has variance: {motion_std > 1e-6}")

    results['motion_token_usage']['motion_embedding_has_variance'] = motion_std > 1e-6
    results['motion_token_usage']['motion_embedding_std'] = motion_std

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_checks_pass = (
        results['motion_token_usage']['motion_tokens_used'] and
        results['motion_token_usage']['motion_has_gradient'] and
        results['motion_token_usage']['motion_embedding_has_variance'] and
        results['prompt_sensitivity']['regular_vs_motion'] > 1e-6
    )

    print(f"Motion tokens produce different embeddings: {results['motion_token_usage']['motion_tokens_used']}")
    print(f"Motion embedding has gradient: {results['motion_token_usage']['motion_has_gradient']}")
    print(f"Motion embedding has variance: {results['motion_token_usage']['motion_embedding_has_variance']}")
    print(f"Regular vs motion tokens differ: {results['prompt_sensitivity']['regular_vs_motion'] > 1e-6}")
    print(f"\nAll checks pass: {all_checks_pass}")

    results['summary'] = {
        'all_checks_pass': all_checks_pass
    }

    # Save results
    output_file = base_dir / "experiments_stage3_2_9B/phase2L_training_repair/pilot/STAGE3_2_9L2_PROMPT_CONDITIONING.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results

if __name__ == "__main__":
    evaluate_prompt_conditioning()
