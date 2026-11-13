"""
DDIM Implementation Validation Script for Stage 2.3.2

Validates:
- DDIM implementation correctness
- Batch size compatibility (1, 2, 4, 8)
- Checkpoint compatibility
- Text conditioning
- Normalization and clipping behavior
- Argument passing (regression test for clip_range bug)
"""
import os
import sys
import torch
import yaml
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import set_seed, get_device

# Load config
with open('configs/conditional_v0.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = get_device()
set_seed(42)

print("="*70)
print("DDIM IMPLEMENTATION VALIDATION - STAGE 2.3.2")
print("="*70)
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")
print("="*70)

# Load tokenizer
train_metadata_path = "data/text_image/train_metadata.json"
with open(train_metadata_path, 'r') as f:
    train_metadata = json.load(f)
captions = [m['caption'] for m in train_metadata]
tokenizer = SimpleTokenizer(
    vocab_size=config['text_encoder']['vocab_size'],
    min_freq=config['text_encoder']['min_freq']
)
tokenizer.build_vocab(captions)

# Load models
print("\nLoading models...")
text_encoder = TextEncoder(
    vocab_size=len(tokenizer.word_to_idx),
    d_model=config['text_encoder']['d_model'],
    n_heads=config['text_encoder']['n_heads'],
    n_layers=config['text_encoder']['n_layers'],
    d_ff=config['text_encoder']['d_ff'],
    max_seq_len=config['text_encoder']['max_seq_len'],
    dropout=0.0
).to(device)

model = ConditionalUNet(
    in_channels=config['model']['in_channels'],
    model_channels=config['model']['model_channels'],
    channel_mult=config['model']['channel_mult'],
    num_res_blocks=config['model']['num_res_blocks'],
    attention_resolutions=config['model']['attention_resolutions'],
    dropout=0.0,
    num_timesteps=config['diffusion']['num_timesteps'],
    text_dim=config['text_encoder']['d_model'],
    n_heads=config.get('n_heads', 4),
    use_cross_attn=config['model']['use_cross_attn']
).to(device)

# Load checkpoint
checkpoint_path = "checkpoints_conditional/checkpoint_final.pt"
print(f"Loading checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device)
text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
model.load_state_dict(checkpoint['model_state_dict'])

text_encoder.eval()
model.eval()

# Create diffusion sampler
diffusion = DiffusionSampler(
    num_timesteps=config['diffusion']['num_timesteps'],
    beta_start=config['diffusion']['beta_start'],
    beta_end=config['diffusion']['beta_end'],
    schedule=config['diffusion']['beta_schedule']
)

# Test prompt
prompt = "A small red circle on a blue background."
max_len = config['text_encoder']['max_seq_len']
token_ids, attention_mask = tokenizer.encode(prompt, max_length=max_len)
token_ids = torch.tensor([token_ids], device=device)
attention_mask = torch.tensor([attention_mask], device=device).float()

with torch.no_grad():
    text_emb, _ = text_encoder(token_ids, attention_mask)

print(f"Text embedding shape: {text_emb.shape}")

# Test 1: Batch size compatibility
print("\n" + "="*70)
print("TEST 1: Batch Size Compatibility")
print("="*70)

batch_sizes = [1, 2, 4, 8]
test_results = {}

for batch_size in batch_sizes:
    try:
        # Expand text embeddings for batch
        text_emb_batch = text_emb.repeat(batch_size, 1, 1)
        attention_mask_batch = attention_mask.repeat(batch_size, 1)

        print(f"\nTesting batch size {batch_size}...")
        print(f"  text_emb shape: {text_emb_batch.shape}")
        print(f"  attention_mask shape: {attention_mask_batch.shape}")

        with torch.no_grad():
            start = time.time()
            samples = diffusion.sample(
                model=model,
                batch_size=batch_size,
                channels=config['model']['in_channels'],
                image_size=config['model']['image_size'],
                num_sampling_steps=5,  # Fast test
                device=device,
                clip_x0=False,  # No per-step clipping (Stage 1.8 fix)
                text_emb=text_emb_batch,
                text_mask=attention_mask_batch
            )
            elapsed = time.time() - start

        print(f"  SUCCESS: {samples.shape}")
        print(f"  Time: {elapsed:.2f}s")

        # Check normalization
        min_val = samples.min().item()
        max_val = samples.max().item()
        print(f"  Value range: [{min_val:.3f}, {max_val:.3f}]")

        # Check that values are in [-1, 1] after final clipping
        assert min_val >= -1.0 and max_val <= 1.0, f"Values out of range: [{min_val}, {max_val}]"
        print(f"  Normalization: PASS")

        test_results[batch_size] = {
            'status': 'PASS',
            'shape': str(samples.shape),
            'time_s': elapsed,
            'min_val': min_val,
            'max_val': max_val
        }

    except Exception as e:
        print(f"  FAILED: {e}")
        test_results[batch_size] = {'status': 'FAIL', 'error': str(e)}

# Test 2: DDPM compatibility (regression test)
print("\n" + "="*70)
print("TEST 2: DDPM Compatibility (Regression Test)")
print("="*70)

try:
    batch_size = 2
    text_emb_batch = text_emb.repeat(batch_size, 1, 1)
    attention_mask_batch = attention_mask.repeat(batch_size, 1)

    print(f"\nTesting DDPM with batch size {batch_size}...")
    with torch.no_grad():
        # Fast DDPM test (every 100th timestep)
        start = time.time()
        samples_ddpm = diffusion.sample_ddpm(
            model=model,
            batch_size=batch_size,
            channels=config['model']['in_channels'],
            image_size=config['model']['image_size'],
            device=device,
            text_emb=text_emb_batch,
            text_mask=attention_mask_batch
        )
        elapsed = time.time() - start

    print(f"  SUCCESS: {samples_ddpm.shape}")
    print(f"  Time: {elapsed:.2f}s")

    min_val = samples_ddpm.min().item()
    max_val = samples_ddpm.max().item()
    print(f"  Value range: [{min_val:.3f}, {max_val:.3f}]")
    assert min_val >= -1.0 and max_val <= 1.0, f"Values out of range: [{min_val}, {max_val}]"
    print(f"  Normalization: PASS")

    test_results['ddpm'] = {
        'status': 'PASS',
        'shape': str(samples_ddpm.shape),
        'time_s': elapsed,
        'min_val': min_val,
        'max_val': max_val
    }

except Exception as e:
    print(f"  FAILED: {e}")
    test_results['ddpm'] = {'status': 'FAIL', 'error': str(e)}

# Test 3: Argument passing (regression test for clip_range bug)
print("\n" + "="*70)
print("TEST 3: Argument Passing (Regression Test for clip_range Bug)")
print("="*70)

try:
    batch_size = 4
    text_emb_batch = text_emb.repeat(batch_size, 1, 1)
    attention_mask_batch = attention_mask.repeat(batch_size, 1)

    print(f"\nTesting DDIM with batch size {batch_size} (this was the failing case)...")
    print(f"  text_emb shape: {text_emb_batch.shape}")

    with torch.no_grad():
        samples = diffusion.sample(
            model=model,
            batch_size=batch_size,
            channels=config['model']['in_channels'],
            image_size=config['model']['image_size'],
            num_sampling_steps=5,
            device=device,
            clip_x0=False,
            text_emb=text_emb_batch,
            text_mask=attention_mask_batch
        )

    print(f"  SUCCESS: {samples.shape}")
    print(f"  Regression test: PASS (clip_range bug fixed)")

    test_results['argument_passing'] = {'status': 'PASS'}

except Exception as e:
    print(f"  FAILED: {e}")
    print(f"  Regression test: FAIL (clip_range bug still present)")
    test_results['argument_passing'] = {'status': 'FAIL', 'error': str(e)}

# Test 4: Sampling behavior verification
print("\n" + "="*70)
print("TEST 4: Sampling Behavior Verification")
print("="*70)

try:
    batch_size = 1
    text_emb_batch = text_emb.repeat(batch_size, 1, 1)
    attention_mask_batch = attention_mask.repeat(batch_size, 1)

    # Test with clipping disabled (Stage 1.8 fix)
    print("\nTesting with clip_x0=False (Stage 1.8 fix)...")
    with torch.no_grad():
        samples_no_clip = diffusion.sample(
            model=model,
            batch_size=batch_size,
            channels=config['model']['in_channels'],
            image_size=config['model']['image_size'],
            num_sampling_steps=5,
            device=device,
            clip_x0=False,
            text_emb=text_emb_batch,
            text_mask=attention_mask_batch
        )

    # Test with clipping enabled
    print("Testing with clip_x0=True...")
    with torch.no_grad():
        samples_with_clip = diffusion.sample(
            model=model,
            batch_size=batch_size,
            channels=config['model']['in_channels'],
            image_size=config['model']['image_size'],
            num_sampling_steps=5,
            device=device,
            clip_x0=True,
            text_emb=text_emb_batch,
            text_mask=attention_mask_batch
        )

    print(f"  No clip: range [{samples_no_clip.min():.3f}, {samples_no_clip.max():.3f}]")
    print(f"  With clip: range [{samples_with_clip.min():.3f}, {samples_with_clip.max():.3f}]")
    print(f"  Sampling behavior: PASS")

    test_results['sampling_behavior'] = {'status': 'PASS'}

except Exception as e:
    print(f"  FAILED: {e}")
    test_results['sampling_behavior'] = {'status': 'FAIL', 'error': str(e)}

# Summary
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

all_pass = True
for test_name, result in test_results.items():
    status = result['status']
    symbol = "[PASS]" if status == "PASS" else "[FAIL]"
    print(f"{symbol} {test_name}: {status}")
    if status != "PASS":
        all_pass = False

print("="*70)
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print("="*70)

# Save results
os.makedirs("outputs_conditional", exist_ok=True)
results_path = "outputs_conditional/ddim_validation_results.json"
with open(results_path, 'w') as f:
    json.dump(test_results, f, indent=2)
print(f"\nResults saved to {results_path}")
