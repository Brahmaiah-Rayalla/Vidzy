"""
Regression test for DDIM argument-shifting bug (Stage 2.3.2)

Bug: ddim_sample_loop() was missing clip_range parameter when calling ddim_sample(),
causing all positional arguments to shift. This caused text_emb to lose its last
dimension when batch_size > 1.

This test ensures the bug remains fixed.
"""
import os
import sys
import torch
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.diffusion.sampler import DiffusionSampler


def test_ddim_argument_passing_regression():
    """
    Regression test for clip_range argument-shifting bug.

    Tests that DDIM sampling works correctly with batch_size > 1,
    which was the failing case before the fix.
    """
    # Load config
    import yaml
    with open('configs/conditional_v0.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cpu')
    torch.manual_seed(42)

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
    checkpoint = torch.load("checkpoints_conditional/checkpoint_final.pt", map_location=device)
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

    # Test with batch size 4 (the failing case)
    batch_size = 4
    text_emb_batch = text_emb.repeat(batch_size, 1, 1)
    attention_mask_batch = attention_mask.repeat(batch_size, 1)

    # This should NOT raise a shape error
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

    # Verify output shape
    assert samples.shape == (batch_size, 3, 64, 64), f"Expected shape (4, 3, 64, 64), got {samples.shape}"

    # Verify normalization
    assert samples.min() >= -1.0 and samples.max() <= 1.0, "Values out of range [-1, 1]"

    print("Regression test PASSED: DDIM argument-shifting bug is fixed")


if __name__ == "__main__":
    test_ddim_argument_passing_regression()
