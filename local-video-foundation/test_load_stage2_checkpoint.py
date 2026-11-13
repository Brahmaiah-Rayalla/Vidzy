"""
Test loading Stage 2 checkpoint into VideoUNet.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import torch
from video_foundation.models.conditional_unet import ConditionalUNet
from video_foundation.models.text_encoder import TextEncoder
from video_foundation.models.video_unet import VideoUNet


def test_load_stage2_checkpoint():
    """Test loading Stage 2 checkpoint."""
    print("Testing Stage 2 checkpoint loading...")

    # Paths to Stage 2 checkpoints
    spatial_checkpoint_path = "checkpoints_conditional/checkpoint_final.pt"
    tokenizer_path = "data/text_image/tokenizer/tokenizer.json"

    # Check if files exist
    if not os.path.exists(spatial_checkpoint_path):
        print(f"ERROR: Spatial checkpoint not found at {spatial_checkpoint_path}")
        return

    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        return

    print(f"Loading checkpoint from {spatial_checkpoint_path}")
    checkpoint = torch.load(spatial_checkpoint_path, map_location='cpu')

    print(f"Checkpoint keys: {checkpoint.keys()}")

    # Reconstruct spatial U-Net
    spatial_unet = ConditionalUNet(
        in_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        attention_resolutions=[],
        dropout=0.0,
        num_timesteps=1000,
        text_dim=256,
        n_heads=4,
        use_cross_attn=True
    )

    # Load U-Net weights
    model_state_dict = checkpoint['model_state_dict']
    missing, unexpected = spatial_unet.load_state_dict(model_state_dict, strict=False)

    print(f"\nLoaded spatial U-Net:")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print(f"  Sample missing keys: {missing[:5]}")

    # Load text encoder from checkpoint
    print(f"\nLoading text encoder from checkpoint...")
    text_encoder_state_dict = checkpoint['text_encoder_state_dict']

    # Get vocab size from state dict
    vocab_size = text_encoder_state_dict['token_emb.weight'].shape[0]
    print(f"Detected vocab size: {vocab_size}")

    # Get config to reconstruct text encoder
    config = checkpoint['config']
    text_encoder_config = config['text_encoder']

    text_encoder = TextEncoder(
        vocab_size=vocab_size,
        d_model=text_encoder_config['d_model'],
        n_heads=text_encoder_config['n_heads'],
        n_layers=text_encoder_config['n_layers'],
        d_ff=text_encoder_config['d_ff'],
        max_seq_len=text_encoder_config['max_seq_len'],
        dropout=text_encoder_config.get('dropout', 0.1)
    )

    text_encoder.load_state_dict(text_encoder_state_dict)

    print(f"Text encoder loaded successfully")
    print(f"  Vocab size: {text_encoder.vocab_size}")
    print(f"  d_model: {text_encoder.d_model}")

    # Load tokenizer
    print(f"\nLoading tokenizer from {tokenizer_path}")
    from video_foundation.models.text_encoder import SimpleTokenizer
    import json

    # The tokenizer JSON has a different format
    # We'll reconstruct it properly
    with open(tokenizer_path, 'r') as f:
        data = json.load(f)

    # Reconstruct from idx_to_word which is more consistent
    # idx_to_word has: "<PAD>": 0, "3": "a", etc.
    # For special tokens, key is token, value is index
    # For regular tokens, key is string of index, value is word
    idx_to_word = {}
    word_to_idx = {}

    for key, value in data['idx_to_word'].items():
        if key.startswith('<'):  # Special token
            idx = int(value)
            token = key
            idx_to_word[idx] = token
            word_to_idx[token] = idx
        else:  # Regular token (key is string number)
            idx = int(key)
            token = value
            idx_to_word[idx] = token
            word_to_idx[token] = idx

    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=data['vocab_size'], min_freq=data['min_freq'])
    tokenizer.word_to_idx = word_to_idx
    tokenizer.idx_to_word = idx_to_word
    tokenizer.word_freq = data['word_freq']

    print(f"Tokenizer loaded successfully")
    print(f"  Vocab size: {len(tokenizer.word_to_idx)}")

    # Create video U-Net
    print(f"\nCreating VideoUNet...")
    video_unet = VideoUNet(
        spatial_unet=spatial_unet,
        text_encoder=text_encoder,
        num_frames=16,
        image_size=64,
        freeze_spatial=False  # Allow fine-tuning for video task
    )

    # Store tokenizer
    video_unet.tokenizer = tokenizer

    print(f"\nVideo U-Net created successfully")
    print(f"Total parameters: {sum(p.numel() for p in video_unet.parameters()):,}")
    print(f"Trainable parameters: {video_unet.trainable_params:,}")

    # Test forward pass with actual text encoding
    print(f"\nTesting forward pass with text encoding...")

    # Encode a prompt
    prompt = "A red circle moves from left to right."
    token_ids, mask = tokenizer.encode(prompt, max_length=32)
    token_ids = torch.tensor([token_ids])
    mask = torch.tensor([mask])

    with torch.no_grad():
        text_emb, _ = text_encoder(token_ids, mask)

    print(f"Text embedding shape: {text_emb.shape}")

    # Create dummy video
    batch_size = 1
    x = torch.randn(batch_size, 3, 16, 64, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))

    # Forward pass
    with torch.no_grad():
        output = video_unet(x, timesteps, text_emb, mask)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    print("\nStage 2 checkpoint loading test PASSED")


if __name__ == "__main__":
    test_load_stage2_checkpoint()
