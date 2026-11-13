"""
Tests for checkpoint saving and loading.
"""

import os
import sys
import tempfile
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_foundation.models.unet import UNet
from video_foundation.utils.checkpoint import CheckpointManager


def test_checkpoint_save_load():
    """Test that checkpoints can be saved and loaded correctly."""
    print("Testing checkpoint save/load...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create model and optimizer
        model = UNet(
            in_channels=3,
            model_channels=32,
            channel_mult=[1, 2],
            num_res_blocks=1,
            attention_resolutions=[],
            dropout=0.0,
            num_timesteps=1000
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)

        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
        config = {"test": "config"}
        seed = 42

        CheckpointManager.save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            step=100,
            config=config,
            seed=seed
        )

        # Check that file exists
        assert os.path.exists(checkpoint_path), "Checkpoint file should exist"

        # Create new model and optimizer
        model2 = UNet(
            in_channels=3,
            model_channels=32,
            channel_mult=[1, 2],
            num_res_blocks=1,
            attention_resolutions=[],
            dropout=0.0,
            num_timesteps=1000
        )
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.0002)

        # Load checkpoint
        metadata = CheckpointManager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model2,
            optimizer=optimizer2
        )

        # Check metadata
        assert metadata['step'] == 100, f"Expected step 100, got {metadata['step']}"
        assert metadata['seed'] == 42, f"Expected seed 42, got {metadata['seed']}"
        assert metadata['config'] == config, "Config should match"

        # Check that model parameters match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Model parameters should match after loading"

        print("[PASS] Checkpoint save/load test passed")


def test_config_save_load():
    """Test that config can be saved and loaded."""
    print("Testing config save/load...")

    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            "model": {
                "in_channels": 3,
                "model_channels": 64
            },
            "training": {
                "num_steps": 2000
            }
        }

        config_path = os.path.join(temp_dir, "config.json")

        # Save config
        CheckpointManager.save_config(config, config_path)

        # Check that file exists
        assert os.path.exists(config_path), "Config file should exist"

        # Load config
        loaded_config = CheckpointManager.load_config(config_path)

        # Check that config matches
        assert loaded_config == config, "Loaded config should match original"

        print("[PASS] Config save/load test passed")


if __name__ == "__main__":
    test_checkpoint_save_load()
    test_config_save_load()

    print("\n" + "="*60)
    print("All checkpoint tests passed!")
    print("="*60)
