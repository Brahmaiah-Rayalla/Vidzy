"""
Stage 3.2.9A - Spatial Integrity Regression Tests

Automated test suite for verifying spatial foundation integrity.
All tests are reproducible and CPU-only.

Author: Devin
Date: 2026-11-01
"""

import sys
import os
import json
import torch
import torch.nn as nn
import unittest
from pathlib import Path
from typing import Dict

# Add paths
parent_dir = Path(__file__).parent.parent
stage3_2_7_dir = parent_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(stage3_2_7_dir))

from video_foundation.models.conditional_unet import ConditionalUNet as OriginalConditionalUNet
from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))
from models.conditional_unet import ConditionalUNet
from models.video_unet_fixed import VideoUNetFixed


class SpatialIntegrityTests(unittest.TestCase):
    """Test suite for spatial foundation integrity."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.parent_dir = Path(__file__).parent.parent
        cls.stage2_checkpoint = str(cls.parent_dir / "checkpoints_conditional" / "checkpoint_final.pt")
        cls.stage3_checkpoint = str(cls.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "checkpoint_stage2_spatial.pt")
        cls.config_path = str(cls.parent_dir / "configs" / "conditional_v0.yaml")
        cls.tokenizer_path = str(cls.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "tokenizer_motion_expanded.json")

        # Load config
        import yaml
        with open(cls.config_path, 'r') as f:
            cls.config = yaml.safe_load(f)

        model_config = cls.config['model']
        text_config = cls.config['text_encoder']

        # Load checkpoint
        checkpoint = torch.load(cls.stage2_checkpoint, map_location='cpu')
        cls.actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

        # Create Stage 2.1 model
        cls.spatial_unet_original = OriginalConditionalUNet(
            in_channels=model_config['in_channels'],
            model_channels=model_config['model_channels'],
            channel_mult=model_config['channel_mult'],
            num_res_blocks=model_config['num_res_blocks'],
            attention_resolutions=model_config['attention_resolutions'],
            dropout=model_config['dropout'],
            num_timesteps=cls.config['diffusion']['num_timesteps'],
            text_dim=text_config['d_model'],
            n_heads=text_config['n_heads'],
            use_cross_attn=model_config['use_cross_attn']
        )

        cls.text_encoder = TextEncoder(
            vocab_size=cls.actual_vocab_size,
            d_model=text_config['d_model'],
            n_heads=text_config['n_heads'],
            n_layers=text_config['n_layers'],
            d_ff=text_config['d_ff'],
            max_seq_len=text_config['max_seq_len'],
            dropout=0.0
        )

        cls.spatial_unet_original.load_state_dict(checkpoint['model_state_dict'])
        cls.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

        # Create Stage 3 model
        cls.spatial_unet_modified = ConditionalUNet(
            in_channels=model_config['in_channels'],
            model_channels=model_config['model_channels'],
            channel_mult=model_config['channel_mult'],
            num_res_blocks=model_config['num_res_blocks'],
            attention_resolutions=model_config['attention_resolutions'],
            dropout=model_config['dropout'],
            num_timesteps=cls.config['diffusion']['num_timesteps'],
            text_dim=text_config['d_model'],
            n_heads=text_config['n_heads'],
            use_cross_attn=model_config['use_cross_attn']
        )
        cls.spatial_unet_modified.load_state_dict(checkpoint['model_state_dict'])

        cls.video_unet = VideoUNetFixed(
            spatial_unet=cls.spatial_unet_modified,
            text_encoder=cls.text_encoder,
            num_frames=4,
            image_size=model_config['image_size'],
            temporal_kernel=3,
            temporal_layers=2,
            freeze_spatial=True,
            use_temporal=True,
            enable_temporal_processing=False
        )

        # Load tokenizer
        expanded_tokenizer = SimpleTokenizer.load(cls.tokenizer_path)
        appearance_vocab_size = 42
        cls.tokenizer = SimpleTokenizer(vocab_size=appearance_vocab_size, min_freq=1)
        cls.tokenizer.word_to_idx = {k: v for k, v in expanded_tokenizer.word_to_idx.items() if v < appearance_vocab_size}
        cls.tokenizer.idx_to_word = {k: v for k, v in expanded_tokenizer.idx_to_word.items() if int(k) < appearance_vocab_size}

        # Set deterministic seed
        torch.manual_seed(42)

    def test_1_checkpoint_loading(self):
        """Test that checkpoints can be loaded without errors."""
        # Load Stage 2.1 checkpoint
        checkpoint2 = torch.load(self.stage2_checkpoint, map_location='cpu')
        self.assertIn('model_state_dict', checkpoint2)
        self.assertIn('text_encoder_state_dict', checkpoint2)

        # Load Stage 3.2.8 checkpoint
        checkpoint3 = torch.load(self.stage3_checkpoint, map_location='cpu')
        self.assertIn('model_state_dict', checkpoint3)

        # Verify they have the same keys
        keys2 = set(checkpoint2['model_state_dict'].keys())
        keys3 = set(checkpoint3['model_state_dict'].keys())
        self.assertEqual(keys2, keys3)

    def test_2_parameter_equality(self):
        """Test that spatial parameters are identical between checkpoints."""
        checkpoint2 = torch.load(self.stage2_checkpoint, map_location='cpu')
        checkpoint3 = torch.load(self.stage3_checkpoint, map_location='cpu')

        state_dict2 = checkpoint2['model_state_dict']
        state_dict3 = checkpoint3['model_state_dict']

        # Check all parameters
        for key in state_dict2.keys():
            diff = torch.max(torch.abs(state_dict2[key] - state_dict3[key])).item()
            self.assertLess(diff, 1e-10, f"Parameter {key} differs by {diff}")

    def test_3_single_frame_equivalence(self):
        """Test that single-frame outputs are identical."""
        # Create deterministic input
        torch.manual_seed(42)
        x = torch.randn(1, 3, 1, 64, 64)
        timesteps = torch.randint(0, 1000, (1,))

        # Encode text
        prompt = "A small red circle on a blue background."
        token_ids, mask = self.tokenizer.encode(prompt, max_length=self.config['text_encoder']['max_seq_len'])
        token_ids_tensor = torch.tensor([token_ids])
        mask_tensor = torch.tensor([mask], dtype=torch.float)
        text_emb, text_mask = self.text_encoder(token_ids_tensor, mask_tensor)

        # Stage 2.1 forward
        x_spatial = x.squeeze(2)
        output_spatial = self.spatial_unet_original(x_spatial, timesteps, text_emb, text_mask)

        # Stage 3 forward (temporal disabled)
        output_video = self.video_unet(x, timesteps, text_emb, text_mask)

        # Compare
        max_diff = torch.max(torch.abs(output_spatial - output_video.squeeze(2))).item()
        self.assertLess(max_diff, 1e-5, f"Max difference: {max_diff}")

    def test_4_multi_frame_equivalence(self):
        """Test that multi-frame outputs are nearly identical."""
        # Create deterministic input
        torch.manual_seed(42)
        x = torch.randn(1, 3, 4, 64, 64)
        timesteps = torch.randint(0, 1000, (1,))

        # Encode text
        prompt = "A small red circle on a blue background."
        token_ids, mask = self.tokenizer.encode(prompt, max_length=self.config['text_encoder']['max_seq_len'])
        token_ids_tensor = torch.tensor([token_ids])
        mask_tensor = torch.tensor([mask], dtype=torch.float)
        text_emb, text_mask = self.text_encoder(token_ids_tensor, mask_tensor)

        # Stage 2.1 forward (frame-by-frame)
        outputs = []
        for f in range(4):
            x_frame = x[:, :, f, :, :]
            output_frame = self.spatial_unet_original(x_frame, timesteps, text_emb, text_mask)
            outputs.append(output_frame)
        output_spatial = torch.stack(outputs, dim=2)

        # Stage 3 forward (temporal disabled)
        output_video = self.video_unet(x, timesteps, text_emb, text_mask)

        # Compare
        max_diff = torch.max(torch.abs(output_spatial - output_video)).item()
        self.assertLess(max_diff, 1e-4, f"Max difference: {max_diff}")

    def test_5_batch_size_variations(self):
        """Test that different batch sizes work correctly."""
        for batch_size in [1, 2, 4]:
            with self.subTest(batch_size=batch_size):
                torch.manual_seed(42)
                x = torch.randn(batch_size, 3, 1, 64, 64)
                timesteps = torch.randint(0, 1000, (batch_size,))

                prompt = "A small red circle on a blue background."
                token_ids, mask = self.tokenizer.encode(prompt, max_length=self.config['text_encoder']['max_seq_len'])
                token_ids_tensor = torch.tensor([token_ids] * batch_size)
                mask_tensor = torch.tensor([mask] * batch_size, dtype=torch.float)
                text_emb, text_mask = self.text_encoder(token_ids_tensor, mask_tensor)

                x_spatial = x.squeeze(2)
                output_spatial = self.spatial_unet_original(x_spatial, timesteps, text_emb, text_mask)
                output_video = self.video_unet(x, timesteps, text_emb, text_mask)

                max_diff = torch.max(torch.abs(output_spatial - output_video.squeeze(2))).item()
                self.assertLess(max_diff, 1e-5, f"Batch {batch_size}: max diff {max_diff}")

    def test_6_temporal_disable(self):
        """Test that temporal processing can be disabled."""
        # Create video model with temporal enabled
        video_unet_enabled = VideoUNetFixed(
            spatial_unet=self.spatial_unet_modified,
            text_encoder=self.text_encoder,
            num_frames=4,
            image_size=self.config['model']['image_size'],
            temporal_kernel=3,
            temporal_layers=2,
            freeze_spatial=True,
            use_temporal=True,
            enable_temporal_processing=True
        )

        torch.manual_seed(42)
        x = torch.randn(1, 3, 4, 64, 64)
        timesteps = torch.randint(0, 1000, (1,))

        prompt = "A small red circle on a blue background."
        token_ids, mask = self.tokenizer.encode(prompt, max_length=self.config['text_encoder']['max_seq_len'])
        token_ids_tensor = torch.tensor([token_ids])
        mask_tensor = torch.tensor([mask], dtype=torch.float)
        text_emb, text_mask = self.text_encoder(token_ids_tensor, mask_tensor)

        with torch.no_grad():
            output_disabled = self.video_unet(x, timesteps, text_emb, text_mask)
            output_enabled = video_unet_enabled(x, timesteps, text_emb, text_mask)

        # Outputs should differ
        max_diff = torch.max(torch.abs(output_disabled - output_enabled)).item()
        self.assertGreater(max_diff, 1e-3, f"Temporal processing should affect output")

    def test_7_text_encoder_determinism(self):
        """Test that text encoder produces deterministic outputs."""
        prompt = "A small red circle on a blue background."
        token_ids, mask = self.tokenizer.encode(prompt, max_length=self.config['text_encoder']['max_seq_len'])
        token_ids_tensor = torch.tensor([token_ids])
        mask_tensor = torch.tensor([mask], dtype=torch.float)

        with torch.no_grad():
            emb1, _ = self.text_encoder(token_ids_tensor, mask_tensor)
            emb2, _ = self.text_encoder(token_ids_tensor, mask_tensor)

        max_diff = torch.max(torch.abs(emb1 - emb2)).item()
        self.assertLess(max_diff, 1e-10, f"Text encoder should be deterministic")

    def test_8_cpu_only_execution(self):
        """Test that all operations run on CPU."""
        self.assertFalse(torch.cuda.is_available(), "CUDA should not be available for CPU-only tests")

        # Ensure models are on CPU
        self.assertEqual(next(self.spatial_unet_original.parameters()).device.type, 'cpu')
        self.assertEqual(next(self.video_unet.parameters()).device.type, 'cpu')

    def test_9_output_shapes(self):
        """Test that output shapes are correct."""
        torch.manual_seed(42)
        x = torch.randn(1, 3, 4, 64, 64)
        timesteps = torch.randint(0, 1000, (1,))

        prompt = "A small red circle on a blue background."
        token_ids, mask = self.tokenizer.encode(prompt, max_length=self.config['text_encoder']['max_seq_len'])
        token_ids_tensor = torch.tensor([token_ids])
        mask_tensor = torch.tensor([mask], dtype=torch.float)
        text_emb, text_mask = self.text_encoder(token_ids_tensor, mask_tensor)

        output = self.video_unet(x, timesteps, text_emb, text_mask)
        self.assertEqual(output.shape, (1, 3, 4, 64, 64))

    def test_10_temporal_modules_exist(self):
        """Test that temporal modules exist and are configured correctly."""
        self.assertTrue(self.video_unet.use_temporal)
        self.assertIn('bottleneck', self.video_unet.temporal_modules)
        self.assertIn('low_res', self.video_unet.temporal_modules)
        self.assertIn('bottleneck', self.video_unet.fusion_modules)
        self.assertIn('low_res', self.video_unet.fusion_modules)


def main():
    """Run the test suite."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(SpatialIntegrityTests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("=" * 80)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
