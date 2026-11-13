"""
Stage 3.2.9B - Integrity Tests

Critical integrity tests for Stage 3.2.9B experiments.
Enforces hard assertions about frozen parameters.

Author: Devin
Date: 2026-11-01
"""

import sys
import os
import json
import hashlib
import torch
import torch.nn as nn
import unittest
from pathlib import Path

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
from models.text_encoder_split import TextEncoderSplit, load_from_standard_encoder


class Stage3_2_9BIntegrityTests(unittest.TestCase):
    """Integrity tests for Stage 3.2.9B."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.parent_dir = Path(__file__).parent.parent
        cls.stage2_checkpoint = str(cls.parent_dir / "checkpoints_conditional" / "checkpoint_final.pt")
        cls.stage3_spatial_checkpoint = str(cls.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "checkpoint_stage2_spatial.pt")
        cls.stage3_motion_checkpoint = str(cls.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "checkpoint_stage2_motion_expanded.pt")
        cls.tokenizer_path = str(cls.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "tokenizer_motion_expanded.json")

        # Record initial hashes
        cls.initial_hashes = {
            'stage2': cls.get_file_hash(cls.stage2_checkpoint),
            'stage3_spatial': cls.get_file_hash(cls.stage3_spatial_checkpoint),
            'stage3_motion': cls.get_file_hash(cls.stage3_motion_checkpoint),
            'tokenizer': cls.get_file_hash(cls.tokenizer_path)
        }

        # Load models
        checkpoint = torch.load(cls.stage2_checkpoint, map_location='cpu')
        cls.actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

        cls.spatial_unet_original = OriginalConditionalUNet(
            in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
            num_res_blocks=2, attention_resolutions=[], dropout=0.0,
            num_timesteps=1000, text_dim=256, n_heads=4, use_cross_attn=True
        )

        cls.text_encoder = TextEncoder(
            vocab_size=cls.actual_vocab_size, d_model=256, n_heads=4,
            n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0
        )

        cls.spatial_unet_original.load_state_dict(checkpoint['model_state_dict'])
        cls.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

        cls.spatial_unet_modified = ConditionalUNet(
            in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
            num_res_blocks=2, attention_resolutions=[], dropout=0.0,
            num_timesteps=1000, text_dim=256, n_heads=4, use_cross_attn=True
        )
        cls.spatial_unet_modified.load_state_dict(checkpoint['model_state_dict'])

        cls.split_encoder = load_from_standard_encoder(
            cls.text_encoder, motion_vocab_size=40, freeze_transformer=True
        )

        cls.video_unet_B = VideoUNetFixed(
            spatial_unet=cls.spatial_unet_modified,
            text_encoder=cls.split_encoder,
            num_frames=16, image_size=64, temporal_kernel=3, temporal_layers=2,
            freeze_spatial=True, use_temporal=True, enable_temporal_processing=False
        )

        cls.video_unet_C = VideoUNetFixed(
            spatial_unet=cls.spatial_unet_modified,
            text_encoder=cls.split_encoder,
            num_frames=16, image_size=64, temporal_kernel=3, temporal_layers=2,
            freeze_spatial=True, use_temporal=True, enable_temporal_processing=True
        )

        # Store initial spatial parameters
        cls.initial_spatial_params = {
            name: param.clone() for name, param in cls.spatial_unet_modified.named_parameters()
        }

        # Store initial appearance embeddings
        cls.initial_appearance_emb = cls.split_encoder.token_emb.get_original_embeddings().clone()

    @staticmethod
    def get_file_hash(filepath):
        if not os.path.exists(filepath):
            return "FILE_NOT_FOUND"
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def test_1_checkpoint_hashes_unchanged(self):
        """Test that checkpoint hashes haven't changed."""
        current_hashes = {
            'stage2': self.get_file_hash(self.stage2_checkpoint),
            'stage3_spatial': self.get_file_hash(self.stage3_spatial_checkpoint),
            'stage3_motion': self.get_file_hash(self.stage3_motion_checkpoint),
            'tokenizer': self.get_file_hash(self.tokenizer_path)
        }

        for key in self.initial_hashes:
            self.assertEqual(
                self.initial_hashes[key],
                current_hashes[key],
                f"Checkpoint hash changed for {key}"
            )

    def test_2_spatial_parameters_frozen(self):
        """Test that spatial parameters are frozen."""
        for name, param in self.spatial_unet_modified.named_parameters():
            self.assertFalse(
                param.requires_grad,
                f"Spatial parameter {name} is not frozen"
            )

    def test_3_appearance_embeddings_frozen(self):
        """Test that appearance embeddings are frozen."""
        appearance_emb = self.split_encoder.token_emb.original_emb
        self.assertFalse(
            appearance_emb.weight.requires_grad,
            "Appearance embeddings are not frozen"
        )

    def test_4_motion_embeddings_trainable(self):
        """Test that motion embeddings are trainable."""
        motion_emb = self.split_encoder.token_emb.motion_emb
        self.assertTrue(
            motion_emb.weight.requires_grad,
            "Motion embeddings are not trainable"
        )

    def test_5_temporal_modules_trainable(self):
        """Test that temporal modules are trainable."""
        for name, param in self.video_unet_C.temporal_modules.named_parameters():
            self.assertTrue(
                param.requires_grad,
                f"Temporal parameter {name} is not trainable"
            )

    def test_6_fusion_modules_trainable(self):
        """Test that fusion modules are trainable."""
        for name, param in self.video_unet_C.fusion_modules.named_parameters():
            self.assertTrue(
                param.requires_grad,
                f"Fusion parameter {name} is not trainable"
            )

    def test_7_optimizer_B_only_motion_params(self):
        """Test that Experiment B optimizer only includes motion parameters."""
        import torch.optim as optim

        motion_params = list(self.split_encoder.token_emb.motion_emb.parameters())
        optimizer_B = optim.AdamW(motion_params, lr=0.0002)

        # Get all parameter names in optimizer
        optimizer_param_names = set()
        for group in optimizer_B.param_groups:
            for param in group['params']:
                # Find the corresponding parameter name
                for name, p in self.video_unet_B.named_parameters():
                    if p is param:
                        optimizer_param_names.add(name)
                        break
                for name, p in self.split_encoder.named_parameters():
                    if p is param:
                        optimizer_param_names.add(f"split_encoder.{name}")
                        break

        # Check that no spatial parameters are in optimizer
        for name in optimizer_param_names:
            self.assertNotIn(
                'spatial', name.lower(),
                f"Spatial parameter {name} found in optimizer B"
            )
            self.assertNotIn(
                'unet', name.lower(),
                f"UNet parameter {name} found in optimizer B"
            )

    def test_8_optimizer_C_no_spatial_params(self):
        """Test that Experiment C optimizer has no spatial parameters."""
        import torch.optim as optim

        trainable_params_C = list(self.split_encoder.token_emb.motion_emb.parameters())
        trainable_params_C.extend(list(self.video_unet_C.temporal_modules.parameters()))
        trainable_params_C.extend(list(self.video_unet_C.fusion_modules.parameters()))

        optimizer_C = optim.AdamW(trainable_params_C, lr=0.0002)

        # Get all parameter names in optimizer
        optimizer_param_names = set()
        for group in optimizer_C.param_groups:
            for param in group['params']:
                for name, p in self.video_unet_C.named_parameters():
                    if p is param:
                        optimizer_param_names.add(name)
                        break
                for name, p in self.split_encoder.named_parameters():
                    if p is param:
                        optimizer_param_names.add(f"split_encoder.{name}")
                        break

        # Check that no spatial parameters are in optimizer
        for name in optimizer_param_names:
            self.assertNotIn(
                'spatial', name.lower(),
                f"Spatial parameter {name} found in optimizer C"
            )

    def test_9_initial_spatial_params_recorded(self):
        """Test that initial spatial parameters are recorded."""
        self.assertEqual(
            len(self.initial_spatial_params),
            sum(1 for _ in self.spatial_unet_modified.named_parameters()),
            "Not all spatial parameters were recorded"
        )

    def test_10_initial_appearance_emb_recorded(self):
        """Test that initial appearance embeddings are recorded."""
        self.assertIsNotNone(self.initial_appearance_emb)
        self.assertEqual(
            self.initial_appearance_emb.shape,
            torch.Size([42, 256]),
            "Appearance embedding shape incorrect"
        )


def main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(Stage3_2_9BIntegrityTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
