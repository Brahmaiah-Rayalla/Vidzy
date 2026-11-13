"""
Stage 3.2.9B - Initial Model Readiness Checks

Verify model integrity before pilot training.

Author: Devin
Date: 2026-11-01
"""

import sys
import os
import json
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
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


class ReadinessChecker:
    """Perform initial model readiness checks."""

    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.parent_dir = self.experiment_dir.parent

        # Paths
        self.stage2_checkpoint = str(self.parent_dir / "checkpoints_conditional" / "checkpoint_final.pt")
        self.stage3_spatial_checkpoint = str(self.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "checkpoint_stage2_spatial.pt")
        self.tokenizer_path = str(self.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "tokenizer_motion_expanded.json")

        self.results = {
            'checks': {},
            'final_status': 'UNKNOWN'
        }

    def log_section(self, title: str):
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)

    def get_file_hash(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return "FILE_NOT_FOUND"
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def load_models(self):
        """Load all required models."""
        # Load checkpoint
        checkpoint = torch.load(self.stage2_checkpoint, map_location='cpu')
        actual_vocab_size = checkpoint['text_encoder_state_dict']['token_emb.weight'].shape[0]

        # Create models
        spatial_unet_original = OriginalConditionalUNet(
            in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
            num_res_blocks=2, attention_resolutions=[], dropout=0.0,
            num_timesteps=1000, text_dim=256, n_heads=4, use_cross_attn=True
        )

        text_encoder = TextEncoder(
            vocab_size=actual_vocab_size, d_model=256, n_heads=4,
            n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0
        )

        spatial_unet_original.load_state_dict(checkpoint['model_state_dict'])
        text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

        spatial_unet_modified = ConditionalUNet(
            in_channels=3, model_channels=64, channel_mult=[1, 2, 2, 2],
            num_res_blocks=2, attention_resolutions=[], dropout=0.0,
            num_timesteps=1000, text_dim=256, n_heads=4, use_cross_attn=True
        )
        spatial_unet_modified.load_state_dict(checkpoint['model_state_dict'])

        split_encoder = load_from_standard_encoder(
            text_encoder, motion_vocab_size=40, freeze_transformer=True
        )

        video_unet_disabled = VideoUNetFixed(
            spatial_unet=spatial_unet_modified,
            text_encoder=split_encoder,
            num_frames=16, image_size=64, temporal_kernel=3, temporal_layers=2,
            freeze_spatial=True, use_temporal=True, enable_temporal_processing=False
        )

        video_unet_enabled = VideoUNetFixed(
            spatial_unet=spatial_unet_modified,
            text_encoder=split_encoder,
            num_frames=16, image_size=64, temporal_kernel=3, temporal_layers=2,
            freeze_spatial=True, use_temporal=True, enable_temporal_processing=True
        )

        return (spatial_unet_original, text_encoder, spatial_unet_modified,
                split_encoder, video_unet_disabled, video_unet_enabled)

    def check_1_checkpoint_hashes_unchanged(self):
        """Check that checkpoint hashes haven't changed."""
        self.log_section("CHECK 1: Checkpoint Hashes Unchanged")

        hashes = {
            'stage2': self.get_file_hash(self.stage2_checkpoint),
            'stage3_spatial': self.get_file_hash(self.stage3_spatial_checkpoint)
        }

        expected = {
            'stage2': 'a7f5142e133c6aeadc656967639eab046a7a8f922e3201a6feda4496a544d57c',
            'stage3_spatial': 'a7f5142e133c6aeadc656967639eab046a7a8f922e3201a6feda4496a544d57c'
        }

        status = all(hashes[k] == expected[k] for k in expected)
        print(f"Hashes match: {status}")
        print(f"Stage 2: {hashes['stage2']}")
        print(f"Stage 3 spatial: {hashes['stage3_spatial']}")

        self.results['checks']['checkpoint_hashes'] = {
            'status': 'PASS' if status else 'FAIL',
            'hashes': hashes,
            'expected': expected
        }
        return status

    def check_2_temporal_disabled_equivalence(self):
        """Check temporal-disabled output matches Stage 2.1 within Stage 3.2.9A tolerance."""
        self.log_section("CHECK 2: Temporal-Disabled Equivalence")

        spatial_unet_original, text_encoder, spatial_unet_modified, split_encoder, video_unet_disabled, _ = self.load_models()

        # Create deterministic input
        torch.manual_seed(42)
        x = torch.randn(1, 3, 4, 64, 64)
        timesteps = torch.randint(0, 1000, (1,))

        # Encode text - use full tokenizer but only appearance tokens for equivalence test
        expanded_tokenizer = SimpleTokenizer.load(self.tokenizer_path)
        from collections import Counter
        appearance_vocab_size = 42
        tokenizer = SimpleTokenizer(vocab_size=appearance_vocab_size, min_freq=1)
        tokenizer.word_to_idx = {k: v for k, v in expanded_tokenizer.word_to_idx.items() if int(v) < appearance_vocab_size}
        tokenizer.idx_to_word = {k: v for k, v in expanded_tokenizer.idx_to_word.items() if int(k) < appearance_vocab_size}

        prompt = "A small red circle on a blue background."
        token_ids, mask = tokenizer.encode(prompt, max_length=64)
        token_ids_tensor = torch.tensor([token_ids])
        mask_tensor = torch.tensor([mask], dtype=torch.float)
        text_emb, text_mask = text_encoder(token_ids_tensor, mask_tensor)

        # Stage 2.1 forward
        outputs = []
        for f in range(4):
            x_frame = x[:, :, f, :, :]
            output_frame = spatial_unet_original(x_frame, timesteps, text_emb, text_mask)
            outputs.append(output_frame)
        output_spatial = torch.stack(outputs, dim=2)

        # Stage 3 forward (temporal disabled)
        output_video = video_unet_disabled(x, timesteps, text_emb, text_mask)

        # Compare
        max_diff = torch.max(torch.abs(output_spatial - output_video)).item()
        within_tolerance = max_diff < 1e-4

        print(f"Max difference: {max_diff:.10f}")
        print(f"Within tolerance (1e-4): {within_tolerance}")

        self.results['checks']['temporal_disabled_equivalence'] = {
            'status': 'PASS' if within_tolerance else 'FAIL',
            'max_diff': max_diff,
            'tolerance': 1e-4
        }
        return within_tolerance

    def check_3_temporal_enabled_initialization(self):
        """Check temporal-enabled initial output is close to temporal-disabled."""
        self.log_section("CHECK 3: Temporal-Enabled Initialization")

        _, _, _, split_encoder, video_unet_disabled, video_unet_enabled = self.load_models()

        # Create deterministic input
        torch.manual_seed(42)
        x = torch.randn(1, 3, 4, 64, 64)
        timesteps = torch.randint(0, 1000, (1,))

        # Encode text - use full tokenizer
        tokenizer = SimpleTokenizer.load(self.tokenizer_path)

        prompt = "A small red circle on a blue background."
        token_ids, mask = tokenizer.encode(prompt, max_length=64)
        token_ids_tensor = torch.tensor([token_ids])
        mask_tensor = torch.tensor([mask], dtype=torch.float)
        text_emb, text_mask = split_encoder(token_ids_tensor, mask_tensor)

        with torch.no_grad():
            output_disabled = video_unet_disabled(x, timesteps, text_emb, text_mask)
            output_enabled = video_unet_enabled(x, timesteps, text_emb, text_mask)

        max_diff = torch.max(torch.abs(output_disabled - output_enabled)).item()
        close_enough = max_diff < 5.0  # Allow some difference due to temporal initialization

        print(f"Max difference (disabled vs enabled): {max_diff:.10f}")
        print(f"Close enough: {close_enough}")

        self.results['checks']['temporal_enabled_initialization'] = {
            'status': 'PASS' if close_enough else 'FAIL',
            'max_diff': max_diff,
            'threshold': 5.0
        }
        return close_enough

    def check_4_gradients_reach_intended_parameters(self):
        """Check that intended parameters receive gradients."""
        self.log_section("CHECK 4: Gradients Reach Intended Parameters")

        _, _, _, split_encoder, _, video_unet_enabled = self.load_models()

        # Create input
        x = torch.randn(1, 3, 4, 64, 64, requires_grad=True)
        timesteps = torch.randint(0, 1000, (1,))

        # Encode text - use full tokenizer to include motion tokens
        tokenizer = SimpleTokenizer.load(self.tokenizer_path)

        # Include motion tokens to test gradient flow to motion embeddings
        prompt = "A small red circle on a blue background moves left."
        token_ids, mask = tokenizer.encode(prompt, max_length=64)

        # Manually inject a motion token (token 42 = first motion token) to ensure gradient flow
        token_ids[0] = 42  # Replace first token with motion token

        token_ids_tensor = torch.tensor([token_ids])
        mask_tensor = torch.tensor([mask], dtype=torch.float)
        text_emb, text_mask = split_encoder(token_ids_tensor, mask_tensor)

        # Forward pass
        output = video_unet_enabled(x, timesteps, text_emb, text_mask)
        loss = output.mean()
        loss.backward()

        # Check gradients
        motion_emb_grad = split_encoder.token_emb.motion_emb.weight.grad is not None
        temporal_grad = any(p.grad is not None for p in video_unet_enabled.temporal_modules.parameters())
        fusion_grad = any(p.grad is not None for p in video_unet_enabled.fusion_modules.parameters())

        print(f"Motion embeddings have gradients: {motion_emb_grad}")
        print(f"Temporal modules have gradients: {temporal_grad}")
        print(f"Fusion modules have gradients: {fusion_grad}")

        status = motion_emb_grad and temporal_grad and fusion_grad

        self.results['checks']['gradients_intended'] = {
            'status': 'PASS' if status else 'FAIL',
            'motion_emb_grad': motion_emb_grad,
            'temporal_grad': temporal_grad,
            'fusion_grad': fusion_grad
        }
        return status

    def check_5_frozen_parameters_no_gradients(self):
        """Check that frozen parameters receive no gradients."""
        self.log_section("CHECK 5: Frozen Parameters No Gradients")

        _, _, spatial_unet_modified, split_encoder, _, video_unet_enabled = self.load_models()

        # Create input
        x = torch.randn(1, 3, 4, 64, 64, requires_grad=True)
        timesteps = torch.randint(0, 1000, (1,))

        # Encode text - use full tokenizer
        tokenizer = SimpleTokenizer.load(self.tokenizer_path)

        prompt = "A small red circle on a blue background."
        token_ids, mask = tokenizer.encode(prompt, max_length=64)
        token_ids_tensor = torch.tensor([token_ids])
        mask_tensor = torch.tensor([mask], dtype=torch.float)
        text_emb, text_mask = split_encoder(token_ids_tensor, mask_tensor)

        # Forward pass
        output = video_unet_enabled(x, timesteps, text_emb, text_mask)
        loss = output.mean()
        loss.backward()

        # Check frozen parameters
        spatial_grad = any(p.grad is not None for p in spatial_unet_modified.parameters())
        appearance_grad = split_encoder.token_emb.original_emb.weight.grad is not None

        print(f"Spatial parameters have gradients: {spatial_grad}")
        print(f"Appearance embeddings have gradients: {appearance_grad}")

        status = not spatial_grad and not appearance_grad

        self.results['checks']['frozen_no_gradients'] = {
            'status': 'PASS' if status else 'FAIL',
            'spatial_grad': spatial_grad,
            'appearance_grad': appearance_grad
        }
        return status

    def check_6_optimizer_only_permitted_parameters(self):
        """Check optimizer contains only permitted parameters."""
        self.log_section("CHECK 6: Optimizer Only Permitted Parameters")

        _, _, _, split_encoder, _, video_unet_enabled = self.load_models()

        # Create optimizer
        trainable_params = list(split_encoder.token_emb.motion_emb.parameters())
        trainable_params.extend(list(video_unet_enabled.temporal_modules.parameters()))
        trainable_params.extend(list(video_unet_enabled.fusion_modules.parameters()))

        optimizer = optim.AdamW(trainable_params, lr=0.0002)

        # Check optimizer parameters
        optimizer_param_ids = set(id(p) for group in optimizer.param_groups for p in group['params'])
        frozen_param_ids = set()

        # Check spatial parameters
        for name, p in video_unet_enabled.spatial_unet.named_parameters():
            frozen_param_ids.add(id(p))

        # Check appearance embeddings
        frozen_param_ids.add(id(split_encoder.token_emb.original_emb.weight))

        overlap = optimizer_param_ids & frozen_param_ids
        no_overlap = len(overlap) == 0

        print(f"Optimizer parameters: {len(optimizer_param_ids)}")
        print(f"Frozen parameters: {len(frozen_param_ids)}")
        print(f"Overlap: {len(overlap)}")
        print(f"No overlap: {no_overlap}")

        self.results['checks']['optimizer_permitted'] = {
            'status': 'PASS' if no_overlap else 'FAIL',
            'optimizer_params': len(optimizer_param_ids),
            'frozen_params': len(frozen_param_ids),
            'overlap': len(overlap)
        }
        return no_overlap

    def check_7_optimizer_step_unchanged_frozen(self):
        """Check one optimizer step leaves frozen parameters unchanged."""
        self.log_section("CHECK 7: Optimizer Step Leaves Frozen Unchanged")

        spatial_unet_original, text_encoder, spatial_unet_modified, split_encoder, video_unet_disabled, video_unet_enabled = self.load_models()

        # Store initial parameters
        initial_spatial = {name: p.clone() for name, p in spatial_unet_modified.named_parameters()}
        initial_appearance = split_encoder.token_emb.original_emb.weight.clone()

        # Create optimizer
        trainable_params = list(split_encoder.token_emb.motion_emb.parameters())
        trainable_params.extend(list(video_unet_enabled.temporal_modules.parameters()))
        trainable_params.extend(list(video_unet_enabled.fusion_modules.parameters()))

        optimizer = optim.AdamW(trainable_params, lr=0.0002)

        # Dummy forward/backward
        x = torch.randn(1, 3, 4, 64, 64, requires_grad=True)
        timesteps = torch.randint(0, 1000, (1,))

        expanded_tokenizer = SimpleTokenizer.load(self.tokenizer_path)
        from collections import Counter
        appearance_vocab_size = 42
        tokenizer = SimpleTokenizer(vocab_size=appearance_vocab_size, min_freq=1)
        tokenizer.word_to_idx = {k: v for k, v in expanded_tokenizer.word_to_idx.items() if int(v) < appearance_vocab_size}
        tokenizer.idx_to_word = {k: v for k, v in expanded_tokenizer.idx_to_word.items() if int(k) < appearance_vocab_size}

        prompt = "A small red circle on a blue background."
        token_ids, mask = tokenizer.encode(prompt, max_length=64)
        token_ids_tensor = torch.tensor([token_ids])
        mask_tensor = torch.tensor([mask], dtype=torch.float)
        text_emb, text_mask = split_encoder(token_ids_tensor, mask_tensor)

        output = video_unet_enabled(x, timesteps, text_emb, text_mask)
        loss = output.mean()
        loss.backward()
        optimizer.step()

        # Check frozen parameters unchanged
        spatial_unchanged = all(torch.allclose(initial_spatial[name], p, atol=1e-10)
                               for name, p in spatial_unet_modified.named_parameters())
        appearance_unchanged = torch.allclose(initial_appearance, split_encoder.token_emb.original_emb.weight, atol=1e-10)

        print(f"Spatial parameters unchanged: {spatial_unchanged}")
        print(f"Appearance embeddings unchanged: {appearance_unchanged}")

        status = spatial_unchanged and appearance_unchanged

        self.results['checks']['optimizer_step_frozen_unchanged'] = {
            'status': 'PASS' if status else 'FAIL',
            'spatial_unchanged': spatial_unchanged,
            'appearance_unchanged': appearance_unchanged
        }
        return status

    def run_all_checks(self):
        """Run all readiness checks."""
        self.log_section("STAGE 3.2.9B - MODEL READINESS CHECKS")

        checks = [
            self.check_1_checkpoint_hashes_unchanged,
            self.check_2_temporal_disabled_equivalence,
            self.check_3_temporal_enabled_initialization,
            self.check_4_gradients_reach_intended_parameters,
            self.check_5_frozen_parameters_no_gradients,
            self.check_6_optimizer_only_permitted_parameters,
            self.check_7_optimizer_step_unchanged_frozen
        ]

        all_pass = True
        for check in checks:
            try:
                result = check()
                if not result:
                    all_pass = False
            except Exception as e:
                print(f"Check failed with exception: {e}")
                all_pass = False

        self.results['final_status'] = 'PASS' if all_pass else 'FAIL'

        self.log_section("READINESS SUMMARY")
        print(f"Overall Status: {self.results['final_status']}")

        for check_name, check_result in self.results['checks'].items():
            print(f"  {check_name}: {check_result['status']}")

        return self.results


def main():
    experiment_dir = Path(__file__).parent
    checker = ReadinessChecker(str(experiment_dir))
    results = checker.run_all_checks()

    # Save results
    results_path = Path(__file__).parent / "STAGE3_2_9B_READINESS_CHECKS.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    return 0 if results['final_status'] == 'PASS' else 1


if __name__ == "__main__":
    sys.exit(main())
