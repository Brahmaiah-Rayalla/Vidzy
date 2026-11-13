"""
Stage 3.2.9B - Phase 1.1: Temporal Initialization Investigation

Detailed analysis of temporal initialization difference with comprehensive metrics.

Author: Devin
Date: 2026-11-02
"""

import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import math

# Add paths
parent_dir = Path(__file__).parent.parent
stage3_2_7_dir = parent_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(stage3_2_7_dir))
sys.path.insert(0, str(stage3_2_7_dir / "video_foundation"))

from video_foundation.models.conditional_unet import ConditionalUNet as OriginalConditionalUNet
from video_foundation.models.text_encoder import TextEncoder, SimpleTokenizer
from models.conditional_unet import ConditionalUNet
from models.video_unet_fixed import VideoUNetFixed
from models.text_encoder_split import TextEncoderSplit, load_from_standard_encoder


class TemporalInitInvestigator:
    """Investigate temporal initialization differences."""

    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.parent_dir = self.experiment_dir.parent

        # Paths
        self.stage2_checkpoint = str(self.parent_dir / "checkpoints_conditional" / "checkpoint_final.pt")
        self.stage3_spatial_checkpoint = str(self.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "checkpoint_stage2_spatial.pt")
        self.tokenizer_path = str(self.parent_dir / "experiments_stage3_2_7" / "checkpoints" / "tokenizer_motion_expanded.json")

        self.results = {
            'baseline_metrics': {},
            'zero_gate_metrics': {},
            'normal_init_metrics': {},
            'fusion_gate_values': {},
            'temporal_residuals': {},
            'stage3_2_9a_tolerance': 1e-4
        }

    def log_section(self, title: str):
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)

    def load_models(self):
        """Load all required models."""
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

        return (spatial_unet_original, text_encoder, spatial_unet_modified, split_encoder)

    def compute_metrics(self, output1: torch.Tensor, output2: torch.Tensor,
                       baseline_output: torch.Tensor = None) -> Dict:
        """Compute comprehensive comparison metrics."""
        diff = output1 - output2

        metrics = {
            'max_abs_diff': torch.max(torch.abs(diff)).item(),
            'mean_abs_diff': torch.mean(torch.abs(diff)).item(),
            'rmse': torch.sqrt(torch.mean(diff ** 2)).item(),
        }

        if baseline_output is not None:
            # Relative L2 error
            baseline_norm = torch.norm(baseline_output)
            metrics['relative_l2_error'] = torch.norm(diff) / (baseline_norm + 1e-10)

            # Cosine similarity
            flat1 = output1.flatten()
            flat2 = output2.flatten()
            metrics['cosine_similarity'] = torch.nn.functional.cosine_similarity(
                flat1.unsqueeze(0), flat2.unsqueeze(0)
            ).item()

            # Baseline statistics
            metrics['baseline_std'] = torch.std(baseline_output).item()
            metrics['baseline_rms'] = torch.sqrt(torch.mean(baseline_output ** 2)).item()

        return metrics

    def extract_fusion_gate_values(self, model: VideoUNetFixed) -> Dict:
        """Extract fusion gate values from the model (raw parameters, not activated)."""
        gate_values = {}

        for name, module in model.fusion_modules.named_children():
            if hasattr(module, 'gate'):
                gate_values[name] = module.gate.data.item()

        return gate_values

    def extract_temporal_residuals(self, model: VideoUNetFixed, x: torch.Tensor,
                                  timesteps: torch.Tensor, text_emb: torch.Tensor,
                                  text_mask: torch.Tensor) -> Dict:
        """Extract temporal residual magnitudes at fusion locations."""
        # This requires modifying the model to capture intermediate values
        # For now, we'll record that this needs implementation
        return {
            'note': 'Requires model modification to capture intermediate temporal residuals',
            'status': 'deferred'
        }

    def force_zero_gates(self, model: VideoUNetFixed):
        """Force all fusion gates to zero."""
        for name, module in model.fusion_modules.named_children():
            if hasattr(module, 'gate'):
                # Set to -10 so that sigmoid(-10) ≈ 0
                module.gate.data.fill_(-10.0)

    def restore_gates(self, model: VideoUNetFixed, original_values: Dict):
        """Restore fusion gates to original values."""
        for name, module in model.fusion_modules.named_children():
            if hasattr(module, 'gate') and name in original_values:
                # Restore the gate parameter value (before sigmoid activation)
                module.gate.data.fill_(original_values[name])

    def run_investigation(self):
        """Run the complete temporal initialization investigation."""
        self.log_section("TEMPORAL INITIALIZATION INVESTIGATION")

        # Load models
        spatial_unet_original, text_encoder, spatial_unet_modified, split_encoder = self.load_models()

        # Create model configurations
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

        # Store original gate values
        original_gate_values = self.extract_fusion_gate_values(video_unet_enabled)
        self.results['fusion_gate_values']['original'] = original_gate_values

        # Create deterministic input
        torch.manual_seed(42)
        x = torch.randn(1, 3, 4, 64, 64)
        timesteps = torch.randint(0, 1000, (1,))

        # Encode text
        tokenizer = SimpleTokenizer.load(self.tokenizer_path)
        prompt = "A small red circle on a blue background."
        token_ids, mask = tokenizer.encode(prompt, max_length=64)
        token_ids_tensor = torch.tensor([token_ids])
        mask_tensor = torch.tensor([mask], dtype=torch.float)
        text_emb, text_mask = split_encoder(token_ids_tensor, mask_tensor)

        # === Configuration A: Temporal Disabled ===
        self.log_section("Configuration A: Temporal Modules Disabled")
        with torch.no_grad():
            output_a = video_unet_disabled(x, timesteps, text_emb, text_mask)

        # Baseline metrics
        baseline_metrics = self.compute_metrics(output_a, output_a, output_a)
        self.results['baseline_metrics'] = baseline_metrics

        print(f"Baseline Output Statistics:")
        print(f"  Std Dev: {baseline_metrics['baseline_std']:.10f}")
        print(f"  RMS: {baseline_metrics['baseline_rms']:.10f}")

        # === Configuration B: Temporal Enabled with Zero Gates ===
        self.log_section("Configuration B: Temporal Enabled with Zero Gates")
        self.force_zero_gates(video_unet_enabled)
        self.results['fusion_gate_values']['forced_zero'] = {k: 0.0 for k in original_gate_values.keys()}

        with torch.no_grad():
            output_b = video_unet_enabled(x, timesteps, text_emb, text_mask)

        # Compare A vs B
        metrics_a_vs_b = self.compute_metrics(output_a, output_b, output_a)
        self.results['zero_gate_metrics'] = metrics_a_vs_b

        print(f"Configuration A vs B (Zero Gates):")
        print(f"  Max Abs Diff: {metrics_a_vs_b['max_abs_diff']:.10f}")
        print(f"  Mean Abs Diff: {metrics_a_vs_b['mean_abs_diff']:.10f}")
        print(f"  RMSE: {metrics_a_vs_b['rmse']:.10f}")
        print(f"  Relative L2 Error: {metrics_a_vs_b['relative_l2_error']:.10f}")
        print(f"  Cosine Similarity: {metrics_a_vs_b['cosine_similarity']:.10f}")

        within_tolerance = metrics_a_vs_b['max_abs_diff'] < self.results['stage3_2_9a_tolerance']
        print(f"  Within Stage 3.2.9A Tolerance (1e-4): {within_tolerance}")

        if not within_tolerance:
            print("  WARNING: Zero-gate configuration does NOT match temporal-disabled output!")
            print("  This indicates a problem with the temporal module implementation.")

        # === Configuration C: Temporal Enabled with Normal Initialization ===
        self.log_section("Configuration C: Temporal Enabled with Normal Initialization")
        self.restore_gates(video_unet_enabled, original_gate_values)
        self.results['fusion_gate_values']['restored'] = original_gate_values

        with torch.no_grad():
            output_c = video_unet_enabled(x, timesteps, text_emb, text_mask)

        # Compare A vs C
        metrics_a_vs_c = self.compute_metrics(output_a, output_c, output_a)
        self.results['normal_init_metrics'] = metrics_a_vs_c

        print(f"Configuration A vs C (Normal Init):")
        print(f"  Max Abs Diff: {metrics_a_vs_c['max_abs_diff']:.10f}")
        print(f"  Mean Abs Diff: {metrics_a_vs_c['mean_abs_diff']:.10f}")
        print(f"  RMSE: {metrics_a_vs_c['rmse']:.10f}")
        print(f"  Relative L2 Error: {metrics_a_vs_c['relative_l2_error']:.10f}")
        print(f"  Cosine Similarity: {metrics_a_vs_c['cosine_similarity']:.10f}")

        # Compare B vs C (effect of gate initialization)
        metrics_b_vs_c = self.compute_metrics(output_b, output_c, output_b)
        self.results['gate_perturbation_metrics'] = metrics_b_vs_c

        print(f"\nConfiguration B vs C (Gate Perturbation Effect):")
        print(f"  Max Abs Diff: {metrics_b_vs_c['max_abs_diff']:.10f}")
        print(f"  Mean Abs Diff: {metrics_b_vs_c['mean_abs_diff']:.10f}")
        print(f"  RMSE: {metrics_b_vs_c['rmse']:.10f}")

        # Print gate values
        self.log_section("Fusion Gate Values")
        print("Original gate parameters (before sigmoid):")
        for name, value in original_gate_values.items():
            print(f"  {name}: {value:.10f}")
        print("Original gate values (after sigmoid):")
        for name, value in original_gate_values.items():
            activated = 1.0 / (1.0 + math.exp(-value))
            print(f"  {name}: {activated:.10f}")

        # Analysis
        self.log_section("ANALYSIS")
        print(f"Stage 3.2.9A Tolerance: {self.results['stage3_2_9a_tolerance']}")
        print(f"\nZero-gate vs Temporal-Disabled:")
        print(f"  Max difference: {metrics_a_vs_b['max_abs_diff']:.10f}")
        print(f"  Expected: < {self.results['stage3_2_9a_tolerance']}")
        print(f"  Status: {'PASS' if within_tolerance else 'FAIL'}")

        print(f"\nNormal-init vs Temporal-Disabled:")
        print(f"  Max difference: {metrics_a_vs_c['max_abs_diff']:.10f}")
        print(f"  This represents the perturbation from near-zero gate initialization")

        # Determine if perturbation is acceptable
        perturbation_ratio = metrics_a_vs_c['max_abs_diff'] / (baseline_metrics['baseline_rms'] + 1e-10)
        print(f"  Perturbation ratio (diff / baseline RMS): {perturbation_ratio:.10f}")

        if perturbation_ratio > 0.1:
            print("  WARNING: Near-zero initialization causes >10% of baseline RMS deviation")
            print("  This may indicate improper initialization or architectural issues")
        elif perturbation_ratio > 0.01:
            print("  CAUTION: Near-zero initialization causes >1% of baseline RMS deviation")
            print("  Monitor training stability")
        else:
            print("  Near-zero initialization perturbation is reasonable")

        # Convert tensors to floats for JSON serialization
        for key in ['baseline_metrics', 'zero_gate_metrics', 'normal_init_metrics', 'gate_perturbation_metrics']:
            if key in self.results:
                self.results[key] = {k: float(v) if hasattr(v, 'item') else v
                                    for k, v in self.results[key].items()}

        # Convert gate values to floats
        for key in ['fusion_gate_values']:
            if key in self.results:
                self.results[key] = {k: float(v) if hasattr(v, 'item') else v
                                    for k, v in self.results[key].items()}

        return self.results


def main():
    experiment_dir = Path(__file__).parent
    investigator = TemporalInitInvestigator(str(experiment_dir))
    results = investigator.run_investigation()

    # Save results
    results_path = Path(__file__).parent / "STAGE3_2_9B_TEMPORAL_INIT_INVESTIGATION.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
