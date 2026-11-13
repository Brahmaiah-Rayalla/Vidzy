"""
Stage 3.2.7B: Parameter Group Configuration

Defines parameter groups for different training scenarios.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from experiments_stage3_2_7.training.text_encoder_split import convert_text_encoder_to_split


class ParameterGroupManager:
    """Manage parameter groups for different training configurations."""

    def __init__(self, model: nn.Module):
        self.model = model

    def setup_split_embeddings(
        self,
        appearance_vocab_size: int = 42,
        motion_vocab_size: int = 40
    ):
        """
        Convert text encoder to use split embeddings for selective training.

        Args:
            appearance_vocab_size: Number of appearance tokens (0-41)
            motion_vocab_size: Number of motion tokens (42-81)
        """
        if hasattr(self.model, 'text_encoder'):
            # Check if already using split embeddings
            if hasattr(self.model.text_encoder, 'token_emb') and \
               hasattr(self.model.text_encoder.token_emb, 'motion_emb'):
                print("Text encoder already uses split embeddings")
                return

            # Convert to split embeddings
            print("Converting text encoder to split embeddings...")
            convert_text_encoder_to_split(
                self.model.text_encoder,
                appearance_vocab_size=appearance_vocab_size,
                motion_vocab_size=motion_vocab_size
            )
            print("Split embeddings configured:")
            print(f"  - Appearance tokens: 0-{appearance_vocab_size-1} (frozen)")
            print(f"  - Motion tokens: {appearance_vocab_size}-{appearance_vocab_size+motion_vocab_size-1} (trainable)")

    def get_experiment_a_groups(self) -> Dict[str, nn.Parameter]:
        """
        Experiment A: Motion vocabulary only
        - Trainable: New motion embeddings (tokens 42-81)
        - Frozen: Original appearance embeddings (0-41), transformer layers, spatial U-Net
        - Temporal: Disabled

        Uses split embedding architecture for proper weight decay handling.
        """
        groups = {}

        # Ensure split embeddings are set up
        self.setup_split_embeddings(appearance_vocab_size=42, motion_vocab_size=40)

        # Only train motion embeddings
        if hasattr(self.model, 'text_encoder'):
            # Check if using split embedding architecture
            if hasattr(self.model.text_encoder, 'token_emb') and hasattr(self.model.text_encoder.token_emb, 'motion_emb'):
                # Split embedding: only motion embeddings are trainable
                groups['motion_embeddings'] = self.model.text_encoder.token_emb.motion_emb.weight
            else:
                # Fallback: try to identify motion indices
                token_emb = self.model.text_encoder.token_emb
                vocab_size = token_emb.weight.shape[0]
                # Motion tokens are indices 42-81 (assuming 82-token vocab)
                if vocab_size > 42:
                    motion_indices = list(range(42, min(82, vocab_size)))
                    motion_mask = torch.zeros(vocab_size, dtype=torch.bool)
                    motion_mask[motion_indices] = True
                    groups['motion_embeddings'] = token_emb.weight
                    groups['motion_mask'] = motion_mask

        return groups

    def get_experiment_b_groups(self) -> Dict[str, nn.Parameter]:
        """
        Experiment B: Temporal fusion only
        - Trainable: Temporal convolution parameters, fusion gates
        - Frozen: All spatial parameters, all text encoder parameters
        - Temporal: Enabled
        """
        groups = {}

        if hasattr(self.model, 'temporal_modules'):
            for name, module in self.model.temporal_modules.items():
                for param_name, param in module.named_parameters():
                    groups[f'temporal_{name}_{param_name}'] = param

        if hasattr(self.model, 'fusion_modules'):
            for name, module in self.model.fusion_modules.items():
                for param_name, param in module.named_parameters():
                    groups[f'fusion_{name}_{param_name}'] = param

        return groups

    def get_experiment_c_groups(self) -> Dict[str, nn.Parameter]:
        """
        Experiment C: Combined architecture
        - Trainable: Motion embeddings, temporal parameters, fusion gates
        - Frozen: Original appearance embeddings, transformer layers, spatial backbone
        - Temporal: Enabled

        Uses split embedding architecture for proper weight decay handling.
        """
        groups = {}

        # Ensure split embeddings are set up
        self.setup_split_embeddings(appearance_vocab_size=42, motion_vocab_size=40)

        # Only train motion embeddings (not full text encoder)
        if hasattr(self.model, 'text_encoder'):
            # Check if using split embedding architecture
            if hasattr(self.model.text_encoder, 'token_emb') and hasattr(self.model.text_encoder.token_emb, 'motion_emb'):
                # Split embedding: only motion embeddings are trainable
                groups['motion_embeddings'] = self.model.text_encoder.token_emb.motion_emb.weight
            else:
                # Fallback: try to identify motion indices
                token_emb = self.model.text_encoder.token_emb
                vocab_size = token_emb.weight.shape[0]
                # Motion tokens are indices 42-81 (assuming 82-token vocab)
                if vocab_size > 42:
                    motion_indices = list(range(42, min(82, vocab_size)))
                    motion_mask = torch.zeros(vocab_size, dtype=torch.bool)
                    motion_mask[motion_indices] = True
                    groups['motion_embeddings'] = token_emb.weight
                    groups['motion_mask'] = motion_mask

        # Temporal parameters
        if hasattr(self.model, 'temporal_modules'):
            for name, module in self.model.temporal_modules.items():
                for param_name, param in module.named_parameters():
                    groups[f'temporal_{name}_{param_name}'] = param

        # Fusion gates
        if hasattr(self.model, 'fusion_modules'):
            for name, module in self.model.fusion_modules.items():
                for param_name, param in module.named_parameters():
                    groups[f'fusion_{name}_{param_name}'] = param

        return groups

    def freeze_all_except(self, trainable_params: Dict[str, nn.Parameter]):
        """Freeze all parameters except those in trainable_params."""
        trainable_set = set(trainable_params.values())

        for param in self.model.parameters():
            if param not in trainable_set:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count total or trainable parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters())

    def get_parameter_summary(self) -> Dict[str, int]:
        """Get summary of parameter counts by module."""
        summary = {}

        for name, param in self.model.named_parameters():
            module_name = name.split('.')[0]
            if module_name not in summary:
                summary[module_name] = {'total': 0, 'trainable': 0}
            summary[module_name]['total'] += param.numel()
            if param.requires_grad:
                summary[module_name]['trainable'] += param.numel()

        return summary
