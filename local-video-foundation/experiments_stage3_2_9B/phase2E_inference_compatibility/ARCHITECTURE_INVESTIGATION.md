# Stage 3.2.9D - Architecture Compatibility Investigation

**Date:** 2026-11-02
**Purpose:** Investigate architecture differences between Stage 2.1 and Stage 3 checkpoints

## Architecture Summary

### Experiment A (Stage 2.1)
**Model Type:** `ConditionalUNet` (plain text-conditioned diffusion model)

**Checkpoint Structure:**
- Top-level keys: `step`, `model_state_dict`, `text_encoder_state_dict`, `optimizer_state_dict`, `config`, `seed`
- Model state dict keys: Direct parameter names (e.g., `conv_in.weight`, `down_blocks.0.0.conv1.weight`)
- Total parameters: 504 keys in model_state_dict

**Configuration:**
```python
{
  'model': {
    'image_size': 64,
    'in_channels': 3,
    'model_channels': 64,
    'channel_mult': [1, 2, 2, 2],
    'attention_resolutions': [],
    'num_res_blocks': 2,
    'dropout': 0.0,
    'use_cross_attn': True,
    'n_heads': 4
  }
}
```

**Key Features:**
- Spatial-only processing (2D convolutions)
- Text conditioning via cross-attention
- No temporal processing
- No video-specific components

### Experiments B & C (Stage 3)
**Model Type:** `VideoUNetFixed` (video generation model with temporal fusion)

**Checkpoint Structure:**
- Top-level keys: `video_unet`, `optimizer`, `step`, `loss`, `config`
- Model state dict keys: Nested under `video_unet` (e.g., `spatial_unet.conv_in.weight`, `text_encoder.*`, `temporal_convs.*`, `fusion_gates.*`)
- Total parameters: 589 keys in video_unet state dict

**Configuration (Experiment B):**
```python
{
  'enable_temporal': False,
  'num_steps': 100,
  'seed': 42,
  'trainable_params': 10240
}
```

**Architecture Components:**
- `spatial_unet`: ConditionalUNet for spatial processing
- `text_encoder`: TextEncoder for text embeddings
- `temporal_convs`: TemporalConv1DBlock for temporal processing
- `fusion_gates`: FeatureLevelTemporalFusion for feature fusion
- `use_temporal`: Boolean flag (False for B, True for C)

**Key Features:**
- Spatial processing via ConditionalUNet
- Text conditioning via TextEncoder
- Temporal convolution layers
- Feature-level fusion gates
- Video generation capability

## Compatibility Analysis

### Experiment A (Stage 2.1 Checkpoint)
- **Architecture:** ConditionalUNet
- **Checkpoint location:** `checkpoints_conditional/checkpoint_final.pt`
- **Compatible implementation:** `video_foundation/models/conditional_unet.py`
- **Loading requirements:** Use ConditionalUNet with model_channels=64

### Experiment B (Stage 3 Checkpoint)
- **Architecture:** VideoUNetFixed with enable_temporal=False
- **Checkpoint location:** `experiments_stage3_2_9B/phase2C/checkpoints/checkpoint_phase2c_B.pt`
- **Compatible implementation:** `experiments_stage3_2_7/video_foundation/models/video_unet_feature_fusion.py`
- **Loading requirements:** Use VideoUNetFixed with use_temporal=False

### Experiment C (Stage 3 Checkpoint)
- **Architecture:** VideoUNetFixed with enable_temporal=True
- **Checkpoint location:** `experiments_stage3_2_9B/phase2C/checkpoints/checkpoint_phase2c_C.pt`
- **Compatible implementation:** `experiments_stage3_2_7/video_foundation/models/video_unet_feature_fusion.py`
- **Loading requirements:** Use VideoUNetFixed with use_temporal=True

## Root Cause of Loading Issues

The previous generation attempt failed because:
1. **Wrong model type:** Attempted to load VideoUNetFixed checkpoints with ConditionalUNet
2. **Nested structure:** VideoUNetFixed checkpoints have nested state dict under `video_unet` key
3. **Missing dependencies:** VideoUNetFixed requires additional components (TextEncoder, temporal convs, fusion gates)

## Solution: Architecture-Specific Adapters

Create separate adapters for each checkpoint type:

1. **Stage2Adapter:** Loads Stage 2.1 checkpoint with ConditionalUNet
2. **Stage3Adapter:** Loads Stage 3 checkpoints with VideoUNetFixed
3. **CommonInterface:** Exposes consistent video generation interface

## Adapter Implementation Strategy

```python
class InferenceAdapter:
    """Common interface for all model types"""
    
    def __init__(self, checkpoint_path, config):
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None
        
    def load(self):
        """Load appropriate model based on checkpoint type"""
        pass
    
    def generate_video(self, text_prompt, num_frames=16):
        """Common video generation interface"""
        pass
```

## Required Dependencies

For Stage 3 adapters, need to copy:
- `video_unet_feature_fusion.py` (VideoUNetFixed implementation)
- `conditional_unet.py` (Spatial UNet component)
- `text_encoder.py` (TextEncoder component)
- `unet.py` (Base UNet components)

## Conclusion

The architecture differences are significant:
- Stage 2.1 uses plain ConditionalUNet for image generation
- Stage 3 uses VideoUNetFixed which wraps ConditionalUNet for video generation
- Checkpoint structures are fundamentally different (flat vs nested)
- Adapter pattern must handle both architectures appropriately

**Recommendation:** Implement Stage2Adapter (ConditionalUNet) and Stage3Adapter (VideoUNetFixed) classes that load checkpoints with their native architectures, then wrap them in a common video generation interface.
