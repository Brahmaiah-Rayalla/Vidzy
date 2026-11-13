# Stage 3 CPU-First Video Architecture Proposal

**Date**: October 31, 2026
**Objective**: Design CPU-optimized synthetic video generation architecture

## Target Specifications
- **Frames**: 8-16 frames
- **Resolution**: 64×64
- **Content**: Synthetic moving shapes
- **Architecture**: Compact temporal convolution + lightweight attention
- **Training**: Random initialization (no pretrained weights)
- **Inference**: DDIM-based fast sampling
- **Platform**: CPU-first, GPU optional

## Proposed Architecture

### Overview
```
Stage 2 Spatial U-Net (frozen or fine-tuned) → Temporal Module → Sampled Video
                              ↓
                        Text Conditioning (shared)
```

### Spatial Backbone (Stage 2 Reuse)
- **Model**: Stage 2.1 conditional U-Net (11.6M parameters)
- **Resolution**: 64×64 per frame
- **Conditioning**: Text embeddings (256-dim)
- **Status**: Already trained and validated
- **Options**:
  - Frozen: Use as spatial feature extractor only
  - Fine-tuned: Continue training with temporal loss

### Temporal Module (New)

#### Design Option A: 1D Temporal Convolution (Recommended)
```python
class TemporalModule(nn.Module):
    """
    Lightweight 1D temporal convolution for video generation.
    Processes temporal dimension of spatial features.
    """
    def __init__(self, spatial_dim=128, temporal_kernel=3, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(spatial_dim, spatial_dim, kernel_size=temporal_kernel, padding=1)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(spatial_dim)
    
    def forward(self, x):  # x: [batch, frames, spatial_dim]
        x = x.transpose(1, 2)  # [batch, spatial_dim, frames]
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)  # [batch, frames, spatial_dim]
        return self.norm(x)
```

**Parameters**: ~50K
**Forward time**: ~5ms per 16-frame window
**Memory**: ~1 MB

#### Design Option B: Temporal Self-Attention
```python
class TemporalAttention(nn.Module):
    def __init__(self, spatial_dim=128, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(spatial_dim, num_heads, batch_first=True)
    
    def forward(self, x):  # x: [batch, frames, spatial_dim]
        x, _ = self.attn(x, x, x)
        return x
```

**Parameters**: ~100K
**Forward time**: ~10ms per 16-frame window
**Memory**: ~2 MB

#### Design Option C: Recurrent (GRU)
```python
class TemporalGRU(nn.Module):
    def __init__(self, spatial_dim=128):
        super().__init__()
        self.gru = nn.GRU(spatial_dim, spatial_dim, batch_first=True)
    
    def forward(self, x):  # x: [batch, frames, spatial_dim]
        x, _ = self.gru(x)
        return x
```

**Parameters**: ~200K
**Forward time**: ~8ms per 16-frame window
**Memory**: ~1.5 MB

**Recommendation**: Start with Option A (1D Conv) - simplest and fastest.

### Complete Pipeline

#### Training
```
Text Prompt → Text Encoder → Text Embedding
                                    ↓
    Noise Frames (N×64×64) → Stage 2 U-Net (per frame) → Spatial Features
                                                      ↓
                                               Temporal Module
                                                      ↓
                                            Frame-wise DDPM Loss
```

#### Inference (DDIM)
```
Text Prompt → Text Encoder → Text Embedding
                                    ↓
    Noise Frames (N×64×64) → Stage 2 U-Net (per frame) → Spatial Features
                                                      ↓
                                               Temporal Module
                                                      ↓
                                            DDIM Update (per frame)
                                                      ↓
                                      Output Video (N×64×64)
```

## CPU Execution Considerations

### Memory Requirements

#### Per-Component Memory
- Stage 2 U-Net (frozen): 387 MB (checkpoint)
- Text encoder: 50 MB
- Temporal module: ~2 MB
- Frame buffer (16 frames): 16 × 3 × 64 × 64 × 4 bytes = 0.5 MB
- Activations (est): ~50 MB
- **Total**: ~490 MB peak

#### Memory Management Strategies
1. **Process frames in batches**: 4 frames at a time (sliding window)
2. **Stream to disk**: Save frames as they're generated
3. **Lower precision**: Use float16 where compatible
4. **Gradient checkpointing**: For training if needed

### Performance Estimates

#### Stage 2 Baseline (Image)
- Forward pass: 128 ms
- DDIM 50 steps: 6.4s
- DDIM 100 steps: 12.8s

#### Stage 3 Estimates (Video, 16 frames)
**Option A: Parallel Frame Processing**
- 16 frames × 128 ms = 2.05s (parallel on 8 threads)
- Temporal module: 5ms
- DDIM 25 steps: 2.05s × 25 = 51s
- **Total**: ~51s per 16-frame video

**Option B: Sequential Frame Processing**
- 16 frames × 128 ms = 2.05s (sequential)
- Temporal module: 5ms per window
- DDIM 25 steps: 2.05s × 25 = 51s
- **Total**: ~51s per 16-frame video

**Scaling**:
- 8 frames: ~25s
- 16 frames: ~51s
- 32 frames: ~102s (1.7 min)
- 64 frames: ~204s (3.4 min)

### Optimization Strategies

#### 1. Temporal Window Processing
```python
def process_temporal_windows(frames, window_size=5, stride=2):
    outputs = []
    for i in range(0, len(frames) - window_size + 1, stride):
        window = frames[i:i+window_size]
        refined = temporal_module(window)
        outputs.append(refined[window_size//2])  # Take center frame
    return outputs
```

**Benefits**:
- Constant memory (window_size instead of total_frames)
- Linear time with total frames
- Enables arbitrary length videos

#### 2. DDIM for Video
```python
def generate_video_ddim(prompt, num_frames=16, num_steps=25):
    timesteps = torch.linspace(999, 0, num_steps, dtype=torch.long)
    
    for t in timesteps:
        # Apply spatial U-Net to all frames
        for i in range(num_frames):
            frames[i] = unet(frames[i], t, text_emb)
        
        # Apply temporal module
        frames = temporal_module(frames)
        
        # DDIM update
        frames = ddim_update(frames, t)
```

**Benefits**:
- 40x faster than DDPM (25 vs 1000 steps)
- Deterministic (same seed = same output)
- Suitable for CPU inference

#### 3. Frame Reuse
- Cache spatial features if temporal stride > 1
- Only recompute for new frames in window
- Reduces computation by ~50% for stride=2

#### 4. Progressive Refinement
- Generate low-res video first (32×32)
- Upscale spatial resolution after generation
- User sees progress faster
- Can stop early if satisfied

## Parameter Count

### Total Architecture
- Stage 2 U-Net: 9.4M (or 11.6M with text encoder)
- Temporal module: 0.05M (1D conv option)
- **Total**: ~11.65M parameters

### Comparison
- Stage 2 (image): 11.6M
- Stage 3 (video): 11.65M
- **Increase**: 0.05M (0.4%)

## Training Strategy

### Phase 1: Synthetic Dataset Generation
- Generate 5,000 synthetic video clips
- Moving shapes with different velocities
- Captions describing motion
- 8-16 frames per clip
- 64×64 resolution

### Phase 2: Temporal Module Training
- Freeze Stage 2 U-Net (initially)
- Train temporal module only
- Loss: Frame-wise DDPM loss + temporal consistency loss
- 5,000-10,000 steps
- Expected: 1-2 hours on CPU

### Phase 3: Fine-tuning (Optional)
- Unfreeze Stage 2 U-Net
- Joint training with low learning rate
- Additional 5,000 steps
- Expected: Improved spatial quality

### Phase 4: Evaluation
- Generate test videos
- Measure temporal coherence
- Measure frame-wise quality
- Compare to baseline (no temporal module)

## Acceptance Criteria

### Minimum Viable Product
- [ ] Generates 8-frame synthetic videos
- [ ] Videos show moving shapes
- [ ] Runs on CPU without CUDA
- [ ] DDIM sampling <60 seconds for 16 frames
- [ ] Peak memory <1 GB
- [ ] Temporal coherence visible

### Full Product
- [ ] Generates 16-frame videos
- [ ] Text conditioning influences motion
- [ ] DDIM sampling <30 seconds for 16 frames
- [ ] Peak memory <500 MB
- [ ] Frame quality comparable to Stage 2
- [ ] Temporal coherence high

## Risks and Mitigations

### Risk 1: DDIM Quality Degradation
**Mitigation**: Use 50-100 steps for video (vs 25 for images), temporal smoothing helps

### Risk 2: Memory Blowup
**Mitigation**: Process in batches, stream to disk, use float16

### Risk 3: Temporal Incoherence
**Mitigation**: Small temporal windows (3-5 frames), overlapping windows

### Risk 4: Training Too Slow
**Mitigation**: Start with 8 frames, 5,000 examples, freeze spatial U-Net

### Risk 5: Text Control Weak
**Mitigation**: Reuse Stage 2 text encoder (already validated), add temporal conditioning

## Implementation Timeline

### Week 1: Architecture and Dataset
- Implement temporal module (1D conv)
- Generate synthetic video dataset
- Create video data loader

### Week 2: Training
- Train temporal module (frozen U-Net)
- Benchmark training speed
- Generate sample videos

### Week 3: Evaluation and Optimization
- Evaluate temporal coherence
- Optimize memory usage
- Implement DDIM video sampling
- Benchmark inference speed

### Week 4: Integration
- Create video inference CLI
- Document usage
- Prepare Stage 3 report

## Conclusion

The proposed Stage 3 architecture is CPU-first by design:
- **Minimal parameters**: 11.65M total (0.4% increase over Stage 2)
- **Efficient inference**: ~51s for 16-frame video with DDIM
- **Low memory**: ~490 MB peak
- **Scalable**: Supports arbitrary length via sliding windows
- **Ready for CPU**: No GPU required

The architecture reuses the validated Stage 2 spatial U-Net and adds minimal temporal processing. This approach minimizes risk while enabling video generation on consumer CPU hardware.

**Recommendation**: Proceed with Option A (1D temporal convolution) and 16-frame target. Scale to 32 frames only if performance allows.
