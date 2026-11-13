# Stage 3 CPU-First Architecture Guide

**Purpose**: Design principles and implementation guidelines for CPU-optimized video generation architecture.

## Core Design Principles

### 1. Factorized Spatial-Temporal Processing
**Rationale**: Processing full 3D spatiotemporal data is too expensive for CPU.

**Design**:
```
Stage 2 Spatial U-Net (per frame) → Stage 3 Temporal Module (across frames)
```

**Benefits**:
- Reuses efficient Stage 2 architecture
- Separates spatial and temporal complexity
- Enables parallel frame processing
- Reduces memory footprint

### 2. Small Temporal Windows
**Rationale**: Processing entire video sequences (e.g., 60 frames) is infeasible on CPU.

**Design**:
- Process 3-5 frames at a time
- Sliding window approach
- Overlap between windows for continuity

**Benefits**:
- Linear memory scaling with window size
- Constant computational complexity per window
- Trivial to parallelize across windows

### 3. DDIM-First Sampling
**Rationale**: DDPM with 1000 steps is impossible for video (would take hours).

**Design**:
- DDIM with 20-30 steps for video
- Fewer steps needed for video (temporal coherence smooths noise)
- Configurable step count for quality/speed tradeoff

**Benefits**:
- 50-100x speedup vs DDPM
- Real-time-ish generation on CPU
- Minimal quality degradation

### 4. Lightweight Temporal Layers
**Rationale**: 3D convolutions are too expensive for CPU.

**Design Options** (in order of preference):
1. **1D Temporal Convolutions**: Process time dimension only on spatial features
2. **Temporal Attention**: Self-attention across frames on flattened features
3. **Recurrent**: GRU/LSTM on frame features (simple but effective)

**Avoid**:
- 3D convolutions (too expensive)
- Large temporal transformers (overkill)
- Frame-wise full U-Net (no temporal efficiency)

### 5. Progressive Refinement
**Rationale**: Generating high-resolution video directly is too slow.

**Design**:
- Generate low-res video first (32×32 or 48×48)
- Upscale spatial resolution after generation
- Optional: refine temporal coherence

**Benefits**:
- Faster initial generation
- User sees progress
- Can stop early if satisfied

### 6. Configurable Frame Counts
**Rationale**: Different use cases need different video lengths.

**Design**:
- Support 8, 16, 32, 64 frame outputs
- User can choose quality vs speed
- Default to 16 frames for testing

**Benefits**:
- Flexible for different hardware
- Scales linearly with frame count
- Enables short clips for quick testing

## Proposed Architecture

### Spatial Backbone (Stage 2 Reuse)
```
Input Frame (64×64) → Stage 2 U-Net → Spatial Features (128-dim)
                        ↓
                  Text Conditioning (shared)
```

**Implementation**:
- Load Stage 2 checkpoint (frozen or fine-tuned)
- Extract intermediate features
- Reuse text encoder for conditioning

### Temporal Module (New)
```
Spatial Features (N frames × 128-dim) → Temporal 1D Conv → Refined Features
                                              ↓
                                      Frame-wise Decoder
```

**Architecture Options**:

#### Option A: 1D Temporal Convolution (Recommended)
```python
class TemporalModule(nn.Module):
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

**Parameters**: ~50K (negligible)
**Forward time**: ~5ms per window
**Memory**: ~1 MB

#### Option B: Temporal Self-Attention
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
**Forward time**: ~10ms per window
**Memory**: ~2 MB

#### Option C: Recurrent (GRU)
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
**Forward time**: ~8ms per window
**Memory**: ~1.5 MB

**Recommendation**: Start with Option A (1D Conv) - simplest and fastest.

### Complete Pipeline
```
Text Prompt → Text Encoder → Text Embedding
                                    ↓
    Noise Frames (N×64×64) → Stage 2 U-Net (per frame) → Spatial Features
                                                      ↓
                                               Temporal Module
                                                      ↓
                                            Frame-wise Upsampling
                                                      ↓
                                      Output Video (N×64×64)
```

## Implementation Guidelines

### 1. Memory Management
```python
# Process frames in batches to limit memory
def generate_video(prompt, num_frames=16, batch_size=4):
    for i in range(0, num_frames, batch_size):
        frame_batch = generate_frames(prompt, batch_size)
        # Store or yield frames
```

### 2. Temporal Window Processing
```python
def process_temporal_windows(frames, window_size=5, stride=2):
    outputs = []
    for i in range(0, len(frames) - window_size + 1, stride):
        window = frames[i:i+window_size]
        refined = temporal_module(window)
        outputs.append(refined[window_size//2])  # Take center frame
    return outputs
```

### 3. DDIM Sampling for Video
```python
def generate_video_ddim(prompt, num_frames=16, num_steps=25):
    # Subsample timesteps for DDIM
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

### 4. Thread Optimization
```python
# At script startup
import torch
torch.set_num_threads(8)  # Use all logical cores
torch.set_num_interop_threads(4)

# For frame generation, can use multiprocessing
from multiprocessing import Pool
with Pool(4) as p:
    frames = p.map(generate_frame, range(num_frames))
```

## Performance Targets

### Baseline (Stage 2)
- Single image (64×64): 6.4s with DDIM (50 steps)
- Memory: 434 MB peak

### Stage 3 Targets
- 16-frame video (64×64): <60 seconds
- Memory: <1 GB peak
- Throughput: >0.25 videos/second

### Scaling
- 8 frames: ~30 seconds
- 16 frames: ~60 seconds
- 32 frames: ~120 seconds (2 minutes)
- 64 frames: ~240 seconds (4 minutes)

## Quality Considerations

### Temporal Coherence
- Small temporal windows (3-5 frames) ensure local coherence
- Overlapping windows smooth transitions
- DDIM's deterministic sampling helps consistency

### Tradeoffs
- **Fewer DDIM steps**: Faster but noisier
- **Smaller windows**: Faster but less global coherence
- **Lower resolution**: Faster but blurrier
- **Fewer frames**: Faster but shorter video

### Recommended Defaults
- **Resolution**: 64×64 (spatial), 16 frames (temporal)
- **DDIM steps**: 25 (vs 50 for images)
- **Temporal window**: 5 frames
- **Temporal stride**: 2 frames

## Optimization Checklist

### Before Implementation
- [ ] Finalize Stage 2 with DDIM default
- [ ] Benchmark Stage 2 final performance
- [ ] Profile Stage 2 forward pass bottlenecks

### During Implementation
- [ ] Use 1D temporal convolutions (not 3D)
- [ ] Process frames in batches (not all at once)
- [ ] Implement DDIM from start (not DDPM)
- [ ] Add thread configuration
- [ ] Profile memory usage per component

### After Implementation
- [ ] Benchmark video generation time
- [ ] Measure peak memory
- [ ] Test with different frame counts
- [ ] Test with different DDIM step counts
- [ ] Evaluate temporal coherence visually

## Potential Bottlenecks & Solutions

### Bottleneck 1: Spatial U-Net Per Frame
**Issue**: Stage 2 U-Net is called N times
**Solution**:
- Parallelize across frames (multiprocessing)
- Cache spatial features if possible
- Consider smaller spatial U-Net for video

### Bottleneck 2: Temporal Module
**Issue**: Linear scaling with frame count
**Solution**:
- Use sliding windows (constant complexity)
- Reduce temporal layers
- Consider recurrent instead of attention

### Bottleneck 3: Memory
**Issue**: Storing all frames in memory
**Solution**:
- Process and save frames incrementally
- Use lower precision (float16)
- Stream to disk during generation

### Bottleneck 4: DDIM Steps
**Issue**: Still sequential across steps
**Solution**:
- Use fewer steps (20-25 for video)
- Consider parallel DDIM variants
- Accept that this is fundamental to diffusion

## Next Steps

1. **Complete Stage 2 optimization**
   - Implement DDIM as default
   - Add thread configuration
   - Final benchmark

2. **Stage 3 prototype**
   - Implement simple 1D temporal module
   - Test with 8-frame videos
   - Benchmark performance

3. **Iterate based on results**
   - If too slow: reduce frame count or steps
   - If poor quality: increase temporal window
   - If memory high: process in batches

4. **Finalize architecture**
   - Document final configuration
   - Create CPU-optimized inference script
   - Provide user configuration options

## Conclusion

The key to CPU-first video generation is **architectural simplicity**:
- Reuse Stage 2 spatial U-Net (already optimized)
- Add minimal temporal processing (1D conv or attention)
- Use DDIM sampling (mandatory)
- Process in small batches (memory management)

With this approach, we can achieve **reasonable video generation on consumer CPU hardware** without requiring GPU acceleration.
