# Stage 3.2.9L.8 - Experiment Design: Extended 250-Step Temporal Learning

**Date:** 2026-11-06
**Status:** Design Document

## Objective

Extend Stage 3.2.9L.7 training from 100 to 250 steps to determine whether:
1. 25% gate initialization advantage persists over longer training
2. Fusion gates show meaningful convergence
3. Denoising improvement translates to better prompt-controlled motion

## Experimental Design

### Starting Points

**Experiment C1 (1% init):**
- Resume from Stage 3.2.9L.7 C1 step-100 checkpoint
- Initial gates: ~1% (bottleneck: 0.010083, low-res: 0.010266)
- Extend to 250 total steps (150 additional steps)

**Experiment C25 (25% init):**
- Resume from Stage 3.2.9L.7 C25 step-100 checkpoint
- Initial gates: ~25% (bottleneck: 0.254079, low-res: 0.254478)
- Extend to 250 total steps (150 additional steps)

### Training Configuration

Both experiments share identical:
- Resolution: 64×64, 16 frames RGB
- Batch size: 1
- CPU-only, 8 threads
- AdamW: lr=0.0002, wd=0.01
- Same dataset and random seeds
- Restored optimizer states

### Milestones

- Step 100: Baseline (existing)
- Step 150: New checkpoint
- Step 200: New checkpoint
- Step 250: New checkpoint

### Monitored Metrics

**Fusion Gates:**
- Raw logits and activated values
- Gate gradients
- Absolute and relative changes

**Temporal Features:**
- Temporal feature RMS
- Spatial feature RMS
- Gated residual RMS
- Residual-to-spatial ratio

**Integrity:**
- Frozen parameter checks
- Loss and gradient finiteness

## Success Criteria

### Denoising Success
- C25 maintains or improves loss advantage over C1
- No loss degradation in either configuration

### Gate Learning Success
- Gate movement > 5% relative change from step 100 to 250
- Gates show consistent movement trend

### Motion Success
- C25 shows better direction accuracy than C1
- C25 shows better motion consistency
- No unacceptable stationary drift

### Stability Success
- No spatial parameter corruption
- No appearance degradation
- No training instability

## Risk Mitigation

### If gate learning remains slow (< 5% change)
- Consider separate gate learning rates in next stage
- Investigate architectural factors

### If C25 advantage disappears
- Test higher initial activation (50%)
- Investigate gate-specific tuning

### If motion evaluation shows no improvement
- Investigate motion supervision
- Consider stronger temporal conditioning

## Deliverables

1. STAGE3_2_9L8_EXPERIMENT_DESIGN.md (this document)
2. STAGE3_2_9L8_TRAINING_RESULTS.json
3. STAGE3_2_9L8_GATE_PROGRESSION.json
4. STAGE3_2_9L8_TEMPORAL_ACTIVATIONS.json
5. STAGE3_2_9L8_LOSS_COMPARISON.json
6. STAGE3_2_9L8_MOTION_EVALUATION.json (deferred - requires DDIM)
7. STAGE3_2_9L8_CHECKPOINT_INTEGRITY.json
8. STAGE3_2_9L8_CPU_PERFORMANCE.json
9. STAGE3_2_9L8_FINAL_REPORT.md

## Next Steps

1. Implement training continuation script
2. Run 150 additional steps for both C1 and C25
3. Analyze gate progression and loss comparison
4. Generate final classification
