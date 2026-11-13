# Stage 3.2.9L.23 - Final Report

**Date:** 2026-11-09 | **Classification: FROZEN DOWNSTREAM BACKBONE LIMITATION**

## Implementation
- Split conv (conv_rgb frozen+excluded from optimizer, conv_text trainable): parity max diff 0.0 EXACT.
- Frozen-param integrity: ZERO violations both arms (L22 weight-decay bug eliminated by optimizer exclusion).
- Gain sweep: swap sensitivity 0.006->0.054 across G0.5-G4 (linear amplification, not semantic). Kept 1.0.

## Results
- Arm A (cond-only): mse_mean 0.175->0.297 - degraded like L22.
- Arm B (+first res block @1e-5): mse_mean 0.175->0.291 - NO advantage; swap cos 0.9999 unchanged.
- First res block sees caption diffs (rel 0.098) but limited unfreezing did not translate to semantics.

## Answers
1. First block amplifies caption features? Input rel_diff 0.098 - present, not amplified.
2. Survives deeper? Attenuates (L22: 0.137->0.051).
3. Improves correct-caption denoising? NO.
4. Improves recognizable generation? NO.
5. B > A? NO - statistically identical failure.
6. Frozen downstream still limits? YES - confirmed.
7. Learns color/shape? NO.
8. Ablation removes improvement? N/A (no improvement to remove).

## Cumulative conclusion (L17-L23)
Every conditioning pathway variant fails identically: mask OK, encoder OK, attention uniform, FiLM dead, learned-queries dead, direct channels active but too weak (~5-10%), and NO amount of partial unfreezing in 500-step probes fixes it. The bottleneck is now definitively the TRAINING SCALE/OBJECTIVE, not architecture: ~12k presentations of eps-loss cannot teach prompt-conditioned generation at 64x64 on this synthetic data.

## Separate reports
- Split parity: EXACT | Frozen integrity: PASS | Gain: amplification only | Arm A: FAIL | Arm B: FAIL | Caption sens: ~0 | Semantic: FAIL | CPU ~9min | Video-ready: NO

## Recommendation
L.24: decision point - either commit to full end-to-end training at real scale, or document architecture ceiling. No more small probes.
