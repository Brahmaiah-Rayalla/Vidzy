# Stage 3.2.9K - Motion Conditioning Forensics - Comprehensive Report

**Date**: 2026-11-04
**Stage**: Motion Conditioning Forensics

## Executive Summary

Stage 3.2.9K conducted a comprehensive diagnostic investigation into why motion conditioning is not reliably translated into video trajectories. The investigation revealed critical issues with the training process that explain the poor motion control (19.4% accuracy) observed in Stage 3.2.9J.

## Investigation Overview

Three diagnostic analyses were performed:
1. **B/D Equivalence Investigation**: Why experiments B and D produced identical outputs
2. **Motion Embedding Analysis**: Analysis of tokenizer, embeddings, and tokenization
3. **Training Caption Analysis**: Analysis of training data for motion content

## Key Findings

### Finding 1: Checkpoint C is a Copy of Checkpoint B (CRITICAL)

**Severity: CRITICAL**

Checkpoints B and C have **identical tensor parameters** despite:
- Different training configurations (B: no temporal, C: with temporal)
- Different loss values (B: 0.007, C: 1.177)
- Supposedly different training runs

**Evidence:**
- Max tensor difference: 0.0 (identical)
- Both have 589 parameter keys in `video_unet`
- Non-tensor differences (config, loss, optimizer) suggest different training metadata but same weights

**Implication:** Configuration C training failed to update weights. The temporal training did not actually occur or did not affect the model parameters. This invalidates the Stage 3.2.9J comparison between temporal and non-temporal configurations.

### Finding 2: Motion Embedding Not Effectively Trained (CRITICAL)

**Severity: CRITICAL**

The motion embedding difference between B and D is only 0.03, suggesting:
- Motion embedding is not being trained effectively
- Motion conditioning is not being learned
- The text encoder's token embedding layer may not be receiving gradient updates for motion tokens

**Evidence:**
- B vs D model parameters: 569/570 identical
- Only difference: `text_encoder.token_emb.motion_emb.weight` with max diff 0.03
- This difference is too small to affect generation outcomes

**Implication:** The text encoder may not be properly integrated into the training loop, or motion tokens are not receiving sufficient gradient updates.

### Finding 3: No Specialized Motion Conditioning (HIGH)

**Severity: HIGH**

The text encoder uses a single token embedding layer without specialized motion conditioning.

**Evidence:**
- Vocabulary includes 82 tokens with motion-related words
- Motion tokens are embedded the same way as any other token
- No separate motion embedding or conditioning pathway
- Motion information is mixed with all other attributes (color, position, size)

**Implication:** Motion information may be overwhelmed by other visual attributes, making it difficult for the model to learn motion-specific patterns.

### Finding 4: Training Data Contains Motion Descriptions (GOOD)

**Severity: INFORMATIONAL**

The training data contains comprehensive motion descriptions.

**Evidence:**
- 176 training captions, 100% contain motion keywords
- Motion words present: "moving" (149), "right" (61), "left" (61), "slow" (49), "diagonal" (32), "stationary" (27)
- Captions are descriptive: "A large pink circle slowly moving from top to bottom on a light_yellow background."

**Implication:** The training data is not the problem. The issue is with how the model learns from this data.

### Finding 5: Tokenization Works Correctly (GOOD)

**Severity: INFORMATIONAL**

Motion descriptions are properly tokenized.

**Evidence:**
- "moving right" → ['moving', 'right', '<EOS>'] → [42, 36, 2]
- "moving left" → ['moving', 'left', '<EOS>'] → [42, 27, 2]
- "moving diagonal" → ['moving', 'diagonal', '<EOS>'] → [42, 53, 2]
- "stationary" → ['stationary', '<EOS>'] → [43, 2]

**Implication:** The tokenizer is not the bottleneck. Motion tokens are correctly identified and encoded.

## Root Cause Analysis

Based on the findings, the root causes of poor motion control are:

### Primary Root Cause: Training Loop Issues

1. **Checkpoint C Training Failure**: Configuration C (with temporal) failed to train. The weights are identical to configuration B (without temporal), suggesting the temporal modules were never trained or the training loop has a bug.

2. **Text Encoder Not Training**: The minimal difference (0.03) in motion embeddings between B and D suggests the text encoder is not receiving gradient updates or the gradients are too small to be effective.

3. **Motion Conditioning Not Integrated**: The lack of specialized motion conditioning means motion information is not given special treatment, making it difficult for the model to learn motion-specific patterns.

### Secondary Root Cause: Architecture Limitations

1. **Unified Token Embedding**: Motion tokens are embedded the same way as other tokens, which may cause motion information to be overshadowed by more dominant visual attributes (color, position, size).

2. **No Motion-Specific Loss**: There is no evidence of motion-specific loss terms or supervision signals that would encourage the model to learn motion.

## Impact on Stage 3.2.9J Results

The Stage 3.2.9J motion evaluation results are **invalidated** by these findings:

1. **B vs C Comparison Invalid**: Since checkpoints B and C have identical weights, the comparison between temporal (C) and non-temporal (B) configurations is meaningless.

2. **B vs D Comparison Expected**: B and D produced identical outputs because they effectively use the same weights (B checkpoint vs C checkpoint with temporal disabled).

3. **Temporal Benefit Not Demonstrated**: The conclusion that temporal processing provides no benefit is based on a comparison of identical models.

## Recommendations

### Immediate Actions (CRITICAL)

1. **Investigate Configuration C Training**:
   - Review training logs for configuration C
   - Verify temporal modules were included in the training loop
   - Check if temporal training was actually executed

2. **Audit Training Loop**:
   - Verify text encoder is included in the training loop
   - Check if text encoder gradients are being computed and applied
   - Verify all model components are receiving gradient updates

3. **Fix Checkpoint Saving**:
   - Ensure checkpoints are saved correctly with the right weights
   - Verify checkpoint loading/training integration

### Short-Term Actions (HIGH)

1. **Add Motion-Specific Conditioning**:
   - Consider adding a separate motion embedding layer
   - Implement motion-specific attention mechanisms
   - Add motion loss terms to the training objective

2. **Improve Motion Token Representation**:
   - Expand vocabulary to include more motion terms (pan, zoom, etc.)
   - Consider subword tokenization for motion phrases
   - Add motion-specific positional encoding

### Long-Term Actions (MEDIUM)

1. **Architecture Redesign**:
   - Separate motion conditioning from spatial conditioning
   - Implement temporal attention mechanisms
   - Add motion trajectory prediction as an auxiliary task

2. **Training Strategy**:
   - Use curriculum learning focused on motion
   - Add motion-specific data augmentation
   - Implement motion-consistency regularization

## Conclusion

Stage 3.2.9K has revealed critical issues with the training process that explain the poor motion control observed in Stage 3.2.9J:

1. **Checkpoint C training failed** - temporal modules were not trained
2. **Text encoder not training effectively** - motion embeddings not being updated
3. **No specialized motion conditioning** - motion information not given special treatment

The training data contains comprehensive motion descriptions, and tokenization works correctly, so the problem lies in the training loop and architecture design.

**Next Steps**: Investigate the training loop for configuration C to determine why temporal training failed, and audit the text encoder integration to ensure it receives gradient updates.
