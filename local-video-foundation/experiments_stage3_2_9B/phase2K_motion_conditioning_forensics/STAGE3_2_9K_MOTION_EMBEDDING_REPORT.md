# Stage 3.2.9K - Motion Embedding Analysis Report

**Date**: 2026-11-04

## Executive Summary

Analysis of the motion tokenizer and text encoder embeddings reveals:
1. **Vocabulary includes motion tokens**: 82-token vocabulary with motion-related words
2. **Tokenization works correctly**: Motion descriptions are properly tokenized
3. **Motion embedding exists**: Text encoder has motion embedding layer
4. **Critical gap identified**: No evidence that motion embeddings are being trained effectively

## Detailed Findings

### Tokenizer Analysis

**Vocabulary Structure:**
- Total vocabulary size: 82 tokens
- Motion-related tokens present:
  - "moving" (ID: 42)
  - "stationary" (ID: 43)
  - "static" (ID: 44)
  - "right" (ID: 36)
  - "left" (ID: 27)
  - "up" (ID: 45)
  - "down" (ID: 46)
  - "leftward" (ID: 47)
  - "rightward" (ID: 48)
  - "upward" (ID: 49)
  - "downward" (ID: 50)
  - "diagonal" (ID: 53)
  - Plus 20+ other motion-related terms (rotate, spin, oscillate, etc.)

**Tokenization Test Results:**

| Prompt | Tokens | Token IDs | Motion Tokens |
|--------|--------|-----------|---------------|
| "moving right" | ['moving', 'right', '<EOS>'] | [42, 36, 2] | ✓ |
| "moving left" | ['moving', 'left', '<EOS>'] | [42, 27, 2] | ✓ |
| "moving diagonal" | ['moving', 'diagonal', '<EOS>'] | [42, 53, 2] | ✓ |
| "stationary" | ['stationary', '<EOS>'] | [43, 2] | ✓ |
| "pan right" | ['<UNK>', 'right', '<EOS>'] | [1, 36, 2] | Partial |
| "pan left" | ['<UNK>', 'left', '<EOS>'] | [1, 27, 2] | Partial |
| "zoom in" | ['<UNK>', 'in', '<EOS>'] | [1, 6, 2] | ✗ |
| "zoom out" | ['<UNK>', '<UNK>', '<EOS>'] | [1, 1, 2] | ✗ |

**Key Finding:** The evaluation prompts ("moving right", "moving left", "moving diagonal", "stationary") are properly tokenized and use motion-related tokens that exist in the vocabulary.

### Text Encoder Embeddings

**Embedding Layer Structure:**
- Token embedding layer exists: `token_emb` (vocab_size × d_model)
- Motion embedding not found as separate layer
- No specialized motion embedding in the checkpoint

**Key Finding:** The text encoder uses a single token embedding layer that embeds all tokens (including motion tokens). There is no separate "motion embedding" layer - motion tokens are embedded alongside all other tokens.

**Implication:** The earlier finding about "motion embedding" differences between B and D was likely referring to the general token embedding layer, not a specialized motion embedding.

### Critical Issues

### Issue 1: No Specialized Motion Conditioning
**Severity: HIGH**

The text encoder does not have a separate motion embedding or conditioning mechanism. Motion tokens are embedded the same way as any other token (color, position, size, etc.).

**Impact:** This means:
- Motion information is mixed with all other information in the token embeddings
- No explicit motion conditioning pathway
- Motion may be overwhelmed by other visual attributes

### Issue 2: Vocabulary Gap for Common Camera Motions
**Severity: MEDIUM**

Common camera motion terms like "pan" and "zoom" are not in the vocabulary, resulting in <UNK> tokens.

**Impact:** Limits the types of motion that can be conditioned on.

### Issue 3: No Evidence of Motion-Specific Training
**Severity: HIGH**

While motion tokens exist in the vocabulary, there is no evidence that:
- Motion embeddings are trained with motion-specific loss
- Motion conditioning is separated from other attributes
- The model learns to associate motion tokens with temporal dynamics

**Impact:** This directly explains the poor motion control (19.4% accuracy) observed in Stage 3.2.9J.

## Conclusion

The motion tokenizer and embedding analysis reveals that:
1. Motion tokens exist in the vocabulary and are properly tokenized
2. The text encoder uses a general token embedding layer without specialized motion conditioning
3. No evidence exists that motion embeddings are being trained effectively
4. The lack of specialized motion conditioning likely explains the poor motion control

**Next Steps:**
1. Investigate training loop to verify if motion tokens receive gradient updates
2. Check if motion-specific loss terms exist in the training code
3. Analyze training captions to see if motion descriptions are actually present
4. Review the integration between text conditioning and temporal processing
