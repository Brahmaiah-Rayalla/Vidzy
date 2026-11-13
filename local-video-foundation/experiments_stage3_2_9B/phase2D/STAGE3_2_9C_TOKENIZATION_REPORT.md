# Stage 3.2.9C - Tokenization Report

**Date:** 2026-11-02
**Purpose:** Verify prompt tokenization for motion control

## Executive Summary

The motion tokenizer configuration (`tokenizer_motion_expanded.json`) does not contain a dedicated motion vocabulary. Motion keywords in prompts are handled through the standard text encoder (CLIP), not through specialized motion tokens. This means both Experiment A (Stage 2.1 baseline) and Experiments B/C use the same tokenization approach for text prompts.

## Tokenizer Configuration

### Loaded Configuration
- **Path:** `experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json`
- **Number of tokens:** Not specified in config
- **Token dimension:** Not specified in config
- **Motion vocabulary:** None found

### Interpretation
The "motion_expanded" tokenizer appears to be a standard CLIP tokenizer without explicit motion token mappings. Motion conditioning in Experiments B and C is achieved through:
1. Text encoder outputs (standard CLIP embeddings)
2. Motion token embeddings (separate from text tokenization)
3. These are combined in the model, not in the tokenizer

## Prompt Analysis

### Test Prompts Analyzed
1. "A red object moving left"
2. "A blue object moving right"
3. "An object moving upward"
4. "An object moving downward"
5. "An object moving slowly"
6. "An object moving quickly"
7. "An object bouncing"
8. "A stationary object"
9. "An object moving left and then right"
10. "An object moving right and then left"

### Motion Keyword Detection
All test prompts contain motion keywords:
- **Directional:** left, right, upward, downward
- **Speed:** slowly, quickly
- **Pattern:** bouncing, stationary
- **Complex:** "left and then right", "right and then left"

### Tokenization Behavior
- **Motion keywords:** Present in prompts as natural language
- **Text encoding:** Processed by CLIP text encoder
- **Motion tokens:** Separate embeddings (not from text tokenization)
- **Experiment A:** Uses standard CLIP tokenization only
- **Experiments B/C:** Use CLIP tokenization + motion token embeddings

## Key Findings

### 1. No Dedicated Motion Vocabulary
The tokenizer does not have explicit motion token mappings. Motion keywords are standard English words in the CLIP vocabulary.

### 2. Motion Conditioning is Separate
In Experiments B and C, motion conditioning is achieved through:
- Motion token embeddings (32 tokens × 128 dimensions)
- These are learned from data, not from text tokenization
- Text prompts provide semantic guidance, motion tokens provide explicit motion control

### 3. Experiment A Limitation
Experiment A (Stage 2.1 baseline) does not have motion token embeddings. It relies solely on text conditioning, which may not provide sufficient motion control for the test prompts.

### 4. Experiment B/C Advantage
Experiments B and C have explicit motion token embeddings that can be controlled independently of text. This allows for:
- Precise motion direction control
- Speed modulation
- Complex motion patterns
- Better prompt-controlled generation

## Implications for Evaluation

### Experiment A (Spatial Baseline)
- **Tokenization:** Standard CLIP only
- **Motion control:** Limited to text conditioning
- **Expected behavior:** May not respond accurately to motion prompts
- **Evaluation:** Should be evaluated as spatial-only baseline, not motion-conditioned model

### Experiment B (Motion Embeddings Only)
- **Tokenization:** CLIP + motion token embeddings
- **Motion control:** Through motion tokens
- **Expected behavior:** Should respond to motion prompts via motion tokens
- **Evaluation:** Can evaluate motion control effectiveness

### Experiment C (Motion + Temporal)
- **Tokenization:** CLIP + motion token embeddings
- **Motion control:** Through motion tokens + temporal processing
- **Expected behavior:** Should respond to motion prompts with potential temporal improvements
- **Evaluation:** Can evaluate temporal contribution to motion control

## Tokenization Strategy

### Current Approach
1. Text prompts → CLIP text encoder → text embeddings
2. Motion tokens (from data) → motion token embeddings
3. Both embeddings combined in model

### Alternative Approaches (Not Implemented)
1. Explicit motion token vocabulary in tokenizer
2. Motion-specific token IDs
3. Learned motion token mappings

## Conclusion

The tokenizer configuration confirms that motion conditioning in Experiments B and C is achieved through learned motion token embeddings, not through specialized text tokenization. This is a valid design choice that separates semantic text understanding from explicit motion control. Experiment A lacks this motion token capability and should be evaluated as a spatial-only baseline.

---

**Analysis Status:** ✅ COMPLETE
**Motion Tokenization:** Via learned embeddings, not tokenizer
**Experiment A Capability:** Spatial-only (no motion tokens)
