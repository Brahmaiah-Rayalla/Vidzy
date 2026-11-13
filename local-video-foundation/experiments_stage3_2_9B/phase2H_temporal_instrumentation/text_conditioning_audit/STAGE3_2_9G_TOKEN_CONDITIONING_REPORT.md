# Stage 3.2.9G - Text Conditioning Audit Report
**Date:** 2026-11-03
**Status:** Complete

## Critical Finding

### CURRENT IMPLEMENTATION USES RANDOM TEXT EMBEDDINGS

The current `comparison_generator.py` implementation uses:
```python
text_emb = torch.randn(batch_size, seq_len, text_dim)
```

**This means:**
- Text prompts have **ZERO effect** on video generation
- All motion prompts are effectively identical to the model
- Direction control failures are **expected**, not surprising
- The observed 32% displacement increase in Experiment C is **unconditional** temporal activity

## Tokenization Analysis

### A red object moving left
- **Tokens:** ['a', 'red', 'object', 'moving', 'left', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>']
- **Unknown Tokens:** 0
- **Motion Tokens Present:** ['left', 'moving']
- **Effective Length:** 6

### A red object moving right
- **Tokens:** ['a', 'red', 'object', 'moving', 'right', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>']
- **Unknown Tokens:** 0
- **Motion Tokens Present:** ['right', 'moving']
- **Effective Length:** 6

### A blue object moving upward
- **Tokens:** ['a', 'blue', 'object', 'moving', 'upward', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>']
- **Unknown Tokens:** 0
- **Motion Tokens Present:** ['upward', 'moving']
- **Effective Length:** 6

### A blue object moving downward
- **Tokens:** ['a', 'blue', 'object', 'moving', 'downward', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>']
- **Unknown Tokens:** 0
- **Motion Tokens Present:** ['downward', 'moving']
- **Effective Length:** 6

### An object moving slowly
- **Tokens:** ['an', 'object', 'moving', 'slowly', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>']
- **Unknown Tokens:** 0
- **Motion Tokens Present:** ['moving', 'slowly']
- **Effective Length:** 5

### An object moving quickly
- **Tokens:** ['an', 'object', 'moving', 'quickly', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>']
- **Unknown Tokens:** 0
- **Motion Tokens Present:** ['moving', 'quickly']
- **Effective Length:** 5

### A stationary object
- **Tokens:** ['a', 'stationary', 'object', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>']
- **Unknown Tokens:** 0
- **Motion Tokens Present:** ['stationary']
- **Effective Length:** 4

## Embedding Analysis

### A red object moving left
- **Embedding Shape:** [1, 20, 256]
- **Mean:** 0.0000
- **Std:** 1.0000

### A red object moving right
- **Embedding Shape:** [1, 20, 256]
- **Mean:** -0.0000
- **Std:** 1.0000

### A blue object moving upward
- **Embedding Shape:** [1, 20, 256]
- **Mean:** -0.0000
- **Std:** 1.0000

### A blue object moving downward
- **Embedding Shape:** [1, 20, 256]
- **Mean:** -0.0000
- **Std:** 1.0000

### An object moving slowly
- **Embedding Shape:** [1, 20, 256]
- **Mean:** 0.0000
- **Std:** 1.0000

### An object moving quickly
- **Embedding Shape:** [1, 20, 256]
- **Mean:** 0.0000
- **Std:** 1.0000

### A stationary object
- **Embedding Shape:** [1, 20, 256]
- **Mean:** -0.0000
- **Std:** 1.0000

## Embedding Comparisons

### A red object moving left vs A red object moving right
- **Cosine Similarity:** 0.9545
- **Difference Norm:** 21.5898

### A blue object moving upward vs A blue object moving downward
- **Cosine Similarity:** 0.9540
- **Difference Norm:** 21.6982

### An object moving slowly vs An object moving quickly
- **Cosine Similarity:** 0.9460
- **Difference Norm:** 23.5119

### A red object moving left vs A stationary object
- **Cosine Similarity:** 0.8416
- **Difference Norm:** 40.2798

## Root Cause Analysis

### Primary Issue: Missing Text Conditioning

The text conditioning pathway is **completely bypassed** in the current implementation:

1. **Prompt to Tokenization:** NOT IMPLEMENTED (random embeddings used)
2. **Tokenization to Embeddings:** NOT IMPLEMENTED (random embeddings used)
3. **Embeddings to Encoder:** NOT IMPLEMENTED (random embeddings used)
4. **Encoder to Temporal Modules:** NOT IMPLEMENTED (random embeddings used)

### Why Direction Control Fails

Given that all prompts receive identical random embeddings:

- **Horizontal motion (50% accuracy):** Random behavior
- **Downward motion (0% accuracy):** Random failure
- **Upward motion (100% accuracy):** Random success
- **Stationary (38% accuracy for C):** Random activity

### Why Experiment C Has More Motion

The 32% displacement increase in Experiment C is **unconditional temporal activity**:

- Temporal modules are active (not receiving conditioning)
- They introduce motion regardless of prompt
- This is **not** prompt-controlled motion
- This explains why directional accuracy is LOWER for C (more chaotic motion)

## Recommendations

### Immediate Fix (Required)

1. **Implement real text encoding** in comparison_generator.py:
   ```python
   # Load text encoder and tokenizer
   text_encoder, tokenizer = TextEncoder.load(
       text_encoder_path,
       tokenizer_path
   )

   # Encode prompt
   indices, mask = tokenizer.encode(prompt, max_length=10)
   token_ids = torch.tensor([indices])
   attention_mask = torch.tensor([mask], dtype=torch.float)
   text_emb, _ = text_encoder(token_ids, attention_mask)
   ```

2. **Verify text encoder checkpoint exists** and is properly trained
3. **Re-run Stage 3.2.9F** with real text conditioning
4. **Re-evaluate** motion metrics with proper conditioning

### Secondary Investigation

After implementing real text conditioning:

1. Verify motion tokens are in vocabulary
2. Verify embeddings differ between direction prompts
3. Verify temporal modules receive text-conditioned features
4. Re-assess whether temporal modules can learn prompt-controlled motion

## Classification

**Primary Issue:** CONDITIONING PATH ISSUE

Motion instructions are not being transmitted correctly to the temporal modules
because the text conditioning pathway is completely bypassed (random embeddings).

## Next Stage Recommendation

**Recommendation:** Fix text conditioning implementation before proceeding

1. Implement real text encoding in inference pipeline
2. Verify text encoder checkpoint is available and trained
3. Re-run motion evaluation with proper conditioning
4. If still failing, investigate temporal module design

