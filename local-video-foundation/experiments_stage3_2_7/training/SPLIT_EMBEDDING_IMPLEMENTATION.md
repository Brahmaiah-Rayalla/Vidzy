# Split Embedding Implementation for Selective Motion Embedding Protection

## Overview

This implementation provides a split embedding architecture that separates the text encoder vocabulary into:
- **Appearance embeddings** (tokens 0-41): Frozen during training to protect pre-trained knowledge
- **Motion embeddings** (tokens 42-81): Trainable during motion-focused experiments

This enables selective fine-tuning of motion vocabulary while preventing catastrophic forgetting of appearance knowledge.

## Files Created/Modified

### 1. `text_encoder_split.py` (NEW)
Implements the `SplitTextEmbedding` class with the following features:

**Key Components:**
- `SplitTextEmbedding`: Main module that separates appearance and motion embeddings
- `convert_text_encoder_to_split()`: Helper function to convert existing text encoders
- `freeze_appearance()` / `freeze_motion()`: Methods to control parameter freezing
- `load_from_unified_embedding()`: Initialize split embeddings from a unified checkpoint
- `merge_to_unified()`: Convert back to unified embedding for deployment

**Architecture:**
```python
class SplitTextEmbedding(nn.Module):
    - appearance_emb: nn.Embedding(42, d_model)  # Frozen by default
    - motion_emb: nn.Embedding(40, d_model)     # Trainable by default
```

**Forward Pass:**
- Input: token_ids [batch, seq_len]
- Output: embeddings [batch, seq_len, d_model]
- Automatically routes tokens 0-41 to appearance_emb
- Automatically routes tokens 42-81 to motion_emb (with index adjustment)

### 2. `parameter_groups.py` (MODIFIED)
Updated to integrate split embedding support:

**New Method:**
- `setup_split_embeddings()`: Automatically converts text encoder to use split embeddings when needed

**Updated Methods:**
- `get_experiment_a_groups()`: Now calls `setup_split_embeddings()` before parameter selection
- `get_experiment_c_groups()`: Now calls `setup_split_embeddings()` before parameter selection

**Parameter Selection Logic:**
```python
# Experiments A and C:
if split_embedding_available:
    groups['motion_embeddings'] = model.text_encoder.token_emb.motion_emb.weight
else:
    # Fallback to mask-based selection
```

## How It Works

### Training Flow for Experiment A (Motion Vocabulary Only)

1. **Initialization:**
   ```python
   trainer.setup_parameters('A')
   ```

2. **Automatic Conversion:**
   - `setup_split_embeddings()` checks if text encoder uses split embeddings
   - If not, converts unified embedding to split architecture
   - Loads weights from checkpoint, freezing appearance, training motion

3. **Parameter Selection:**
   - Only `motion_emb.weight` is marked as trainable
   - All other parameters are frozen via `freeze_all_except()`

4. **Training:**
   - Gradients only flow through motion embeddings
   - Appearance embeddings remain frozen, preventing degradation

### Training Flow for Experiment C (Combined)

Same as Experiment A, but also includes:
- Temporal module parameters
- Fusion gate parameters

## Benefits

1. **Catastrophic Forgetting Prevention:**
   - Appearance embeddings are explicitly frozen
   - No risk of degrading pre-trained appearance knowledge

2. **Memory Efficiency:**
   - Only motion embeddings require gradient computation
   - Reduces memory usage during training

3. **Clean Architecture:**
   - Explicit separation of concerns
   - No need for complex masking logic
   - Easy to verify which parameters are trainable

4. **Compatibility:**
   - Can load from unified checkpoints
   - Can merge back to unified for deployment
   - Fallback logic for non-split models

## Testing

The implementation includes a self-test in `text_encoder_split.py`:

```bash
python text_encoder_split.py
```

Tests:
- Forward pass with mixed appearance/motion tokens
- Freezing appearance embeddings
- Merging back to unified embedding

## Usage Example

```python
from experiments_stage3_2_7.training.parameter_groups import ParameterGroupManager

# Initialize parameter manager
param_manager = ParameterGroupManager(model)

# Setup for Experiment A (automatically converts to split embeddings)
groups = param_manager.get_experiment_a_groups()

# Freeze all except selected parameters
param_manager.freeze_all_except(groups)

# Train only motion embeddings
```

## Notes

- The split embedding conversion is idempotent - calling it multiple times is safe
- Appearance embeddings are frozen by default in `SplitTextEmbedding.__init__()`
- Motion embeddings are trainable by default
- The architecture assumes 82 total tokens (42 appearance + 40 motion)
- Token indices can be customized via `appearance_vocab_size` and `motion_vocab_size`
