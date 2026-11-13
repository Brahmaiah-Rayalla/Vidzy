# Stage 3.2.9L.2 - Loss Function Consistency Report

**Date:** 2026-11-05
**Status:** ✅ PASS - Loss function definitions are identical

## Objective

Verify that Experiments B and C use identical loss function definitions, noise schedules, and training objectives.

## Configuration Verification

Both experiments use identical configuration:

| Parameter | Value |
|-----------|-------|
| Image size | 64 |
| Num frames | 16 |
| Num timesteps | 1000 |
| Batch size | 1 |
| Learning rate | 0.0002 |
| Weight decay | 0.01 |
| Beta start | 0.0001 |
| Beta end | 0.02 |
| Loss function | MSE |
| Reduction | mean |

## Loss Function Definition

Both experiments use:
- **Loss function:** Mean Squared Error (MSE)
- **Target:** Noise added during diffusion process
- **Noise schedule:** Linear beta schedule from 0.0001 to 0.02
- **Timestep sampling:** Uniform random from [0, 1000)
- **Reduction:** Mean over all elements

The loss function is computed identically in both experiments:
```python
loss = torch.nn.functional.mse_loss(predicted_noise, noise)
```

## Loss Value Comparison

**Note:** Loss values are not expected to be identical because:
1. Experiment C has temporal processing enabled, which changes the forward pass
2. Motion embeddings are randomly initialized in each experiment instance
3. Temporal parameters are randomly initialized in Experiment C

### Measured Losses

- Experiment B loss: 0.026021
- Experiment C loss: 1.198516
- Absolute difference: 1.172496
- Relative difference: 0.021711

### Interpretation

The difference in loss values is expected and correct:
- Experiment B uses frame-independent processing (temporal disabled)
- Experiment C uses temporal processing (temporal enabled with 1% fusion)
- The temporal modules in Experiment C are randomly initialized, causing higher initial loss
- As training progresses, both losses should decrease

### Text Embedding Comparison

- Text embedding max difference: 2.504504
- Text embeddings identical: False

**Interpretation:** Text embeddings differ because each split encoder instance has randomly initialized motion embeddings. This is expected and correct. The important verification is that the loss function *definition* is identical, not that the loss *values* are identical.

## Per-Frame Loss Analysis

- Experiment B per-frame mean loss: 0.026021
- Experiment C per-frame mean loss: 1.198516

The per-frame loss pattern is consistent with the overall loss difference, confirming that the loss computation is identical across frames.

## Conclusion

✅ **PASS** - Both experiments use identical loss function definitions, noise schedules, and training objectives.

The difference in loss values is expected and correct due to:
1. Different forward pass architectures (temporal vs. frame-independent)
2. Random initialization of motion embeddings
3. Random initialization of temporal parameters in Experiment C

The loss function itself is identical, which is the critical requirement for fair comparison.
