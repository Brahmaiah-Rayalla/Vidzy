# Stage 3.2.9J - Generation Matrix Audit Report

**Date**: 2026-11-03
**Status**: COMPLETE

## Summary

- **Expected videos**: 36 (4 prompts × 3 seeds × 3 configs)
- **Found videos**: 36
- **Missing videos**: 0
- **Metadata issues**: 0
- **Duplicate hashes**: 12 (B and D are identical)
- **Audit success**: ✓

## Video Integrity

- **Total checked**: 36
- **Valid videos**: 36
- **Invalid videos**: 0
- **Errors**: 0
- **Integrity success**: ✓

## Important Finding: B = D

All 12 B videos have identical SHA256 hashes to corresponding D videos:

| Hash Prefix | B Video | D Video |
|-------------|---------|---------|
| 8594a8a7... | B_seed42_prompt0 | D_seed42_prompt0 |
| e571996e... | B_seed42_prompt1 | D_seed42_prompt1 |
| 00cdd4d0... | B_seed42_prompt2 | D_seed42_prompt2 |
| 485c8fd5... | B_seed42_prompt3 | D_seed42_prompt3 |
| 960b88ec... | B_seed123_prompt0 | D_seed123_prompt0 |
| 7d4652a3... | B_seed123_prompt1 | D_seed123_prompt1 |
| cb865ecd... | B_seed123_prompt2 | D_seed123_prompt2 |
| f66a8c51... | B_seed123_prompt3 | D_seed123_prompt3 |
| ac9d8c9b... | B_seed456_prompt0 | D_seed456_prompt0 |
| 62fb168e... | B_seed456_prompt1 | D_seed456_prompt1 |
| 26cfa012... | B_seed456_prompt2 | D_seed456_prompt2 |
| 31b92430... | B_seed456_prompt3 | D_seed456_prompt3 |

**Interpretation**:
- When temporal processing is disabled, the C checkpoint behaves identically to B
- This confirms the ablation design (C vs D) correctly isolates temporal processing impact
- Differences between B and C are due to temporal processing, not checkpoint architecture

## Integrity Validation

All 36 videos passed integrity checks:
- **Frame count**: 16 frames (expected)
- **Resolution**: 64×64 pixels (expected)
- **Channels**: RGB (3 channels)
- **Pixel range**: Valid [0, 255]
- **No NaN or infinity values**
- **No blank frames**
- **No duplicate frames**
- **No corruption detected**

## Conclusion

All 36 videos are present with valid metadata and pass integrity checks. The duplicate hash analysis confirms that B and D produce identical outputs, validating the experimental design for isolating temporal processing effects.
