# Stage 3.2.9J - Visual Assessment

**Date**: 2026-11-03

## Visual Assessment Table

| Seed | Prompt | B Behavior | B Adherence | B Quality | B Failures | C Behavior | C Adherence | C Quality | C Failures | D Behavior | D Adherence | D Quality | D Failures |
|------|--------|------------|-------------|------------|------------|------------|-------------|------------|------------|------------|-------------|------------|------------|
| 42 | A red object moving left | Minimal movement | Poor (minimal movement) | Good | Wrong direction | Minimal movement | Poor (minimal movement) | Good | Wrong direction | Minimal movement | Poor (minimal movement) | Good | Wrong direction |
| 42 | A red object moving right | Minimal movement | Poor (minimal movement) | Good | Wrong direction | Correct direction | Good | Good | None | Minimal movement | Poor (minimal movement) | Good | Wrong direction |
| 42 | A blue object moving diagonally | Minimal movement | Poor (minimal movement) | Good | Wrong direction | Minimal movement | Poor (minimal movement) | Good | Wrong direction | Minimal movement | Poor (minimal movement) | Good | Wrong direction |
| 42 | A yellow object remaining stationary | Random/wrong direction | Poor | Good | None | Random/wrong direction | Poor | Good | Excessive drift | Random/wrong direction | Poor | Good | None |
| 123 | A red object moving left | Minimal movement | Poor (minimal movement) | Good | Wrong direction | Minimal movement | Poor (minimal movement) | Good | Wrong direction | Minimal movement | Poor (minimal movement) | Good | Wrong direction |
| 123 | A red object moving right | Correct direction | Good | Good | None | Correct direction | Good | Good | None | Correct direction | Good | Good | None |
| 123 | A blue object moving diagonally | Minimal movement | Poor (minimal movement) | Variable | Wrong direction, Inconsistent motion | Minimal movement | Poor (minimal movement) | Variable | Wrong direction, Inconsistent motion | Minimal movement | Poor (minimal movement) | Variable | Wrong direction, Inconsistent motion |
| 123 | A yellow object remaining stationary | Random/wrong direction | Poor | Variable | Excessive drift, Inconsistent motion | Random/wrong direction | Poor | Variable | Excessive drift, Inconsistent motion | Random/wrong direction | Poor | Variable | Excessive drift, Inconsistent motion |
| 456 | A red object moving left | Minimal movement | Poor (minimal movement) | Good | Wrong direction | Minimal movement | Poor (minimal movement) | Good | Wrong direction | Minimal movement | Poor (minimal movement) | Good | Wrong direction |
| 456 | A red object moving right | Correct direction | Good | Good | None | Correct direction | Good | Good | None | Correct direction | Good | Good | None |
| 456 | A blue object moving diagonally | Minimal movement | Poor (minimal movement) | Good | Wrong direction | Minimal movement | Poor (minimal movement) | Good | Wrong direction | Minimal movement | Poor (minimal movement) | Good | Wrong direction |
| 456 | A yellow object remaining stationary | Random/wrong direction | Poor | Good | None | Random/wrong direction | Poor | Good | None | Random/wrong direction | Poor | Good | None |

## Summary

### Key Observations

- Direction accuracy is very low across all configurations
- Stationary control fails completely (all videos show significant drift)
- B and D show identical behavior (confirmed by duplicate hashes)
- C shows marginal improvement in direction accuracy but worse stationary control
- Visual quality is consistent across configurations

### Comparison Notes

**B vs C**: C has slightly better direction accuracy (25% vs 16.7%) but worse stationary control.

**C vs D**: C has marginal direction improvement but higher drift. Temporal processing does not provide meaningful benefits.

**B vs D**: Identical outputs, confirming temporal-disabled behavior is consistent.

### Limitations

- Visual assessment is based on quantitative metrics (tracking, direction accuracy)
- Contact sheets not generated due to library limitations
- Manual visual inspection of videos recommended for qualitative assessment

