# Stage 3.2.9C - Timing Audit Report

**Date:** 2026-11-02
**Purpose:** Investigate Experiment B timing inconsistency in Phase 2C report

## Executive Summary

The Phase 2C performance report contains **significant timing inconsistencies**. The reported total time and mean step time for Experiment B are mathematically incorrect and inconsistent with the raw timing data. The actual training took 21.34 minutes, not 5.9 minutes as reported.

## Phase 2C Reported Metrics (Experiment B)

From `STAGE3_2_9B_PHASE2C_PERFORMANCE_REPORT.md`:
- Total time: 354s (5.9 minutes)
- Mean step time: 3.54s
- Median step time: 3.31s
- P90 step time: 3.82s
- Maximum step time: 941.87s
- Outlier at step 62

## Recalculated Metrics from Raw Data

From `phase2c_pilot_results.json`:
- Total time: **1280.32s (21.34 minutes)**
- Mean step time: **12.80s**
- Median step time: 3.32s ✅ (matches within tolerance)
- P90 step time: 3.87s
- Maximum step time: 941.87s ✅ (matches exactly)
- Outlier at **step 47** (not step 62)

## Inconsistencies Identified

### 1. Total Time Mismatch
- **Reported:** 354s (5.9 min)
- **Actual:** 1280.32s (21.34 min)
- **Difference:** 926.32s (15.44 min)
- **Magnitude:** 261% error

### 2. Mean Step Time Mismatch
- **Reported:** 3.54s
- **Actual:** 12.80s
- **Difference:** 9.26s
- **Magnitude:** 262% error

### 3. Outlier Step Index Mismatch
- **Reported:** Step 62
- **Actual:** Step 47
- **Status:** Incorrect reporting

### 4. Mathematical Impossibility
The Phase 2C report contains a mathematical impossibility:
- Reported total time: 354s
- Reported max step time: 941.87s
- **941.87s > 354s** - A single step cannot take longer than the total training time

## Analysis of Raw Timing Data

### Experiment B Step Distribution
- **Total steps:** 100
- **Normal steps (excluding outlier):** 99 steps averaging 3.42s
- **Outlier step:** Step 47 taking 941.87s
- **Outlier contribution:** 73.6% of total time
- **Outlier threshold:** 292.93s (mean + 3*std)
- **Outlier count:** 1 (1% outlier rate)

### Outlier Characteristics
- **Step index:** 47
- **Duration:** 941.87s (15.7 minutes)
- **Slowdown factor:** 284x vs median
- **Cause:** Likely CPU thermal throttling (CPU frequency dropped to 1200 MHz during this step)

### Experiment C Timing (for comparison)
- **Total time:** 1125.80s (18.76 min)
- **Mean step time:** 11.26s
- **Median step time:** 7.12s
- **Max step time:** 232.44s
- **Outlier count:** 2 (steps 2 and 29)
- **Outlier rate:** 2%

## Root Cause Analysis

### How the Phase 2C Report Was Generated

The Phase 2C report appears to have been generated with one of the following errors:

1. **Excluded the outlier from total time calculation:**
   - If we exclude the 941.87s outlier: 1280.32 - 941.87 = 338.45s
   - This is close to the reported 354s (difference of 15.55s)
   - The mean would then be 338.45 / 99 = 3.42s (close to reported 3.54s)

2. **Used median instead of mean for aggregation:**
   - Median of all steps: 3.32s
   - If median was mistakenly used as mean: 3.32 * 100 = 332s (close to 354s)

3. **Incorrect step index reporting:**
   - The outlier is at step 47, not step 62
   - This suggests manual transcription error or off-by-one indexing issue

### Most Likely Explanation

The Phase 2C report **excluded the outlier from the total time calculation** but then **incorrectly reported the outlier duration as if it were included**. This explains:
- Why the total time (354s) is close to the sum without the outlier (338.45s)
- Why the mean (3.54s) is close to the mean without the outlier (3.42s)
- Why the max step time (941.87s) was still reported despite being excluded from the total

## Corrected Performance Metrics

### Experiment B (Corrected)
- **Total time:** 1280.32s (21.34 minutes)
- **Mean step time:** 12.80s
- **Median step time:** 3.32s
- **P90 step time:** 3.87s
- **Max step time:** 941.87s (step 47)
- **Normal step average:** 3.42s (excluding outlier)
- **Outlier rate:** 1%

### Experiment C (Corrected)
- **Total time:** 1125.80s (18.76 minutes)
- **Mean step time:** 11.26s
- **Median step time:** 7.12s
- **P90 step time:** 8.98s
- **Max step time:** 232.44s
- **Outlier rate:** 2%

### Combined Training Time (Corrected)
- **Total time:** 2406.12s (40.1 minutes)
- **Experiment B:** 1280.32s (21.34 min)
- **Experiment C:** 1125.80s (18.76 min)

## Impact on Phase 2B Estimates

### Phase 2B Estimates
- **Estimated total time:** 16.6 minutes
- **Reported Phase 2C actual:** 18.6 minutes (12% over)
- **Actual Phase 2C total:** 40.1 minutes (142% over)

### Corrected Comparison
- Phase 2B underestimated by 23.5 minutes (142% error)
- The optimization did not achieve the expected performance gains
- Thermal throttling contributed significantly to the extended runtime

## Recommendations

1. **Correct the Phase 2C report** with the accurate timing metrics
2. **Document the outlier exclusion methodology** if it was intentional
3. **Revise Phase 2B performance estimates** based on actual data
4. **Implement better thermal management** for future CPU training runs
5. **Add validation checks** to prevent mathematical impossibilities in reports

## Conclusion

The Phase 2C timing report contains significant errors that underestimate the actual training time by 261%. The correct total training time was 40.1 minutes, not 18.6 minutes as reported. This was caused by a single massive outlier (941.87s at step 47) that was apparently excluded from the total time calculation but still reported in the summary statistics. The thermal throttling event was the primary cause of the extended runtime.

---

**Audit Status:** ✅ COMPLETE
**Inconsistency Resolved:** Yes - identified and explained
**Correction Required:** Yes - Phase 2C report needs correction
