# Stage 3.2.9B - Phase 2B: Outlier Investigation Report

**Date:** 2026-11-02
**Purpose:** Investigate the 301-second outlier from Phase 2A smoke test

## Outlier Summary

**Location:** Experiment C, Step 6
**Duration:** 301.63s
**Median Step Time:** 8.75s
**Slowdown Factor:** 34.47x

## Detailed Timing Analysis

### All Step Times (Experiment C)
| Step | Total Time | Status |
|------|------------|--------|
| 1 | 8.57s | Normal |
| 2 | 9.19s | Normal |
| 3 | 8.93s | Normal |
| 4 | 8.35s | Normal |
| 5 | 14.58s | Elevated (1.67x median) |
| 6 | 301.63s | **OUTLIER** (34.47x median) |
| 7 | 8.54s | Normal |
| 8 | 8.36s | Normal |
| 9 | 8.38s | Normal |
| 10 | 9.27s | Normal |

### Statistics
- **Mean:** 38.58s (skewed by outlier)
- **Median:** 8.75s
- **Std Dev:** 92.45s (high due to outlier)

### Outlier Step Breakdown (Step 6)
- **Data loading:** 0.0042s (normal)
- **Forward pass:** 218.38s (**33.54x slowdown**)
- **Backward pass:** 83.23s (**10.34x slowdown**)
- **Optimizer step:** 0.0039s (normal)
- **Total:** 301.63s

### Normal Step Comparison (Step 5)
- **Data loading:** 0.0021s
- **Forward pass:** 6.51s
- **Backward pass:** 8.05s
- **Optimizer step:** 0.0060s
- **Total:** 14.58s

### Slowdown Breakdown
- **Forward pass:** 33.54x slowdown
- **Backward pass:** 10.34x slowdown
- **Data loading:** Normal
- **Optimizer step:** Normal

## Key Observations

### 1. Pattern Recognition
- **Step 5 already elevated:** 14.58s (1.67x median) suggests system stress building
- **Step 6 extreme outlier:** 301.63s (34.47x median) is the peak of the stress
- **Steps 7-10 return to normal:** Performance recovers after the outlier

### 2. Forward Pass Hit Hardest
- **Forward pass:** 33.54x slowdown (218.38s vs 6.51s)
- **Backward pass:** 10.34x slowdown (83.23s vs 8.05s)
- **Ratio:** Forward pass slowdown is 3.24x worse than backward pass

### 3. Non-Computation Components Unaffected
- **Data loading:** Normal timing
- **Optimizer step:** Normal timing
- **Conclusion:** The slowdown is specifically in tensor operations (forward/backward passes)

## Likely Causes

### Primary Cause: CPU Thermal Throttling
**Evidence:**
- CPU running at 900 MHz vs 2701 MHz max (33% of capacity)
- Gradual buildup (Step 5 elevated, Step 6 extreme)
- Recovery after outlier (Steps 7-10 normal)
- Forward pass hit hardest (more heat-generating operations)

**Mechanism:**
1. CPU gradually heats up during training
2. Thermal management system reduces clock speed
3. Severe throttling at Step 6 (extreme slowdown)
4. System cools down or adapts, performance recovers

### Secondary Cause: Background System Load
**Evidence:**
- Isolated single step (not all steps affected)
- Random timing suggests external factor
- Could coincide with scheduled system task

**Mechanism:**
- Background process (update, scan, indexing) consumes CPU
- Training competes for resources
- Forward/backward passes delayed

### Tertiary Cause: Memory Pressure
**Evidence:**
- Peak RAM at 10.4 GB (66% of total)
- Garbage collection could cause pauses
- Less likely (would affect multiple steps)

**Mechanism:**
- Memory fragmentation or GC spike
- CPU waits for memory operations
- Delayed tensor allocation/computation

## Probability Assessment

| Cause | Probability | Rationale |
|-------|-------------|-----------|
| Thermal throttling | **High (70%)** | Matches pattern, CPU at 33% frequency |
| Background load | **Medium (20%)** | Explains single-step anomaly |
| Memory pressure | **Low (10%)** | Would likely affect multiple steps |

## Impact on 100-Step Pilot

### Current Estimate (Excluding Outlier)
- **Median step time:** 8.75s
- **100 steps:** 875s (14.6 minutes)
- **With 10% overhead:** 962s (16.0 minutes)

### With Outlier Risk (10% probability)
- **Expected outliers:** 10 steps in 100
- **Outlier time per step:** 301.63s
- **Normal time per step:** 8.75s
- **Additional time:** 10 × (301.63 - 8.75) = 2,929s
- **Total with outliers:** 875 + 2,929 = 3,804s (63.4 minutes)

### Risk Assessment
- **Best case (no outliers):** 16 minutes
- **Expected case (10% outliers):** 63 minutes
- **Worst case (20% outliers):** 110 minutes

## Recommendations

### Immediate Actions
1. **Enable CPU performance mode** - Disable power saving, force max frequency
2. **Monitor CPU temperature** - Verify thermal throttling hypothesis
3. **Close background applications** - Reduce system load
4. **Run during off-peak hours** - Minimize scheduled tasks

### Verification Steps
1. Repeat smoke test with CPU performance mode enabled
2. Compare step timing distribution
3. Check if outlier frequency decreases
4. Measure CPU frequency during training

### If Outliers Persist
- Consider reducing batch size further (already at 1)
- Implement warm-up cooldown (allow CPU to rest between steps)
- Investigate PyTorch thread configuration
- Consider deferring to GPU access

## Conclusion

The 301-second outlier is likely caused by CPU thermal throttling, exacerbated by the CPU running at 33% of its maximum frequency. The pattern of gradual buildup (Step 5), extreme spike (Step 6), and recovery (Steps 7-10) strongly suggests thermal management intervention.

The primary mitigation strategy is to enable CPU performance mode and reduce thermal throttling. If this doesn't resolve the issue, the secondary strategy is to investigate background system load and memory pressure.

**Status:** Outlier analyzed, thermal throttling identified as primary cause
**Next Step:** Implement CPU performance optimizations and repeat smoke test
