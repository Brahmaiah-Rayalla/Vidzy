# Stage 3.2.9B - Phase 2B: Baseline System Information

**Date:** 2026-11-02
**Purpose:** Record baseline system state before performance optimization

## Baseline Hardware Specifications

### CPU
- **Processor:** Intel64 Family 6 Model 140 Stepping 1, GenuineIntel
- **Physical cores:** 4
- **Logical cores:** 8
- **Max frequency:** 2701 MHz
- **Current frequency:** 900 MHz (33% of max - indicates power-saving mode or thermal throttling)

### Memory
- **Total RAM:** 15.65 GB
- **Available RAM:** 7.23 GB (53.8% used at time of measurement)
- **Peak during training:** 10.40 GB (66% of total)

### PyTorch Configuration
- **Version:** 2.7.1+cpu
- **CUDA:** Not available
- **Intel XPU:** Not available
- **Execution device:** CPU (mandatory)
- **Thread configuration:** 4 threads, 4 interop threads

## Baseline Performance Metrics (Phase 2A Smoke Test)

### Experiment B (Motion Embeddings Only)
- **Median step time:** 3.92s
- **Forward pass:** 2.00s (51.0%)
- **Backward pass:** 1.86s (47.4%)
- **Total time (10 steps):** 39.2s
- **No outliers detected**

### Experiment C (Motion + Temporal Learning)
- **Median step time (excluding outlier):** 8.54s
- **Forward pass:** 4.31s (50.5%)
- **Backward pass:** 4.40s (51.5%)
- **Total time (10 steps):** 85.4s (excluding outlier)
- **Outlier detected:** Step 6 took 301.63s (35.3x slowdown)

### Temporal Overhead
- **Overhead factor:** 2.18x (Experiment C vs Experiment B)
- **Additional time per step:** 4.62s

## Baseline 100-Step Estimates (Current Performance)

### Experiment B
- **Median estimate:** 392s (6.5 minutes)
- **With 20% buffer:** 475s (7.9 minutes)

### Experiment C
- **Median estimate:** 854s (14.2 minutes)
- **With 20% buffer:** 1127s (18.8 minutes)

### Combined (Both Experiments)
- **Total estimated time:** 1602s (26.7 minutes)
- **Risk:** 10% outlier probability could significantly extend time

## Known Issues

### 1. CPU Thermal Throttling
- **Current frequency:** 900 MHz
- **Max frequency:** 2701 MHz
- **Gap:** 3x slower than potential
- **Likely cause:** Power-saving mode or thermal constraints

### 2. Performance Outlier
- **Frequency:** 1/10 steps in Experiment C
- **Severity:** 35.3x slowdown (301s vs 8.5s)
- **Possible causes:**
  - CPU thermal throttling
  - Background system load
  - OS scheduling anomaly
  - Garbage collection spike
  - Memory pressure

### 3. Thread Configuration
- **Current:** 4 threads
- **Physical cores:** 4
- **Logical cores:** 8
- **Opportunity:** May benefit from hyperthreading (8 threads)

## Optimization Targets

### Safe Optimizations to Attempt
1. **CPU Performance Mode** (High Impact - 3x expected)
   - Disable power saving
   - Enable performance mode
   - Target: 900 MHz → 2701 MHz

2. **Thread Tuning** (Medium Impact - 10-30% expected)
   - Experiment with 8 threads
   - Test hyperthreading utilization

3. **Background Load Reduction** (Low Impact - 5-10% expected)
   - Close background applications
   - Minimize system load

4. **PyTorch CPU Optimizations** (Low Impact - 5-15% expected)
   - Enable MKL-DNN (if available)
   - Set environment variables

### Target Performance
- **Current:** 26.7 minutes (with buffer)
- **Target:** < 10 minutes (2.7x improvement)
- **Ideal:** < 5 minutes (5.3x improvement)

## Checkpoint Hashes (Baseline)

- **Stage 2 checkpoint:** a7f5142e133c6aeadc656967639eab046a7a8f922e3201a6feda4496a544d57c
- **Stage 3 spatial checkpoint:** a7f5142e133c6aeadc656967639eab046a7a8f922e3201a6feda4496a544d57c
- **Stage 3 motion checkpoint:** 9a75aadc6bae18b9083797fa4bf2424f58fdf73118eac620e259e092096fa28a
- **Tokenizer:** 9227a1001b4d3503fa59729c04a26a01307d4e0a050b579e858ee8a809d679e8

## Next Steps

1. Investigate the 301-second outlier from Phase 2A
2. Implement safe CPU performance optimizations
3. Repeat smoke test with optimizations
4. Compare performance improvements
5. Establish reliable execution time estimates
