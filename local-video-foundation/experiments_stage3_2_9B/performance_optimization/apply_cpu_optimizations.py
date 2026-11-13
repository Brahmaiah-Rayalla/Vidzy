"""
Stage 3.2.9B - Phase 2B: CPU Performance Optimizations

Apply safe CPU performance optimizations to PyTorch training.

Author: Devin
Date: 2026-11-02
"""

import os
import sys
import torch
import psutil
from pathlib import Path

def apply_pytorch_optimizations():
    """Apply PyTorch-specific CPU optimizations."""

    print("=" * 80)
    print("  PYTORCH CPU OPTIMIZATIONS")
    print("=" * 80)

    optimizations = {}

    # 1. Set thread count to logical cores (hyperthreading)
    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)

    print(f"\nThread Configuration:")
    print(f"  Physical cores: {physical_cores}")
    print(f"  Logical cores: {logical_cores}")
    print(f"  Current threads: {torch.get_num_threads()}")

    # Use logical cores for hyperthreading
    torch.set_num_threads(logical_cores)
    print(f"  New thread count: {torch.get_num_threads()}")
    optimizations['num_threads'] = logical_cores

    # 2. Set interop threads
    torch.set_num_interop_threads(logical_cores)
    print(f"  Interop threads: {torch.get_num_interop_threads()}")
    optimizations['num_interop_threads'] = logical_cores

    # 3. Enable MKL-DNN if available (oneDNN)
    os.environ['MKL_NUM_THREADS'] = str(logical_cores)
    os.environ['OMP_NUM_THREADS'] = str(logical_cores)
    print(f"\nEnvironment Variables:")
    print(f"  MKL_NUM_THREADS: {os.environ['MKL_NUM_THREADS']}")
    print(f"  OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
    optimizations['mkl_threads'] = logical_cores
    optimizations['omp_threads'] = logical_cores

    # 4. Enable PyTorch deterministic mode (for reproducibility)
    # Note: This may slightly reduce performance but ensures consistency
    # torch.use_deterministic_algorithms(True)  # Commented out for performance

    # 5. Disable benchmarking (slower first pass, more consistent)
    torch.backends.cudnn.benchmark = False  # No effect on CPU, but good practice
    torch.backends.cudnn.deterministic = True  # No effect on CPU, but good practice

    print(f"\nPyTorch Backend:")
    print(f"  CuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  CuDNN deterministic: {torch.backends.cudnn.deterministic}")

    return optimizations


def print_manual_steps():
    """Print manual optimization steps that require user action."""

    print("\n" + "=" * 80)
    print("  MANUAL OPTIMIZATION STEPS")
    print("=" * 80)

    print("\n1. Enable CPU Performance Mode (Windows)")
    print("   - Open Control Panel > Power Options")
    print("   - Select 'High performance' or 'Ultimate performance' plan")
    print("   - This will prevent CPU from throttling to 900 MHz")

    print("\n2. Close Background Applications")
    print("   - Close web browsers, editors, and other applications")
    print("   - Check Task Manager for high CPU usage processes")
    print("   - Disable antivirus temporarily (if safe)")

    print("\n3. Run During Off-Peak Hours")
    print("   - Avoid times when Windows updates might run")
    print("   - Close any scheduled tasks")
    print("   - Ensure no backup or sync operations are running")

    print("\n4. Monitor CPU Temperature (Optional)")
    print("   - Use HWMonitor or similar tool")
    print("   - Check if CPU is thermal throttling")
    print("   - Ensure adequate cooling/ventilation")

    print("\n5. Disable Windows Indexing (Optional)")
    print("   - Services > Windows Search > Stop")
    print("   - Reduces background disk/CPU activity")


def verify_optimizations():
    """Verify that optimizations are applied."""

    print("\n" + "=" * 80)
    print("  OPTIMIZATION VERIFICATION")
    print("=" * 80)

    print(f"\nCurrent PyTorch Configuration:")
    print(f"  Thread count: {torch.get_num_threads()}")
    print(f"  Interop threads: {torch.get_num_interop_threads()}")
    print(f"  CPU frequency: {psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'} MHz")
    print(f"  CPU max frequency: {psutil.cpu_freq().max if psutil.cpu_freq() else 'N/A'} MHz")

    print(f"\nEnvironment Variables:")
    print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'Not set')}")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")


def main():
    """Main function to apply optimizations."""

    print("\n" + "=" * 80)
    print("  CPU PERFORMANCE OPTIMIZATION")
    print("=" * 80)

    # Apply PyTorch optimizations
    optimizations = apply_pytorch_optimizations()

    # Print manual steps
    print_manual_steps()

    # Verify optimizations
    verify_optimizations()

    # Save optimization configuration
    config = {
        'optimizations_applied': optimizations,
        'manual_steps_required': [
            'Enable CPU performance mode in Windows power settings',
            'Close background applications',
            'Run during off-peak hours',
            'Monitor CPU temperature (optional)',
            'Disable Windows indexing (optional)'
        ]
    }

    output_path = Path(__file__).parent / "optimization_config.json"
    import json
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved optimization config to {output_path}")

    print("\n" + "=" * 80)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Apply manual optimization steps (above)")
    print("2. Run optimized smoke test")
    print("3. Compare performance with baseline")

    return 0


if __name__ == "__main__":
    sys.exit(main())
