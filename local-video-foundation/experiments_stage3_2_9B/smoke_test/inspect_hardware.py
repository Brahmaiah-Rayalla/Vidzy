"""
Stage 3.2.9B - Phase 2A: Hardware Inspection

Report available hardware specs before training.

Author: Devin
Date: 2026-11-02
"""

import sys
import os
import json
import platform
import psutil
import torch
from pathlib import Path

def inspect_hardware():
    """Inspect and report hardware specifications."""

    print("=" * 80)
    print("  HARDWARE INSPECTION")
    print("=" * 80)

    specs = {}

    # CPU info
    print("\nCPU Information:")
    cpu_info = {
        'processor': platform.processor(),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'cpu_freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else None,
        'cpu_freq_current': psutil.cpu_freq().current if psutil.cpu_freq() else None
    }
    for key, value in cpu_info.items():
        print(f"  {key}: {value}")
    specs['cpu'] = cpu_info

    # RAM info
    print("\nMemory Information:")
    mem_info = {
        'total_gb': psutil.virtual_memory().total / (1024**3),
        'available_gb': psutil.virtual_memory().available / (1024**3),
        'percent_used': psutil.virtual_memory().percent
    }
    for key, value in mem_info.items():
        print(f"  {key}: {value}")
    specs['memory'] = mem_info

    # PyTorch info
    print("\nPyTorch Information:")
    torch_info = {
        'version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'intel_xpu_available': hasattr(torch, 'xpu') and torch.xpu.is_available() if hasattr(torch, 'xpu') else False
    }
    for key, value in torch_info.items():
        print(f"  {key}: {value}")
    specs['pytorch'] = torch_info

    # CUDA device info (if available)
    if torch.cuda.is_available():
        print("\nCUDA Devices:")
        cuda_devices = []
        for i in range(torch.cuda.device_count()):
            device_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'capability': torch.cuda.get_device_capability(i),
                'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3)
            }
            cuda_devices.append(device_info)
            print(f"  Device {i}: {device_info['name']}")
            print(f"    Capability: {device_info['capability']}")
            print(f"    Memory: {device_info['memory_total_gb']:.2f} GB")
        specs['cuda_devices'] = cuda_devices

    # Intel XPU info (if available)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("\nIntel XPU Devices:")
        xpu_devices = []
        for i in range(torch.xpu.device_count()):
            device_info = {
                'id': i,
                'name': torch.xpu.get_device_name(i)
            }
            xpu_devices.append(device_info)
            print(f"  Device {i}: {device_info['name']}")
        specs['xpu_devices'] = xpu_devices

    # PyTorch thread configuration
    print("\nPyTorch Thread Configuration:")
    thread_info = {
        'num_threads': torch.get_num_threads(),
        'num_interop_threads': torch.get_num_interop_threads()
    }
    for key, value in thread_info.items():
        print(f"  {key}: {value}")
    specs['threads'] = thread_info

    # Current execution device
    print("\nExecution Device:")
    device = torch.device('cpu')  # CPU is mandatory
    print(f"  Current device: {device}")
    print(f"  Device type: {device.type}")
    specs['execution_device'] = {
        'type': device.type,
        'index': device.index
    }

    return specs


def record_checkpoint_hashes(parent_dir):
    """Record checkpoint hashes before training."""
    import hashlib

    print("\n" + "=" * 80)
    print("  CHECKPOINT HASHES (Before Training)")
    print("=" * 80)

    hashes = {}

    stage2_checkpoint = parent_dir / "checkpoints_conditional" / "checkpoint_final.pt"
    stage3_spatial_checkpoint = parent_dir / "experiments_stage3_2_7" / "checkpoints" / "checkpoint_stage2_spatial.pt"
    stage3_motion_checkpoint = parent_dir / "experiments_stage3_2_7" / "checkpoints" / "checkpoint_stage2_motion_expanded.pt"
    tokenizer_path = parent_dir / "experiments_stage3_2_7" / "checkpoints" / "tokenizer_motion_expanded.json"

    for path, name in [
        (stage2_checkpoint, 'stage2_checkpoint'),
        (stage3_spatial_checkpoint, 'stage3_spatial_checkpoint'),
        (stage3_motion_checkpoint, 'stage3_motion_checkpoint'),
        (tokenizer_path, 'tokenizer')
    ]:
        if path.exists():
            with open(path, 'rb') as f:
                hash_value = hashlib.sha256(f.read()).hexdigest()
            hashes[name] = hash_value
            print(f"{name}: {hash_value}")
        else:
            hashes[name] = "FILE_NOT_FOUND"
            print(f"{name}: FILE_NOT_FOUND")

    return hashes


def main():
    parent_dir = Path(__file__).parent.parent.parent
    smoke_test_dir = Path(__file__).parent

    # Inspect hardware
    specs = inspect_hardware()

    # Record checkpoint hashes
    hashes = record_checkpoint_hashes(parent_dir)

    # Save results
    results = {
        'hardware': specs,
        'checkpoint_hashes': hashes
    }

    results_path = smoke_test_dir / "STAGE3_2_9B_HARDWARE_SPECS.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved hardware specs to {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
