"""
Stage 3.2.9H.2 - Small Scale Video Generation with Corrected Text Conditioning

Generates a small set of videos to verify the corrected end-to-end pipeline.

Author: Devin
Date: 2026-11-03
"""

import torch
import sys
from pathlib import Path
import json

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "experiments_stage3_2_7" / "video_foundation"))
sys.path.insert(0, str(Path(__file__).parent))

from video_generation_runner import VideoGenerationRunner


def main():
    """Run small-scale video generation with corrected text conditioning."""
    print("\n" + "=" * 80)
    print("STAGE 3.2.9H.2 - SMALL SCALE VIDEO GENERATION")
    print("Testing Corrected Text Conditioning End-to-End")
    print("=" * 80)
    
    base_dir = Path(__file__).parent.parent.parent.parent
    
    # Configuration
    config = {
        'output_dir': Path(__file__).parent / "outputs_small",
        'device': 'cpu',
        'resolution': 64,
        'num_frames': 16,
        'sampling_steps': 10,  # Reduced for faster testing
        'seed': 42
    }
    
    # Checkpoint paths
    checkpoints = {
        'B': str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_B.pt"),
        'C': str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_C.pt")
    }
    
    # Minimal prompts for testing
    prompts = [
        "Red object moving left",
        "Blue object moving right",
        "Stationary object"
    ]
    
    # Create runner
    runner = VideoGenerationRunner(**config)
    
    # Run Experiment B
    print("\n" + "=" * 80)
    print("EXPERIMENT B: Motion-Conditioned (Temporal Disabled)")
    print("=" * 80)
    
    results_B = runner.run_experiment(
        'B', checkpoints['B'], prompts, "Motion-Conditioned (Temporal Disabled)"
    )
    
    # Run Experiment C
    print("\n" + "=" * 80)
    print("EXPERIMENT C: Temporal-Enabled")
    print("=" * 80)
    
    results_C = runner.run_experiment(
        'C', checkpoints['C'], prompts, "Temporal-Enabled"
    )
    
    # Compile results
    all_results = {
        'experiment_B': results_B,
        'experiment_C': results_C,
        'text_conditioning': 'CORRECTED',
        'sampling_steps': config['sampling_steps']
    }
    
    # Save results
    results_path = config['output_dir'] / "generation_results_small.json"
    with open(results_path, 'w') as f:
        json.dump(runner.convert_to_serializable(all_results), f, indent=2)
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to {results_path}")
    print(f"Text conditioning: CORRECTED (actual text encoding)")
    print(f"Sampling steps: {config['sampling_steps']}")
    
    return all_results


if __name__ == "__main__":
    main()
