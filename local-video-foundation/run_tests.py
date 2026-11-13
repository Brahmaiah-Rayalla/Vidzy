"""
Run all automated tests.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_dataset import test_synthetic_generation, test_dataset_loading, test_reproducibility
from tests.test_model import test_model_output_shape, test_model_initialization, test_model_gradient_flow
from tests.test_diffusion import test_schedule_computation, test_forward_diffusion, test_training_loss, test_ddim_sampling
from tests.test_checkpoint import test_checkpoint_save_load, test_config_save_load


def main():
    print("="*60)
    print("Running Automated Tests")
    print("="*60 + "\n")

    try:
        # Dataset tests
        print("Dataset Tests")
        print("-" * 60)
        test_synthetic_generation()
        test_dataset_loading()
        test_reproducibility()
        print()

        # Model tests
        print("Model Tests")
        print("-" * 60)
        test_model_output_shape()
        test_model_initialization()
        test_model_gradient_flow()
        print()

        # Diffusion tests
        print("Diffusion Tests")
        print("-" * 60)
        test_schedule_computation()
        test_forward_diffusion()
        test_training_loss()
        test_ddim_sampling()
        print()

        # Checkpoint tests
        print("Checkpoint Tests")
        print("-" * 60)
        test_checkpoint_save_load()
        test_config_save_load()
        print()

        print("="*60)
        print("All tests passed successfully!")
        print("="*60)
        return 0

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Test failed with error: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
