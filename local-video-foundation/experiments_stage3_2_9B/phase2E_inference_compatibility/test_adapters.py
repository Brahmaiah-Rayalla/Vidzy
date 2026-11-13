"""
Stage 3.2.9D - Test Inference Adapters

Test that all three checkpoints can be loaded with their respective adapters.

Author: Devin
Date: 2026-11-02
"""

import torch
from pathlib import Path
from inference_adapters import create_adapter


def test_adapter(experiment_type: str, checkpoint_path: str):
    """Test loading a checkpoint with its adapter"""
    print("=" * 80)
    print(f"  TESTING {experiment_type} ADAPTER")
    print("=" * 80)
    
    if not Path(checkpoint_path).exists():
        print(f"[X] Checkpoint not found: {checkpoint_path}")
        return False
        
    try:
        # Create adapter
        adapter = create_adapter(checkpoint_path, experiment_type)
        
        # Load model
        model = adapter.load()
        
        # Test forward pass with dummy data
        batch_size = 1
        num_frames = 16
        channels = 3
        height = 32
        width = 32
        text_dim = 256
        
        # Create dummy inputs based on model type
        if experiment_type == 'A':
            # Stage 2.1: Image generation (2D input)
            x = torch.randn(batch_size, channels, height, width).to(adapter.device)
            t = torch.randint(0, 1000, (batch_size,)).to(adapter.device)
            text_emb = torch.randn(batch_size, 10, text_dim).to(adapter.device)  # [batch, seq_len, text_dim]
            print(f"\nTesting forward pass (image generation)...")
            print(f"  Input shape: {x.shape}")
            print(f"  Timesteps shape: {t.shape}")
            print(f"  Text embedding shape: {text_emb.shape}")
        else:
            # Stage 3: Video generation (5D input)
            # VideoUNetFixed expects [batch, channels, frames, height, width]
            x = torch.randn(batch_size, channels, num_frames, height, width).to(adapter.device)
            t = torch.randint(0, 1000, (batch_size,)).to(adapter.device)
            text_emb = torch.randn(batch_size, 10, text_dim).to(adapter.device)  # [batch, seq_len, text_dim]
            print(f"\nTesting forward pass (video generation)...")
            print(f"  Input shape: {x.shape}")
            print(f"  Timesteps shape: {t.shape}")
            print(f"  Text embedding shape: {text_emb.shape}")
        
        # Test denoising
        output = adapter.denoise(x, t, text_emb)
        
        print(f"  Output shape: {output.shape}")
        print(f"  [OK] Forward pass successful")
        
        # Verify output shape matches input
        if output.shape == x.shape:
            print(f"  [OK] Output shape matches input")
        else:
            print(f"  [X] Output shape mismatch: expected {x.shape}, got {output.shape}")
            return False
            
        print(f"\n[OK] {experiment_type} adapter test PASSED")
        return True
        
    except Exception as e:
        print(f"\n[X] {experiment_type} adapter test FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all three adapters"""
    base_dir = Path(__file__).parent.parent.parent
    
    checkpoints = {
        'A': str(base_dir / "checkpoints_conditional" / "checkpoint_final.pt"),
        'B': str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_B.pt"),
        'C': str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_C.pt")
    }
    
    print("\n" + "=" * 80)
    print("  STAGE 3.2.9D - INFERENCE ADAPTER TESTS")
    print("=" * 80)
    
    results = {}
    for exp_type, checkpoint_path in checkpoints.items():
        results[exp_type] = test_adapter(exp_type, checkpoint_path)
        print()
    
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    for exp_type, success in results.items():
        status = "[OK]" if success else "[X]"
        print(f"  {status} Experiment {exp_type}: {'PASSED' if success else 'FAILED'}")
    
    all_passed = all(results.values())
    print("=" * 80)
    if all_passed:
        print("  [OK] ALL ADAPTER TESTS PASSED")
    else:
        print("  [X] SOME ADAPTER TESTS FAILED")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
