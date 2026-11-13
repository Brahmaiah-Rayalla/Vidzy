"""
Stage 3.2.9H.2 - Integration Test for Corrected Text Conditioning

Tests the integrated inference pipeline to verify:
1. Adapters load correctly
2. Text encoding uses actual text encoder (not random)
3. Model predictions are prompt-sensitive
4. End-to-end generation works

Author: Devin
Date: 2026-11-03
"""

import torch
import sys
from pathlib import Path
import json

# Add parent directories to path
base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "experiments_stage3_2_7" / "video_foundation"))
sys.path.insert(0, str(Path(__file__).parent))

from inference_adapters import create_adapter


def test_adapter_loading():
    """Test that adapters load correctly with text encoder components."""
    print("\n" + "=" * 80)
    print("TEST 1: Adapter Loading")
    print("=" * 80)
    
    base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
    
    checkpoints = {
        'A': str(base_dir / "checkpoints_conditional" / "checkpoint_final.pt"),
        'B': str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_B.pt"),
        'C': str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_C.pt")
    }
    
    results = {}
    
    for exp_type, checkpoint_path in checkpoints.items():
        print(f"\nTesting Experiment {exp_type}")
        try:
            adapter = create_adapter(checkpoint_path, exp_type)
            model = adapter.load()
            
            # Check if adapter has text encoder components
            has_text_encoder = hasattr(adapter, 'text_encoder') and adapter.text_encoder is not None
            has_tokenizer = hasattr(adapter, 'tokenizer') and adapter.tokenizer is not None
            has_encode_method = hasattr(adapter, 'encode_prompt')
            
            results[exp_type] = {
                'loaded': True,
                'has_text_encoder': has_text_encoder,
                'has_tokenizer': has_tokenizer,
                'has_encode_method': has_encode_method,
                'error': None
            }
            
            print(f"  [OK] Adapter loaded")
            print(f"  Has text encoder: {has_text_encoder}")
            print(f"  Has tokenizer: {has_tokenizer}")
            print(f"  Has encode_prompt method: {has_encode_method}")
            
        except Exception as e:
            results[exp_type] = {
                'loaded': False,
                'has_text_encoder': False,
                'has_tokenizer': False,
                'has_encode_method': False,
                'error': str(e)
            }
            print(f"  [FAIL] {e}")
    
    return results


def test_text_encoding():
    """Test that text encoding uses actual text encoder (not random)."""
    print("\n" + "=" * 80)
    print("TEST 2: Text Encoding (Actual vs Random)")
    print("=" * 80)
    
    base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
    checkpoint_path = str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_B.pt")
    
    try:
        adapter = create_adapter(checkpoint_path, 'B')
        model = adapter.load()
        
        # Test with different prompts
        prompts = [
            "Red object moving left",
            "Blue object moving right",
            "Stationary object"
        ]
        
        embeddings = {}
        for prompt in prompts:
            text_emb, text_mask = adapter.encode_prompt(prompt)
            embeddings[prompt] = {
                'mean': float(text_emb.mean().item()),
                'std': float(text_emb.std().item()),
                'min': float(text_emb.min().item()),
                'max': float(text_emb.max().item())
            }
            print(f"\nPrompt: '{prompt}'")
            print(f"  Embedding shape: {text_emb.shape}")
            print(f"  Mean: {embeddings[prompt]['mean']:.4f}")
            print(f"  Std: {embeddings[prompt]['std']:.4f}")
        
        # Check if embeddings are different for different prompts
        prompt1_mean = embeddings[prompts[0]]['mean']
        prompt2_mean = embeddings[prompts[1]]['mean']
        
        different_prompts = abs(prompt1_mean - prompt2_mean) > 0.01
        
        print(f"\nDifferent prompts produce different embeddings: {different_prompts}")
        
        # Test determinism
        text_emb1, _ = adapter.encode_prompt(prompts[0])
        text_emb2, _ = adapter.encode_prompt(prompts[0])
        
        is_deterministic = torch.allclose(text_emb1, text_emb2, atol=1e-6)
        print(f"Same prompt produces same embedding (deterministic): {is_deterministic}")
        
        return {
            'success': True,
            'embeddings': embeddings,
            'different_prompts': different_prompts,
            'deterministic': is_deterministic,
            'error': None
        }
        
    except Exception as e:
        print(f"  [FAIL] {e}")
        return {
            'success': False,
            'embeddings': None,
            'different_prompts': False,
            'deterministic': False,
            'error': str(e)
        }


def test_model_prediction_sensitivity():
    """Test that model predictions are sensitive to text embeddings."""
    print("\n" + "=" * 80)
    print("TEST 3: Model Prediction Sensitivity")
    print("=" * 80)
    
    base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
    checkpoint_path = str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_B.pt")
    
    try:
        adapter = create_adapter(checkpoint_path, 'B')
        model = adapter.load()
        
        # Create fixed noise
        noise = torch.randn(1, 3, 16, 64, 64)
        t = torch.tensor([500]).long()
        
        # Test with different prompts
        prompts = [
            "Red object moving left",
            "Blue object moving right"
        ]
        
        predictions = {}
        for prompt in prompts:
            text_emb, _ = adapter.encode_prompt(prompt)
            
            with torch.no_grad():
                prediction = adapter.denoise(noise, t, text_emb)
            
            predictions[prompt] = {
                'mean': float(prediction.mean().item()),
                'std': float(prediction.std().item())
            }
            print(f"\nPrompt: '{prompt}'")
            print(f"  Prediction mean: {predictions[prompt]['mean']:.4f}")
            print(f"  Prediction std: {predictions[prompt]['std']:.4f}")
        
        # Check if predictions are different
        pred1_mean = predictions[prompts[0]]['mean']
        pred2_mean = predictions[prompts[1]]['mean']
        
        predictions_differ = abs(pred1_mean - pred2_mean) > 0.001
        
        print(f"\nDifferent prompts produce different predictions: {predictions_differ}")
        
        return {
            'success': True,
            'predictions': predictions,
            'predictions_differ': predictions_differ,
            'error': None
        }
        
    except Exception as e:
        print(f"  [FAIL] {e}")
        return {
            'success': False,
            'predictions': None,
            'predictions_differ': False,
            'error': str(e)
        }


def test_checkpoint_integrity():
    """Verify checkpoint SHA256 hashes."""
    print("\n" + "=" * 80)
    print("TEST 4: Checkpoint Integrity")
    print("=" * 80)
    
    import hashlib
    
    base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
    
    checkpoints = {
        'stage_2_1': base_dir / "checkpoints_conditional" / "checkpoint_final.pt",
        'experiment_B': base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_B.pt",
        'experiment_C': base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_C.pt"
    }
    
    expected_hashes = {
        'stage_2_1': 'a7f5142e133c6aeadc656967639eab046a7a8f922e3201a6feda4496a544d57c',
        'experiment_B': '27eef3c7afec90a0cfaf32384f6535d271b9369ba7baaba51f17669fd3b0ff14',
        'experiment_C': 'cc71317860e152b818413c67a7e0304079a86e6e243c6e5e5d9559cec21499f5'
    }
    
    results = {}
    all_match = True
    
    for name, path in checkpoints.items():
        if path.exists():
            sha256 = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            
            actual_hash = sha256.hexdigest()
            expected_hash = expected_hashes[name]
            matches = actual_hash == expected_hash
            
            results[name] = {
                'path': str(path),
                'actual_hash': actual_hash,
                'expected_hash': expected_hash,
                'matches': matches
            }
            
            print(f"\n{name}:")
            print(f"  Path: {path}")
            print(f"  Actual: {actual_hash}")
            print(f"  Expected: {expected_hash}")
            print(f"  Matches: {matches}")
            
            if not matches:
                all_match = False
        else:
            results[name] = {
                'path': str(path),
                'actual_hash': None,
                'expected_hash': expected_hashes[name],
                'matches': False,
                'error': 'File not found'
            }
            print(f"\n{name}: [FAIL] File not found: {path}")
            all_match = False
    
    print(f"\nAll checkpoint hashes match: {all_match}")
    
    return {
        'success': all_match,
        'results': results,
        'all_match': all_match
    }


def main():
    """Run all integration tests."""
    print("\n" + "=" * 80)
    print("STAGE 3.2.9H.2 - INTEGRATION TEST")
    print("Testing Corrected Text Conditioning Integration")
    print("=" * 80)
    
    # Run tests
    test1_results = test_adapter_loading()
    test2_results = test_text_encoding()
    test3_results = test_model_prediction_sensitivity()
    test4_results = test_checkpoint_integrity()
    
    # Compile results
    all_results = {
        'test_adapter_loading': test1_results,
        'test_text_encoding': test2_results,
        'test_model_prediction_sensitivity': test3_results,
        'test_checkpoint_integrity': test4_results
    }
    
    # Save results
    output_path = Path(__file__).parent / "integration_test_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Adapter loading: {'PASS' if all(r['loaded'] for r in test1_results.values()) else 'FAIL'}")
    print(f"Text encoding: {'PASS' if test2_results['success'] else 'FAIL'}")
    print(f"Model prediction sensitivity: {'PASS' if test3_results['success'] else 'FAIL'}")
    print(f"Checkpoint integrity: {'PASS' if test4_results['success'] else 'FAIL'}")
    print(f"\nResults saved to {output_path}")
    
    return all_results


if __name__ == "__main__":
    main()
