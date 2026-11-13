"""
Stage 3.2.9E - DDIM Sampler Validation

Progressive validation tests for DDIM sampling before full video generation.

Author: Devin
Date: 2026-11-02
"""

import torch
import numpy as np
from pathlib import Path
import json
import time
from collections import defaultdict
from typing import Dict, Any

from video_generation_runner import DDIMSampler, VideoGenerationRunner
from inference_adapters import create_adapter


class DDIMValidator:
    """Progressive DDIM validation tests."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = defaultdict(list)
    
    def convert_to_serializable(self, obj):
        """Convert tensors and other non-serializable objects to JSON-serializable types."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
        
    def test_single_denoising_step(self, adapter, model) -> Dict[str, Any]:
        """Test 1: Single denoising step."""
        print("\n" + "=" * 80)
        print("  TEST 1: SINGLE DENOISING STEP")
        print("=" * 80)
        
        sampler = DDIMSampler(sampling_steps=25)
        
        # Initialize noise
        batch_size = 1
        channels = 3
        num_frames = 16
        height = 64
        width = 64
        
        if hasattr(adapter.model, 'use_temporal'):
            shape = (batch_size, channels, num_frames, height, width)
        else:
            shape = (batch_size, channels, height, width)
        
        noise = torch.randn(shape)
        t = torch.tensor([999])  # Start timestep
        text_emb = torch.randn(batch_size, 10, 256)
        
        print(f"  Input shape: {noise.shape}")
        print(f"  Timestep: {t.item()}")
        print(f"  Text embedding shape: {text_emb.shape}")
        
        # Single denoising step
        start_time = time.time()
        with torch.no_grad():
            predicted_noise = adapter.denoise(noise, t, text_emb)
        step_time = time.time() - start_time
        
        print(f"  Predicted noise shape: {predicted_noise.shape}")
        print(f"  Step time: {step_time:.4f}s")
        
        # Validation
        test_result = {
            'test': 'single_denoising_step',
            'passed': True,
            'errors': [],
            'warnings': [],
            'metrics': {
                'input_shape': list(noise.shape),
                'output_shape': list(predicted_noise.shape),
                'timestep': float(t.item()),
                'text_emb_shape': list(text_emb.shape),
                'step_time': float(step_time),
                'predicted_noise_mean': float(predicted_noise.mean().item()),
                'predicted_noise_std': float(predicted_noise.std().item()),
                'has_nan': bool(torch.isnan(predicted_noise).any().item()),
                'has_inf': bool(torch.isinf(predicted_noise).any().item())
            }
        }
        
        # Check shape
        if predicted_noise.shape != noise.shape:
            test_result['passed'] = False
            test_result['errors'].append(f"Shape mismatch: {noise.shape} vs {predicted_noise.shape}")
        
        # Check for NaN/Inf
        if test_result['metrics']['has_nan']:
            test_result['passed'] = False
            test_result['errors'].append("Predicted noise contains NaN")
        
        if test_result['metrics']['has_inf']:
            test_result['passed'] = False
            test_result['errors'].append("Predicted noise contains Inf")
        
        # Check finite
        if not torch.isfinite(predicted_noise).all():
            test_result['passed'] = False
            test_result['errors'].append("Predicted noise contains non-finite values")
        
        print(f"  Result: {'PASSED' if test_result['passed'] else 'FAILED'}")
        
        return test_result
    
    def test_two_ddim_steps(self, adapter, model) -> Dict[str, Any]:
        """Test 2: Two DDIM steps."""
        print("\n" + "=" * 80)
        print("  TEST 2: TWO DDIM STEPS")
        print("=" * 80)
        
        sampler = DDIMSampler(sampling_steps=25)
        
        # Initialize noise
        batch_size = 1
        channels = 3
        num_frames = 16
        height = 64
        width = 64
        
        if hasattr(adapter.model, 'use_temporal'):
            shape = (batch_size, channels, num_frames, height, width)
        else:
            shape = (batch_size, channels, height, width)
        
        x = torch.randn(shape)
        text_emb = torch.randn(batch_size, 10, 256)
        
        # Two DDIM steps
        timesteps = sampler.timesteps[:2]
        x_values = []
        diagnostics = []
        
        for i, (t, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_tensor = torch.tensor([t])
            t_prev_tensor = torch.tensor([t_prev])
            
            x_prev, diag = sampler.p_sample_ddim(model, x, t_tensor, t_prev_tensor, text_emb, adapter)
            
            x_values.append(x.clone())
            diagnostics.append(diag)
            x = x_prev
            
            print(f"  Step {i+1}: t={t.item()}, t_prev={t_prev.item()}")
            print(f"    x_mean: {diag['x_prev_mean']:.4f}, x_std: {diag['x_prev_std']:.4f}")
        
        test_result = {
            'test': 'two_ddim_steps',
            'passed': True,
            'errors': [],
            'warnings': [],
            'metrics': {
                'timesteps': [float(t.item()) for t in timesteps],
                'step_diagnostics': diagnostics,
                'x_shape': list(x.shape),
                'has_nan': any(d['has_nan'] for d in diagnostics),
                'has_inf': any(d['has_inf'] for d in diagnostics)
            }
        }
        
        # Check for NaN/Inf
        if test_result['metrics']['has_nan']:
            test_result['passed'] = False
            test_result['errors'].append("NaN detected in sampling")
        
        if test_result['metrics']['has_inf']:
            test_result['passed'] = False
            test_result['errors'].append("Inf detected in sampling")
        
        # Check timestep progression
        if timesteps[0] <= timesteps[1]:
            test_result['passed'] = False
            test_result['errors'].append(f"Timesteps not decreasing: {timesteps}")
        
        print(f"  Result: {'PASSED' if test_result['passed'] else 'FAILED'}")
        
        return test_result
    
    def test_five_ddim_steps(self, adapter, model) -> Dict[str, Any]:
        """Test 3: Five DDIM steps."""
        print("\n" + "=" * 80)
        print("  TEST 3: FIVE DDIM STEPS")
        print("=" * 80)
        
        sampler = DDIMSampler(sampling_steps=25)
        
        # Initialize noise
        batch_size = 1
        channels = 3
        num_frames = 16
        height = 64
        width = 64
        
        if hasattr(adapter.model, 'use_temporal'):
            shape = (batch_size, channels, num_frames, height, width)
        else:
            shape = (batch_size, channels, height, width)
        
        x = torch.randn(shape)
        text_emb = torch.randn(batch_size, 10, 256)
        
        # Five DDIM steps
        timesteps = sampler.timesteps[:5]
        x_values = []
        diagnostics = []
        
        start_time = time.time()
        for i, (t, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_tensor = torch.tensor([t])
            t_prev_tensor = torch.tensor([t_prev])
            
            x_prev, diag = sampler.p_sample_ddim(model, x, t_tensor, t_prev_tensor, text_emb, adapter)
            
            x_values.append(x.clone())
            diagnostics.append(diag)
            x = x_prev
            
            if (i + 1) % 2 == 0:
                print(f"  Step {i+1}: t={t.item()}, x_mean={diag['x_prev_mean']:.4f}")
        
        total_time = time.time() - start_time
        
        # Normalize final output
        x_final = (x.clamp(-1, 1) + 1) / 2
        
        test_result = {
            'test': 'five_ddim_steps',
            'passed': True,
            'errors': [],
            'warnings': [],
            'metrics': {
                'num_steps': len(diagnostics),
                'total_time': total_time,
                'time_per_step': total_time / len(diagnostics),
                'final_shape': list(x_final.shape),
                'final_mean': x_final.mean().item(),
                'final_std': x_final.std().item(),
                'final_min': x_final.min().item(),
                'final_max': x_final.max().item(),
                'has_nan': any(d['has_nan'] for d in diagnostics),
                'has_inf': any(d['has_inf'] for d in diagnostics),
                'valid_range': (x_final.min() >= 0 and x_final.max() <= 1),
                'step_diagnostics': diagnostics
            }
        }
        
        # Check for NaN/Inf
        if test_result['metrics']['has_nan']:
            test_result['passed'] = False
            test_result['errors'].append("NaN detected in sampling")
        
        if test_result['metrics']['has_inf']:
            test_result['passed'] = False
            test_result['errors'].append("Inf detected in sampling")
        
        # Check output range
        if not test_result['metrics']['valid_range']:
            test_result['warnings'].append(f"Output outside [0,1]: [{x_final.min():.3f}, {x_final.max():.3f}]")
        
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Time per step: {test_result['metrics']['time_per_step']:.4f}s")
        print(f"  Final range: [{x_final.min():.3f}, {x_final.max():.3f}]")
        print(f"  Result: {'PASSED' if test_result['passed'] else 'FAILED'}")
        
        return test_result
    
    def test_twenty_five_ddim_steps(self, adapter, model) -> Dict[str, Any]:
        """Test 4: Twenty-five DDIM steps (full schedule)."""
        print("\n" + "=" * 80)
        print("  TEST 4: TWENTY-FIVE DDIM STEPS (FULL SCHEDULE)")
        print("=" * 80)
        
        sampler = DDIMSampler(sampling_steps=25)
        
        # Initialize noise
        batch_size = 1
        channels = 3
        num_frames = 16
        height = 64
        width = 64
        
        if hasattr(adapter.model, 'use_temporal'):
            shape = (batch_size, channels, num_frames, height, width)
        else:
            shape = (batch_size, channels, height, width)
        
        x = torch.randn(shape)
        text_emb = torch.randn(batch_size, 10, 256)
        
        # Full DDIM sampling
        timesteps = sampler.timesteps
        diagnostics = []
        
        start_time = time.time()
        for i, (t, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_tensor = torch.tensor([t])
            t_prev_tensor = torch.tensor([t_prev])
            
            x_prev, diag = sampler.p_sample_ddim(model, x, t_tensor, t_prev_tensor, text_emb, adapter)
            
            diagnostics.append(diag)
            x = x_prev
            
            if diag['has_nan'] or diag['has_inf']:
                print(f"  [ERROR] Step {i}: NaN or Inf detected")
                break
            
            if (i + 1) % 5 == 0:
                print(f"  Step {i+1}/{len(timesteps)-1}: x_mean={diag['x_prev_mean']:.4f}")
        
        total_time = time.time() - start_time
        
        # Normalize final output
        x_final = (x.clamp(-1, 1) + 1) / 2
        
        test_result = {
            'test': 'twenty_five_ddim_steps',
            'passed': True,
            'errors': [],
            'warnings': [],
            'metrics': {
                'num_steps': len(diagnostics),
                'total_time': total_time,
                'time_per_step': total_time / len(diagnostics),
                'final_shape': list(x_final.shape),
                'final_mean': x_final.mean().item(),
                'final_std': x_final.std().item(),
                'final_min': x_final.min().item(),
                'final_max': x_final.max().item(),
                'has_nan': any(d['has_nan'] for d in diagnostics),
                'has_inf': any(d['has_inf'] for d in diagnostics),
                'valid_range': (x_final.min() >= 0 and x_final.max() <= 1),
                'step_diagnostics': diagnostics
            }
        }
        
        # Check for NaN/Inf
        if test_result['metrics']['has_nan']:
            test_result['passed'] = False
            test_result['errors'].append("NaN detected in sampling")
        
        if test_result['metrics']['has_inf']:
            test_result['passed'] = False
            test_result['errors'].append("Inf detected in sampling")
        
        # Check output range
        if not test_result['metrics']['valid_range']:
            test_result['warnings'].append(f"Output outside [0,1]: [{x_final.min():.3f}, {x_final.max():.3f}]")
        
        print(f"  Steps completed: {len(diagnostics)}/{len(timesteps)-1}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Time per step: {test_result['metrics']['time_per_step']:.4f}s")
        print(f"  Final range: [{x_final.min():.3f}, {x_final.max():.3f}]")
        print(f"  Result: {'PASSED' if test_result['passed'] else 'FAILED'}")
        
        return test_result
    
    def run_all_tests(self, experiment_type: str, checkpoint_path: str) -> Dict[str, Any]:
        """Run all validation tests for an experiment."""
        print("\n" + "=" * 80)
        print(f"  VALIDATING EXPERIMENT {experiment_type}")
        print("=" * 80)
        
        # Load adapter
        print(f"\nLoading adapter for Experiment {experiment_type}")
        adapter = create_adapter(checkpoint_path, experiment_type)
        model = adapter.load()
        
        # Run tests
        results = {
            'experiment_type': experiment_type,
            'tests': []
        }
        
        # Test 1
        result1 = self.test_single_denoising_step(adapter, model)
        results['tests'].append(result1)
        
        # Test 2
        result2 = self.test_two_ddim_steps(adapter, model)
        results['tests'].append(result2)
        
        # Test 3
        result3 = self.test_five_ddim_steps(adapter, model)
        results['tests'].append(result3)
        
        # Test 4
        result4 = self.test_twenty_five_ddim_steps(adapter, model)
        results['tests'].append(result4)
        
        # Overall result
        all_passed = all(t['passed'] for t in results['tests'])
        results['overall_passed'] = all_passed
        
        # Save results
        output_path = self.output_dir / f"ddim_validation_{experiment_type}.json"
        with open(output_path, 'w') as f:
            json.dump(self.convert_to_serializable(results), f, indent=2)
        
        print("\n" + "=" * 80)
        print(f"  VALIDATION SUMMARY FOR EXPERIMENT {experiment_type}")
        print("=" * 80)
        for test in results['tests']:
            status = "PASSED" if test['passed'] else "FAILED"
            print(f"  {test['test']}: {status}")
        print(f"\n  Overall: {'PASSED' if all_passed else 'FAILED'}")
        print(f"  Results saved to {output_path}")
        
        return results


def main():
    """Main execution function."""
    base_dir = Path(__file__).parent.parent.parent
    
    # Output directory
    output_dir = Path(__file__).parent / "validation_results"
    
    # Create validator
    validator = DDIMValidator(output_dir)
    
    # Checkpoint paths
    checkpoints = {
        'A': str(base_dir / "checkpoints_conditional" / "checkpoint_final.pt"),
        'B': str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_B.pt"),
        'C': str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_C.pt")
    }
    
    # Run validation for each experiment
    all_results = {}
    for exp_type, checkpoint_path in checkpoints.items():
        results = validator.run_all_tests(exp_type, checkpoint_path)
        all_results[exp_type] = results
    
    # Save overall results
    output_path = output_dir / "ddim_validation_all.json"
    with open(output_path, 'w') as f:
        json.dump(validator.convert_to_serializable(all_results), f, indent=2)
    
    print("\n" + "=" * 80)
    print("  ALL VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
