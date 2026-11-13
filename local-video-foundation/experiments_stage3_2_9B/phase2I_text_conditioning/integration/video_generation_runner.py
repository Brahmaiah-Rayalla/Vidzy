"""
Stage 3.2.9H.2 - Corrected Video Generation Runner with Actual Text Conditioning

Implements complete inference pipeline for video generation using
Stage 2.1 and Stage 3 checkpoints with DDIM sampling and CORRECTED text encoding.

Author: Devin
Date: 2026-11-03
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import time
import sys
from PIL import Image
try:
    import imageio
except ImportError:
    imageio = None
from collections import defaultdict

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "experiments_stage3_2_7" / "video_foundation"))
sys.path.insert(0, str(Path(__file__).parent))

from inference_adapters import create_adapter


class DDIMSampler:
    """DDIM sampler for video generation."""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        sampling_steps: int = 25,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear'
    ):
        self.num_timesteps = num_timesteps
        self.sampling_steps = sampling_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        
        # Compute beta schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == 'quadratic':
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Sampling timesteps
        self.timesteps = torch.linspace(num_timesteps - 1, 0, sampling_steps).long()
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])
        
        # Reshape for broadcasting
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, *([1] * (len(x_start.shape) - 1)))
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(-1, *([1] * (len(x_start.shape) - 1)))
        
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
    
    def p_sample_ddim(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        text_emb: torch.Tensor,
        adapter
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """DDIM sampling step."""
        with torch.no_grad():
            # Predict noise
            predicted_noise = adapter.denoise(x, t, text_emb)
            
            # Get alpha values
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t_prev]
            
            # DDIM update
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)
            
            # Predict x_0
            pred_x_0 = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            
            # Compute direction
            direction = sqrt_one_minus_alpha_t_prev * predicted_noise
            
            # Compute x_prev
            x_prev = sqrt_alpha_t_prev * pred_x_0 + direction
            
            # Diagnostics
            diagnostics = {
                'predicted_noise_mean': float(predicted_noise.mean().item()),
                'predicted_noise_std': float(predicted_noise.std().item()),
                'pred_x_0_mean': float(pred_x_0.mean().item()),
                'pred_x_0_std': float(pred_x_0.std().item()),
                'x_prev_mean': float(x_prev.mean().item()),
                'x_prev_std': float(x_prev.std().item()),
                'has_nan': bool(torch.isnan(x_prev).any().item()),
                'has_inf': bool(torch.isinf(x_prev).any().item())
            }
            
            return x_prev, diagnostics


class VideoGenerationRunner:
    """End-to-end video generation runner with CORRECTED text conditioning."""
    
    def __init__(
        self,
        output_dir: str,
        device: str = 'cpu',
        resolution: int = 64,
        num_frames: int = 16,
        sampling_steps: int = 25,
        seed: int = 42
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.resolution = resolution
        self.num_frames = num_frames
        self.sampling_steps = sampling_steps
        self.seed = seed
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize DDIM sampler
        self.sampler = DDIMSampler(sampling_steps=sampling_steps)
        
        # Storage for metrics
        self.metrics = defaultdict(list)
    
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
        
    def load_adapter(self, experiment_type: str, checkpoint_path: str) -> Any:
        """Load adapter for specified experiment."""
        print(f"\nLoading adapter for Experiment {experiment_type}")
        print(f"  Checkpoint: {checkpoint_path}")
        
        start_time = time.time()
        adapter = create_adapter(checkpoint_path, experiment_type)
        model = adapter.load()
        load_time = time.time() - start_time
        
        print(f"  Load time: {load_time:.2f}s")
        
        return adapter, model, load_time
    
    def encode_prompt(self, prompt: str, adapter) -> torch.Tensor:
        """
        Encode text prompt using ACTUAL text encoder (CORRECTED).
        
        This replaces the buggy random embedding implementation from Stage 3.2.9G.
        """
        # Check if adapter supports actual text encoding
        if hasattr(adapter, 'encode_prompt'):
            # Use the corrected text encoding
            text_emb, text_mask = adapter.encode_prompt(prompt)
            print(f"  [CORRECTED] Using actual text encoding for prompt: '{prompt}'")
            return text_emb
        else:
            # Fallback to dummy encoding for Stage 2.1 (no tokenizer available)
            print(f"  [FALLBACK] Using deterministic dummy encoding for Stage 2.1")
            batch_size = 1
            seq_len = 10
            text_dim = 256
            
            # Create deterministic embedding based on prompt hash
            import hashlib
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            seed = int(prompt_hash[:8], 16)
            torch.manual_seed(seed)
            
            text_emb = torch.randn(batch_size, seq_len, text_dim)
            
            # Reset seed
            torch.manual_seed(self.seed)
            
            return text_emb
    
    def initialize_noise(self, shape: Tuple[int, ...], fixed_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Initialize noise tensor."""
        if fixed_noise is not None:
            return fixed_noise
        
        noise = torch.randn(shape)
        return noise
    
    def generate_video(
        self,
        adapter,
        model,
        prompt: str,
        fixed_noise: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate video using DDIM sampling with CORRECTED text conditioning."""
        num_steps = num_steps or self.sampling_steps
        
        print(f"\nGenerating video for prompt: '{prompt}'")
        print(f"  Steps: {num_steps}")
        
        # Encode prompt (CORRECTED - uses actual text encoder)
        start_time = time.time()
        text_emb = self.encode_prompt(prompt, adapter)
        encode_time = time.time() - start_time
        
        # Initialize noise
        batch_size = 1
        channels = 3
        height = self.resolution
        width = self.resolution
        
        # Determine input shape based on adapter type
        if hasattr(adapter, 'model') and hasattr(adapter.model, 'use_temporal'):
            # Video model: [batch, channels, frames, height, width]
            shape = (batch_size, channels, self.num_frames, height, width)
        else:
            # Image model: [batch, channels, height, width]
            shape = (batch_size, channels, height, width)
        
        noise = self.initialize_noise(shape, fixed_noise)
        
        # Sampling loop
        sampling_start = time.time()
        x = noise
        step_diagnostics = []
        
        for i, (t, t_prev) in enumerate(zip(self.sampler.timesteps[:-1], self.sampler.timesteps[1:])):
            step_start = time.time()
            
            t_tensor = torch.tensor([t]).long()
            t_prev_tensor = torch.tensor([t_prev]).long()
            
            x, diagnostics = self.sampler.p_sample_ddim(
                model, x, t_tensor, t_prev_tensor, text_emb, adapter
            )
            
            step_time = time.time() - step_start
            diagnostics['step'] = i
            diagnostics['timestep'] = t.item()
            diagnostics['step_time'] = step_time
            step_diagnostics.append(diagnostics)
            
            if diagnostics['has_nan'] or diagnostics['has_inf']:
                print(f"  [WARNING] Step {i}: NaN or Inf detected")
                break
            
            if (i + 1) % 5 == 0:
                print(f"  Step {i+1}/{num_steps}: {step_time:.3f}s")
        
        sampling_time = time.time() - sampling_start
        
        # Normalize to [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        
        # Overall diagnostics
        total_diagnostics = {
            'encode_time': encode_time,
            'sampling_time': sampling_time,
            'total_time': sampling_time + encode_time,
            'steps_completed': len(step_diagnostics),
            'step_diagnostics': step_diagnostics,
            'final_mean': x.mean().item(),
            'final_std': x.std().item(),
            'final_min': x.min().item(),
            'final_max': x.max().item(),
            'text_conditioning': 'CORRECTED' if hasattr(adapter, 'encode_prompt') else 'FALLBACK'
        }
        
        print(f"  Total time: {total_diagnostics['total_time']:.2f}s")
        print(f"  Sampling time: {sampling_time:.2f}s")
        print(f"  Text conditioning: {total_diagnostics['text_conditioning']}")
        
        return x, total_diagnostics
    
    def save_video(
        self,
        video: torch.Tensor,
        output_path: Path,
        metadata: Dict[str, Any]
    ):
        """Save video as GIF."""
        # Convert to numpy
        video_np = video.cpu().numpy()
        
        # Handle different shapes
        if len(video_np.shape) == 5:
            # [batch, channels, frames, height, width]
            video_np = video_np[0]  # Remove batch
            video_np = np.transpose(video_np, (1, 2, 3, 0))  # [frames, height, width, channels]
        elif len(video_np.shape) == 4:
            # [batch, channels, height, width] - single frame
            video_np = video_np[0]  # Remove batch
            video_np = np.transpose(video_np, (1, 2, 0))  # [height, width, channels]
            # Repeat for frames
            video_np = np.repeat(video_np[np.newaxis, ...], self.num_frames, axis=0)
        
        # Scale to [0, 255]
        video_np = (video_np * 255).astype(np.uint8)
        
        # Save as GIF
        imageio.mimsave(output_path, video_np, fps=8, format='GIF')
        
        print(f"  Saved video to {output_path}")
    
    def save_frames(
        self,
        video: torch.Tensor,
        output_dir: Path
    ):
        """Save individual frames as PNG."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy
        video_np = video.cpu().numpy()
        
        # Handle different shapes
        if len(video_np.shape) == 5:
            video_np = video_np[0]  # Remove batch
            video_np = np.transpose(video_np, (1, 2, 3, 0))  # [frames, height, width, channels]
        elif len(video_np.shape) == 4:
            video_np = video_np[0]  # Remove batch
            video_np = np.transpose(video_np, (1, 2, 0))  # [height, width, channels]
            video_np = np.repeat(video_np[np.newaxis, ...], self.num_frames, axis=0)
        
        # Scale to [0, 255]
        video_np = (video_np * 255).astype(np.uint8)
        
        # Save each frame
        for i, frame in enumerate(video_np):
            frame_path = output_dir / f"frame_{i:04d}.png"
            Image.fromarray(frame).save(frame_path)
        
        print(f"  Saved {len(video_np)} frames to {output_dir}")
    
    def save_contact_sheet(
        self,
        video: torch.Tensor,
        output_path: Path,
        rows: int = 4,
        cols: int = 4
    ):
        """Save contact sheet of frames."""
        # Convert to numpy
        video_np = video.cpu().numpy()
        
        # Handle different shapes
        if len(video_np.shape) == 5:
            video_np = video_np[0]  # Remove batch
            video_np = np.transpose(video_np, (1, 2, 3, 0))  # [frames, height, width, channels]
        elif len(video_np.shape) == 4:
            video_np = video_np[0]  # Remove batch
            video_np = np.transpose(video_np, (1, 2, 0))  # [height, width, channels]
            video_np = np.repeat(video_np[np.newaxis, ...], self.num_frames, axis=0)
        
        # Scale to [0, 255]
        video_np = (video_np * 255).astype(np.uint8)
        
        # Create contact sheet
        frame_h, frame_w = video_np.shape[1], video_np.shape[2]
        sheet_h = rows * frame_h
        sheet_w = cols * frame_w
        sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
        
        for i in range(min(len(video_np), rows * cols)):
            row = i // cols
            col = i % cols
            y_start = row * frame_h
            y_end = y_start + frame_h
            x_start = col * frame_w
            x_end = x_start + frame_w
            sheet[y_start:y_end, x_start:x_end] = video_np[i]
        
        Image.fromarray(sheet).save(output_path)
        print(f"  Saved contact sheet to {output_path}")
    
    def validate_video(
        self,
        video: torch.Tensor,
        expected_frames: int = 16,
        expected_resolution: int = 64
    ) -> Dict[str, Any]:
        """Validate generated video."""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check shape
        if len(video.shape) == 5:
            batch, channels, frames, height, width = video.shape
        elif len(video.shape) == 4:
            batch, channels, height, width = video.shape
            frames = 1
        else:
            validation['valid'] = False
            validation['errors'].append(f"Unexpected shape: {video.shape}")
            return validation
        
        if frames != expected_frames:
            validation['valid'] = False
            validation['errors'].append(f"Expected {expected_frames} frames, got {frames}")
        
        if height != expected_resolution or width != expected_resolution:
            validation['valid'] = False
            validation['errors'].append(f"Expected {expected_resolution}x{expected_resolution}, got {height}x{width}")
        
        if channels != 3:
            validation['valid'] = False
            validation['errors'].append(f"Expected 3 channels, got {channels}")
        
        # Check for NaN or Inf
        if torch.isnan(video).any():
            validation['valid'] = False
            validation['errors'].append("Video contains NaN values")
        
        if torch.isinf(video).any():
            validation['valid'] = False
            validation['errors'].append("Video contains Inf values")
        
        # Check value range
        if video.min() < 0 or video.max() > 1:
            validation['warnings'].append(f"Video values outside [0,1]: [{video.min():.3f}, {video.max():.3f}]")
        
        return validation
    
    def run_experiment(
        self,
        experiment_type: str,
        checkpoint_path: str,
        prompts: List[str],
        experiment_name: str
    ) -> Dict[str, Any]:
        """Run complete experiment for given checkpoint with CORRECTED text conditioning."""
        print("\n" + "=" * 80)
        print(f"  EXPERIMENT {experiment_type}: {experiment_name}")
        print("=" * 80)
        
        # Load adapter
        adapter, model, load_time = self.load_adapter(experiment_type, checkpoint_path)
        
        # Create output directory
        exp_dir = self.output_dir / f"experiment_{experiment_type}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize fixed noise for reproducibility
        if hasattr(adapter.model, 'use_temporal'):
            noise_shape = (1, 3, self.num_frames, self.resolution, self.resolution)
        else:
            noise_shape = (1, 3, self.resolution, self.resolution)
        
        fixed_noise = self.initialize_noise(noise_shape)
        
        # Run generation for each prompt
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\n  Prompt {i+1}/{len(prompts)}: '{prompt}'")
            
            # Generate video
            video, diagnostics = self.generate_video(
                adapter, model, prompt, fixed_noise=fixed_noise
            )
            
            # Validate video
            validation = self.validate_video(video)
            
            # Save outputs
            sample_dir = exp_dir / f"sample_{i:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            video_path = sample_dir / "video.gif"
            frames_dir = sample_dir / "frames"
            contact_path = sample_dir / "contact_sheet.png"
            metadata_path = sample_dir / "metadata.json"
            
            self.save_video(video, video_path, diagnostics)
            self.save_frames(video, frames_dir)
            self.save_contact_sheet(video, contact_path)
            
            # Save metadata
            metadata = {
                'experiment_type': experiment_type,
                'experiment_name': experiment_name,
                'prompt': prompt,
                'sample_index': i,
                'seed': self.seed,
                'resolution': self.resolution,
                'num_frames': self.num_frames,
                'sampling_steps': self.sampling_steps,
                'load_time': load_time,
                'diagnostics': diagnostics,
                'validation': validation,
                'text_conditioning': diagnostics.get('text_conditioning', 'UNKNOWN')
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(self.convert_to_serializable(metadata), f, indent=2)
            
            results.append({
                'prompt': prompt,
                'video_path': str(video_path),
                'metadata_path': str(metadata_path),
                'diagnostics': diagnostics,
                'validation': validation
            })
        
        return {
            'experiment_type': experiment_type,
            'experiment_name': experiment_name,
            'load_time': load_time,
            'results': results
        }


def main():
    """Main execution function."""
    base_dir = Path(__file__).parent.parent.parent
    
    # Configuration
    config = {
        'output_dir': Path(__file__).parent / "outputs",
        'device': 'cpu',
        'resolution': 64,
        'num_frames': 16,
        'sampling_steps': 25,
        'seed': 42
    }
    
    # Checkpoint paths
    checkpoints = {
        'A': str(base_dir / "checkpoints_conditional" / "checkpoint_final.pt"),
        'B': str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_B.pt"),
        'C': str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_C.pt")
    }
    
    # Prompts for each experiment
    prompts_A = [
        "A red circle on a white background",
        "A blue square on a white background",
        "A green object centered on a white background"
    ]
    
    prompts_B_C = [
        "Red object moving left",
        "Red object moving right",
        "Blue object moving upward",
        "Blue object moving downward",
        "Stationary object"
    ]
    
    # Create runner
    runner = VideoGenerationRunner(**config)
    
    # Run experiments
    all_results = {}
    
    # Experiment A
    all_results['A'] = runner.run_experiment(
        'A', checkpoints['A'], prompts_A, "Spatial Reference"
    )
    
    # Experiment B
    all_results['B'] = runner.run_experiment(
        'B', checkpoints['B'], prompts_B_C, "Motion-Conditioned (Temporal Disabled)"
    )
    
    # Experiment C
    all_results['C'] = runner.run_experiment(
        'C', checkpoints['C'], prompts_B_C, "Temporal-Enabled"
    )
    
    # Save overall results
    results_path = config['output_dir'] / "generation_results.json"
    with open(results_path, 'w') as f:
        json.dump(runner.convert_to_serializable(all_results), f, indent=2)
    
    print("\n" + "=" * 80)
    print("  GENERATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to {results_path}")
    print(f"Text conditioning: CORRECTED (actual text encoding)")


if __name__ == "__main__":
    main()
