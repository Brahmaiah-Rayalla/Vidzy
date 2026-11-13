"""
Stage 3.2.9F - Controlled Comparison Generator

Generate matched comparison samples for Experiments A, B, C, and D.

Author: Devin
Date: 2026-11-02
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import json
import time
from PIL import Image
import imageio

from video_generation_runner import VideoGenerationRunner, DDIMSampler
from inference_adapters import create_adapter
from experiment_d_adapter import create_experiment_d_adapter


class ComparisonGenerator:
    """Generate controlled comparison samples for motion evaluation."""
    
    def __init__(self, output_dir: str, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # DDIM sampler
        self.sampler = DDIMSampler(sampling_steps=25)
        
    def load_adapters(self, base_dir: Path) -> Dict[str, str]:
        """Return checkpoint paths (load adapters on-demand to avoid memory issues)."""
        checkpoints = {
            'A': str(base_dir / "checkpoints_conditional" / "checkpoint_final.pt"),
            'B': str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_B.pt"),
            'C': str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_C.pt")
        }
        return checkpoints
    
    def generate_comparison_sample(
        self,
        checkpoints: Dict[str, str],
        prompt: str,
        seed: int,
        experiment_types: List[str] = ['B', 'C', 'D']
    ) -> Dict[str, Any]:
        """Generate comparison sample for given prompt and seed."""
        print(f"\n{'='*80}")
        print(f"Generating comparison for: '{prompt}' (seed={seed})")
        print(f"{'='*80}")
        
        results = {}
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create fixed noise for video models
        noise_shape = (1, 3, 16, 64, 64)
        fixed_noise = torch.randn(noise_shape)
        
        for exp_type in experiment_types:
            print(f"\n  Experiment {exp_type}:")
            
            # Load adapter on-demand
            if exp_type == 'D':
                print(f"    Loading adapter (uses C checkpoint with temporal disabled)...")
                adapter = create_experiment_d_adapter(checkpoints['C'])
            else:
                print(f"    Loading adapter...")
                adapter = create_adapter(checkpoints[exp_type], exp_type)
            
            model = adapter.load()
            
            # Encode prompt (deterministic based on seed)
            text_emb = self.encode_prompt(prompt, seed)
            
            # Generate video
            video, diagnostics = self.generate_video(
                adapter, prompt, fixed_noise, text_emb
            )
            
            # Free memory
            del adapter
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Save outputs (include prompt in path for uniqueness)
            prompt_safe = prompt.replace(' ', '_')
            sample_dir = self.output_dir / f"seed_{seed}" / f"experiment_{exp_type}" / prompt_safe
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            video_path = sample_dir / "video.gif"
            frames_dir = sample_dir / "frames"
            contact_path = sample_dir / "contact_sheet.png"
            metadata_path = sample_dir / "metadata.json"
            
            self.save_video(video, video_path)
            self.save_frames(video, frames_dir)
            self.save_contact_sheet(video, contact_path)
            
            # Save metadata
            metadata = {
                'experiment_type': exp_type,
                'prompt': prompt,
                'seed': seed,
                'diagnostics': diagnostics
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(self.convert_to_serializable(metadata), f, indent=2)
            
            results[exp_type] = {
                'video_path': str(video_path),
                'frames_dir': str(frames_dir),
                'metadata_path': str(metadata_path),
                'diagnostics': diagnostics
            }
        
        return results
    
    def encode_prompt(self, prompt: str, seed: int) -> torch.Tensor:
        """Encode text prompt deterministically."""
        batch_size = 1
        seq_len = 10
        text_dim = 256
        
        torch.manual_seed(seed)
        text_emb = torch.randn(batch_size, seq_len, text_dim)
        torch.manual_seed(self.seed)
        
        return text_emb
    
    def generate_video(
        self,
        adapter,
        prompt: str,
        fixed_noise: torch.Tensor,
        text_emb: torch.Tensor
    ) -> tuple:
        """Generate video using DDIM sampling."""
        x = fixed_noise
        step_diagnostics = []
        
        for i, (t, t_prev) in enumerate(zip(self.sampler.timesteps[:-1], self.sampler.timesteps[1:])):
            t_tensor = torch.tensor([t])
            t_prev_tensor = torch.tensor([t_prev])
            
            x, diag = self.sampler.p_sample_ddim(
                adapter.model, x, t_tensor, t_prev_tensor, text_emb, adapter
            )
            
            diag['step'] = i
            diag['timestep'] = t.item()
            step_diagnostics.append(diag)
            
            if diag['has_nan'] or diag['has_inf']:
                break
        
        # Normalize
        x = (x.clamp(-1, 1) + 1) / 2
        
        total_diagnostics = {
            'steps_completed': len(step_diagnostics),
            'step_diagnostics': step_diagnostics,
            'final_mean': float(x.mean().item()),
            'final_std': float(x.std().item())
        }
        
        return x, total_diagnostics
    
    def save_video(self, video: torch.Tensor, output_path: Path):
        """Save video as GIF."""
        video_np = video.cpu().numpy()
        video_np = video_np[0]  # Remove batch
        video_np = np.transpose(video_np, (1, 2, 3, 0))  # [frames, h, w, c]
        video_np = (video_np * 255).astype(np.uint8)
        imageio.mimsave(output_path, video_np, fps=8, format='GIF')
        print(f"    Saved: {output_path.name}")
    
    def save_frames(self, video: torch.Tensor, output_dir: Path):
        """Save individual frames."""
        output_dir.mkdir(parents=True, exist_ok=True)
        video_np = video.cpu().numpy()
        video_np = video_np[0]
        video_np = np.transpose(video_np, (1, 2, 3, 0))
        video_np = (video_np * 255).astype(np.uint8)
        
        for i, frame in enumerate(video_np):
            frame_path = output_dir / f"frame_{i:04d}.png"
            Image.fromarray(frame).save(frame_path)
    
    def save_contact_sheet(self, video: torch.Tensor, output_path: Path):
        """Save contact sheet."""
        video_np = video.cpu().numpy()
        video_np = video_np[0]
        video_np = np.transpose(video_np, (1, 2, 3, 0))
        video_np = (video_np * 255).astype(np.uint8)
        
        rows, cols = 4, 4
        frame_h, frame_w = video_np.shape[1], video_np.shape[2]
        sheet = np.zeros((rows * frame_h, cols * frame_w, 3), dtype=np.uint8)
        
        for i in range(min(len(video_np), rows * cols)):
            row = i // cols
            col = i % cols
            y_start = row * frame_h
            y_end = y_start + frame_h
            x_start = col * frame_w
            x_end = x_start + frame_w
            sheet[y_start:y_end, x_start:x_end] = video_np[i]
        
        Image.fromarray(sheet).save(output_path)
    
    def convert_to_serializable(self, obj):
        """Convert to JSON-serializable format."""
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


def main():
    """Main execution function."""
    base_dir = Path(__file__).parent.parent.parent
    output_dir = Path(__file__).parent / "comparison_outputs"
    
    # Prompts for motion evaluation
    prompts = [
        "Red object moving left",
        "Red object moving right",
        "Blue object moving upward",
        "Blue object moving downward",
        "Stationary object"
    ]
    
    # Two seeds per prompt
    seeds = [42, 43]
    
    # Create generator
    generator = ComparisonGenerator(str(output_dir), seed=42)
    
    # Get checkpoint paths
    checkpoints = generator.load_adapters(base_dir)
    
    # Generate comparisons
    all_results = {}
    
    for prompt in prompts:
        for seed in seeds:
            result = generator.generate_comparison_sample(
                checkpoints, prompt, seed, experiment_types=['B', 'C', 'D']
            )
            key = f"{prompt.replace(' ', '_')}_seed{seed}"
            all_results[key] = result
    
    # Save results
    results_path = output_dir / "comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(generator.convert_to_serializable(all_results), f, indent=2)
    
    print(f"\n{'='*80}")
    print("COMPARISON GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
