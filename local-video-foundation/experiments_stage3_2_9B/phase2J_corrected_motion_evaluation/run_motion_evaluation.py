"""
Stage 3.2.9I - Motion Evaluation with Corrected Text Conditioning

Generates videos for B, C, and D configurations and evaluates motion control.

Author: Devin
Date: 2026-11-03
"""

import torch
import sys
from pathlib import Path
import json
import time
import numpy as np
from PIL import Image
try:
    import imageio
except ImportError:
    imageio = None

# Add parent directories to path
base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "experiments_stage3_2_7" / "video_foundation"))
sys.path.insert(0, str(Path(__file__).parent))

from inference_adapters import create_adapter
from video_generation_runner import VideoGenerationRunner, DDIMSampler


class MotionEvaluationRunner:
    """Runner for motion evaluation with corrected text conditioning."""
    
    def __init__(self, output_dir: str, device: str = 'cpu'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Load prompt mapping
        with open(Path(__file__).parent / "STAGE3_2_9I_PROMPT_MAPPING.json") as f:
            self.prompt_mapping = json.load(f)
        
        # Get evaluation prompts from supported motion categories
        self.prompts = [cat["evaluation_prompt"] for cat in self.prompt_mapping["supported_motion_categories"].values()
                        if cat.get("supported", False)]
        self.seeds = self.prompt_mapping["evaluation_seeds"]
        
        print(f"Loaded {len(self.prompts)} evaluation prompts")
        print(f"Using {len(self.seeds)} seeds: {self.seeds}")
    
    def create_experiment_d_config(self, experiment_c_config: dict) -> dict:
        """Create Experiment D config (C with temporal disabled)."""
        config = experiment_c_config.copy()
        config['use_temporal'] = False
        return config
    
    def generate_videos_for_config(self, config_name: str, checkpoint_path: str, 
                                   config: dict, enable_temporal: bool):
        """Generate videos for a specific configuration."""
        print("\n" + "=" * 80)
        print(f"GENERATING VIDEOS FOR {config_name}")
        print("=" * 80)
        
        # Create adapter
        adapter = create_adapter(checkpoint_path, config_name)
        model = adapter.load()
        
        # Initialize sampler
        sampler = DDIMSampler(sampling_steps=25)
        
        # Create config directory
        config_dir = self.output_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for seed in self.seeds:
            print(f"\nSeed {seed}")
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Create fixed noise
            noise_shape = (1, 3, 16, 64, 64)
            fixed_noise = torch.randn(noise_shape)
            
            for prompt_idx, prompt in enumerate(self.prompts):
                print(f"  Prompt {prompt_idx + 1}/{len(self.prompts)}: '{prompt}'")
                
                # Encode prompt
                text_emb, text_mask = adapter.encode_prompt(prompt)
                
                # Initialize noise
                x = fixed_noise.clone()
                
                # Sampling loop
                step_times = []
                for i, (t, t_prev) in enumerate(zip(sampler.timesteps[:-1], sampler.timesteps[1:])):
                    step_start = time.time()
                    
                    t_tensor = torch.tensor([t]).long()
                    t_prev_tensor = torch.tensor([t_prev]).long()
                    
                    with torch.no_grad():
                        predicted_noise = adapter.denoise(x, t_tensor, text_emb)
                        
                        alpha_t = sampler.alphas_cumprod[t]
                        alpha_t_prev = sampler.alphas_cumprod[t_prev]
                        
                        sqrt_alpha_t = torch.sqrt(alpha_t)
                        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
                        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                        sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)
                        
                        pred_x_0 = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
                        direction = sqrt_one_minus_alpha_t_prev * predicted_noise
                        x = sqrt_alpha_t_prev * pred_x_0 + direction
                    
                    step_time = time.time() - step_start
                    step_times.append(step_time)
                
                # Normalize
                x = (x.clamp(-1, 1) + 1) / 2
                
                # Save video
                sample_dir = config_dir / f"seed_{seed}" / f"prompt_{prompt_idx:02d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                
                video_path = sample_dir / "video.gif"
                metadata_path = sample_dir / "metadata.json"
                
                # Save as GIF
                video_np = x.cpu().numpy()[0]  # Remove batch
                video_np = np.transpose(video_np, (1, 2, 3, 0))  # [frames, height, width, channels]
                video_np = (video_np * 255).astype(np.uint8)
                imageio.mimsave(video_path, video_np, fps=8, format='GIF')
                
                # Save metadata
                metadata = {
                    'config_name': config_name,
                    'checkpoint_path': checkpoint_path,
                    'prompt': prompt,
                    'seed': seed,
                    'enable_temporal': enable_temporal,
                    'sampling_steps': 25,
                    'total_time': sum(step_times),
                    'avg_step_time': np.mean(step_times),
                    'text_conditioning': 'CORRECTED',
                    'video_shape': list(x.shape),
                    'video_mean': float(x.mean().item()),
                    'video_std': float(x.std().item())
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                results.append(metadata)
                print(f"    Saved to {video_path}")
        
        return results
    
    def run_evaluation(self):
        """Run complete motion evaluation."""
        print("\n" + "=" * 80)
        print("STAGE 3.2.9I - MOTION EVALUATION")
        print("=" * 80)
        
        base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
        
        # Checkpoint paths
        checkpoints = {
            'B': str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_B.pt"),
            'C': str(base_dir / "experiments_stage3_2_9B" / "phase2C" / "checkpoints" / "checkpoint_phase2c_C.pt")
        }
        
        # Configs
        config_B = {
            'in_channels': 3,
            'model_channels': 64,
            'channel_mult': [1, 2, 2, 2],
            'num_res_blocks': 2,
            'attention_resolutions': [],
            'dropout': 0.0,
            'num_timesteps': 1000,
            'text_dim': 256,
            'n_heads': 4,
            'use_cross_attn': True,
            'vocab_size': 42,
            'motion_vocab_size': 40,
            'n_layers': 4,
            'd_ff': 512,
            'max_seq_len': 64,
            'num_frames': 16,
            'image_size': 64,
            'temporal_kernel': 3,
            'temporal_layers': 2,
            'freeze_spatial': True,
            'use_temporal': False,
            'device': 'cpu'
        }
        
        config_C = config_B.copy()
        config_C['use_temporal'] = True
        
        config_D = config_C.copy()
        config_D['use_temporal'] = False
        
        all_results = {}
        
        # Generate for Experiment B
        results_B = self.generate_videos_for_config(
            'B', checkpoints['B'], config_B, enable_temporal=False
        )
        all_results['B'] = results_B
        
        # Generate for Experiment C
        results_C = self.generate_videos_for_config(
            'C', checkpoints['C'], config_C, enable_temporal=True
        )
        all_results['C'] = results_C
        
        # Generate for Experiment D (C with temporal disabled)
        results_D = self.generate_videos_for_config(
            'D', checkpoints['C'], config_D, enable_temporal=False
        )
        all_results['D'] = results_D
        
        # Save results
        results_path = self.output_dir / "generation_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print(f"Results saved to {results_path}")
        print(f"Total videos generated: {len(results_B) + len(results_C) + len(results_D)}")
        
        return all_results


def main():
    """Run motion evaluation."""
    output_dir = Path(__file__).parent / "outputs_evaluation"
    runner = MotionEvaluationRunner(str(output_dir))
    results = runner.run_evaluation()
    return results


if __name__ == "__main__":
    main()
