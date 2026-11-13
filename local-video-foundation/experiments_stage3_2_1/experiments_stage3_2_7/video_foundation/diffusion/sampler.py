"""
Convenience sampler for diffusion models.

Provides a high-level interface for the diffusion process.
"""

import torch
from typing import Optional
from .schedule import GaussianDiffusionSchedule
from .process import DiffusionProcess


class DiffusionSampler:
    """
    High-level sampler combining schedule and process.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear"
    ):
        """
        Initialize the sampler.

        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            schedule: Type of schedule (linear, quadratic, or sigmoid)
        """
        self.schedule = GaussianDiffusionSchedule(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule=schedule
        )
        self.process = DiffusionProcess(self.schedule)

    def get_training_loss(
        self,
        model: torch.nn.Module,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            model: Denoising model
            x_0: Clean images
            timesteps: Diffusion timesteps
            noise: Optional pre-sampled noise

        Returns:
            Loss scalar
        """
        return self.process.p_losses(model, x_0, timesteps, noise)

    def sample(
        self,
        model: torch.nn.Module,
        batch_size: int = 1,
        channels: int = 3,
        image_size: int = 64,
        num_sampling_steps: int = 50,
        device: torch.device = torch.device('cpu'),
        progress_callback: callable = None,
        clip_x0: bool = True,
        text_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample images using DDIM with optional text conditioning.

        Args:
            model: Trained denoising model
            batch_size: Number of images to generate
            channels: Number of channels
            image_size: Image size
            num_sampling_steps: Number of DDIM steps
            device: Device to run on
            progress_callback: Optional progress callback
            clip_x0: Whether to clip x_0 predictions
            text_emb: Optional text embeddings for conditional generation
            text_mask: Optional text attention mask

        Returns:
            Generated images
        """
        return self.process.ddim_sample_loop(
            model=model,
            batch_size=batch_size,
            channels=channels,
            image_size=image_size,
            num_sampling_steps=num_sampling_steps,
            device=device,
            progress_callback=progress_callback,
            clip_x0=clip_x0,
            text_emb=text_emb,
            text_mask=text_mask
        )

    def sample_ddpm(
        self,
        model: torch.nn.Module,
        batch_size: int = 1,
        channels: int = 3,
        image_size: int = 64,
        device: torch.device = torch.device('cpu'),
        progress_callback: callable = None,
        text_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample images using DDPM (ancestral sampling).

        Args:
            model: Trained denoising model
            batch_size: Number of images to generate
            channels: Number of channels
            image_size: Image size
            device: Device to run on
            progress_callback: Optional progress callback
            text_emb: Optional text embeddings for conditional generation
            text_mask: Optional text attention mask

        Returns:
            Generated images
        """
        return self.process.ddpm_sample_loop(
            model=model,
            batch_size=batch_size,
            channels=channels,
            image_size=image_size,
            device=device,
            progress_callback=progress_callback,
            text_emb=text_emb,
            text_mask=text_mask
        )
