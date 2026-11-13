"""
Diffusion noise schedule.

Implements beta schedules and computes alpha, alpha_bar, and related quantities.
"""

import torch
import numpy as np
from typing import Literal


class GaussianDiffusionSchedule:
    """
    Gaussian diffusion noise schedule.

    Computes the noise schedule parameters for DDPM/DDIM:
    - beta_t: noise variance at timestep t
    - alpha_t: 1 - beta_t
    - alpha_bar_t: cumulative product of alphas up to timestep t
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: Literal["linear", "quadratic", "sigmoid"] = "linear"
    ):
        """
        Initialize the noise schedule.

        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            schedule: Type of schedule (linear, quadratic, or sigmoid)
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule

        # Compute beta schedule
        self.betas = self._get_beta_schedule()

        # Compute derived quantities
        # alpha_t = 1 - beta_t
        self.alphas = 1.0 - self.betas

        # alpha_bar_t = cumulative product of alphas
        # alpha_bar_t = prod_{s=1}^t alpha_s
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # For DDIM sampling: 1 - alpha_bar_t
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod

        # For posterior variance (not used in DDIM but kept for reference)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

    def _get_beta_schedule(self) -> torch.Tensor:
        """Compute the beta schedule based on the specified type."""
        if self.schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.schedule == "quadratic":
            return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_timesteps) ** 2
        elif self.schedule == "sigmoid":
            betas = torch.linspace(-6, 6, self.num_timesteps)
            return torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def get(
        self,
        timestep: torch.Tensor,
        device: torch.device = None
    ) -> tuple:
        """
        Get schedule values for specific timesteps.

        Args:
            timestep: Tensor of timesteps of shape [batch_size]
            device: Device to place tensors on

        Returns:
            Tuple of (alpha, alpha_bar, one_minus_alpha_bar) for the given timesteps
        """
        if device is None:
            device = timestep.device

        # Ensure timestep is a long tensor for indexing
        timestep = timestep.long()

        # Gather values for the given timesteps
        alpha = self.alphas[timestep].to(device)
        alpha_bar = self.alphas_cumprod[timestep].to(device)
        one_minus_alpha_bar = self.one_minus_alphas_cumprod[timestep].to(device)

        # Reshape for broadcasting: [batch_size] -> [batch_size, 1, 1, 1]
        alpha = alpha.view(-1, 1, 1, 1)
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)
        one_minus_alpha_bar = one_minus_alpha_bar.view(-1, 1, 1, 1)

        return alpha, alpha_bar, one_minus_alpha_bar
