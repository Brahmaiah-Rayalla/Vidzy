"""
Diffusion forward and reverse processes.

Implements:
- Forward diffusion: adding noise to clean images
- Reverse diffusion: DDIM and DDPM sampling for generation
"""

import torch
from typing import Optional
from .schedule import GaussianDiffusionSchedule


class DiffusionProcess:
    """
    Diffusion process for training and sampling.

    Forward process (training):
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    Reverse process (DDIM sampling):
        x_{t-1} = sqrt(alpha_bar_{t-1}) * (x_t - sqrt(1 - alpha_bar_t) * epsilon_pred) / sqrt(alpha_bar_t)
                  + sqrt(1 - alpha_bar_{t-1}) * epsilon_pred

    Reverse process (DDPM sampling):
        x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * epsilon_pred)
                  + sqrt(beta_t) * z
    """

    def __init__(self, schedule: GaussianDiffusionSchedule):
        """
        Initialize the diffusion process.

        Args:
            schedule: GaussianDiffusionSchedule instance
        """
        self.schedule = schedule

    def q_sample(
        self,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Forward diffusion: add noise to clean images.

        Args:
            x_0: Clean images of shape [batch, channels, height, width]
            timesteps: Diffusion timesteps of shape [batch]
            noise: Optional pre-sampled noise of same shape as x_0

        Returns:
            Tuple of (noisy_images, noise) where:
            - noisy_images: x_t of shape [batch, channels, height, width]
            - noise: epsilon of shape [batch, channels, height, width]
        """
        device = x_0.device
        batch_size = x_0.shape[0]

        # Sample random noise if not provided
        if noise is None:
            noise = torch.randn_like(x_0)

        # Get schedule values for the given timesteps
        _, alpha_bar, one_minus_alpha_bar = self.schedule.get(timesteps, device)

        # Schedules are returned as [B,1,1,1] for images. Extend their rank for
        # video tensors [B,C,T,H,W] without changing the image behavior.
        if x_0.ndim != alpha_bar.ndim:
            target_shape = (batch_size,) + (1,) * (x_0.ndim - 1)
            alpha_bar = alpha_bar.reshape(target_shape)
            one_minus_alpha_bar = one_minus_alpha_bar.reshape(target_shape)

        # Compute noisy images:
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(one_minus_alpha_bar)

        noisy_images = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        return noisy_images, noise

    def p_losses(
        self,
        model: torch.nn.Module,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute training loss for noise prediction.

        Args:
            model: Denoising model (U-Net)
            x_0: Clean images of shape [batch, channels, height, width]
            timesteps: Diffusion timesteps of shape [batch]
            noise: Optional pre-sampled noise

        Returns:
            Loss scalar (mean squared error)
        """
        # Forward diffusion
        x_t, noise = self.q_sample(x_0, timesteps, noise)

        # Predict noise
        predicted_noise = model(x_t, timesteps)

        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        return loss

    def ddim_sample(
        self,
        model: torch.nn.Module,
        shape: tuple,
        num_sampling_steps: int = 50,
        device: torch.device = torch.device('cpu'),
        progress_callback: Optional[callable] = None,
        clip_x0: bool = True,
        clip_range: tuple = (-1.0, 1.0),
        text_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        initial_noise: Optional[torch.Tensor] = None,
        recompute_noise_after_clipping: bool = False,
        prediction_type: str = "epsilon",
        class_labels: Optional[torch.Tensor] = None,
        color_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        DDIM sampling for image generation with optional text conditioning.

        Args:
            model: Trained denoising model
            shape: Shape of the output tensor [batch, channels, height, width]
            num_sampling_steps: Number of DDIM sampling steps
            device: Device to run on
            progress_callback: Optional callback for progress updates
            clip_x0: Whether to clip x_0 predictions to prevent instability
            clip_range: Range to clip x_0 predictions to
            text_emb: Optional text embeddings for conditional generation
            text_mask: Optional text attention mask

        Returns:
            Generated images of shape [batch, channels, height, width]
        """
        batch_size, channels, height, width = shape

        # Start from pure Gaussian noise
        x = torch.randn(shape, device=device) if initial_noise is None else initial_noise.to(device).clone()
        if tuple(x.shape) != tuple(shape):
            raise ValueError(f"initial_noise must have shape {shape}, got {tuple(x.shape)}")

        # Create subsampled timesteps
        # We want to go from T to 0 in num_sampling_steps
        total_timesteps = self.schedule.num_timesteps
        timesteps = torch.linspace(total_timesteps - 1, 0, num_sampling_steps, dtype=torch.long, device=device)

        # Reverse diffusion loop
        for i, t in enumerate(timesteps):
            if progress_callback:
                progress_callback(i, num_sampling_steps)

            # Predict noise at current timestep (with or without conditioning)
            t_batch = t.expand(batch_size)
            if class_labels is not None or color_labels is not None:
                labels = class_labels.to(device=device, dtype=torch.long) if class_labels is not None else None
                colors = color_labels.to(device=device, dtype=torch.long) if color_labels is not None else None
                for name, values in (("class", labels), ("color", colors)):
                    if values is not None and values.numel() != batch_size:
                        raise ValueError(f"Expected {batch_size} {name} labels, got {values.numel()}")
                predicted_noise = model(x, t_batch, labels, colors)
            elif text_emb is not None:
                predicted_noise = model(x, t_batch, text_emb, text_mask)
            else:
                predicted_noise = model(x, t_batch)

            # Get schedule values for current and previous timesteps
            _, alpha_bar_t, _ = self.schedule.get(t_batch, device)

            # Previous timestep (or 0 for the last step)
            if i < num_sampling_steps - 1:
                t_prev = timesteps[i + 1]
                t_prev_batch = t_prev.expand(batch_size)
                _, alpha_bar_t_prev, _ = self.schedule.get(t_prev_batch, device)
            else:
                alpha_bar_t_prev = torch.ones_like(alpha_bar_t)

            # DDIM update:
            # x_{t-1} = sqrt(alpha_bar_{t-1}) * (x_t - sqrt(1 - alpha_bar_t) * epsilon_pred) / sqrt(alpha_bar_t)
            #           + sqrt(1 - alpha_bar_{t-1}) * epsilon_pred
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1 - alpha_bar_t_prev)

            if prediction_type == "epsilon":
                epsilon_pred = predicted_noise
                x_0_pred = (x - sqrt_one_minus_alpha_bar_t * epsilon_pred) / sqrt_alpha_bar_t
            elif prediction_type == "v_prediction":
                # v = sqrt(alpha_bar)*epsilon - sqrt(1-alpha_bar)*x0
                x_0_pred = sqrt_alpha_bar_t * x - sqrt_one_minus_alpha_bar_t * predicted_noise
                epsilon_pred = sqrt_one_minus_alpha_bar_t * x + sqrt_alpha_bar_t * predicted_noise
            else:
                raise ValueError(f"Unsupported prediction_type: {prediction_type}")

            # Clip x_0 prediction to prevent instability at high noise levels
            # Note: Per Stage 1.8, we avoid per-step clipping to prevent mode collapse
            # Only clip if explicitly requested (e.g., for numerical stability)
            if clip_x0:
                x_0_pred = torch.clamp(x_0_pred, clip_range[0], clip_range[1])
                if recompute_noise_after_clipping:
                    # Keep the clipped x0 estimate and epsilon estimate
                    # algebraically consistent before the DDIM update.
                    epsilon_pred = (
                        x - sqrt_alpha_bar_t * x_0_pred
                    ) / sqrt_one_minus_alpha_bar_t.clamp_min(1e-8)

            # Compute x_{t-1}
            x = sqrt_alpha_bar_t_prev * x_0_pred + sqrt_one_minus_alpha_bar_t_prev * epsilon_pred

        # Final clipping after sampling complete (Stage 1.8 fix)
        x = torch.clamp(x, clip_range[0], clip_range[1])

        return x

    def ddpm_sample(
        self,
        model: torch.nn.Module,
        shape: tuple,
        device: torch.device = torch.device('cpu'),
        progress_callback: Optional[callable] = None,
        text_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Standard DDPM (ancestral) sampling for image generation.

        Args:
            model: Trained denoising model
            shape: Shape of the output tensor [batch, channels, height, width]
            device: Device to run on
            progress_callback: Optional callback for progress updates
            text_emb: Optional text embeddings for conditional generation
            text_mask: Optional text attention mask

        Returns:
            Generated images of shape [batch, channels, height, width]
        """
        batch_size, channels, height, width = shape

        # Start from pure Gaussian noise
        x = torch.randn(shape, device=device)

        # Reverse diffusion loop (go through all timesteps)
        timesteps = torch.arange(self.schedule.num_timesteps - 1, -1, -1, device=device, dtype=torch.long)

        for i, t in enumerate(timesteps):
            if progress_callback:
                progress_callback(i, len(timesteps))

            t_batch = t.expand(batch_size)

            # Predict noise (with or without conditioning)
            if text_emb is not None:
                predicted_noise = model(x, t_batch, text_emb, text_mask)
            else:
                predicted_noise = model(x, t_batch)

            # Get schedule values
            alpha, alpha_bar, one_minus_alpha_bar = self.schedule.get(t_batch, device)
            beta = 1 - alpha

            # Compute posterior mean and variance
            sqrt_alpha = torch.sqrt(alpha)
            sqrt_alpha_bar = torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha_bar = torch.sqrt(one_minus_alpha_bar)

            # Predict x_0
            x_0_pred = (x - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar

            # Note: We do NOT clip x_0_pred here to avoid mode collapse
            # Clipping during sampling destroys the signal at high timesteps
            # Final clipping will be applied after sampling is complete

            # Get previous alpha_bar
            if t > 0:
                alpha_bar_prev = self.schedule.alphas_cumprod[t - 1].to(device)
            else:
                alpha_bar_prev = torch.tensor(1.0).to(device)

            sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)

            # Compute posterior mean
            posterior_mean = (sqrt_alpha_bar_prev * beta / one_minus_alpha_bar) * x_0_pred + \
                           (sqrt_alpha * (1 - alpha_bar_prev) / one_minus_alpha_bar) * x

            # Compute posterior variance
            if t > 0:
                posterior_variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
            else:
                posterior_variance = torch.tensor(0.0).to(device)

            # Sample from posterior
            if t > 0:
                noise = torch.randn_like(x)
                x = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                x = posterior_mean

        return x

    def ddim_sample_loop(
        self,
        model: torch.nn.Module,
        batch_size: int = 1,
        channels: int = 3,
        image_size: int = 64,
        num_sampling_steps: int = 50,
        device: torch.device = torch.device('cpu'),
        progress_callback: Optional[callable] = None,
        clip_x0: bool = True,
        text_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        initial_noise: Optional[torch.Tensor] = None,
        recompute_noise_after_clipping: bool = False,
        prediction_type: str = "epsilon",
        class_labels: Optional[torch.Tensor] = None,
        color_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convenience method for DDIM sampling with explicit dimensions and optional text conditioning.

        Args:
            model: Trained denoising model
            batch_size: Number of images to generate
            channels: Number of channels (3 for RGB)
            image_size: Size of the images
            num_sampling_steps: Number of DDIM sampling steps
            device: Device to run on
            progress_callback: Optional progress callback
            clip_x0: Whether to clip x_0 predictions
            text_emb: Optional text embeddings for conditional generation
            text_mask: Optional text attention mask

        Returns:
            Generated images of shape [batch_size, channels, image_size, image_size]
        """
        shape = (batch_size, channels, image_size, image_size)
        return self.ddim_sample(
            model, shape, num_sampling_steps, device, progress_callback,
            clip_x0, (-1.0, 1.0), text_emb, text_mask, initial_noise,
            recompute_noise_after_clipping, prediction_type, class_labels, color_labels
        )

    def ddpm_sample_loop(
        self,
        model: torch.nn.Module,
        batch_size: int = 1,
        channels: int = 3,
        image_size: int = 64,
        device: torch.device = torch.device('cpu'),
        progress_callback: Optional[callable] = None,
        text_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convenience method for DDPM sampling with explicit dimensions.

        Args:
            model: Trained denoising model
            batch_size: Number of images to generate
            channels: Number of channels (3 for RGB)
            image_size: Size of the images
            device: Device to run on
            progress_callback: Optional progress callback
            text_emb: Optional text embeddings for conditional generation
            text_mask: Optional text attention mask

        Returns:
            Generated images of shape [batch_size, channels, image_size, image_size]
        """
        shape = (batch_size, channels, image_size, image_size)
        samples = self.ddpm_sample(model, shape, device, progress_callback, text_emb, text_mask)
        # Apply final clipping to ensure output is in [-1, 1]
        return torch.clamp(samples, -1.0, 1.0)
