"""CPU-oriented video denoiser built from a trained 2D U-Net."""

import torch
from torch import nn
import torch.nn.functional as F


class CPUVideoUNet(nn.Module):
    """Denoise in motion-canonical coordinates and restore video coordinates.

    One pretrained 2D U-Net is shared over frames. A small per-pixel Conv1d
    refines canonical v-predictions in time, and a nearest-neighbor trajectory
    warp maps them back to requested positions without smoothing diffusion
    noise. This avoids 3D convolutions and keeps the path CPU friendly.
    """

    def __init__(self, spatial_unet: nn.Module, num_frames: int, freeze_spatial: bool = True,
                 use_motion: bool = True, canonicalize_motion: bool = True,
                 gate_temporal_foreground: bool = False):
        super().__init__()
        self.spatial_unet = spatial_unet
        self.num_frames = num_frames
        self.use_motion = use_motion
        self.canonicalize_motion = canonicalize_motion
        self.gate_temporal_foreground = gate_temporal_foreground
        betas = torch.linspace(0.0001, 0.02, 1000)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0), persistent=False)
        if freeze_spatial:
            for parameter in self.spatial_unet.parameters():
                parameter.requires_grad_(False)
        temporal_inputs = 17 if use_motion else 3
        self.temporal = nn.Sequential(
            nn.Conv1d(temporal_inputs, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, 3, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.temporal[-1].weight)
        nn.init.zeros_(self.temporal[-1].bias)
        if use_motion:
            self.motion_embed = nn.Embedding(7, 8)
            self.speed_embed = nn.Embedding(2, 4)
            # Synthetic clip motion is constant velocity. Learn one 2D velocity
            # per direction/speed pair, then integrate it over normalized time.
            self.motion_velocity = nn.Embedding(14, 2)
            nn.init.zeros_(self.motion_velocity.weight)

    def predict_offsets(self, motion_labels, speed_labels, speed_scale=1.0,
                        velocity_px_per_frame=None, image_size=None, reference_frames=None):
        """Predict normalized per-frame displacement relative to clip center."""
        if not self.use_motion:
            raise RuntimeError("This video model has no learned motion head")
        batch = motion_labels.shape[0]
        positions = torch.linspace(-0.5, 0.5, self.num_frames, device=motion_labels.device)
        positions = positions.view(1, self.num_frames, 1)
        if velocity_px_per_frame is not None:
            if image_size is None:
                raise ValueError("image_size is required with velocity_px_per_frame")
            directions = motion_labels.new_tensor((
                (0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0),
                (0.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
            ), dtype=torch.float32)
            direction = directions[motion_labels.long()]
            if isinstance(image_size, (int, float)):
                image_size = (image_size, image_size)
            size_xy = positions.new_tensor(image_size).view(1, 1, 2)
            span_frames = (reference_frames or self.num_frames) - 1
            speed = torch.as_tensor(velocity_px_per_frame, device=positions.device, dtype=positions.dtype).view(-1, 1, 1)
            velocity = direction[:, None, :] * speed * span_frames / size_xy
            return (velocity * positions).clamp(-0.4, 0.4)
        pair_ids = motion_labels.long() * 2 + speed_labels.long()
        velocities = self.motion_velocity(pair_ids).unsqueeze(1)
        offsets = (velocities * positions).clamp(-0.4, 0.4).expand(-1, self.num_frames, -1)
        return offsets * speed_scale

    @staticmethod
    def _translate_video(video, offsets_px, to_canonical, padding_mode="zeros", mode="bilinear"):
        """Translate [B,T,C,H,W] features using per-frame [dx,dy] offsets."""
        batch, frames, channels, height, width = video.shape
        flat = video.reshape(batch * frames, channels, height, width)
        offsets = offsets_px.reshape(batch * frames, 2).to(dtype=video.dtype)
        theta = torch.zeros(batch * frames, 2, 3, device=video.device, dtype=video.dtype)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        sign = 1.0 if to_canonical else -1.0
        theta[:, 0, 2] = sign * 2.0 * offsets[:, 0] / max(width - 1, 1)
        theta[:, 1, 2] = sign * 2.0 * offsets[:, 1] / max(height - 1, 1)
        grid = F.affine_grid(theta, flat.shape, align_corners=True)
        translated = F.grid_sample(flat, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
        return translated.reshape(batch, frames, channels, height, width)

    def forward(self, video, timesteps, shape_labels, color_labels, motion_labels=None, speed_labels=None,
                motion_scale=1.0, velocity_px_per_frame=None, reference_frames=None, size_labels=None):
        batch, channels, frames, height, width = video.shape
        if channels != 3 or frames != self.num_frames:
            raise ValueError(f"Expected [B,3,{self.num_frames},H,W], got {tuple(video.shape)}")
        video_frames = video.permute(0, 2, 1, 3, 4)
        offsets_px = None
        if self.use_motion:
            if motion_labels is None or speed_labels is None:
                raise ValueError("motion_labels and speed_labels are required by this motion-conditioned video denoiser")
            offset_scale = video.new_tensor([width, height]).view(1, 1, 2)
            offsets_px = torch.round(
                self.predict_offsets(
                    motion_labels, speed_labels, speed_scale=motion_scale,
                    velocity_px_per_frame=velocity_px_per_frame,
                    image_size=(width, height), reference_frames=reference_frames,
                ) * offset_scale
            ).detach()
        # The image prior was trained on centered objects. Align the noisy clip
        # before applying it; temporal and explicit background corrections
        # compensate for resampled noise on the transported output.
        denoiser_input = self._translate_video(
            video_frames, offsets_px, to_canonical=True, mode="nearest"
        ) if offsets_px is not None and self.canonicalize_motion else video_frames
        images = denoiser_input.reshape(batch * frames, channels, height, width)
        frame_steps = timesteps.repeat_interleave(frames)
        frame_shapes = shape_labels.repeat_interleave(frames)
        frame_colors = color_labels.repeat_interleave(frames)
        frame_sizes = size_labels.repeat_interleave(frames) if size_labels is not None else None
        if all(not parameter.requires_grad for parameter in self.spatial_unet.parameters()):
            with torch.no_grad():
                spatial = self.spatial_unet(images, frame_steps, frame_shapes, frame_colors, frame_sizes)
        else:
            spatial = self.spatial_unet(images, frame_steps, frame_shapes, frame_colors, frame_sizes)
        spatial = spatial.reshape(batch, frames, channels, height, width)
        # Shared lightweight temporal filtering runs on motion-aligned clean
        # estimates rather than an expensive 3D feature volume.
        pixels = spatial.permute(0, 3, 4, 2, 1).reshape(batch * height * width, channels, frames)
        if self.use_motion:
            motion_features = self.motion_embed(motion_labels.long())
            motion_features = motion_features[:, None, :, None, None].expand(-1, frames, -1, height, width)
            speed_features = self.speed_embed(speed_labels.long())
            speed_features = speed_features[:, None, :, None, None].expand(-1, frames, -1, height, width)
            motion_features = torch.cat((motion_features, speed_features), dim=2)
            motion_features = motion_features.permute(0, 3, 4, 2, 1).reshape(batch * height * width, 12, frames)
            frame_positions = torch.linspace(-0.5, 0.5, frames, device=video.device)
            frame_positions = frame_positions.view(1, 1, frames).expand(batch * height * width, -1, -1)
            timestep_features = (timesteps.to(video.dtype) / 999.0).view(batch, 1, 1, 1, 1)
            timestep_features = timestep_features.expand(-1, height, width, 1, frames)
            timestep_features = timestep_features.reshape(batch * height * width, 1, frames)
            pixels = torch.cat((pixels, motion_features, frame_positions, timestep_features), dim=1)
        temporal_v = self.temporal(pixels).reshape(batch, height, width, channels, frames)
        temporal_v = temporal_v.permute(0, 4, 3, 1, 2).contiguous()
        canonical_v = spatial + temporal_v
        if offsets_px is not None and self.canonicalize_motion:
            alpha_bar = self.alphas_cumprod[timesteps.long()].view(batch, 1, 1, 1, 1).to(video.dtype)
            sqrt_alpha = alpha_bar.sqrt()
            sqrt_one_minus = (1.0 - alpha_bar).clamp_min(1e-8).sqrt()
            canonical_x0 = sqrt_alpha * denoiser_input - sqrt_one_minus * canonical_v
            palette = video.new_tensor(((1, -1, -1), (-1, 1, -1), (-1, -1, 1), (1, 1, -1)))
            foreground_rgb = palette[color_labels.long()].view(batch, 1, channels, 1, 1)
            distance_to_black = (canonical_x0 + 1.0).square().mean(dim=2, keepdim=True)
            distance_to_color = (canonical_x0 - foreground_rgb).square().mean(dim=2, keepdim=True)
            foreground_confidence = torch.sigmoid((distance_to_black - distance_to_color - 0.25) / 0.15)
            # A small dilation keeps the complete antialiased boundary inside
            # the transported object mask while rejecting isolated noise.
            foreground_confidence = F.max_pool2d(
                foreground_confidence.reshape(batch * frames, 1, height, width),
                kernel_size=3, stride=1, padding=1,
            ).reshape(batch, frames, 1, height, width).clamp_(0.0, 1.0)
            # A rigid object has one clean appearance across frames. Share a
            # small fraction of its canonical x0 estimate in v space.
            object_weights = foreground_confidence.detach()
            shared_x0 = (
                (canonical_x0 * object_weights).sum(dim=1, keepdim=True)
                / object_weights.sum(dim=1, keepdim=True).clamp_min(1e-4)
            )
            shared_x0_correction = 0.25 * object_weights * (shared_x0 - canonical_x0)
            canonical_v = canonical_v - shared_x0_correction / sqrt_one_minus.clamp_min(0.05)
            warped_v = self._translate_video(
                canonical_v, offsets_px, to_canonical=False, padding_mode="zeros", mode="nearest"
            )
            foreground_confidence = self._translate_video(
                foreground_confidence, offsets_px, to_canonical=False, mode="nearest"
            )
            black_background_v = sqrt_alpha * (video_frames + sqrt_alpha) / sqrt_one_minus + sqrt_one_minus
            predicted_v = foreground_confidence * warped_v + (1.0 - foreground_confidence) * black_background_v
        else:
            if self.gate_temporal_foreground and self.use_motion:
                # Keep the temporal residual on the moving colored object.
                # The frozen spatial prior remains responsible for the static
                # background, preventing temporal filters from adding flicker
                # over most of the image area.
                alpha_bar = self.alphas_cumprod[timesteps.long()].view(batch, 1, 1, 1, 1).to(video.dtype)
                sqrt_alpha = alpha_bar.sqrt()
                sqrt_one_minus = (1.0 - alpha_bar).clamp_min(1e-8).sqrt()
                spatial_x0 = sqrt_alpha * video_frames - sqrt_one_minus * spatial
                palette = video.new_tensor(((1, -1, -1), (-1, 1, -1), (-1, -1, 1), (1, 1, -1)))
                foreground_rgb = palette[color_labels.long()].view(batch, 1, channels, 1, 1)
                distance_to_black = (spatial_x0 + 1.0).square().mean(dim=2, keepdim=True)
                distance_to_color = (spatial_x0 - foreground_rgb).square().mean(dim=2, keepdim=True)
                foreground_confidence = torch.sigmoid(
                    (distance_to_black - distance_to_color - 0.25) / 0.15
                )
                foreground_confidence = F.max_pool2d(
                    foreground_confidence.reshape(batch * frames, 1, height, width),
                    kernel_size=3, stride=1, padding=1,
                ).reshape(batch, frames, 1, height, width).clamp_(0.0, 1.0)
                predicted_v = spatial + temporal_v * foreground_confidence
            else:
                predicted_v = canonical_v
        output = predicted_v.permute(0, 2, 1, 3, 4).contiguous()  # [B,C,T,H,W]
        if output.shape != video.shape:
            raise RuntimeError(f"Video denoiser returned {tuple(output.shape)} for input {tuple(video.shape)}")
        return output
