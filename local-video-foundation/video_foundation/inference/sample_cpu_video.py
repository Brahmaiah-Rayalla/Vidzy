"""Generate a short CPU video from a compact color-conditioned checkpoint."""

import argparse
import os
import sys

import torch
import json
import numpy as np
import time
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from video_foundation.models.unet import UNet
from video_foundation.models.cpu_video_unet import CPUVideoUNet
from video_foundation.diffusion.schedule import GaussianDiffusionSchedule
from video_foundation.inference.shape_rendering import (
    close_binary_mask, fill_binary_mask_holes, render_canonical_shape, resize_binary_mask,
)
from video_foundation.utils.experiment_metadata import build_generation_manifest, sha256_file
from video_foundation.inference.prompt_conditions import parse_scene_prompt


def main():
    total_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--spatial_checkpoint", default=None,
                        help="Optional compatible image-diffusion checkpoint to use as the video model's spatial denoiser")
    parser.add_argument("--small_spatial_checkpoint", default=None,
                        help="Optional raw-diffusion spatial prior selected for small triangles")
    parser.add_argument("--medium_spatial_checkpoint", default=None,
                        help="Optional raw-diffusion spatial prior selected for medium triangles")
    parser.add_argument("--large_spatial_checkpoint", default=None,
                        help="Optional raw-diffusion spatial prior selected for large triangles")
    parser.add_argument("--medium_large_spatial_checkpoint", default=None,
                        help="Fallback raw-diffusion prior for medium and large triangles")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default=None,
                        help="Closed-curriculum prompt, e.g. 'a medium blue triangle moving left to right quickly'")
    parser.add_argument("--shape", type=int, default=None, choices=(0, 1, 2), help="0 circle, 1 square, 2 triangle")
    parser.add_argument("--color", type=int, default=None, choices=(0, 1, 2, 3), help="0 red, 1 green, 2 blue, 3 yellow")
    parser.add_argument("--size", choices=("small", "medium", "large"), default=None,
                        help="Requested size for checkpoints trained with size conditioning (defaults to medium)")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--duration_seconds", type=float, default=None,
                        help="Optional output duration up to 6 seconds; overrides --frames when set")
    parser.add_argument("--fps", type=float, default=12.0,
                        help="Output frame rate (MP4 always; GIF when --duration_seconds is set)")
    parser.add_argument("--mp4_pixel_format", choices=("yuv444p", "yuv420p"), default="yuv444p",
                        help="MP4 color sampling; yuv444p preserves sharp colored edges, yuv420p favors broad player compatibility")
    parser.add_argument("--image_size", type=int, default=None, help="Defaults to the checkpoint's training resolution")
    parser.add_argument("--display_scale", type=int, default=None,
                        help="Integer output enlargement for encoded frames; defaults to the path/model-size preset")
    parser.add_argument("--display_resampling", choices=("nearest", "bilinear", "bicubic", "lanczos"),
                        default="lanczos",
                        help="Resampling filter for raw diffusion anchor enlargement and rigid translation")
    parser.add_argument("--steps", type=int, default=None,
                        help="DDIM steps (defaults to 12; use 25 for a slower quality comparison)")
    parser.add_argument("--diffusion_ensemble_samples", type=int, default=1,
                        help="Average this many independent raw diffusion anchors before rigid motion; 1 keeps a single sample")
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU inference threads; 1-4 is usually best for this compact model")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise_correlation", type=float, default=0.0,
                        help="Shared component in initial frame noise: 0 matches the iid canonical-noise training distribution")
    parser.add_argument("--noise_alignment", choices=("canonical", "video"), default=None,
                        help="Coordinate system for initial noise; canonical matches the learned-mode training alignment")
    parser.add_argument("--metrics_json", default=None, help="Optional path for temporal consistency metrics")
    parser.add_argument("--manifest_json", default=None,
                        help="Optional provenance path (defaults to a sibling of the metrics file)")
    parser.add_argument("--motion_mode", choices=("rigid", "learned"), default="rigid",
                        help="Rigid: denoise one frame then render the path; learned: predict motion in the video denoiser")
    parser.add_argument("--refine_motion_geometry", action="store_true",
                        help="For learned mode, rerasterize the requested primitive along the predicted trajectory (already the default for size-conditioned checkpoints)")
    parser.add_argument("--raw_learned_geometry", action="store_true",
                        help="Keep the learned per-frame silhouette in learned mode instead of refining it to the requested primitive")
    parser.add_argument("--rigid_appearance", choices=("palette", "denoised"), default="palette",
                        help="Rigid mode: snap the detected object to the curriculum's exact color and black background, or preserve denoised pixels")
    parser.add_argument("--rigid_geometry", choices=("canonical", "learned"), default="canonical",
                        help="Size-conditioned rigid palette mode: rasterize the requested primitive exactly, or keep the learned silhouette")
    parser.add_argument("--motion", choices=("stationary", "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top", "diagonal_down_right", "diagonal_down_left"), default=None)
    parser.add_argument("--speed", choices=("slowly", "quickly"), default=None)
    parser.add_argument("--speed_scale", type=float, default=1.0,
                        help="Scale the learned speed for finer control (0.5 slower, 1.0 trained mean, 1.5 faster)")
    parser.add_argument("--velocity_px_per_frame", type=float, default=None,
                        help="Optional exact synthetic velocity in pixels per training frame; overrides speed label mean")
    args = parser.parse_args()
    prompt_conditions = None
    if args.prompt is not None:
        try:
            prompt_conditions = parse_scene_prompt(args.prompt)
        except ValueError as exc:
            parser.error(str(exc))
    defaults = {"shape": 0, "color": 0, "motion": "left_to_right", "speed": "quickly"}
    for name in defaults:
        explicit = getattr(args, name)
        parsed = prompt_conditions.get(name) if prompt_conditions else None
        setattr(args, name, explicit if explicit is not None else (parsed if parsed is not None else defaults[name]))
    if args.size is None and prompt_conditions and prompt_conditions["size"] is not None:
        args.size = prompt_conditions["size"]
    if args.steps is None:
        args.steps = 12
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if args.display_scale is not None and args.display_scale < 1:
        parser.error("--display_scale must be a positive integer")
    display_resampling = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }[args.display_resampling]
    # Pillow supports Lanczos for resizing, but not for affine transforms.
    # Keep the sharp Lanczos anchor enlargement and use bicubic for the
    # subpixel per-frame translation when Lanczos was requested.
    motion_resampling = (
        Image.Resampling.BICUBIC
        if display_resampling == Image.Resampling.LANCZOS
        else display_resampling
    )
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.diffusion_ensemble_samples < 1:
        parser.error("--diffusion_ensemble_samples must be at least 1")
    if args.diffusion_ensemble_samples > 1 and not (
        args.motion_mode == "rigid" and args.rigid_appearance == "denoised"
    ):
        parser.error("Diffusion anchor ensembling requires --motion_mode rigid --rigid_appearance denoised")
    if not 0.0 <= args.noise_correlation <= 1.0:
        parser.error("--noise_correlation must be between 0 and 1")
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if not 0.25 <= args.speed_scale <= 2.0:
        parser.error("--speed_scale must be between 0.25 and 2.0")
    if args.velocity_px_per_frame is not None and args.velocity_px_per_frame < 0:
        parser.error("--velocity_px_per_frame must be nonnegative")
    if args.velocity_px_per_frame is not None and args.speed_scale != 1.0:
        parser.error("Use either --velocity_px_per_frame or --speed_scale, not both")
    if args.duration_seconds is not None:
        if not 0 < args.duration_seconds <= 6:
            parser.error("--duration_seconds must be greater than 0 and at most 6")
        if not 1 <= args.fps <= 30:
            parser.error("--fps must be between 1 and 30")
        args.frames = max(2, int(round(args.duration_seconds * args.fps)))
        frame_duration_ms = args.duration_seconds * 1000.0 / args.frames
    else:
        frame_duration_ms = 120.0
    output_format = os.path.splitext(args.output)[1].lower()
    if output_format not in (".gif", ".mp4"):
        parser.error("--output must end in .gif or .mp4")
    if output_format == ".mp4" and not 1 <= args.fps <= 60:
        parser.error("MP4 --fps must be between 1 and 60")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    routed_spatial_checkpoints = (
        args.small_spatial_checkpoint, args.medium_spatial_checkpoint,
        args.large_spatial_checkpoint, args.medium_large_spatial_checkpoint,
    )
    if args.spatial_checkpoint and any(routed_spatial_checkpoints):
        parser.error("Use --spatial_checkpoint or size-routed spatial checkpoints, not both")
    if args.medium_large_spatial_checkpoint and (
        args.medium_spatial_checkpoint or args.large_spatial_checkpoint
    ):
        parser.error("Use --medium_large_spatial_checkpoint as a fallback or specify separate medium/large routes")
    reference_frames = int(checkpoint.get("video", {}).get("frames", 8))
    cfg = checkpoint["config"]
    mc = cfg["model"]
    num_shape_classes = int(mc.get("num_classes", 2))
    num_size_classes = int(mc.get("num_size_classes", 0))
    if args.shape >= num_shape_classes:
        raise ValueError(
            f"Shape label {args.shape} is unsupported by this checkpoint, which has "
            f"{num_shape_classes} shape classes. Train with a three-class shape curriculum to generate triangles."
        )
    if num_size_classes > 0:
        size_ids = {"small": 0, "medium": 1, "large": 2}
        size_label = torch.tensor([size_ids[args.size or "medium"]])
    else:
        if args.size is not None:
            raise ValueError("This checkpoint has no size conditioning; train a size-conditioned checkpoint first")
        size_label = None
    if args.refine_motion_geometry and args.raw_learned_geometry:
        raise ValueError("Choose either --refine_motion_geometry or --raw_learned_geometry")
    if args.refine_motion_geometry and (args.motion_mode != "learned" or size_label is None):
        raise ValueError("--refine_motion_geometry requires learned motion and a size-conditioned checkpoint")
    refine_motion_geometry = (
        args.refine_motion_geometry
        or (args.motion_mode == "learned" and size_label is not None and not args.raw_learned_geometry)
    )
    direct_geometry_video = (
        size_label is not None
        and args.motion_mode == "rigid"
        and args.rigid_appearance == "palette"
        and args.rigid_geometry == "canonical"
    )
    if direct_geometry_video:
        generation_mode = "direct_geometry_video"
    elif refine_motion_geometry:
        generation_mode = "diffusion_anchor_plus_renderer_motion"
    elif args.motion_mode == "rigid":
        generation_mode = "diffusion_anchor_rigid_motion"
    else:
        generation_mode = "learned_pixel_video"
    image_size = args.image_size or mc.get("image_size", 64)
    spatial = UNet(in_channels=3, model_channels=mc["model_channels"], channel_mult=mc["channel_mult"],
                   num_res_blocks=mc["num_res_blocks"], num_classes=mc["num_classes"],
                   num_color_classes=mc["num_color_classes"], num_size_classes=num_size_classes,
                   spatial_size_conditioning=mc.get("spatial_size_conditioning", False),
                   high_resolution_refinement=mc.get("high_resolution_refinement", False),
                   size_embedding_scale=mc.get("size_embedding_scale", 1.0),
                   middle_dilation=int(mc.get("middle_dilation", 1)))
    checkpoint_video = checkpoint.get("video", {})
    canonicalize_motion = not checkpoint_video.get("no_canonicalize_motion", False)
    if args.noise_alignment is None:
        args.noise_alignment = "canonical" if canonicalize_motion else "video"
    model = CPUVideoUNet(
        spatial, args.frames, freeze_spatial=True, use_motion=True,
        canonicalize_motion=canonicalize_motion,
        gate_temporal_foreground=checkpoint_video.get("gate_temporal_foreground", False),
    )
    model_state = checkpoint["model_state_dict"]
    expected_state = model.state_dict()
    incompatible_temporal = [
        key for key, value in model_state.items()
        if key.startswith("temporal.") and key in expected_state
        and value.shape != expected_state[key].shape
    ]
    if incompatible_temporal:
        if args.motion_mode == "learned":
            raise ValueError(
                "This checkpoint predates the motion-canonical denoiser. "
                "Use a motion_canonical_v_prediction_v1 checkpoint for --motion_mode learned."
            )
        # Rigid mode uses the spatial prior and trajectory table only; old
        # temporal filters can be ignored safely for this inference path.
        model_state = {key: value for key, value in model_state.items() if not key.startswith("temporal.")}
        incompatible = model.load_state_dict(model_state, strict=False)
        unexpected = incompatible.unexpected_keys
        missing = [key for key in incompatible.missing_keys if not key.startswith("temporal.")]
        if unexpected or missing:
            raise RuntimeError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")
    else:
        model.load_state_dict(model_state)
    selected_spatial_checkpoint_path = args.spatial_checkpoint
    spatial_route = "explicit" if args.spatial_checkpoint else None
    if any(routed_spatial_checkpoints):
        if args.shape != 2:
            raise ValueError("Size-routed raw triangle priors require --shape 2 (triangle)")
        if size_label is None:
            raise ValueError("Size-routed raw triangle priors require a size-conditioned video checkpoint")
        size_id = int(size_label.item())
        if size_id == 0:
            selected_spatial_checkpoint_path = args.small_spatial_checkpoint
            spatial_route = "triangle_small"
        elif size_id == 1:
            selected_spatial_checkpoint_path = (
                args.medium_spatial_checkpoint or args.medium_large_spatial_checkpoint
            )
            spatial_route = "triangle_medium"
        else:
            selected_spatial_checkpoint_path = (
                args.large_spatial_checkpoint or args.medium_large_spatial_checkpoint
            )
            spatial_route = "triangle_large"
        if not selected_spatial_checkpoint_path:
            route_name = ("small", "medium", "large")[size_id]
            raise ValueError(f"No raw spatial checkpoint was configured for the {route_name} triangle route")
    spatial_checkpoint = None
    if selected_spatial_checkpoint_path:
        spatial_checkpoint = torch.load(selected_spatial_checkpoint_path, map_location="cpu", weights_only=False)
        spatial_config = spatial_checkpoint.get("config", {})
        spatial_model_config = spatial_config.get("model", {})
        expected_model_config = cfg.get("model", {})
        expected_model_config.setdefault("middle_dilation", 1)
        spatial_model_config.setdefault("middle_dilation", 1)
        structural_fields = (
            "image_size", "in_channels", "model_channels", "channel_mult", "num_res_blocks",
            "num_classes", "num_color_classes", "num_size_classes", "spatial_size_conditioning",
            "high_resolution_refinement",
            "middle_dilation",
        )
        mismatches = {
            key: (expected_model_config.get(key, False if key in (
                "spatial_size_conditioning", "high_resolution_refinement"
            ) else (1 if key == "middle_dilation" else None)), spatial_model_config.get(key, False if key in (
                "spatial_size_conditioning", "high_resolution_refinement"
            ) else (1 if key == "middle_dilation" else None)))
            for key in structural_fields
            if expected_model_config.get(key, False if key in (
                "spatial_size_conditioning", "high_resolution_refinement"
            ) else (1 if key == "middle_dilation" else None)) != spatial_model_config.get(key, False if key in (
                "spatial_size_conditioning", "high_resolution_refinement"
            ) else (1 if key == "middle_dilation" else None))
        }
        video_prediction_type = cfg.get("diffusion", {}).get("prediction_type")
        spatial_prediction_type = spatial_config.get("diffusion", {}).get("prediction_type")
        if mismatches or spatial_prediction_type != video_prediction_type:
            raise ValueError(
                "Spatial checkpoint is incompatible with the video checkpoint: "
                f"model differences={mismatches}, prediction types="
                f"{spatial_prediction_type!r} vs {video_prediction_type!r}"
            )
        spatial_state = spatial_checkpoint.get("model_state_dict")
        if not isinstance(spatial_state, dict):
            raise ValueError("Spatial checkpoint has no model_state_dict")
        if spatial_state and all(key.startswith("spatial_unet.") for key in spatial_state):
            spatial_state = {key.removeprefix("spatial_unet."): value
                              for key, value in spatial_state.items()}
        try:
            model.spatial_unet.load_state_dict(spatial_state, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Spatial checkpoint weights do not match the video denoiser: {exc}") from exc
        model.spatial_unet.size_embedding_scale = float(
            spatial_model_config.get("size_embedding_scale", 1.0)
        )
        print(
            f"Using spatial prior {selected_spatial_checkpoint_path} (step {spatial_checkpoint.get('step')}, "
            f"route={spatial_route}); "
            "video temporal/motion weights remain from --checkpoint"
        )
    model.eval()
    schedule = GaussianDiffusionSchedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule="linear")
    generator = torch.Generator().manual_seed(args.seed)
    ts = torch.linspace(999, 0, args.steps, dtype=torch.long)
    shape = torch.tensor([args.shape])
    color = torch.tensor([args.color])
    motion_ids = {
        "stationary": 0, "left_to_right": 1, "right_to_left": 2,
        "top_to_bottom": 3, "bottom_to_top": 4,
        "diagonal_down_right": 5, "diagonal_down_left": 6,
    }
    motion = torch.tensor([motion_ids[args.motion]])
    speed = torch.tensor([{"slowly": 0, "quickly": 1}[args.speed]])
    with torch.no_grad():
        if args.velocity_px_per_frame is None:
            predicted_offsets = model.predict_offsets(motion, speed, speed_scale=args.speed_scale)
        else:
            explicit_velocity = torch.tensor([args.velocity_px_per_frame], dtype=torch.float32)
            predicted_offsets = model.predict_offsets(
                motion, speed, velocity_px_per_frame=explicit_velocity,
                image_size=image_size, reference_frames=reference_frames,
            )
        offsets = predicted_offsets[0].numpy() * image_size
    sampling_started = time.perf_counter()
    if direct_geometry_video:
        # Shape, palette, and size are explicit controls in this curriculum.
        # Render that known appearance directly instead of spending CPU time
        # denoising pixels that the exact geometry renderer will replace.
        target_extent = max(1, round((15, 29, 42)[int(size_label.item())] * image_size / 64))
        center = ((image_size - 1) / 2.0, (image_size - 1) / 2.0)
        base_mask = render_canonical_shape(args.shape, center, target_extent, image_size)
        palette_rgb = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
        base_rgb = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        base_rgb[base_mask] = palette_rgb[args.color]
        base_tensor = torch.from_numpy(base_rgb).permute(2, 0, 1).float() / 127.5 - 1.0
        x = base_tensor.view(1, 3, 1, image_size, image_size).expand(
            1, 3, args.frames, image_size, image_size
        ).contiguous()
    else:
        with torch.no_grad():
            if args.motion_mode == "rigid" or refine_motion_geometry:
                # Denoise one anchor, or average independent raw anchors to
                # reduce uncorrelated RGB speckle for deterministic shape clips.
                anchor_samples = []
                for sample_index in range(args.diffusion_ensemble_samples):
                    anchor_generator = generator
                    if sample_index > 0:
                        anchor_seed = (args.seed + sample_index * 1_000_003) % (2**63 - 1)
                        anchor_generator = torch.Generator().manual_seed(anchor_seed)
                    x = torch.randn(
                        (1, 3, image_size, image_size), generator=anchor_generator
                    )
                    for i, timestep in enumerate(ts):
                        t = timestep.expand(1)
                        v = model.spatial_unet(x, t, shape, color, size_label)
                        _, alpha, one_minus = schedule.get(t, x.device)
                        alpha = alpha.view(1, 1, 1, 1)
                        one_minus = one_minus.view(1, 1, 1, 1)
                        x0 = (alpha.sqrt() * x - one_minus.sqrt() * v).clamp(-1, 1)
                        eps = (x - alpha.sqrt() * x0) / one_minus.sqrt().clamp_min(1e-8)
                        if i + 1 < len(ts):
                            _, alpha_prev, one_minus_prev = schedule.get(ts[i + 1].expand(1), x.device)
                            alpha_prev = alpha_prev.view(1, 1, 1, 1)
                            one_minus_prev = one_minus_prev.view(1, 1, 1, 1)
                            x = alpha_prev.sqrt() * x0 + one_minus_prev.sqrt() * eps
                        else:
                            x = x0
                    anchor_samples.append(x[0])
                x = torch.stack(anchor_samples, dim=0).mean(dim=0, keepdim=True).clamp(-1, 1)
                x = x.unsqueeze(2).expand(-1, -1, args.frames, -1, -1).clone()
            else:
                # Correlate in object-centered coordinates so shared structure
                # moves with the object, then transport noise into screen space.
                shared = torch.randn((1, 3, 1, image_size, image_size), generator=generator)
                independent = torch.randn((1, 3, args.frames, image_size, image_size), generator=generator)
                canonical_noise = (args.noise_correlation ** 0.5) * shared.expand_as(independent)
                canonical_noise = canonical_noise + ((1.0 - args.noise_correlation) ** 0.5) * independent
                if args.noise_alignment == "canonical":
                    offsets_px = torch.round(torch.as_tensor(offsets, dtype=canonical_noise.dtype)).view(1, args.frames, 2)
                    x = model._translate_video(
                        canonical_noise.permute(0, 2, 1, 3, 4), offsets_px,
                        to_canonical=False, mode="nearest",
                    ).permute(0, 2, 1, 3, 4).contiguous()
                else:
                    x = canonical_noise
                for i, timestep in enumerate(ts):
                    t = timestep.expand(1)
                    v = model(
                        x, t, shape, color, motion, speed, motion_scale=args.speed_scale,
                        velocity_px_per_frame=(None if args.velocity_px_per_frame is None
                                               else torch.tensor([args.velocity_px_per_frame])),
                        reference_frames=reference_frames,
                        size_labels=size_label,
                    )
                    _, alpha, one_minus = schedule.get(t, x.device)
                    alpha = alpha.view(1, 1, 1, 1, 1)
                    one_minus = one_minus.view(1, 1, 1, 1, 1)
                    x0 = (alpha.sqrt() * x - one_minus.sqrt() * v).clamp(-1, 1)
                    eps = (x - alpha.sqrt() * x0) / one_minus.sqrt().clamp_min(1e-8)
                    if i + 1 < len(ts):
                        _, alpha_prev, one_minus_prev = schedule.get(ts[i + 1].expand(1), x.device)
                        alpha_prev = alpha_prev.view(1, 1, 1, 1, 1)
                        one_minus_prev = one_minus_prev.view(1, 1, 1, 1, 1)
                        x = alpha_prev.sqrt() * x0 + one_minus_prev.sqrt() * eps
                    else:
                        x = x0
    sampling_seconds = time.perf_counter() - sampling_started
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    frames = []
    rendered_pixels = []
    observed_centroids = []
    base_centroids = []
    foreground_areas = []
    raw_foreground_areas = []
    raw_bbox_extents = []
    rendered_bbox_extents = []
    applied_offsets = []
    crop_fractions = []
    observed_masks = []

    def foreground_mask(rgb):
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        if args.color == 0:
            return (red > green * 1.35) & (red > blue * 1.35) & (red > 55)
        if args.color == 1:
            return (green > red * 1.2) & (green > blue * 1.2) & (green > 55)
        if args.color == 2:
            return (blue > red * 1.25) & (blue > green * 1.15) & (blue > 55)
        return (red > blue * 1.3) & (green > blue * 1.15) & (red > 55) & (green > 55)

    def largest_component(mask):
        """Discard isolated color-noise pixels before estimating shape bounds."""
        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        largest = []
        for seed_y, seed_x in zip(*np.where(mask)):
            if visited[seed_y, seed_x]:
                continue
            visited[seed_y, seed_x] = True
            stack = [(int(seed_y), int(seed_x))]
            component = []
            while stack:
                y, x_coord = stack.pop()
                component.append((y, x_coord))
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    for nx in range(max(0, x_coord - 1), min(width, x_coord + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if len(component) > len(largest):
                largest = component
        cleaned = np.zeros_like(mask, dtype=bool)
        if largest:
            ys, xs = zip(*largest)
            cleaned[np.asarray(ys), np.asarray(xs)] = True
        return cleaned

    def centroid(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return (image_size - 1) / 2, (image_size - 1) / 2
        return float(xs.mean()), float(ys.mean())

    for frame_index, frame in enumerate(x[0].permute(1, 2, 3, 0)):
        pixels = ((frame.clamp(-1, 1) + 1) * 127.5).byte().numpy()
        mask = largest_component(foreground_mask(pixels))
        raw_foreground_areas.append(int(mask.sum()))
        raw_ys, raw_xs = np.where(mask)
        raw_bbox_extents.append(
            int(max(raw_xs.max() - raw_xs.min() + 1, raw_ys.max() - raw_ys.min() + 1))
            if len(raw_xs) else 0
        )
        if (args.motion_mode == "rigid" and args.rigid_appearance == "palette"
                and size_label is not None):
            # The source curriculum's measured mean object extents are 15, 29,
            # and 42 px at 64 px resolution. Resize the detected silhouette so
            # size remains controllable even when the compact diffusion prior
            # has not learned a strong size embedding.
            target_extent = (15, 29, 42)[int(size_label.item())]
            target_extent = max(1, round(target_extent * image_size / 64))
            if args.rigid_geometry == "canonical":
                mask = render_canonical_shape(args.shape, centroid(mask), target_extent, image_size)
            else:
                mask = resize_binary_mask(
                    fill_binary_mask_holes(close_binary_mask(mask, kernel_size=3)),
                    target_extent,
                )
        if args.motion_mode == "rigid" and args.rigid_appearance == "palette":
            palette_rgb = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
            clean_pixels = np.zeros_like(pixels)
            clean_pixels[mask] = palette_rgb[args.color]
            pixels = clean_pixels
        rendered_ys, rendered_xs = np.where(mask)
        rendered_bbox_extents.append(
            int(max(rendered_xs.max() - rendered_xs.min() + 1,
                    rendered_ys.max() - rendered_ys.min() + 1))
            if len(rendered_xs) else 0
        )
        image = Image.fromarray(pixels)
        base_area = int(mask.sum())
        base_center = centroid(mask)
        base_centroids.append(base_center)
        foreground_areas.append(base_area)
        if args.motion_mode == "rigid":
            # The canonical-frame path needs an explicit final render. The
            # learned path already predicts translated x0 frames in-model.
            row_counts, col_counts = mask.sum(axis=1), mask.sum(axis=0)
            rows = np.flatnonzero(row_counts >= max(2, row_counts.max(initial=0) * 0.08))
            cols = np.flatnonzero(col_counts >= max(2, col_counts.max(initial=0) * 0.08))
            if len(rows) and len(cols):
                min_x, max_x = int(cols[0]), int(cols[-1])
                min_y, max_y = int(rows[0]), int(rows[-1])
                dx = float(np.clip(offsets[frame_index, 0], -min_x, image_size - 1 - max_x))
                dy = float(np.clip(offsets[frame_index, 1], -min_y, image_size - 1 - max_y))
            else:
                dx = dy = 0.0
            applied_offsets.append((dx, dy))
            moved = image.transform(image.size, Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy),
                                    resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        else:
            dx = dy = 0.0
            applied_offsets.append((float(offsets[frame_index, 0]), float(offsets[frame_index, 1])))
            moved = image
        moved_pixels = np.asarray(moved)
        rendered_pixels.append(moved_pixels.astype(np.float32) / 127.5 - 1.0)
        moved_mask = largest_component(foreground_mask(moved_pixels))
        observed_masks.append(moved_mask)
        observed_centroids.append(centroid(moved_mask))
        moved_area = int(moved_mask.sum())
        crop_fractions.append(max(0.0, 1.0 - moved_area / max(base_area, 1)))
        if (args.motion_mode == "rigid" and args.rigid_appearance == "palette"
                and size_label is not None and args.rigid_geometry == "canonical"):
            # Render the exact primitive above the model's 64px working grid,
            # then downsample for display. Nearest-neighbor enlargement makes
            # small diagonal shapes visibly stair-stepped in the default mode.
            display_scale = args.display_scale or 8
            render_scale = display_scale * 2
            target_extent = max(1, round((15, 29, 42)[int(size_label.item())] * image_size / 64))
            high_center = tuple(
                (coordinate + offset + 0.5) * render_scale - 0.5
                for coordinate, offset in zip(base_center, (dx, dy))
            )
            high_mask = render_canonical_shape(
                args.shape, high_center, target_extent * render_scale,
                image_size * render_scale,
            )
            high_pixels = np.zeros((image_size * render_scale, image_size * render_scale, 3), dtype=np.uint8)
            high_pixels[high_mask] = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))[args.color]
            frames.append(Image.fromarray(high_pixels).resize(
                (image_size * display_scale, image_size * display_scale), Image.Resampling.LANCZOS
            ))
        else:
            if args.motion_mode == "rigid" and args.rigid_appearance == "denoised":
                # Preserve the raw diffusion anchor and perform display
                # resampling only once, at the final output scale. Warping a
                # 64px frame first blurs its contour before nearest-neighbor
                # enlargement; the high-resolution affine keeps subpixel
                # motion smooth without replacing RGB with a rendered mask.
                # Keep high-resolution anchor rendering within the CPU video
                # budget. The 64px path retains its established 512px output;
                # 128px anchors render at 384px instead of expanding every
                # frame to 1024px before affine motion and H.264 encoding.
                display_scale = args.display_scale or (
                    8 if image_size <= 64
                    else max(1, min(8, 384 // image_size))
                )
                high_resolution = image.resize(
                    (image_size * display_scale, image_size * display_scale),
                    display_resampling,
                )
                high_moved = high_resolution.transform(
                    high_resolution.size,
                    Image.Transform.AFFINE,
                    (1, 0, -dx * display_scale, 0, 1, -dy * display_scale),
                    resample=motion_resampling,
                    fillcolor=(0, 0, 0),
                )
                frames.append(high_moved)
            else:
                learned_palette_mask = (
                    args.motion_mode == "rigid"
                    and args.rigid_appearance == "palette"
                    and args.rigid_geometry == "learned"
                    and size_label is not None
                )
                upscale_filter = (
                    Image.Resampling.LANCZOS if learned_palette_mask else Image.Resampling.NEAREST
                )
                frames.append(moved.resize((image_size * 8, image_size * 8), upscale_filter))
    if refine_motion_geometry:
        # Keep the diffusion-denoised anchor, while the learned motion table
        # supplies the trajectory. Reconstructing this simple curriculum
        # primitive removes per-frame shape/background drift without requiring
        # a learned per-frame silhouette to stay stable.
        trajectory_anchor = np.mean(np.asarray(base_centroids), axis=0)
        target_extent = (15, 29, 42)[int(size_label.item())]
        target_extent = max(1, round(target_extent * image_size / 64))
        palette_rgb = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
        base_mask = render_canonical_shape(args.shape, trajectory_anchor, target_extent, image_size)
        trajectory_anchor = centroid(base_mask)
        base_pixels = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        base_pixels[base_mask] = palette_rgb[args.color]
        base_image = Image.fromarray(base_pixels)
        # Supersample geometry at 2x the final GIF resolution. Drawing at
        # 64x64 and enlarging with nearest-neighbor creates 8-pixel stair steps
        # on diagonal edges, especially on large triangles.
        display_scale = args.display_scale or 8
        antialias_scale = 2
        render_scale = display_scale * antialias_scale
        render_size = image_size * render_scale
        high_center = tuple(
            (coordinate + 0.5) * render_scale - 0.5 for coordinate in trajectory_anchor
        )
        high_mask = render_canonical_shape(
            args.shape, high_center, target_extent * render_scale, render_size
        )
        high_pixels = np.zeros((render_size, render_size, 3), dtype=np.uint8)
        high_pixels[high_mask] = palette_rgb[args.color]
        high_image = Image.fromarray(high_pixels)
        high_ys, high_xs = np.where(high_mask)
        high_bbox = (int(high_xs.min()), int(high_ys.min()),
                     int(high_xs.max()) + 1, int(high_ys.max()) + 1)
        high_shape = high_image.crop(high_bbox)
        base_area = int(base_mask.sum())
        base_centroids = [trajectory_anchor for _ in range(args.frames)]
        rendered_pixels, frames = [], []
        observed_centroids, foreground_areas, observed_masks = [], [], []
        rendered_bbox_extents, crop_fractions = [], []
        for offset in applied_offsets:
            moved = base_image.transform(
                base_image.size, Image.Transform.AFFINE,
                (1, 0, -float(offset[0]), 0, 1, -float(offset[1])),
                resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0),
            )
            pixels = np.asarray(moved)
            moved_mask = largest_component(foreground_mask(pixels))
            observed_masks.append(moved_mask)
            ys, xs = np.where(moved_mask)
            moved_area = int(moved_mask.sum())
            observed_centroids.append(centroid(moved_mask))
            foreground_areas.append(moved_area)
            rendered_bbox_extents.append(
                int(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)) if len(xs) else 0
            )
            crop_fractions.append(max(0.0, 1.0 - moved_area / max(base_area, 1)))
            rendered_pixels.append(pixels.astype(np.float32) / 127.5 - 1.0)
            high_moved = Image.new("RGB", high_image.size, (0, 0, 0))
            paste_position = (
                int(round(high_bbox[0] + float(offset[0]) * render_scale)),
                int(round(high_bbox[1] + float(offset[1]) * render_scale)),
            )
            high_moved.paste(high_shape, paste_position)
            frames.append(high_moved.resize(
                (image_size * display_scale, image_size * display_scale),
                Image.Resampling.LANCZOS,
            ))
    trajectory_center = np.mean(np.asarray(base_centroids), axis=0)
    expected_centroids = [
        (trajectory_center[0] + applied_offsets[i][0], trajectory_center[1] + applied_offsets[i][1])
        for i in range(args.frames)
    ]
    output_trajectory_rmse = float(np.mean([
        (observed_centroids[i][0] - expected_centroids[i][0]) ** 2
        + (observed_centroids[i][1] - expected_centroids[i][1]) ** 2
        for i in range(args.frames)
    ]) ** 0.5)
    raw_trajectory_rmse = float(np.mean([
        (base_centroids[i][0] - expected_centroids[i][0]) ** 2
        + (base_centroids[i][1] - expected_centroids[i][1]) ** 2
        for i in range(args.frames)
    ]) ** 0.5)
    output_centroid_span = [
        float(max(point[axis] for point in observed_centroids) - min(point[axis] for point in observed_centroids))
        for axis in range(2)
    ]
    observed_areas = np.asarray([int(mask.sum()) for mask in observed_masks], dtype=np.float64)
    observed_area_mean = float(observed_areas.mean()) if len(observed_areas) else 0.0
    observed_area_std = float(observed_areas.std()) if len(observed_areas) else 0.0
    observed_area_cv = (observed_area_std / observed_area_mean
                        if observed_area_mean > 0 else None)
    adjacent_aligned_ious = []
    for index in range(1, len(observed_masks)):
        previous_mask = observed_masks[index - 1]
        current_mask = observed_masks[index]
        previous_center = observed_centroids[index - 1]
        current_center = observed_centroids[index]
        shift_x = int(round(previous_center[0] - current_center[0]))
        shift_y = int(round(previous_center[1] - current_center[1]))
        height, width = current_mask.shape
        aligned_mask = np.zeros_like(current_mask)
        dst_y0, dst_y1 = max(0, shift_y), min(height, height + shift_y)
        dst_x0, dst_x1 = max(0, shift_x), min(width, width + shift_x)
        src_y0, src_y1 = max(0, -shift_y), min(height, height - shift_y)
        src_x0, src_x1 = max(0, -shift_x), min(width, width - shift_x)
        if dst_y1 > dst_y0 and dst_x1 > dst_x0:
            aligned_mask[dst_y0:dst_y1, dst_x0:dst_x1] = current_mask[
                src_y0:src_y1, src_x0:src_x1
            ]
        union = np.logical_or(previous_mask, aligned_mask).sum()
        intersection = np.logical_and(previous_mask, aligned_mask).sum()
        adjacent_aligned_ious.append(float(intersection / max(int(union), 1)))
    centroid_aligned_adjacent_mask_iou = (
        float(np.mean(adjacent_aligned_ious)) if adjacent_aligned_ious else None
    )
    rendered_video = np.stack(rendered_pixels)
    temporal_l1 = float(np.abs(rendered_video[1:] - rendered_video[:-1]).mean())
    # GIF delays use 10 ms units. Distribute rounding error across the frames
    # instead of truncating every delay (for example, 83 ms to 80 ms).
    gif_durations_ms = None
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if output_format == ".gif":
        # GIF delays use 10 ms units. Distribute rounding error across frames.
        gif_boundaries = [int(round(index * frame_duration_ms / 10.0)) * 10
                          for index in range(args.frames + 1)]
        gif_durations_ms = [max(10, gif_boundaries[index + 1] - gif_boundaries[index])
                            for index in range(args.frames)]
        frames[0].save(args.output, save_all=True, append_images=frames[1:],
                       duration=gif_durations_ms, loop=0)
        output_duration_seconds = sum(gif_durations_ms) / 1000.0
        output_fps = None
    else:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError("MP4 output requires imageio-ffmpeg; install project requirements with 'pip install -r requirements.txt'") from exc
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        width, height = frames[0].size
        writer = imageio_ffmpeg.write_frames(
            args.output, (width, height), fps=float(args.fps), codec="libx264",
            pix_fmt_in="rgb24", pix_fmt_out=args.mp4_pixel_format, macro_block_size=16,
            output_params=["-preset", "ultrafast", "-crf", "18"],
        )
        try:
            writer.send(None)
            for frame in frames:
                writer.send(np.ascontiguousarray(frame, dtype=np.uint8))
        finally:
            writer.close()
        output_fps = float(args.fps)
        output_duration_seconds = args.frames / output_fps
    total_seconds = time.perf_counter() - total_started
    metrics = {
        "schema_version": 3,
        "generation_mode": generation_mode,
        "device": "cpu",
        "cpu_threads": args.threads,
        "seed": args.seed,
        "checkpoint_path": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "spatial_checkpoint_path": (os.path.abspath(selected_spatial_checkpoint_path)
                                    if selected_spatial_checkpoint_path else None),
        "spatial_checkpoint_sha256": (sha256_file(selected_spatial_checkpoint_path)
                                      if selected_spatial_checkpoint_path else None),
        "spatial_checkpoint_step": (spatial_checkpoint.get("step")
                                    if spatial_checkpoint is not None else None),
        "spatial_checkpoint_route": spatial_route,
        "output_path": os.path.abspath(args.output),
        "prompt": args.prompt,
        "input_dataset": None,
        "manifest_path": None,
        "frames": args.frames,
        "output_format": output_format.lstrip("."),
        "mp4_pixel_format": args.mp4_pixel_format if output_format == ".mp4" else None,
        "fps": output_fps,
        "frame_duration_ms": (round(sum(gif_durations_ms) / len(gif_durations_ms), 3)
                               if gif_durations_ms else round(1000.0 / output_fps, 3)),
        "requested_frame_duration_ms": round(frame_duration_ms, 3),
        "output_duration_seconds": output_duration_seconds,
        "sampling_seconds": sampling_seconds,
        "total_seconds": total_seconds,
        "sampling_steps": 0 if direct_geometry_video else args.steps,
        "requested_sampling_steps": args.steps,
        "diffusion_ensemble_samples": args.diffusion_ensemble_samples,
        "direct_geometry_render": direct_geometry_video,
        "canonical_geometry_renderer": direct_geometry_video or refine_motion_geometry,
        "noise_correlation": args.noise_correlation,
        "noise_alignment": args.noise_alignment,
        "motion_mode": args.motion_mode,
        "motion_geometry_refined": refine_motion_geometry,
        "rigid_appearance": args.rigid_appearance if args.motion_mode == "rigid" else None,
        "rigid_geometry": (args.rigid_geometry if args.motion_mode == "rigid"
                           and args.rigid_appearance == "palette" and size_label is not None else None),
        "mask_cleanup": (
            "3x3 morphological close, fill enclosed holes, and normalize predicted mask extent"
            if args.motion_mode == "rigid" and args.rigid_appearance == "palette"
            and args.rigid_geometry == "learned" and size_label is not None else None
        ),
        "adjacent_frame_mean_abs_difference": temporal_l1,
        "motion": args.motion,
        "speed": args.speed,
        "size": args.size or ("medium" if size_label is not None else None),
        "speed_scale": args.speed_scale,
        "velocity_px_per_frame": args.velocity_px_per_frame,
        "image_size": image_size,
        "encoded_frame_size_pixels": list(frames[0].size) if frames else None,
        "mean_foreground_crop_fraction": float(np.mean(crop_fractions)),
        "mean_observed_foreground_pixels": observed_area_mean,
        "observed_foreground_area_std_pixels": observed_area_std,
        "observed_foreground_area_cv": observed_area_cv,
        "centroid_aligned_adjacent_mask_iou": centroid_aligned_adjacent_mask_iou,
        "temporal_shape_metric_method": (
            "largest color-connected component; adjacent masks translated to align centroids"
        ),
        "raw_trajectory_centroid_rmse_pixels": raw_trajectory_rmse,
        "rendered_centroid_vs_applied_offset_rmse_pixels": output_trajectory_rmse,
        # Backward-compatible key; this is renderer residual, not learned
        # trajectory accuracy. The training evaluator compares prediction to
        # held-out metadata for true trajectory accuracy.
        "output_trajectory_centroid_rmse_pixels": output_trajectory_rmse,
        "output_centroid_span_pixels": output_centroid_span,
        "mean_raw_foreground_pixels": float(np.mean(foreground_areas)),
        "mean_unadjusted_foreground_pixels": float(np.mean(raw_foreground_areas)),
        "mean_unadjusted_bbox_extent_pixels": float(np.mean(raw_bbox_extents)),
        "mean_rendered_bbox_extent_pixels": float(np.mean(rendered_bbox_extents)),
        "predicted_offsets_pixels": [[float(x), float(y)] for x, y in applied_offsets],
        "mean_observed_centroid_pixels": [
            float(np.mean([point[0] for point in observed_centroids])),
            float(np.mean([point[1] for point in observed_centroids])),
        ],
    }
    metrics_path = os.path.abspath(
        args.metrics_json or os.path.splitext(args.output)[0] + ".metrics.json"
    )
    manifest_path = os.path.abspath(
        args.manifest_json or os.path.splitext(metrics_path)[0] + "_manifest.json"
    )
    metrics["manifest_path"] = manifest_path
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    manifest = build_generation_manifest(
        task="cpu_video_generation",
        generation_mode=generation_mode,
        args=vars(args),
        checkpoint_path=args.checkpoint,
        checkpoint_payload=checkpoint,
        source_paths=(
            __file__,
            os.path.join(ROOT, "video_foundation", "models", "unet.py"),
            os.path.join(ROOT, "video_foundation", "models", "cpu_video_unet.py"),
            os.path.join(ROOT, "video_foundation", "inference", "shape_rendering.py"),
            os.path.join(ROOT, "video_foundation", "inference", "prompt_conditions.py"),
            os.path.join(ROOT, "video_foundation", "utils", "experiment_metadata.py"),
        ),
        elapsed_seconds=total_seconds,
        torch_version=torch.__version__,
        cpu_threads=torch.get_num_threads(),
    )
    if spatial_checkpoint is not None:
        manifest["spatial_prior_checkpoint"] = {
            "path": os.path.abspath(selected_spatial_checkpoint_path),
            "sha256": sha256_file(selected_spatial_checkpoint_path),
            "step": spatial_checkpoint.get("step"),
            "config": spatial_checkpoint.get("config", {}),
            "route": spatial_route,
        }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    sampling_summary = (
        "exact geometry render (diffusion skipped)" if direct_geometry_video
        else (f"diffusion anchor={sampling_seconds:.2f}s/{args.steps} steps + canonical geometry renderer"
              if refine_motion_geometry else f"sampling={sampling_seconds:.2f}s/{args.steps} steps")
    )
    print(f"Saved {len(frames)} {args.speed} {args.motion} frames ({output_duration_seconds:.2f}s) at {image_size}px to {args.output}; "
          f"adjacent L1={temporal_l1:.5f}, render residual={output_trajectory_rmse:.2f}px, "
          f"foreground crop={np.mean(crop_fractions):.1%}, {sampling_summary}, "
          f"total={total_seconds:.2f}s on CPU/{args.threads} threads")


if __name__ == "__main__":
    main()
