"""
Training script for Stage 1 image generation model.
"""

import os
import sys
import argparse
import csv
import copy
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

# Add project root to path for package imports when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from video_foundation.data.synthetic import SyntheticShapeGenerator
from video_foundation.data.dataset import SyntheticImageDataset
from video_foundation.models.unet import UNet
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.utils.reproducibility import set_seed, get_device
from video_foundation.utils.checkpoint import CheckpointManager


def save_image_grid(images: torch.Tensor, save_path: str, nrow: int = 4):
    """
    Save a grid of images.

    Args:
        images: Tensor of shape [batch, channels, height, width] in range [-1, 1]
        save_path: Path to save the grid
        nrow: Number of images per row
    """
    # Denormalize from [-1, 1] to [0, 255]
    images = (images + 1) / 2.0
    images = torch.clamp(images, 0, 1)
    images = (images * 255).byte()

    # Create grid
    from torchvision.utils import make_grid
    grid = make_grid(images, nrow=nrow, padding=2)

    # Convert to PIL Image and save
    grid = grid.permute(1, 2, 0).cpu().numpy()
    grid = Image.fromarray(grid)
    grid.save(save_path)


def translate_foreground_batch(images: torch.Tensor, max_translation: int) -> torch.Tensor:
    """Translate each palette image by integer pixels without cropping its shape."""
    if max_translation <= 0:
        return images
    batch, _, height, width = images.shape
    translated = torch.full_like(images, -1.0)
    for index in range(batch):
        foreground = (images[index] > -0.95).any(dim=0)
        ys, xs = torch.where(foreground)
        if not xs.numel():
            translated[index].copy_(images[index])
            continue
        x_limit = min(max_translation, int(xs.min()), width - 1 - int(xs.max()))
        y_limit = min(max_translation, int(ys.min()), height - 1 - int(ys.max()))
        dx = int(torch.randint(-x_limit, x_limit + 1, ()).item()) if x_limit > 0 else 0
        dy = int(torch.randint(-y_limit, y_limit + 1, ()).item()) if y_limit > 0 else 0
        source_x0, source_x1 = max(0, -dx), min(width, width - dx)
        source_y0, source_y1 = max(0, -dy), min(height, height - dy)
        target_x0, target_x1 = max(0, dx), min(width, width + dx)
        target_y0, target_y1 = max(0, dy), min(height, height + dy)
        translated[index, :, target_y0:target_y1, target_x0:target_x1] = images[
            index, :, source_y0:source_y1, source_x0:source_x1
        ]
    return translated


def stratified_timesteps(batch_size: int, num_timesteps: int, device: torch.device) -> torch.Tensor:
    """Sample uniform timesteps with low-variance coverage in small batches."""
    if batch_size <= 0 or num_timesteps <= 0:
        raise ValueError("batch_size and num_timesteps must be positive")
    cell_width = num_timesteps / batch_size
    offsets = torch.rand(batch_size, device=device)
    timesteps = ((torch.arange(batch_size, device=device, dtype=torch.float32) + offsets)
                 * cell_width).long().clamp_(max=num_timesteps - 1)
    return timesteps[torch.randperm(batch_size, device=device)]


def train(config: dict, args: argparse.Namespace):
    """
    Train the diffusion model.

    Args:
        config: Configuration dictionary
        args: Command-line arguments
    """
    # Set seed for reproducibility
    seed = args.seed if args.seed is not None else config['training']['seed']
    set_seed(seed)

    cpu_threads = int(config.get('training', {}).get('cpu_threads', min(4, os.cpu_count() or 1)))
    if cpu_threads < 1:
        raise ValueError("training.cpu_threads must be at least 1")
    torch.set_num_threads(cpu_threads)

    # Get device
    device = get_device(args.device)

    # Print configuration
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Device: {device}")
    print(f"Image size: {config['model']['image_size']}")
    print(f"Batch size: {args.batch_size if args.batch_size else config['training']['batch_size']}")
    print(f"Training steps: {args.num_steps if args.num_steps else config['training']['num_steps']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Seed: {seed}")
    print(f"Position augmentation: {args.position_augmentation_pixels}px, without shape cropping")
    print("="*60 + "\n")

    # Generate or load dataset
    data_dir = args.data_dir if args.data_dir else "data/synthetic"
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"Generating synthetic dataset in {data_dir}...")
        generator = SyntheticShapeGenerator(
            image_size=config['dataset']['image_size'],
            seed=config['dataset']['seed']
        )
        generator.generate_dataset(
            num_images=config['dataset']['num_images'],
            output_dir=data_dir,
            save_preview=True
        )

    # Create dataset and dataloader
    print("Loading dataset...")
    image_size = args.image_size if args.image_size is not None else config['model']['image_size']
    model_channels = args.model_channels if args.model_channels is not None else config['model']['model_channels']
    num_classes = int(config['model'].get('num_classes', 0))
    num_color_classes = int(config['model'].get('num_color_classes', 0))
    num_size_classes = int(config['model'].get('num_size_classes', 0))
    dataset = SyntheticImageDataset(
        data_dir=data_dir,
        image_size=image_size,
        normalize=True,
        cache_in_memory=config.get('dataset', {}).get('cache_in_memory', len(os.listdir(data_dir)) <= 1024),
        return_label=num_classes > 0,
        return_color_label=num_color_classes > 0,
        return_size_label=num_size_classes > 0
    )

    if args.target_shape_class is not None:
        if not dataset.return_label:
            raise ValueError("--target_shape_class requires a class-conditioned dataset")
        target_class = int(args.target_shape_class)
        if target_class < 0 or target_class >= num_classes:
            raise ValueError(f"--target_shape_class must be in [0, {num_classes - 1}]")
        keep_indices = [i for i, label in enumerate(dataset.class_labels)
                        if int(label) == target_class]
        if not keep_indices:
            raise ValueError(f"No training examples found for shape class {target_class}")
        dataset.image_files = [dataset.image_files[i] for i in keep_indices]
        for label_attribute in ("class_labels", "color_labels", "size_labels"):
            labels = getattr(dataset, label_attribute, None)
            if labels is not None:
                setattr(dataset, label_attribute, [labels[i] for i in keep_indices])
        print(f"Focused fine-tune: shape class {target_class}, {len(dataset)} images")

    if args.target_size_class is not None:
        if not dataset.return_size_label:
            raise ValueError("--target_size_class requires a size-conditioned dataset")
        target_size = int(args.target_size_class)
        if target_size < 0 or target_size >= num_size_classes:
            raise ValueError(f"--target_size_class must be in [0, {num_size_classes - 1}]")
        keep_indices = [i for i, label in enumerate(dataset.size_labels)
                        if int(label) == target_size]
        if not keep_indices:
            raise ValueError(f"No training examples found for size class {target_size}")
        dataset.image_files = [dataset.image_files[i] for i in keep_indices]
        for label_attribute in ("class_labels", "color_labels", "size_labels"):
            labels = getattr(dataset, label_attribute, None)
            if labels is not None:
                setattr(dataset, label_attribute, [labels[i] for i in keep_indices])
        print(f"Focused fine-tune: size class {target_size}, {len(dataset)} images")

    if args.overfit_samples is not None:
        if args.overfit_samples < 1 or args.overfit_samples > len(dataset):
            raise ValueError(f"--overfit_samples must be in [1, {len(dataset)}]")
        dataset.image_files = dataset.image_files[:args.overfit_samples]
        print(f"Overfit mode: restricting training to {len(dataset)} fixed images")

    batch_size = args.batch_size if args.batch_size else config['training']['batch_size']
    condition_sampler = None
    if args.balance_condition_combinations:
        if not (num_classes > 0 and num_color_classes > 0 and num_size_classes > 0):
            raise ValueError("Balanced condition sampling requires shape, color, and size labels")
        condition_keys = list(zip(dataset.class_labels, dataset.color_labels, dataset.size_labels))
        condition_counts = Counter(condition_keys)
        sample_weights = torch.tensor(
            [1.0 / condition_counts[key] for key in condition_keys], dtype=torch.double
        )
        condition_sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(dataset), replacement=True
        )
        print(
            f"Balanced sampling across {len(condition_counts)} shape/color/size combinations "
            f"(source counts {min(condition_counts.values())}-{max(condition_counts.values())})"
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=condition_sampler is None,
        sampler=condition_sampler,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False
    )

    # Create model
    print("Creating model...")
    prediction_type = config.get('diffusion', {}).get('prediction_type', 'epsilon')
    if prediction_type not in ('epsilon', 'v_prediction'):
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")
    structure_loss_weight = float(config.get('training', {}).get('structure_loss_weight', 0.0))
    structure_area_weight = float(config.get('training', {}).get('structure_area_weight', 0.25))
    foreground_threshold = float(config.get('training', {}).get('foreground_threshold', -0.95))
    occupancy_threshold = float(config.get('training', {}).get('occupancy_threshold', 0.0))
    foreground_occupancy_scale = float(
        config.get('training', {}).get('foreground_occupancy_scale', 4.0)
    )
    foreground_weight = float(
        args.foreground_weight
        if args.foreground_weight is not None
        else config.get('training', {}).get('foreground_weight', 0.0)
    )
    foreground_x0_weight = float(
        args.foreground_x0_weight
        if args.foreground_x0_weight is not None
        else config.get('training', {}).get('foreground_x0_weight', 0.0)
    )
    triangle_background_x0_weight = float(
        args.triangle_background_x0_weight
        if args.triangle_background_x0_weight is not None
        else config.get('training', {}).get('triangle_background_x0_weight', 0.0)
    )
    triangle_edge_weight = float(
        args.triangle_edge_weight
        if args.triangle_edge_weight is not None
        else config.get('training', {}).get('triangle_edge_weight', 0.0)
    )
    triangle_silhouette_weight = float(
        args.triangle_silhouette_weight
        if args.triangle_silhouette_weight is not None
        else config.get('training', {}).get('triangle_silhouette_weight', 0.0)
    )
    triangle_boundary_weight = float(
        args.triangle_boundary_weight
        if args.triangle_boundary_weight is not None
        else config.get('training', {}).get('triangle_boundary_weight', 0.0)
    )
    triangle_centroid_weight = float(
        args.triangle_centroid_weight
        if args.triangle_centroid_weight is not None
        else config.get('training', {}).get('triangle_centroid_weight', 0.0)
    )
    if structure_loss_weight < 0:
        raise ValueError("training.structure_loss_weight must be nonnegative")
    if structure_area_weight < 0:
        raise ValueError("training.structure_area_weight must be nonnegative")
    if not -1.0 < foreground_threshold < 1.0:
        raise ValueError("training.foreground_threshold must be between -1 and 1")
    if not -1.0 < occupancy_threshold < 1.0:
        raise ValueError("training.occupancy_threshold must be between -1 and 1")
    if foreground_occupancy_scale <= 0:
        raise ValueError("training.foreground_occupancy_scale must be positive")
    if structure_loss_weight > 0 and prediction_type != 'v_prediction':
        raise ValueError("Structure guidance currently requires v_prediction")
    if min(foreground_weight, foreground_x0_weight, triangle_background_x0_weight,
           triangle_edge_weight,
           triangle_silhouette_weight, triangle_boundary_weight,
           triangle_centroid_weight) < 0:
        raise ValueError("Auxiliary loss weights must be nonnegative")
    if (triangle_background_x0_weight > 0 or triangle_edge_weight > 0
            or triangle_silhouette_weight > 0
            or triangle_boundary_weight > 0 or triangle_centroid_weight > 0) and (
        num_classes <= 0 or args.triangle_class_id < 0 or args.triangle_class_id >= num_classes
    ):
        raise ValueError("Triangle silhouette/edge guidance requires a valid --triangle_class_id")
    if ((foreground_x0_weight > 0 or triangle_background_x0_weight > 0
         or triangle_edge_weight > 0
         or triangle_silhouette_weight > 0
         or triangle_boundary_weight > 0 or triangle_centroid_weight > 0)
            and prediction_type != 'v_prediction'):
        raise ValueError("Clean-pixel/silhouette guidance currently requires v_prediction")
    if args.position_augmentation_pixels < 0:
        raise ValueError("--position_augmentation_pixels must be nonnegative")
    config.setdefault('training', {})['position_augmentation_pixels'] = args.position_augmentation_pixels
    config['training']['structure_area_weight'] = structure_area_weight
    config['training']['foreground_threshold'] = foreground_threshold
    config['training']['occupancy_threshold'] = occupancy_threshold
    config['training']['foreground_occupancy_scale'] = foreground_occupancy_scale
    config['training']['foreground_weight'] = foreground_weight
    config['training']['foreground_x0_weight'] = foreground_x0_weight
    config['training']['triangle_background_x0_weight'] = triangle_background_x0_weight
    config['training']['triangle_edge_weight'] = triangle_edge_weight
    config['training']['triangle_silhouette_weight'] = triangle_silhouette_weight
    config['training']['triangle_boundary_weight'] = triangle_boundary_weight
    config['training']['triangle_centroid_weight'] = triangle_centroid_weight
    config['training']['balance_condition_combinations'] = bool(args.balance_condition_combinations)
    ema_decay = (float(args.ema_decay) if args.ema_decay is not None
                 else float(config.get('training', {}).get('ema_decay', 0.0)))
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    config['training']['ema_decay'] = ema_decay
    config.setdefault('model', {})['high_resolution_refinement'] = bool(
        config['model'].get('high_resolution_refinement', False)
        or args.high_resolution_refinement
        or args.high_resolution_refinement_only
    )
    if args.size_embedding_scale is not None:
        if args.size_embedding_scale <= 0:
            raise ValueError("--size_embedding_scale must be positive")
        config['model']['size_embedding_scale'] = float(args.size_embedding_scale)
    else:
        config['model'].setdefault('size_embedding_scale', 1.0)
    model = UNet(
        in_channels=config['model']['in_channels'],
        model_channels=model_channels,
        num_classes=num_classes,
        num_color_classes=num_color_classes,
        num_size_classes=num_size_classes,
        spatial_size_conditioning=config['model'].get('spatial_size_conditioning', False),
        high_resolution_refinement=config['model']['high_resolution_refinement'],
        size_embedding_scale=config['model']['size_embedding_scale'],
        middle_dilation=int(config['model'].get('middle_dilation', 1)),
        channel_mult=config['model']['channel_mult'],
        num_res_blocks=config['model']['num_res_blocks'],
        attention_resolutions=config['model']['attention_resolutions'],
        dropout=config['model']['dropout'],
        num_timesteps=config['diffusion']['num_timesteps']
    )
    if args.init_checkpoint:
        init_data = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        state = dict(init_data.get('model_state_dict', init_data))
        # Allow curriculum expansion (for example, circle/square to add
        # triangle) while preserving learned rows of conditioning embeddings.
        target_state = model.state_dict()
        for key in ('class_embed.weight', 'color_embed.weight'):
            if key in state and key in target_state and state[key].shape != target_state[key].shape:
                expanded = target_state[key].clone()
                rows = min(state[key].shape[0], expanded.shape[0])
                cols = min(state[key].shape[1], expanded.shape[1])
                expanded[:rows, :cols] = state[key][:rows, :cols]
                state[key] = expanded
        if ('conv_in.weight' in state and 'conv_in.weight' in target_state
                and state['conv_in.weight'].shape != target_state['conv_in.weight'].shape):
            expanded = target_state['conv_in.weight'].clone()
            old_weight = state['conv_in.weight']
            if (old_weight.shape[0] == expanded.shape[0]
                    and old_weight.shape[2:] == expanded.shape[2:]
                    and old_weight.shape[1] <= expanded.shape[1]):
                expanded[:, :old_weight.shape[1]] = old_weight
                state['conv_in.weight'] = expanded
        incompatible = model.load_state_dict(state, strict=False)
        allowed_missing = {key for key in incompatible.missing_keys
                           if key.startswith(('class_embed.', 'color_embed.', 'size_embed.',
                                              'high_res_refine.'))}
        if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Initialization checkpoint is structurally incompatible: "
                f"missing={incompatible.missing_keys[:5]}, "
                f"unexpected={incompatible.unexpected_keys[:5]}"
            )
        previous_prediction_type = init_data.get('config', {}).get('diffusion', {}).get('prediction_type', 'epsilon')
        if previous_prediction_type != prediction_type:
            raise ValueError(
                f"Cannot initialize {prediction_type} training from a "
                f"{previous_prediction_type} checkpoint"
            )
        print(f"Initialized model weights from {args.init_checkpoint}")
    model = model.to(device)

    if args.conditioning_only and args.high_resolution_refinement_only:
        raise ValueError("--conditioning_only and --high_resolution_refinement_only are mutually exclusive")
    if args.high_resolution_refinement_only:
        if model.high_res_refine is None:
            raise ValueError("--high_resolution_refinement_only requires the high-resolution refinement block")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('high_res_refine.'))
    elif args.conditioning_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(('class_embed.', 'size_embed.')))
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("The selected fine-tune has no trainable parameters")
    config.setdefault('training', {})['conditioning_only'] = bool(args.conditioning_only)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_parameters)
    print(f"Model parameters: {num_params:,} total; {trainable_count:,} trainable")
    ema_model = copy.deepcopy(model).eval() if ema_decay > 0 else None
    if ema_model is not None:
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        print(f"EMA enabled with decay {ema_decay:.4f}")

    sampling_model = ema_model if ema_model is not None else model

    def update_ema():
        if ema_model is None:
            return
        with torch.no_grad():
            for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
                ema_parameter.mul_(ema_decay).add_(parameter.detach(), alpha=1.0 - ema_decay)
            for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
                ema_buffer.copy_(buffer)

    # Create diffusion sampler
    diffusion = DiffusionSampler(
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        schedule=config['diffusion']['beta_schedule']
    )

    # Create optimizer
    optimizer = optim.AdamW(
        trainable_parameters,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Create checkpoint directory
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create output directory
    output_dir = args.output_dir if args.output_dir else "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Setup CSV logging
    csv_path = os.path.join(output_dir, "training_metrics.csv")
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'loss', 'structure_loss', 'foreground_x0_loss',
                         'triangle_background_x0_loss',
                         'triangle_edge_loss', 'triangle_silhouette_loss',
                         'triangle_boundary_loss', 'triangle_centroid_loss',
                         'low_t_loss', 'mid_t_loss', 'high_t_loss', 'grad_norm'])

    # Training loop
    num_steps = args.num_steps if args.num_steps else config['training']['num_steps']
    save_interval = config['training']['save_interval']
    log_interval = config['training']['log_interval']
    sample_interval = config['training']['sample_interval']

    print(f"\nStarting training for {num_steps} steps...")
    print(f"Save interval: {save_interval} steps")
    print(f"Log interval: {log_interval} steps")
    print(f"Sample interval: {sample_interval} steps\n")

    step = 0
    epoch = 0
    losses = []

    progress_bar = tqdm(total=num_steps, desc="Training")

    sample_config = config.get('training', {})
    sample_count = int(sample_config.get('sample_count', 4))
    if sample_count < 1:
        raise ValueError("training.sample_count must be positive")
    sample_generator = torch.Generator(device=device).manual_seed(seed + 2024)
    fixed_sample_noise = torch.randn(
        sample_count, config['model']['in_channels'], image_size, image_size,
        generator=sample_generator, device=device
    )

    def make_sample_labels(name, class_count):
        if class_count <= 0:
            return None
        values = sample_config.get(f'sample_{name}_ids')
        if values is not None:
            if len(values) != sample_count:
                raise ValueError(
                    f"training.sample_{name}_ids must contain exactly {sample_count} values"
                )
            labels = torch.as_tensor(values, device=device, dtype=torch.long)
        else:
            fixed_id = sample_config.get(f'sample_{name}_id')
            if fixed_id is not None:
                labels = torch.full((sample_count,), int(fixed_id), device=device, dtype=torch.long)
            else:
                labels = torch.arange(sample_count, device=device) % class_count
        if bool(((labels < 0) | (labels >= class_count)).any()):
            raise ValueError(f"training sample {name} IDs must be in [0, {class_count - 1}]")
        return labels

    fixed_sample_labels = make_sample_labels('shape', num_classes)
    fixed_sample_colors = make_sample_labels('color', num_color_classes)
    fixed_sample_sizes = make_sample_labels('size', num_size_classes)

    def save_fixed_samples(filename):
        sampling_model.eval()
        with torch.no_grad():
            samples = diffusion.process.ddim_sample_loop(
                model=sampling_model,
                batch_size=fixed_sample_noise.shape[0],
                channels=config['model']['in_channels'],
                image_size=image_size,
                num_sampling_steps=50,
                device=device,
                initial_noise=fixed_sample_noise.clone(),
                prediction_type=prediction_type,
                recompute_noise_after_clipping=True,
                class_labels=fixed_sample_labels,
                color_labels=fixed_sample_colors,
                size_labels=fixed_sample_sizes
            )
        path = os.path.join(output_dir, filename)
        save_image_grid(samples, path, nrow=2)
        model.train()
        return path

    # Save initial samples (step 0, before training)
    print("Generating initial samples (step 0)...")
    model.eval()
    sample_path = save_fixed_samples(f"samples_step_{step:06d}.png")
    print(f"Initial samples saved to {sample_path}")
    model.train()

    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            # Move batch to device
            if isinstance(batch, (tuple, list)):
                x_0 = batch[0].to(device)
                cursor = 1
                class_labels = batch[cursor].to(device=device, dtype=torch.long) if num_classes > 0 else None
                cursor += int(num_classes > 0)
                color_labels = batch[cursor].to(device=device, dtype=torch.long) if num_color_classes > 0 else None
                cursor += int(num_color_classes > 0)
                size_labels = batch[cursor].to(device=device, dtype=torch.long) if num_size_classes > 0 else None
            else:
                x_0 = batch.to(device)
                class_labels = None
                color_labels = None
                size_labels = None

            # Video clips contain translated objects, while the original image
            # curriculum recentered every training example. Teach the spatial
            # prior translation equivariance with exact integer shifts so this
            # augmentation adds no interpolation artifacts or shape cropping.
            x_0 = translate_foreground_batch(x_0, args.position_augmentation_pixels)

            # Sample random timesteps
            batch_size = x_0.shape[0]
            # Stratification keeps a tiny CPU batch from accidentally sampling
            # only one noise regime, while each example remains marginally uniform.
            timesteps = stratified_timesteps(
                batch_size, config['diffusion']['num_timesteps'], device
            )

            # One forward pass provides both the standard objective and
            # timestep-band diagnostics without adding CPU inference cost.
            x_t, target_noise = diffusion.process.q_sample(x_0, timesteps)
            if prediction_type == 'v_prediction':
                _, alpha_bar, one_minus_alpha_bar = diffusion.schedule.get(timesteps, device)
                target_noise = torch.sqrt(alpha_bar) * target_noise - torch.sqrt(one_minus_alpha_bar) * x_0
            prediction = model(x_t, timesteps, class_labels, color_labels, size_labels)
            foreground_target = (x_0 > foreground_threshold).any(dim=1, keepdim=True).to(prediction.dtype)
            pixel_error = (prediction - target_noise).square().mean(dim=1, keepdim=True)
            pixel_weights = 1.0 + foreground_weight * foreground_target
            per_sample_loss = (
                (pixel_error * pixel_weights).flatten(1).sum(1)
                / pixel_weights.flatten(1).sum(1).clamp_min(1.0)
            )
            structure_loss = prediction.new_zeros(())
            foreground_x0_loss = prediction.new_zeros(())
            triangle_background_x0_loss = prediction.new_zeros(())
            triangle_edge_loss = prediction.new_zeros(())
            triangle_silhouette_loss = prediction.new_zeros(())
            triangle_boundary_loss = prediction.new_zeros(())
            triangle_centroid_loss = prediction.new_zeros(())
            predicted_x0 = None
            if (structure_loss_weight > 0 or foreground_x0_weight > 0
                    or triangle_background_x0_weight > 0 or triangle_edge_weight > 0
                    or triangle_silhouette_weight > 0 or triangle_boundary_weight > 0
                    or triangle_centroid_weight > 0):
                predicted_x0 = torch.sqrt(alpha_bar) * x_t - torch.sqrt(one_minus_alpha_bar) * prediction
            # Auxiliary clean-image losses backpropagate through
            # x0 = sqrt(alpha_bar) * x_t - sqrt(1-alpha_bar) * v. Their
            # gradient therefore shrinks with sqrt(1-alpha_bar), especially
            # at the low-noise timesteps that determine final RGB/edge quality.
            # Compensate that conversion factor while suppressing very noisy
            # estimates and capping the largest low-noise weight.
            x0_gradient_confidence = None
            if predicted_x0 is not None:
                x0_gradient_confidence = (
                    alpha_bar.flatten().clamp(0.0, 1.0).sqrt()
                    / (1.0 - alpha_bar.flatten()).clamp_min(0.0).sqrt().clamp_min(0.05)
                ).clamp(max=8.0)
            if foreground_x0_weight > 0:
                foreground_x0_error = F.smooth_l1_loss(
                    predicted_x0, x_0, reduction='none', beta=0.1
                ).mean(dim=1, keepdim=True)
                foreground_x0_per_sample = (
                    (foreground_x0_error * foreground_target).flatten(1).sum(1)
                    / foreground_target.flatten(1).sum(1).clamp_min(1.0)
                )
                foreground_x0_loss = (
                    (foreground_x0_per_sample * x0_gradient_confidence).sum()
                    / x0_gradient_confidence.sum().clamp_min(1e-6)
                )
            if triangle_background_x0_weight > 0:
                if class_labels is None:
                    raise ValueError("Triangle background reconstruction requires class-conditioned labels")
                triangle_samples = (class_labels == int(args.triangle_class_id)).to(prediction.dtype)
                background_target = 1.0 - foreground_target
                background_x0_error = F.smooth_l1_loss(
                    predicted_x0, x_0, reduction='none', beta=0.1
                ).mean(dim=1, keepdim=True)
                background_per_sample = (
                    (background_x0_error * background_target).flatten(1).sum(1)
                    / background_target.flatten(1).sum(1).clamp_min(1.0)
                )
                background_confidence = x0_gradient_confidence * triangle_samples
                triangle_background_x0_loss = (
                    (background_per_sample * background_confidence).sum()
                    / background_confidence.sum().clamp_min(1e-6)
                )
            if structure_loss_weight > 0:
                structure_target = foreground_target.squeeze(1)
                foreground_probability = torch.sigmoid(
                    (predicted_x0.max(dim=1).values - occupancy_threshold)
                    * foreground_occupancy_scale
                )
                intersection = (foreground_probability * structure_target).flatten(1).sum(1)
                predicted_area = foreground_probability.flatten(1).sum(1)
                target_area = structure_target.flatten(1).sum(1)
                dice_loss = 1.0 - (2.0 * intersection + 1.0) / (predicted_area + target_area + 1.0)
                area_loss = F.smooth_l1_loss(
                    torch.log1p(predicted_area), torch.log1p(target_area), reduction='none'
                )
                per_sample_structure = dice_loss + structure_area_weight * area_loss
                # Mask and area estimates are most meaningful when x_t still
                # contains enough signal to recover the clean foreground.
                structure_loss = (
                    (per_sample_structure * x0_gradient_confidence).sum()
                    / x0_gradient_confidence.sum().clamp_min(1e-6)
                )
            if triangle_edge_weight > 0:
                # Finite differences directly supervise crisp contour transitions.
                # Restrict this term to triangle examples so it improves the weak
                # triangle prior without changing circle/square geometry.
                if class_labels is None:
                    raise ValueError("Triangle edge guidance requires class-conditioned labels")
                triangle_samples = (class_labels == int(args.triangle_class_id)).to(prediction.dtype)
                pred_dx = predicted_x0[:, :, :, 1:] - predicted_x0[:, :, :, :-1]
                target_dx = x_0[:, :, :, 1:] - x_0[:, :, :, :-1]
                pred_dy = predicted_x0[:, :, 1:, :] - predicted_x0[:, :, :-1, :]
                target_dy = x_0[:, :, 1:, :] - x_0[:, :, :-1, :]
                per_sample_edge = (
                    F.smooth_l1_loss(pred_dx, target_dx, reduction='none', beta=0.1).mean(dim=(1, 2, 3))
                    + F.smooth_l1_loss(pred_dy, target_dy, reduction='none', beta=0.1).mean(dim=(1, 2, 3))
                )
                edge_confidence = x0_gradient_confidence * triangle_samples
                triangle_edge_loss = (
                    (per_sample_edge * edge_confidence).sum()
                    / edge_confidence.sum().clamp_min(1e-6)
                )
            if triangle_silhouette_weight > 0 or triangle_boundary_weight > 0:
                if class_labels is None:
                    raise ValueError("Triangle silhouette guidance requires class-conditioned labels")
                triangle_samples = (class_labels == int(args.triangle_class_id)).to(prediction.dtype)
                # The RGB x0 head already represents foreground as a bright
                # palette pixel against a black background. Turn that signal
                # into a one-channel occupancy probability without adding a
                # second learned head or model parameters.
                occupancy_probability = torch.sigmoid(
                    (predicted_x0.max(dim=1, keepdim=True).values - occupancy_threshold)
                    * foreground_occupancy_scale
                )
                silhouette_confidence = x0_gradient_confidence * triangle_samples
                if triangle_silhouette_weight > 0:
                    pixel_bce = F.binary_cross_entropy(
                        occupancy_probability.clamp(1e-6, 1.0 - 1e-6),
                        foreground_target,
                        reduction='none',
                    )
                    positive = foreground_target
                    negative = 1.0 - foreground_target
                    positive_error = (pixel_bce * positive).flatten(1).sum(1) / positive.flatten(1).sum(1).clamp_min(1.0)
                    negative_error = (pixel_bce * negative).flatten(1).sum(1) / negative.flatten(1).sum(1).clamp_min(1.0)
                    per_sample_silhouette = 0.5 * (positive_error + negative_error)
                    triangle_silhouette_loss = (
                        (per_sample_silhouette * silhouette_confidence).sum()
                        / silhouette_confidence.sum().clamp_min(1e-6)
                    )
                if triangle_boundary_weight > 0:
                    # A 3x3 min-pool erosion gives the inner contour; max-pool
                    # expands it to a one-pixel band on both sides. These
                    # kernels are cheap at the 64x64 CPU curriculum resolution.
                    padded_target = F.pad(foreground_target, (1, 1, 1, 1), value=0.0)
                    eroded_target = -F.max_pool2d(-padded_target, kernel_size=3, stride=1)
                    target_boundary = foreground_target * (1.0 - eroded_target)
                    boundary_band = F.max_pool2d(target_boundary, kernel_size=3, stride=1, padding=1)
                    boundary_error = F.smooth_l1_loss(
                        occupancy_probability, foreground_target, reduction='none', beta=0.1
                    )
                    per_sample_boundary = (
                        (boundary_error * boundary_band).flatten(1).sum(1)
                        / boundary_band.flatten(1).sum(1).clamp_min(1.0)
                    )
                    triangle_boundary_loss = (
                        (per_sample_boundary * silhouette_confidence).sum()
                        / silhouette_confidence.sum().clamp_min(1e-6)
                    )
            if triangle_centroid_weight > 0:
                if class_labels is None:
                    raise ValueError("Triangle centroid guidance requires class-conditioned labels")
                triangle_samples = (class_labels == int(args.triangle_class_id)).to(prediction.dtype)
                # Match the soft foreground center of mass in image coordinates.
                # Remove the low-probability sigmoid background floor first so
                # a large black canvas cannot pull the estimate toward center.
                center_probability = (
                    torch.sigmoid(
                        (predicted_x0.max(dim=1).values - occupancy_threshold)
                        * foreground_occupancy_scale
                    ) - 0.05
                ).clamp_min(0.0)
                target_center_mask = foreground_target.squeeze(1)
                batch_height, batch_width = target_center_mask.shape[-2:]
                # Express centroid coordinates in the 64px curriculum scale
                # so this loss keeps the same magnitude when training a
                # higher-resolution model.
                x_coordinates = (
                    torch.arange(batch_width, device=x_0.device, dtype=x_0.dtype)
                    * (64.0 / batch_width)
                ).view(1, 1, batch_width)
                y_coordinates = (
                    torch.arange(batch_height, device=x_0.device, dtype=x_0.dtype)
                    * (64.0 / batch_height)
                ).view(1, batch_height, 1)
                predicted_mass = center_probability.sum(dim=(-2, -1)).clamp_min(1e-4)
                target_mass = target_center_mask.sum(dim=(-2, -1)).clamp_min(1.0)
                predicted_center = torch.stack((
                    (center_probability * x_coordinates).sum(dim=(-2, -1)) / predicted_mass,
                    (center_probability * y_coordinates).sum(dim=(-2, -1)) / predicted_mass,
                ), dim=-1)
                target_center = torch.stack((
                    (target_center_mask * x_coordinates).sum(dim=(-2, -1)) / target_mass,
                    (target_center_mask * y_coordinates).sum(dim=(-2, -1)) / target_mass,
                ), dim=-1)
                per_sample_centroid = F.smooth_l1_loss(
                    predicted_center, target_center, reduction='none', beta=1.0
                ).mean(dim=-1)
                centroid_confidence = x0_gradient_confidence * triangle_samples
                triangle_centroid_loss = (
                    (per_sample_centroid * centroid_confidence).sum()
                    / centroid_confidence.sum().clamp_min(1e-6)
                )
            loss = (per_sample_loss.mean() + structure_loss_weight * structure_loss
                    + foreground_x0_weight * foreground_x0_loss
                    + triangle_background_x0_weight * triangle_background_x0_loss
                    + triangle_edge_weight * triangle_edge_loss
                    + triangle_silhouette_weight * triangle_silhouette_loss
                    + triangle_boundary_weight * triangle_boundary_loss
                    + triangle_centroid_weight * triangle_centroid_loss)
            bands = [timesteps < 333, (timesteps >= 333) & (timesteps < 666), timesteps >= 666]
            band_losses = [
                per_sample_loss[mask].mean().item() if bool(mask.any()) else None
                for mask in bands
            ]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0).item()
            optimizer.step()
            update_ema()

            # Log loss
            losses.append(loss.item())

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'structure': f'{structure_loss.item():.4f}',
                'triangle_bg_x0': f'{triangle_background_x0_loss.item():.4f}',
                'triangle_edge': f'{triangle_edge_loss.item():.4f}',
                'triangle_silhouette': f'{triangle_silhouette_loss.item():.4f}',
                'triangle_boundary': f'{triangle_boundary_loss.item():.4f}',
                'triangle_centroid': f'{triangle_centroid_loss.item():.4f}'
            })

            # Log to CSV
            if step % log_interval == 0:
                csv_writer.writerow([step, loss.item(), structure_loss.item(),
                                     foreground_x0_loss.item(), triangle_background_x0_loss.item(),
                                     triangle_edge_loss.item(),
                                     triangle_silhouette_loss.item(), triangle_boundary_loss.item(),
                                     triangle_centroid_loss.item(), *band_losses, grad_norm])
                csv_file.flush()

            # Save samples
            if step % sample_interval == 0 and step > 0:
                print(f"\nGenerating samples at step {step}...")
                sampling_model.eval()
                with torch.no_grad():
                    samples = diffusion.process.ddim_sample_loop(
                        model=sampling_model,
                        batch_size=fixed_sample_noise.shape[0],
                        channels=config['model']['in_channels'],
                        image_size=image_size,
                        num_sampling_steps=50,
                        device=device,
                        initial_noise=fixed_sample_noise.clone(),
                        prediction_type=prediction_type,
                        recompute_noise_after_clipping=True,
                        class_labels=fixed_sample_labels,
                        color_labels=fixed_sample_colors,
                        size_labels=fixed_sample_sizes
                    )
                sample_path = os.path.join(output_dir, f"samples_step_{step:06d}.png")
                save_image_grid(samples, sample_path, nrow=2)
                print(f"Samples saved to {sample_path}")
                model.train()

            # Save checkpoint
            if step % save_interval == 0 and step > 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step:06d}.pt")
                CheckpointManager.save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=sampling_model,
                    optimizer=optimizer,
                    step=step,
                    config=config,
                    seed=seed,
                    extra_data=(
                        {"raw_model_state_dict": model.state_dict()}
                        if ema_model is not None else None
                    ),
                )

            step += 1

        epoch += 1

    progress_bar.close()

    # Save final checkpoint
    print("\nSaving final checkpoint...")
    final_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
    CheckpointManager.save_checkpoint(
        checkpoint_path=final_checkpoint_path,
        model=sampling_model,
        optimizer=optimizer,
        step=step,
        config=config,
        seed=seed,
        extra_data=(
            {"raw_model_state_dict": model.state_dict()}
            if ema_model is not None else None
        ),
    )

    # Save final samples
    print("Generating final samples...")
    sampling_model.eval()
    with torch.no_grad():
        samples = diffusion.process.ddim_sample_loop(
            model=sampling_model,
            batch_size=fixed_sample_noise.shape[0],
            channels=config['model']['in_channels'],
            image_size=image_size,
            num_sampling_steps=50,
            device=device,
            initial_noise=fixed_sample_noise.clone(),
            prediction_type=prediction_type,
            recompute_noise_after_clipping=True,
            class_labels=fixed_sample_labels,
            color_labels=fixed_sample_colors,
            size_labels=fixed_sample_sizes
        )
    final_sample_path = os.path.join(output_dir, "samples_final.png")
    save_image_grid(samples, final_sample_path, nrow=2)
    print(f"Final samples saved to {final_sample_path}")

    # Close CSV file
    csv_file.close()

    # Print summary
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Total steps: {step}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Average loss (last 100 steps): {np.mean(losses[-100:]):.6f}")
    print(f"Checkpoint saved to: {final_checkpoint_path}")
    print(f"Metrics saved to: {csv_path}")
    print(f"Samples saved to: {final_sample_path}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 image generation model")

    # Configuration
    parser.add_argument("--config", type=str, default="configs/image_v0.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                        help="Execution device; choose cpu to force native CPU training")

    # Training parameters
    parser.add_argument("--num_steps", type=int, default=None, help="Override number of training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--overfit_samples", type=int, default=None, help="Train repeatedly on this many initial dataset images")
    parser.add_argument("--image_size", type=int, default=None, help="Override image resolution")
    parser.add_argument("--model_channels", type=int, default=None, help="Override base U-Net width (must be divisible by 8)")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Initialize model weights from a compatible checkpoint (fresh optimizer)")
    parser.add_argument("--conditioning_only", action="store_true",
                        help="Fine-tune only shape and size embeddings; freeze the denoising backbone and color embedding")
    parser.add_argument("--target_shape_class", type=int, default=None,
                        help="Restrict a focused fine-tune to one shape label")
    parser.add_argument("--target_size_class", type=int, default=None,
                        help="Restrict a focused fine-tune to one size label")
    parser.add_argument("--balance_condition_combinations", action="store_true",
                        help="Sample shape/color/size combinations uniformly during training")
    parser.add_argument("--ema_decay", type=float, default=None,
                        help="Exponential moving average decay for checkpoint and sample weights (0 disables EMA)")
    parser.add_argument("--position_augmentation_pixels", type=int, default=0,
                        help="Random integer translation radius for clean images; shifts preserve the complete foreground")
    parser.add_argument("--foreground_weight", type=float, default=None,
                        help="Extra weight on foreground pixels; overrides the config when supplied")
    parser.add_argument("--foreground_x0_weight", type=float, default=None,
                        help="Low-noise clean-pixel loss over foreground pixels; defaults to config")
    parser.add_argument("--triangle_background_x0_weight", type=float, default=None,
                        help="Low-noise clean-pixel loss over triangle background pixels; defaults to config")
    parser.add_argument("--triangle_edge_weight", type=float, default=None,
                        help="Low-noise finite-difference triangle contour loss; defaults to config")
    parser.add_argument("--triangle_silhouette_weight", type=float, default=None,
                        help="Balanced binary silhouette loss from the predicted clean RGB image")
    parser.add_argument("--triangle_boundary_weight", type=float, default=None,
                        help="Local contour-band silhouette loss from the predicted clean RGB image")
    parser.add_argument("--triangle_centroid_weight", type=float, default=None,
                        help="Low-noise soft-centroid loss in pixel units for triangle placement")
    parser.add_argument("--high_resolution_refinement", action="store_true",
                        help="Add an identity-initialized full-resolution residual detail block")
    parser.add_argument("--high_resolution_refinement_only", action="store_true",
                        help="Train only the identity-initialized full-resolution detail block; implies --high_resolution_refinement")
    parser.add_argument("--size_embedding_scale", type=float, default=None,
                        help="Scale size-label embeddings (default preserves checkpoint config)")
    parser.add_argument("--triangle_class_id", type=int, default=2,
                        help="Class label selected by --triangle_edge_weight")

    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Train
    train(config, args)


if __name__ == "__main__":
    main()
