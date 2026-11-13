"""
Audit video preprocessing and normalization for Stage 3.2.3 Task 1.

This script inspects the complete video dataset loading and preprocessing pipeline.
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.training.train_video import SyntheticVideoDataset


def audit_normalization(data_dir, split='train'):
    """Audit video normalization pipeline."""
    print("=" * 80)
    print("VIDEO NORMALIZATION AUDIT")
    print("=" * 80)

    # Load dataset
    print(f"\nLoading dataset from {data_dir}/{split}...")
    dataset = SyntheticVideoDataset(data_dir, split=split, num_frames=16, image_size=64)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Collect statistics
    all_videos = []
    all_frames = []

    print(f"\nProcessing {len(dataset)} videos...")
    for batch in tqdm(dataloader):
        video = batch['video']  # [batch, channels, frames, h, w]
        all_videos.append(video)
        # Collect individual frames
        for frame_idx in range(video.shape[2]):
            frame = video[0, :, frame_idx, :, :]  # [channels, h, w]
            all_frames.append(frame)

    # Stack all videos and frames
    all_videos_tensor = torch.cat(all_videos, dim=0)  # [num_videos, channels, frames, h, w]
    all_frames_tensor = torch.stack(all_frames, dim=0)  # [num_frames_total, channels, h, w]

    print("\n" + "=" * 80)
    print("VIDEO-LEVEL STATISTICS")
    print("=" * 80)

    # Overall statistics
    video_flat = all_videos_tensor.flatten()
    print(f"\nOverall Video Statistics:")
    print(f"  Shape: {all_videos_tensor.shape}")
    print(f"  Min: {video_flat.min().item():.6f}")
    print(f"  Max: {video_flat.max().item():.6f}")
    print(f"  Mean: {video_flat.mean().item():.6f}")
    print(f"  Std: {video_flat.std().item():.6f}")
    print(f"  Median: {video_flat.median().item():.6f}")

    # Per-channel statistics
    print(f"\nPer-Channel Statistics:")
    for c in range(all_videos_tensor.shape[1]):
        channel_data = all_videos_tensor[:, c, :, :, :].flatten()
        print(f"  Channel {c}:")
        print(f"    Min: {channel_data.min().item():.6f}")
        print(f"    Max: {channel_data.max().item():.6f}")
        print(f"    Mean: {channel_data.mean().item():.6f}")
        print(f"    Std: {channel_data.std().item():.6f}")

    # Per-frame statistics
    print(f"\nPer-Frame Statistics (aggregated across all videos):")
    frame_stats = []
    for f in range(all_videos_tensor.shape[2]):
        frame_data = all_videos_tensor[:, :, f, :, :].flatten()
        frame_stats.append({
            'frame': f,
            'min': frame_data.min().item(),
            'max': frame_data.max().item(),
            'mean': frame_data.mean().item(),
            'std': frame_data.std().item()
        })
        print(f"  Frame {f}:")
        print(f"    Min: {frame_data.min().item():.6f}")
        print(f"    Max: {frame_data.max().item():.6f}")
        print(f"    Mean: {frame_data.mean().item():.6f}")
        print(f"    Std: {frame_data.std().item():.6f}")

    # Frame-level statistics
    print("\n" + "=" * 80)
    print("FRAME-LEVEL STATISTICS (across all videos)")
    print("=" * 80)

    frame_flat = all_frames_tensor.flatten()
    print(f"\nOverall Frame Statistics:")
    print(f"  Number of frames: {all_frames_tensor.shape[0]}")
    print(f"  Shape per frame: {all_frames_tensor.shape[1:]}")
    print(f"  Min: {frame_flat.min().item():.6f}")
    print(f"  Max: {frame_flat.max().item():.6f}")
    print(f"  Mean: {frame_flat.mean().item():.6f}")
    print(f"  Std: {frame_flat.std().item():.6f}")

    # Check if normalization is correct
    print("\n" + "=" * 80)
    print("NORMALIZATION VALIDATION")
    print("=" * 80)

    expected_range = [-1.0, 1.0]
    actual_min = video_flat.min().item()
    actual_max = video_flat.max().item()

    print(f"\nExpected range: [{expected_range[0]:.1f}, {expected_range[1]:.1f}]")
    print(f"Actual range:   [{actual_min:.6f}, {actual_max:.6f}]")

    range_ok = (expected_range[0] - 0.1 <= actual_min <= expected_range[0] + 0.1 and
                expected_range[1] - 0.1 <= actual_max <= expected_range[1] + 0.1)

    print(f"\nRange check: {'[PASS]' if range_ok else '[FAIL]'}")

    # Note about mean/std
    print(f"\nNote: Mean and standard deviation depend on dataset composition.")
    print(f"  Do not expect mean=0 and std=0.5 for arbitrary datasets.")
    print(f"  These values depend on object coverage and background color.")

    # Check sample raw frames for verification
    print("\n" + "=" * 80)
    print("RAW FRAME INSPECTION")
    print("=" * 80)

    # Load a raw frame from disk to verify transformation
    sample_video_dir = os.path.join(data_dir, split, dataset.video_dirs[0])
    sample_frame_path = os.path.join(sample_video_dir, "frame_0000.png")

    print(f"\nLoading raw frame from: {sample_frame_path}")
    raw_image = Image.open(sample_frame_path)
    raw_array = np.array(raw_image)

    print(f"\nRaw frame statistics:")
    print(f"  Data type: {raw_array.dtype}")
    print(f"  Shape: {raw_array.shape}")
    print(f"  Min: {raw_array.min()}")
    print(f"  Max: {raw_array.max()}")
    print(f"  Mean: {raw_array.mean():.2f}")

    # Apply the normalization transformation manually
    print(f"\nApplying normalization: x = x.float() / 127.5 - 1.0")
    normalized = raw_array.astype(np.float32) / 127.5 - 1.0
    print(f"  Normalized min: {normalized.min():.6f}")
    print(f"  Normalized max: {normalized.max():.6f}")
    print(f"  Normalized mean: {normalized.mean():.6f}")
    print(f"  Normalized std: {normalized.std():.6f}")

    # Compare with dataset output
    dataset_frame = all_frames_tensor[0].numpy()  # First frame, all channels [channels, h, w]
    print(f"\nDataset output frame statistics:")
    print(f"  Shape: {dataset_frame.shape}")
    print(f"  Min: {dataset_frame.min():.6f}")
    print(f"  Max: {dataset_frame.max():.6f}")
    print(f"  Mean: {dataset_frame.mean():.6f}")
    print(f"  Std: {dataset_frame.std():.6f}")

    # Transpose normalized to match dataset format [channels, h, w]
    normalized_transposed = normalized.transpose(2, 0, 1)
    transformation_match = np.allclose(normalized_transposed, dataset_frame, atol=1e-5)
    print(f"\nTransformation match: {'[PASS]' if transformation_match else '[FAIL]'}")

    # Check for double normalization
    if normalized.min() < -1.1 or normalized.max() > 1.1:
        print(f"\n[WARNING] Values outside [-1, 1] range - possible double normalization")
    if transformation_match:
        print(f"\n[INFO] Dataset uses correct normalization: x = x.float() / 127.5 - 1.0")

    print("\n" + "=" * 80)

    return {
        'video_stats': {
            'min': float(video_flat.min().item()),
            'max': float(video_flat.max().item()),
            'mean': float(video_flat.mean().item()),
            'std': float(video_flat.std().item()),
            'shape': list(all_videos_tensor.shape)
        },
        'channel_stats': [
            {
                'channel': c,
                'min': float(all_videos_tensor[:, c, :, :, :].flatten().min().item()),
                'max': float(all_videos_tensor[:, c, :, :, :].flatten().max().item()),
                'mean': float(all_videos_tensor[:, c, :, :, :].flatten().mean().item()),
                'std': float(all_videos_tensor[:, c, :, :, :].flatten().std().item())
            }
            for c in range(all_videos_tensor.shape[1])
        ],
        'frame_stats': frame_stats,
        'raw_frame': {
            'dtype': str(raw_array.dtype),
            'shape': list(raw_array.shape),
            'min': int(raw_array.min()),
            'max': int(raw_array.max()),
            'mean': float(raw_array.mean())
        },
        'normalized_raw': {
            'min': float(normalized.min()),
            'max': float(normalized.max()),
            'mean': float(normalized.mean()),
            'std': float(normalized.std())
        },
        'dataset_frame': {
            'min': float(dataset_frame.min()),
            'max': float(dataset_frame.max()),
            'mean': float(dataset_frame.mean()),
            'std': float(dataset_frame.std())
        },
        'transformation_match': transformation_match,
        'range_ok': range_ok
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Audit video normalization")
    parser.add_argument("--data_dir", type=str, default="data/controlled_video", help="Data directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    args = parser.parse_args()

    results = audit_normalization(args.data_dir, args.split)

    # Save results
    output_path = "normalization_audit.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
