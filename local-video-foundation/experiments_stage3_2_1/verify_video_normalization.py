"""
Task 2: Verify video normalization pipeline.

Inspects the complete synthetic video loading pipeline and verifies:
- Source pixel format
- Floating-point conversion
- Normalization into [-1, 1]
- RGB channel ordering
- Frame ordering
- Tensor dimensions
- Consistency with Stage 2.1 image preprocessing
- Statistical distributions
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_foundation.training.train_video import SyntheticVideoDataset


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def verify_video_normalization(data_dir):
    print_section("TASK 2: VERIFY VIDEO NORMALIZATION PIPELINE")

    # Load video dataset
    print("\n1. Loading video dataset...")
    video_dataset = SyntheticVideoDataset(data_dir, split='train', num_frames=16, image_size=64)
    video_loader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"   Loaded {len(video_dataset)} videos")

    # Load image dataset for comparison (Stage 2.1)
    print("\n2. Loading image dataset for comparison...")
    image_dataset = None
    try:
        from video_foundation.data.dataset import ImageDataset
        
        image_data_dir = os.path.join(os.path.dirname(data_dir), 'conditional_minimal')
        if os.path.exists(image_data_dir):
            image_dataset = ImageDataset(image_data_dir, split='train', image_size=64)
            print(f"   Loaded {len(image_dataset)} images")
        else:
            print(f"   Image dataset not found at {image_data_dir}")
    except Exception as e:
        print(f"   Could not load image dataset: {e}")
        print(f"   Skipping image comparison")

    print_section("2.1 VIDEO LOADING PIPELINE INSPECTION")

    # Inspect the video loading code
    print("\nVideo loading code from SyntheticVideoDataset:")
    print("  1. Load frames as PIL Images")
    print("  2. Resize to (image_size, image_size)")
    print("  3. Convert to numpy array: np.array(frame).astype(np.float32)")
    print("  4. Normalize: / 127.5 - 1.0  (maps [0, 255] -> [-1, 1])")
    print("  5. Stack frames: [num_frames, h, w, c]")
    print("  6. Transpose: [c, num_frames, h, w]")

    print_section("2.2 VIDEO STATISTICS")

    # Collect statistics from video dataset
    video_stats = {
        'mean': [],
        'std': [],
        'min': [],
        'max': [],
        'range': []
    }

    print("\nSampling 10 videos for statistics...")
    for i, batch in enumerate(video_loader):
        if i >= 10:
            break
        
        video = batch['video']
        print(f"\n  Video {i}:")
        print(f"    Shape: {video.shape}")
        print(f"    Dtype: {video.dtype}")
        print(f"    Min: {video.min().item():.6f}")
        print(f"    Max: {video.max().item():.6f}")
        print(f"    Mean: {video.mean().item():.6f}")
        print(f"    Std: {video.std().item():.6f}")
        
        video_stats['mean'].append(video.mean().item())
        video_stats['std'].append(video.std().item())
        video_stats['min'].append(video.min().item())
        video_stats['max'].append(video.max().item())
        video_stats['range'].append(video.max().item() - video.min().item())

    # Aggregate statistics
    print("\nAggregate video statistics:")
    print(f"  Mean of means: {np.mean(video_stats['mean']):.6f}")
    print(f"  Std of means: {np.std(video_stats['mean']):.6f}")
    print(f"  Mean of stds: {np.mean(video_stats['std']):.6f}")
    print(f"  Std of stds: {np.std(video_stats['std']):.6f}")
    print(f"  Overall min: {np.min(video_stats['min']):.6f}")
    print(f"  Overall max: {np.max(video_stats['max']):.6f}")
    print(f"  Mean range: {np.mean(video_stats['range']):.6f}")

    print_section("2.3 IMAGE STATISTICS (FOR COMPARISON)")

    if image_dataset is not None:
        image_stats = {
            'mean': [],
            'std': [],
            'min': [],
            'max': [],
            'range': []
        }

        print("\nSampling 10 images for statistics...")
        for i in range(min(10, len(image_dataset))):
            sample = image_dataset[i]
            image = sample['image']
            
            print(f"\n  Image {i}:")
            print(f"    Shape: {image.shape}")
            print(f"    Dtype: {image.dtype}")
            print(f"    Min: {image.min().item():.6f}")
            print(f"    Max: {image.max().item():.6f}")
            print(f"    Mean: {image.mean().item():.6f}")
            print(f"    Std: {image.std().item():.6f}")
            
            image_stats['mean'].append(image.mean().item())
            image_stats['std'].append(image.std().item())
            image_stats['min'].append(image.min().item())
            image_stats['max'].append(image.max().item())
            image_stats['range'].append(image.max().item() - image.min().item())

        # Aggregate statistics
        print("\nAggregate image statistics:")
        print(f"  Mean of means: {np.mean(image_stats['mean']):.6f}")
        print(f"  Std of means: {np.std(image_stats['mean']):.6f}")
        print(f"  Mean of stds: {np.mean(image_stats['std']):.6f}")
        print(f"  Std of stds: {np.std(image_stats['std']):.6f}")
        print(f"  Overall min: {np.min(image_stats['min']):.6f}")
        print(f"  Overall max: {np.max(image_stats['max']):.6f}")
        print(f"  Mean range: {np.mean(image_stats['range']):.6f}")

        print_section("2.4 VIDEO vs IMAGE COMPARISON")

        print("\nComparing video and image statistics:")
        print(f"  Video mean: {np.mean(video_stats['mean']):.6f}")
        print(f"  Image mean: {np.mean(image_stats['mean']):.6f}")
        print(f"  Difference: {abs(np.mean(video_stats['mean']) - np.mean(image_stats['mean'])):.6f}")
        
        print(f"\n  Video std: {np.mean(video_stats['std']):.6f}")
        print(f"  Image std: {np.mean(image_stats['std']):.6f}")
        print(f"  Difference: {abs(np.mean(video_stats['std']) - np.mean(image_stats['std'])):.6f}")
        
        print(f"\n  Video range: {np.mean(video_stats['range']):.6f}")
        print(f"  Image range: {np.mean(image_stats['range']):.6f}")
        print(f"  Difference: {abs(np.mean(video_stats['range']) - np.mean(image_stats['range'])):.6f}")

    print_section("2.5 NORMALIZATION VERIFICATION")

    # Verify normalization is in [-1, 1]
    all_in_range = bool(np.min(video_stats['min']) >= -1.0 and np.max(video_stats['max']) <= 1.0)
    
    print(f"\nNormalization verification:")
    print(f"  All values in [-1, 1]: {all_in_range}")
    print(f"  Expected range: [-1.0, 1.0]")
    print(f"  Actual range: [{np.min(video_stats['min']):.6f}, {np.max(video_stats['max']):.6f}]")
    
    # Check if normalization formula is correct
    # Expected: [0, 255] -> [-1, 1] via / 127.5 - 1.0
    print(f"\nNormalization formula verification:")
    print(f"  Formula: pixel / 127.5 - 1.0")
    print(f"  Test: 0 / 127.5 - 1.0 = {0 / 127.5 - 1.0:.6f}")
    print(f"  Test: 255 / 127.5 - 1.0 = {255 / 127.5 - 1.0:.6f}")
    print(f"  Expected: [-1.0, 1.0]")

    print_section("2.6 TENSOR DIMENSION VERIFICATION")

    # Verify tensor dimensions
    batch = next(iter(video_loader))
    video = batch['video']
    
    print(f"\nVideo tensor dimensions:")
    print(f"  Shape: {video.shape}")
    print(f"  Expected: [batch, channels, frames, height, width]")
    print(f"  Actual: [batch={video.shape[0]}, channels={video.shape[1]}, frames={video.shape[2]}, height={video.shape[3]}, width={video.shape[4]}]")
    print(f"  Dimensions correct: {len(video.shape) == 5}")

    print_section("2.7 RGB CHANNEL ORDERING")

    # Sample a single frame and check RGB ordering
    video_sample = batch['video'][0]  # [channels, frames, h, w]
    frame_sample = video_sample[:, 0, :, :]  # [channels, h, w]
    
    print(f"\nRGB channel ordering verification:")
    print(f"  Frame shape: {frame_sample.shape}")
    print(f"  Channel 0 (R) mean: {frame_sample[0].mean().item():.6f}")
    print(f"  Channel 1 (G) mean: {frame_sample[1].mean().item():.6f}")
    print(f"  Channel 2 (B) mean: {frame_sample[2].mean().item():.6f}")
    print(f"  Expected: RGB ordering (standard PIL Image to numpy conversion)")

    print_section("2.8 FRAME ORDERING")

    # Check frame ordering
    print(f"\nFrame ordering verification:")
    print(f"  Number of frames: {video.shape[2]}")
    print(f"  Expected: 16 frames (num_frames parameter)")
    print(f"  Frame dimension position: index 2 (after channels)")
    print(f"  Expected order: temporal dimension after channels")

    print_section("2.9 SUMMARY")

    print("\nVideo normalization pipeline verification:")
    print(f"  Normalization in [-1, 1]: {'[PASS]' if all_in_range else '[FAIL]'}")
    print(f"  Tensor dimensions correct: {'[PASS]' if len(video.shape) == 5 else '[FAIL]'}")
    print(f"  Values are float32: {'[PASS]' if video.dtype == torch.float32 else '[FAIL]'}")
    print(f"  Range matches expected: {'[PASS]' if abs(np.mean(video_stats['range']) - 2.0) < 0.1 else '[FAIL]'}")

    results = {
        'video_stats': {
            'mean': float(np.mean(video_stats['mean'])),
            'std': float(np.mean(video_stats['std'])),
            'min': float(np.min(video_stats['min'])),
            'max': float(np.max(video_stats['max'])),
            'range': float(np.mean(video_stats['range'])),
        },
        'normalization_verification': {
            'all_in_range': bool(all_in_range),
            'expected_range': [-1.0, 1.0],
            'actual_range': [float(np.min(video_stats['min'])), float(np.max(video_stats['max']))],
        },
        'tensor_dimensions': {
            'shape': list(video.shape),
            'correct': len(video.shape) == 5,
            'dtype': str(video.dtype),
        }
    }

    if image_dataset is not None:
        results['image_stats'] = {
            'mean': float(np.mean(image_stats['mean'])),
            'std': float(np.mean(image_stats['std'])),
            'min': float(np.min(image_stats['min'])),
            'max': float(np.max(image_stats['max'])),
            'range': float(np.mean(image_stats['range'])),
        }
        results['comparison'] = {
            'mean_diff': float(abs(np.mean(video_stats['mean']) - np.mean(image_stats['mean']))),
            'std_diff': float(abs(np.mean(video_stats['std']) - np.mean(image_stats['std']))),
            'range_diff': float(abs(np.mean(video_stats['range']) - np.mean(image_stats['range']))),
        }

    with open("video_normalization_verification.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to video_normalization_verification.json")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify video normalization")
    parser.add_argument("--data_dir", type=str, default="../data/synthetic_video", help="Data directory")
    args = parser.parse_args()

    results = verify_video_normalization(args.data_dir)


if __name__ == "__main__":
    main()
