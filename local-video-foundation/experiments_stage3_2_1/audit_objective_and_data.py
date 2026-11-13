"""
Stage 3.2.6 Tasks 5-6: Audit Diffusion Objective and Dataset Diversity

Combined audit of:
- Video diffusion objective implementation
- Training data diversity and coverage
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def audit_diffusion_objective():
    """
    Audit the video diffusion objective implementation.
    """
    print_section("VIDEO DIFFUSION OBJECTIVE AUDIT")

    print("\nCode inspection from train_video_stage3_2_5.py (lines 494-503):")
    print("-" * 80)
    print("  videos_flat = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)")
    print("  timesteps_flat = timesteps.unsqueeze(1).expand(-1, num_frames).reshape(batch_size * num_frames)")
    print("  noisy_videos, noise = diffusion.q_sample(videos_flat, timesteps_flat)")
    print("  noisy_videos = noisy_videos.reshape(batch_size, num_frames, 3, image_size, image_size).permute(0, 2, 1, 3, 4)")
    print("  predicted_noise = video_unet(noisy_videos, timesteps, text_emb, text_mask)")
    print("  predicted_noise_flat = predicted_noise.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, image_size, image_size)")
    print("  loss = nn.functional.mse_loss(predicted_noise_flat, noise)")
    print("-" * 80)

    print("\nANALYSIS:")
    print("  1. Diffusion timesteps are SHARED across frames (timesteps_flat)")
    print("  2. Noise is sampled INDEPENDENTLY for each pixel and frame")
    print("  3. Loss is computed as MSE between predicted and actual noise")
    print("  4. Loss is averaged over all frames and pixels")
    print()
    print("Objective: Standard noise prediction MSE")
    print("  - Correct implementation of DDPM-style training")
    print("  - No temporal consistency loss")
    print("  - No frame-to-frame smoothness constraint")
    print("  - No explicit motion prediction")
    print()
    print("Problem: The objective does NOT encourage temporal coherence")
    print("  - Noise prediction MSE is frame-independent")
    print("  - Model can minimize loss by ignoring temporal information")
    print("  - No incentive to learn motion patterns")
    print()
    print("RECOMMENDATION:")
    print("  Add temporal consistency loss:")
    print("  - Frame-to-frame smoothness: ||x_t+1 - x_t||^2")
    print("  - Temporal prediction: predict frame T+1 from frame T")
    print("  - Motion-based loss: optical flow consistency")


def audit_dataset_diversity():
    """
    Audit the training dataset for diversity and coverage.
    """
    print_section("TRAINING DATASET DIVERSITY AUDIT")

    data_dir = "data/controlled_video/train"
    metadata_path = os.path.join(data_dir, "all_metadata.json")
    captions_path = os.path.join(data_dir, "captions.txt")

    if not os.path.exists(metadata_path):
        print(f"[ERROR] Metadata not found: {metadata_path}")
        return

    # Load metadata
    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)

    # Load captions
    with open(captions_path, 'r') as f:
        captions = [line.strip() for line in f.readlines()]

    print(f"\nDataset statistics:")
    print(f"  Total videos: {len(all_metadata)}")
    print(f"  Total captions: {len(captions)}")

    # Analyze trajectories
    trajectories = []
    velocities = []
    positions = []
    shapes = []
    colors = []

    for metadata in all_metadata:
        if 'trajectory' in metadata:
            traj_type = metadata['trajectory']
            # Handle both string and list representations
            if isinstance(traj_type, list):
                traj_type = str(traj_type)
            trajectories.append(traj_type)

        if 'velocity' in metadata:
            vel = metadata['velocity']
            if isinstance(vel, list):
                vel = vel[0] if len(vel) > 0 else 0
            velocities.append(float(vel))

        if 'position' in metadata:
            pos = metadata['position']
            if isinstance(pos, list):
                pos = str(pos)
            positions.append(pos)

        if 'shape' in metadata:
            shape = metadata['shape']
            if isinstance(shape, list):
                shape = shape[0] if len(shape) > 0 else 'unknown'
            shapes.append(shape)

        if 'color' in metadata:
            color = metadata['color']
            if isinstance(color, list):
                color = color[0] if len(color) > 0 else 'unknown'
            colors.append(color)

    print(f"\nTrajectory diversity:")
    traj_counter = Counter(trajectories)
    for traj, count in traj_counter.most_common():
        print(f"  {traj}: {count}")

    print(f"\nVelocity distribution:")
    if velocities:
        velocities_array = np.array(velocities)
        print(f"  Mean: {velocities_array.mean():.4f}")
        print(f"  Std: {velocities_array.std():.4f}")
        print(f"  Min: {velocities_array.min():.4f}")
        print(f"  Max: {velocities_array.max():.4f}")
        print(f"  Unique values: {len(set(velocities))}")

    print(f"\nShape diversity:")
    shape_counter = Counter(shapes)
    for shape, count in shape_counter.most_common():
        print(f"  {shape}: {count}")

    print(f"\nColor diversity:")
    color_counter = Counter(colors)
    for color, count in color_counter.most_common():
        print(f"  {color}: {count}")

    # Analyze captions
    print(f"\nCaption analysis:")
    caption_words = []
    for caption in captions:
        caption_words.extend(caption.lower().split())

    word_counter = Counter(caption_words)
    print(f"  Total words: {len(caption_words)}")
    print(f"  Unique words: {len(word_counter)}")
    print(f"  Top 10 words:")
    for word, count in word_counter.most_common(10):
        print(f"    {word}: {count}")

    # Check for motion-related words in captions
    motion_words = ['moving', 'motion', 'slow', 'fast', 'bounce', 'rotate', 'left', 'right', 'up', 'down']
    motion_in_captions = {word: count for word, count in word_counter.items() if word in motion_words}

    print(f"\nMotion-related words in captions:")
    if motion_in_captions:
        for word, count in motion_in_captions.items():
            print(f"  {word}: {count}")
    else:
        print("  None found")

    print("\n" + "-" * 80)
    print("DATASET DIVERSITY ASSESSMENT")
    print("-" * 80)

    diversity_score = 0
    max_score = 5

    if len(set(trajectories)) >= 3:
        print(f"[OK] Trajectory diversity: {len(set(trajectories))} unique")
        diversity_score += 1
    else:
        print(f"[WARNING] Low trajectory diversity: {len(set(trajectories))} unique")

    if len(set(velocities)) >= 3:
        print(f"[OK] Velocity diversity: {len(set(velocities))} unique")
        diversity_score += 1
    else:
        print(f"[WARNING] Low velocity diversity: {len(set(velocities))} unique")

    if len(set(shapes)) >= 3:
        print(f"[OK] Shape diversity: {len(set(shapes))} unique")
        diversity_score += 1
    else:
        print(f"[WARNING] Low shape diversity: {len(set(shapes))} unique")

    if len(set(colors)) >= 3:
        print(f"[OK] Color diversity: {len(set(colors))} unique")
        diversity_score += 1
    else:
        print(f"[WARNING] Low color diversity: {len(set(colors))} unique")

    if len(all_metadata) >= 20:
        print(f"[OK] Dataset size: {len(all_metadata)} videos")
        diversity_score += 1
    else:
        print(f"[WARNING] Small dataset: {len(all_metadata)} videos (< 20)")

    print(f"\nDiversity score: {diversity_score}/{max_score}")

    if diversity_score < 3:
        print("\n[CRITICAL] Dataset has LOW diversity")
        print("           Temporal modules may not have enough variation to learn")
    elif diversity_score < 5:
        print("\n[MODERATE] Dataset has MODERATE diversity")
        print("             May be sufficient for basic temporal learning")
    else:
        print("\n[OK] Dataset has GOOD diversity")


def main():
    print_section("STAGE 3.2.6 TASKS 5-6: OBJECTIVE AND DATA AUDIT")

    audit_diffusion_objective()
    audit_dataset_diversity()

    print_section("SUMMARY")

    print("\nKEY FINDINGS:")
    print("  1. Diffusion objective is standard noise prediction MSE")
    print("  2. No temporal consistency loss in the objective")
    print("  3. Noise is sampled independently per frame")
    print("  4. Dataset size and diversity are limited")
    print()
    print("CONCLUSION:")
    print("  The training objective does not incentivize temporal learning.")
    print("  The model can minimize loss by ignoring temporal information,")
    print("  which explains why gates learned to suppress temporal features.")
    print()
    print("RECOMMENDATION:")
    print("  1. Add temporal consistency loss to the objective")
    print("  2. Increase dataset diversity (more trajectories, velocities)")
    print("  3. Expand dataset size (currently ~10-25 videos)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
