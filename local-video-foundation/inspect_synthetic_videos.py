"""
Inspect synthetic video samples to verify quality and motion consistency.
"""

import os
import json
from PIL import Image
import numpy as np


def inspect_video_dataset(data_dir, split='train', num_videos=5):
    """Inspect a few video samples."""
    split_dir = os.path.join(data_dir, split)

    # Load captions
    captions_path = os.path.join(split_dir, 'captions.txt')
    with open(captions_path, 'r') as f:
        captions = f.readlines()

    print(f"Inspecting {num_videos} videos from {split} split...")
    print(f"Total videos in split: {len(captions)}")
    print()

    for i in range(min(num_videos, len(captions))):
        video_dir = os.path.join(split_dir, f"video_{i:05d}")
        metadata_path = os.path.join(video_dir, 'metadata.json')

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"Video {i}:")
        print(f"  Caption: {captions[i].strip()}")
        print(f"  Shape: {metadata['shape']}")
        print(f"  Color: {metadata['color']}")
        print(f"  Motion: {metadata['motion']}")
        print(f"  Size: {metadata['size_adjective']} ({metadata['object_size']}px)")
        print(f"  Speed: {metadata['speed_adjective']} ({metadata['velocity']:.2f} px/frame)")
        print(f"  Background: {metadata['background_color']}")
        print(f"  Start position: ({metadata['start_position'][0]:.1f}, {metadata['start_position'][1]:.1f})")
        print(f"  End position: ({metadata['trajectory'][-1][0]:.1f}, {metadata['trajectory'][-1][1]:.1f})")
        print(f"  Total displacement: {np.linalg.norm(np.array(metadata['trajectory'][-1]) - np.array(metadata['start_position'])):.1f} px")
        print()

        # Load and check a few frames
        frame_paths = sorted([f for f in os.listdir(video_dir) if f.endswith('.png')])
        print(f"  Number of frames: {len(frame_paths)}")

        # Check frame consistency
        frames = []
        for frame_path in frame_paths[:5]:  # Check first 5 frames
            frame = Image.open(os.path.join(video_dir, frame_path))
            frames.append(np.array(frame))

        print(f"  Frame shape: {frames[0].shape}")
        print(f"  Frame dtype: {frames[0].dtype}")
        print(f"  Value range: [{frames[0].min()}, {frames[0].max()}]")
        print()

    print("Inspection complete.")


if __name__ == "__main__":
    inspect_video_dataset("data/synthetic_video", split='train', num_videos=3)
