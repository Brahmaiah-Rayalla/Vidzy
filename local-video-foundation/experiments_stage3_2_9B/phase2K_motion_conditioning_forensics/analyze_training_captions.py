"""
Stage 3.2.9K - Training Caption Analysis

Analyzes training captions to determine if motion descriptions are present.

Author: Devin
Date: 2026-11-04
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import re

def analyze_captions():
    """Analyze training captions for motion content."""
    base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")

    # Look for caption files
    caption_files = [
        base_dir / "experiments_stage3_2_7/data_synthetic/train_metadata.json",
        base_dir / "experiments_stage3_2_7/train_captions.json",
        base_dir / "experiments_stage3_2_7/captions.json",
        base_dir / "experiments_stage3_2_7/video_foundation/train_captions.json",
        base_dir / "experiments_stage3_2_7/video_foundation/captions.json",
    ]

    print("=" * 80)
    print("ANALYZING TRAINING CAPTIONS")
    print("=" * 80)

    captions = None
    caption_file = None

    for cf in caption_files:
        if cf.exists():
            print(f"\nFound caption file: {cf}")
            with open(cf, 'r') as f:
                captions = json.load(f)
            caption_file = cf
            break

    if captions is None:
        print("No caption file found")
        return None

    analysis = {
        'caption_file': str(caption_file),
        'total_captions': len(captions),
        'motion_word_counts': {},
        'motion_caption_count': 0,
        'sample_captions': [],
        'caption_length_stats': {}
    }

    # Motion-related keywords
    motion_keywords = [
        'moving', 'stationary', 'static', 'right', 'left', 'up', 'down',
        'leftward', 'rightward', 'upward', 'downward', 'horizontal', 'vertical',
        'diagonal', 'slow', 'fast', 'slowly', 'quickly', 'accelerating',
        'decelerating', 'bounce', 'bouncing', 'rotate', 'rotating', 'rotation',
        'oscillate', 'oscillating', 'swing', 'swinging', 'slide', 'sliding',
        'spin', 'spinning', 'forward', 'backward', 'motion', 'movement',
        'velocity', 'speed', 'clockwise', 'counterclockwise'
    ]

    print(f"\nTotal captions: {len(captions)}")

    # Analyze captions
    caption_lengths = []
    motion_captions = []

    for idx, item in enumerate(captions):
        if isinstance(item, dict):
            # Handle different caption formats
            if 'caption' in item:
                text = item['caption']
            elif 'text' in item:
                text = item['text']
            else:
                text = str(item)
        else:
            text = str(item)

        caption_lengths.append(len(text.split()))

        # Check for motion keywords
        text_lower = text.lower()
        has_motion = any(keyword in text_lower for keyword in motion_keywords)

        if has_motion:
            analysis['motion_caption_count'] += 1
            motion_captions.append(text)

            # Count motion keywords
            for keyword in motion_keywords:
                if keyword in text_lower:
                    analysis['motion_word_counts'][keyword] = analysis['motion_word_counts'].get(keyword, 0) + 1

        # Sample some captions
        if idx < 10:
            analysis['sample_captions'].append({
                'index': idx,
                'text': text,
                'has_motion': has_motion,
                'motion_type': item.get('motion_type', 'N/A') if isinstance(item, dict) else 'N/A'
            })

    # Calculate statistics
    analysis['caption_length_stats'] = {
        'mean': sum(caption_lengths) / len(caption_lengths),
        'min': min(caption_lengths),
        'max': max(caption_lengths)
    }

    analysis['motion_caption_percentage'] = (analysis['motion_caption_count'] / len(captions)) * 100

    print(f"\nMotion captions: {analysis['motion_caption_count']} ({analysis['motion_caption_percentage']:.2f}%)")
    print(f"Average caption length: {analysis['caption_length_stats']['mean']:.2f} words")

    print("\nMotion keyword counts:")
    for keyword, count in sorted(analysis['motion_word_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {keyword}: {count}")

    print("\nSample captions:")
    for sample in analysis['sample_captions']:
        motion_marker = "[MOTION]" if sample['has_motion'] else "[NO MOTION]"
        print(f"  {motion_marker} {sample['text']}")

    # Show some motion captions
    print("\nSample motion captions:")
    for i, caption in enumerate(motion_captions[:10]):
        print(f"  {i+1}. {caption}")

    return analysis

def main():
    """Run training caption analysis."""
    print("=" * 80)
    print("STAGE 3.2.9K - TRAINING CAPTION ANALYSIS")
    print("=" * 80)

    analysis = analyze_captions()

    if analysis is None:
        print("\nAnalysis failed: No caption file found")
        return None

    # Save results
    output_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation/experiments_stage3_2_9B/phase2K_motion_conditioning_forensics")

    with open(output_dir / "STAGE3_2_9K_TRAINING_CAPTION_ANALYSIS.json", 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to {output_dir}")

    return analysis

if __name__ == "__main__":
    main()
