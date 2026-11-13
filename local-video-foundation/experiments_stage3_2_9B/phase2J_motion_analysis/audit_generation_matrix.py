"""
Stage 3.2.9J - Generation Matrix Audit

Audits the complete generation matrix of 36 videos and validates their integrity.

Author: Devin
Date: 2026-11-03
"""

import json
import hashlib
from pathlib import Path
from PIL import Image
import numpy as np
try:
    import imageio
except ImportError:
    imageio = None

def compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def audit_generation_matrix():
    """Audit the complete generation matrix."""
    base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation/experiments_stage3_2_9B/phase2J_corrected_motion_evaluation/outputs_evaluation")
    prompt_mapping_path = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation/experiments_stage3_2_9B/phase2J_corrected_motion_evaluation/STAGE3_2_9I_PROMPT_MAPPING.json")

    # Load prompt mapping
    with open(prompt_mapping_path) as f:
        prompt_mapping = json.load(f)

    # Expected configuration
    configs = ['B', 'C', 'D']
    seeds = [42, 123, 456]
    prompts = [
        "A red object moving left",
        "A red object moving right",
        "A blue object moving diagonally",
        "A yellow object remaining stationary"
    ]

    expected_count = len(configs) * len(seeds) * len(prompts)  # 36

    audit_results = {
        'expected_count': expected_count,
        'configs': configs,
        'seeds': seeds,
        'prompts': prompts,
        'videos': {},
        'missing': [],
        'duplicate_hashes': [],
        'metadata_issues': [],
        'integrity_issues': []
    }

    # Check each expected combination
    for config in configs:
        for seed in seeds:
            for prompt_idx, prompt in enumerate(prompts):
                video_path = base_dir / config / f"seed_{seed}" / f"prompt_{prompt_idx:02d}" / "video.gif"
                metadata_path = base_dir / config / f"seed_{seed}" / f"prompt_{prompt_idx:02d}" / "metadata.json"

                video_key = f"{config}_seed{seed}_prompt{prompt_idx}"

                if not video_path.exists():
                    audit_results['missing'].append({
                        'key': video_key,
                        'config': config,
                        'seed': seed,
                        'prompt_idx': prompt_idx,
                        'prompt': prompt,
                        'video_path': str(video_path)
                    })
                    continue

                if not metadata_path.exists():
                    audit_results['metadata_issues'].append({
                        'key': video_key,
                        'config': config,
                        'seed': seed,
                        'prompt_idx': prompt_idx,
                        'prompt': prompt,
                        'metadata_path': str(metadata_path),
                        'issue': 'metadata file missing'
                    })
                    continue

                # Compute video hash
                video_hash = compute_file_hash(video_path)

                # Load metadata
                with open(metadata_path) as f:
                    metadata = json.load(f)

                # Validate metadata
                metadata_valid = True
                issues = []

                if metadata.get('config_name') != config:
                    issues.append(f"config_name mismatch: expected {config}, got {metadata.get('config_name')}")
                    metadata_valid = False

                if metadata.get('seed') != seed:
                    issues.append(f"seed mismatch: expected {seed}, got {metadata.get('seed')}")
                    metadata_valid = False

                if metadata.get('prompt') != prompt:
                    issues.append(f"prompt mismatch: expected '{prompt}', got '{metadata.get('prompt')}'")
                    metadata_valid = False

                if metadata.get('text_conditioning') != 'CORRECTED':
                    issues.append(f"text_conditioning mismatch: expected 'CORRECTED', got {metadata.get('text_conditioning')}")
                    metadata_valid = False

                if not metadata_valid:
                    audit_results['metadata_issues'].append({
                        'key': video_key,
                        'config': config,
                        'seed': seed,
                        'prompt_idx': prompt_idx,
                        'prompt': prompt,
                        'issues': issues
                    })

                # Store video info
                audit_results['videos'][video_key] = {
                    'config': config,
                    'seed': seed,
                    'prompt_idx': prompt_idx,
                    'prompt': prompt,
                    'video_path': str(video_path),
                    'metadata_path': str(metadata_path),
                    'video_hash': video_hash,
                    'metadata': metadata,
                    'metadata_valid': metadata_valid
                }

    # Check for duplicate hashes
    hash_to_keys = {}
    for key, info in audit_results['videos'].items():
        h = info['video_hash']
        if h not in hash_to_keys:
            hash_to_keys[h] = []
        hash_to_keys[h].append(key)

    for h, keys in hash_to_keys.items():
        if len(keys) > 1:
            audit_results['duplicate_hashes'].append({
                'hash': h,
                'keys': keys
            })

    # Summary
    audit_results['summary'] = {
        'total_found': len(audit_results['videos']),
        'total_missing': len(audit_results['missing']),
        'total_metadata_issues': len(audit_results['metadata_issues']),
        'total_duplicate_hashes': len(audit_results['duplicate_hashes']),
        'success': len(audit_results['missing']) == 0 and len(audit_results['metadata_issues']) == 0
    }

    return audit_results

def validate_video_integrity():
    """Validate video integrity for all videos."""
    base_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation/experiments_stage3_2_9B/phase2J_corrected_motion_evaluation/outputs_evaluation")

    integrity_results = {
        'videos': {},
        'invalid_videos': [],
        'summary': {}
    }

    configs = ['B', 'C', 'D']
    seeds = [42, 123, 456]
    prompts = [0, 1, 2, 3]

    for config in configs:
        for seed in seeds:
            for prompt_idx in prompts:
                video_path = base_dir / config / f"seed_{seed}" / f"prompt_{prompt_idx:02d}" / "video.gif"
                video_key = f"{config}_seed{seed}_prompt{prompt_idx}"

                if not video_path.exists():
                    continue

                try:
                    # Load video
                    if imageio is None:
                        integrity_results['videos'][video_key] = {
                            'status': 'skipped',
                            'reason': 'imageio not available'
                        }
                        continue

                    video = imageio.mimread(video_path)

                    # Check basic properties
                    issues = []

                    # Frame count
                    if len(video) != 16:
                        issues.append(f"frame_count mismatch: expected 16, got {len(video)}")

                    # Resolution
                    if video[0].shape[0] != 64 or video[0].shape[1] != 64:
                        issues.append(f"resolution mismatch: expected 64x64, got {video[0].shape[0]}x{video[0].shape[1]}")

                    # RGB channels
                    if len(video[0].shape) != 3 or video[0].shape[2] != 3:
                        issues.append(f"channel mismatch: expected RGB (3), got shape {video[0].shape}")

                    # Pixel range
                    video_array = np.array(video)
                    if video_array.min() < 0 or video_array.max() > 255:
                        issues.append(f"pixel range invalid: min={video_array.min()}, max={video_array.max()}")

                    # Check for NaN or infinity
                    if np.isnan(video_array).any():
                        issues.append("contains NaN values")

                    if np.isinf(video_array).any():
                        issues.append("contains infinity values")

                    # Check for blank frames
                    for i, frame in enumerate(video):
                        if frame.std() < 1.0:  # Very low variance
                            issues.append(f"frame {i} appears blank (std={frame.std():.2f})")

                    # Check for duplicate frames
                    for i in range(len(video) - 1):
                        if np.array_equal(video[i], video[i+1]):
                            issues.append(f"frames {i} and {i+1} are identical")

                    if issues:
                        integrity_results['invalid_videos'].append({
                            'key': video_key,
                            'video_path': str(video_path),
                            'issues': issues
                        })
                        integrity_results['videos'][video_key] = {
                            'status': 'invalid',
                            'issues': issues
                        }
                    else:
                        integrity_results['videos'][video_key] = {
                            'status': 'valid',
                            'frame_count': len(video),
                            'resolution': f"{video[0].shape[0]}x{video[0].shape[1]}",
                            'channels': video[0].shape[2],
                            'pixel_range': f"{video_array.min()}-{video_array.max()}",
                            'mean': float(video_array.mean()),
                            'std': float(video_array.std())
                        }

                except Exception as e:
                    integrity_results['invalid_videos'].append({
                        'key': video_key,
                        'video_path': str(video_path),
                        'issues': [f"exception during validation: {str(e)}"]
                    })
                    integrity_results['videos'][video_key] = {
                        'status': 'error',
                        'error': str(e)
                    }

    # Summary
    integrity_results['summary'] = {
        'total_checked': len(integrity_results['videos']),
        'total_valid': sum(1 for v in integrity_results['videos'].values() if v.get('status') == 'valid'),
        'total_invalid': len(integrity_results['invalid_videos']),
        'total_errors': sum(1 for v in integrity_results['videos'].values() if v.get('status') == 'error'),
        'success': len(integrity_results['invalid_videos']) == 0
    }

    return integrity_results

def main():
    """Run audit and validation."""
    print("=" * 80)
    print("STAGE 3.2.9J - GENERATION MATRIX AUDIT")
    print("=" * 80)

    # Audit generation matrix
    print("\nAuditing generation matrix...")
    audit_results = audit_generation_matrix()

    print(f"Expected videos: {audit_results['expected_count']}")
    print(f"Found videos: {audit_results['summary']['total_found']}")
    print(f"Missing videos: {audit_results['summary']['total_missing']}")
    print(f"Metadata issues: {audit_results['summary']['total_metadata_issues']}")
    print(f"Duplicate hashes: {audit_results['summary']['total_duplicate_hashes']}")
    print(f"Audit success: {audit_results['summary']['success']}")

    if audit_results['missing']:
        print("\nMissing videos:")
        for m in audit_results['missing']:
            print(f"  - {m['key']}: {m['config']}, seed {m['seed']}, prompt {m['prompt_idx']}")

    if audit_results['metadata_issues']:
        print("\nMetadata issues:")
        for m in audit_results['metadata_issues']:
            print(f"  - {m['key']}: {', '.join(m.get('issues', []))}")

    if audit_results['duplicate_hashes']:
        print("\nDuplicate hashes:")
        for d in audit_results['duplicate_hashes']:
            print(f"  - Hash {d['hash'][:16]}...: {', '.join(d['keys'])}")

    # Validate video integrity
    print("\n" + "=" * 80)
    print("VALIDATING VIDEO INTEGRITY")
    print("=" * 80)

    print("\nValidating video integrity...")
    integrity_results = validate_video_integrity()

    print(f"Total checked: {integrity_results['summary']['total_checked']}")
    print(f"Valid videos: {integrity_results['summary']['total_valid']}")
    print(f"Invalid videos: {integrity_results['summary']['total_invalid']}")
    print(f"Errors: {integrity_results['summary']['total_errors']}")
    print(f"Integrity success: {integrity_results['summary']['success']}")

    if integrity_results['invalid_videos']:
        print("\nInvalid videos:")
        for v in integrity_results['invalid_videos']:
            print(f"  - {v['key']}: {', '.join(v['issues'])}")

    # Save results
    output_dir = Path("C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation/experiments_stage3_2_9B/phase2J_motion_analysis")

    with open(output_dir / "STAGE3_2_9J_GENERATION_MATRIX_AUDIT.json", 'w') as f:
        json.dump(audit_results, f, indent=2)

    with open(output_dir / "STAGE3_2_9J_VIDEO_INTEGRITY.json", 'w') as f:
        json.dump(integrity_results, f, indent=2)

    # Create markdown report
    with open(output_dir / "STAGE3_2_9J_GENERATION_MATRIX_AUDIT.md", 'w') as f:
        f.write("# Stage 3.2.9J - Generation Matrix Audit Report\n\n")
        f.write(f"**Date**: 2026-11-03\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Expected videos: {audit_results['expected_count']}\n")
        f.write(f"- Found videos: {audit_results['summary']['total_found']}\n")
        f.write(f"- Missing videos: {audit_results['summary']['total_missing']}\n")
        f.write(f"- Metadata issues: {audit_results['summary']['total_metadata_issues']}\n")
        f.write(f"- Duplicate hashes: {audit_results['summary']['total_duplicate_hashes']}\n")
        f.write(f"- Audit success: {audit_results['summary']['success']}\n\n")
        f.write("## Video Integrity\n\n")
        f.write(f"- Total checked: {integrity_results['summary']['total_checked']}\n")
        f.write(f"- Valid videos: {integrity_results['summary']['total_valid']}\n")
        f.write(f"- Invalid videos: {integrity_results['summary']['total_invalid']}\n")
        f.write(f"- Errors: {integrity_results['summary']['total_errors']}\n")
        f.write(f"- Integrity success: {integrity_results['summary']['success']}\n\n")

        if audit_results['missing']:
            f.write("## Missing Videos\n\n")
            for m in audit_results['missing']:
                f.write(f"- {m['key']}: {m['config']}, seed {m['seed']}, prompt {m['prompt_idx']}\n")
            f.write("\n")

        if audit_results['metadata_issues']:
            f.write("## Metadata Issues\n\n")
            for m in audit_results['metadata_issues']:
                f.write(f"- {m['key']}: {', '.join(m.get('issues', []))}\n")
            f.write("\n")

        if integrity_results['invalid_videos']:
            f.write("## Invalid Videos\n\n")
            for v in integrity_results['invalid_videos']:
                f.write(f"- {v['key']}: {', '.join(v['issues'])}\n")
            f.write("\n")

        f.write("## Conclusion\n\n")
        if audit_results['summary']['success'] and integrity_results['summary']['success']:
            f.write("All 36 videos are present with valid metadata and pass integrity checks.\n")
        else:
            f.write("Issues detected. See above for details.\n")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"Results saved to {output_dir}")

    return audit_results, integrity_results

if __name__ == "__main__":
    main()
