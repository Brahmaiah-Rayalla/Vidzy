"""
Audit generated samples for quality and check against training data.
"""

import os
import numpy as np
from PIL import Image
import torch

print("="*60)
print("Sample Quality Audit")
print("="*60)

# Load generated samples
print("\n1. Loading generated samples...")
gen_dir = "outputs/final_samples"
gen_files = sorted([f for f in os.listdir(gen_dir) if f.endswith('.png') and f != 'grid.png'])
print(f"   Found {len(gen_files)} generated samples")

# Load training samples
print("\n2. Loading training samples...")
train_dir = "data/synthetic"
train_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.png') and f != 'preview_grid.png'])
print(f"   Found {len(train_files)} training images")

# Check for exact matches
print("\n3. Checking for exact matches with training data...")
gen_hashes = {}
for f in gen_files:
    img = Image.open(os.path.join(gen_dir, f))
    gen_hashes[f] = hash(img.tobytes())

train_hashes = {}
for f in train_files:
    img = Image.open(os.path.join(train_dir, f))
    train_hashes[f] = hash(img.tobytes())

matches = 0
for gen_f, gen_hash in gen_hashes.items():
    if gen_hash in train_hashes.values():
        matches += 1
        print(f"   WARNING: {gen_f} matches a training image")

if matches == 0:
    print(f"   [OK] No exact matches found")

# Check pixel similarity
print("\n4. Checking pixel similarity...")
# Sample a few training images for comparison
sample_train_files = train_files[:100]

similarities = []
for gen_f in gen_files[:16]:  # Check first 16 generated
    gen_img = np.array(Image.open(os.path.join(gen_dir, gen_f)))

    max_sim = 0
    for train_f in sample_train_files:
        train_img = np.array(Image.open(os.path.join(train_dir, train_f)))
        if gen_img.shape == train_img.shape:
            # Compute normalized correlation
            diff = np.abs(gen_img.astype(float) - train_img.astype(float))
            sim = 1.0 - diff.mean() / 255.0
            max_sim = max(max_sim, sim)

    similarities.append(max_sim)

print(f"   Maximum similarity to training data: {max(similarities):.4f}")
print(f"   Average maximum similarity: {np.mean(similarities):.4f}")

if max(similarities) < 0.95:
    print(f"   [OK] No highly similar copies found (threshold: 0.95)")
else:
    print(f"   WARNING: Some samples are very similar to training data")

# Check diversity among generated samples
print("\n5. Checking diversity among generated samples...")
diversity_scores = []
for i in range(len(gen_files)):
    for j in range(i+1, len(gen_files)):
        img1 = np.array(Image.open(os.path.join(gen_dir, gen_files[i])))
        img2 = np.array(Image.open(os.path.join(gen_dir, gen_files[j])))
        if img1.shape == img2.shape:
            diff = np.abs(img1.astype(float) - img2.astype(float))
            sim = 1.0 - diff.mean() / 255.0
            diversity_scores.append(sim)

if diversity_scores:
    print(f"   Average similarity between generated samples: {np.mean(diversity_scores):.4f}")
    print(f"   Minimum similarity: {min(diversity_scores):.4f}")
    print(f"   Maximum similarity: {max(diversity_scores):.4f}")

    if np.mean(diversity_scores) < 0.9:
        print(f"   [OK] Generated samples show good diversity")
    else:
        print(f"   WARNING: Generated samples are very similar to each other")

# Check for collapse to simple patterns
print("\n6. Checking for mode collapse...")
# Compute standard deviation across generated samples
all_pixels = []
for gen_f in gen_files:
    img = np.array(Image.open(os.path.join(gen_dir, gen_f)))
    all_pixels.append(img.flatten())

all_pixels = np.concatenate(all_pixels)
print(f"   Pixel mean: {all_pixels.mean():.2f}")
print(f"   Pixel std: {all_pixels.std():.2f}")

if all_pixels.std() > 30:
    print(f"   [OK] Pixel distribution has reasonable variance")
else:
    print(f"   WARNING: Low pixel variance - possible mode collapse")

# Check for visual patterns
print("\n7. Visual inspection notes:")
print(f"   Review the grid images in outputs/final_samples/grid.png")
print(f"   and outputs/final_samples2/grid.png")
print(f"   Look for:")
print(f"   - Recognizable geometric shapes (circles, squares, triangles)")
print(f"   - Clear boundaries between shapes and background")
print(f"   - Variation in colors and positions")
print(f"   - Absence of severe artifacts or noise patterns")

print(f"\n8. Summary:")
print(f"   - Total generated samples: {len(gen_files)}")
print(f"   - Exact matches with training: {matches}")
print(f"   - Maximum similarity to training: {max(similarities):.4f}")
if diversity_scores:
    print(f"   - Average similarity between generated: {np.mean(diversity_scores):.4f}")
else:
    print(f"   - Average similarity between generated: N/A")
print(f"   - Pixel std dev: {all_pixels.std():.2f}")
