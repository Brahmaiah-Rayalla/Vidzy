"""
Analyze training metrics and generate loss curve.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

# Read metrics
steps = []
losses = []

with open('outputs/training_metrics.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row['step']))
        losses.append(float(row['loss']))

steps = np.array(steps)
losses = np.array(losses)

print(f"Number of logged steps: {len(steps)}")
print(f"Step range: {steps[0]} to {steps[-1]}")
print(f"\nLoss statistics:")
print(f"  Initial loss (step 0): {losses[0]:.6f}")
print(f"  Final loss (step {steps[-1]}): {losses[-1]:.6f}")
print(f"  Minimum loss: {losses.min():.6f}")
print(f"  Maximum loss: {losses.max():.6f}")
print(f"  Mean loss (all): {losses.mean():.6f}")

# First 100 steps
first_100 = losses[:10]  # Only have every 10 steps
print(f"\nFirst 100 steps (logged every 10):")
print(f"  Mean: {first_100.mean():.6f}")
print(f"  Min: {first_100.min():.6f}")
print(f"  Max: {first_100.max():.6f}")

# Last 100 steps
last_100 = losses[-10:]
print(f"\nLast 100 steps (logged every 10):")
print(f"  Mean: {last_100.mean():.6f}")
print(f"  Min: {last_100.min():.6f}")
print(f"  Max: {last_100.max():.6f}")

# Check for NaN or infinite
print(f"\nData quality check:")
print(f"  NaN values: {np.isnan(losses).sum()}")
print(f"  Infinite values: {np.isinf(losses).sum()}")

# Mean loss per 100-step interval
print(f"\nMean loss per 100-step interval:")
for i in range(0, 2000, 100):
    interval_losses = losses[(steps >= i) & (steps < i + 100)]
    if len(interval_losses) > 0:
        print(f"  Steps {i:4d}-{i+99:4d}: {interval_losses.mean():.6f} (n={len(interval_losses)})")

# Plot loss curve
plt.figure(figsize=(12, 6))
plt.plot(steps, losses, 'b-', linewidth=1, alpha=0.7, label='Training loss')

# Add moving average
window = 10
if len(losses) >= window:
    moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
    plt.plot(steps[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving average (window={window})')

plt.xlabel('Training Step')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss Curve - Stage 1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.savefig('outputs/loss_curve.png', dpi=150, bbox_inches='tight')
print(f"\nLoss curve saved to outputs/loss_curve.png")
plt.close()
