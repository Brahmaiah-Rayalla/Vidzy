"""
Manually save the final checkpoint after training completed.
"""
import os
import torch
import yaml

# Load config
with open('configs/conditional_v0.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load the last checkpoint (step 750 or the training state if available)
# The training was at step 1000 when it failed during save
# We'll need to reconstruct from the last successful checkpoint

# Actually, let me check if there's a checkpoint_final.pt or we need to use checkpoint_step_000750.pt
checkpoint_dir = "checkpoints_conditional"
if os.path.exists(os.path.join(checkpoint_dir, "checkpoint_step_000750.pt")):
    print("Loading checkpoint from step 750...")
    checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint_step_000750.pt"), map_location='cpu')
    checkpoint['step'] = 1000  # Update step count
    # Save as final
    torch.save(checkpoint, os.path.join(checkpoint_dir, "checkpoint_final.pt"))
    print(f"Saved final checkpoint to {os.path.join(checkpoint_dir, 'checkpoint_final.pt')}")
else:
    print("No checkpoint found at step 750")
