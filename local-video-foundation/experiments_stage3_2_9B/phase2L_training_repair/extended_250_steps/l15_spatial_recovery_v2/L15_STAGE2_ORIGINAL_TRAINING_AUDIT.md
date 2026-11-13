# L.15 Phase 1 — Stage 2.1 Original Training Audit

## Provenance (from checkpoint metadata, `checkpoint_stage2_motion_expanded.pt`)

- **Completed steps:** 1000 (`step: 1000`)
- **Config:** image_size 64, UNet model_channels 64, channel_mult [1,2,2,2],
  num_res_blocks 2, cross-attn (d_model 256, 4 heads, 4 layers), dropout 0.0
- **Text encoder:** vocab_size 1000 in config (tokenizer has 42 used + motion
  expansion later), dropout 0.1
- **Diffusion:** 1000 timesteps, linear beta 0.0001→0.02, eps-prediction
- **Training config:** batch_size 4, lr 2e-4, weight_decay 0.01, seed 42,
  save/val/sample interval 250
- **Dataset:** `num_images: 1000` synthetic images, image_size 64, max_objects 2
  → matches `data/text_image/` (800 metadata entries present; 1000 generated,
  200 likely held out or metadata truncated — limitation noted)
- **Optimizer state:** `optimizer_state_dict` IS saved in checkpoint
- **EMA:** none found in config
- **Checkpoint selection:** save_interval 250 → step-1000 is the final
  scheduled checkpoint, not val-selected (val_interval existed but no best-ckpt
  selection logic evident)

## Assessment

- Stage2.1 completed its full 1000-step schedule at batch 4 → ~4000 image
  presentations over a 1000-image dataset ≈ 4 epochs. That is VERY light
  training for a text→image diffusion model — abstract-texture output is
  consistent with severe undertraining rather than architecture incapacity.
- The checkpoint is a completed run, not an early stop.
- Optimizer state exists but belongs to a joint UNet+text-encoder run at
  lr 2e-4/batch 4; L.15 uses a different parameterization (UNet-only,
  lr 1e-4, batch 1) → classified as **warm-start with new optimizer**, NOT
  exact continuation.
- Limitation: original training logs are not present in the checkpoint dir;
  reconstruction is from config + checkpoint metadata only.
