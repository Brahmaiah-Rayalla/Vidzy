"""Diagnose a Stage 0 checkpoint at fixed timesteps and DDIM step counts."""

import argparse
import csv
import os

import torch
from PIL import Image

from video_foundation.data.dataset import SyntheticImageDataset
from video_foundation.diffusion.sampler import DiffusionSampler
from video_foundation.models.unet import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = UNet(
        in_channels=config["model"]["in_channels"],
        model_channels=config["model"]["model_channels"],
        channel_mult=config["model"]["channel_mult"],
        num_res_blocks=config["model"]["num_res_blocks"],
        attention_resolutions=config["model"]["attention_resolutions"],
        dropout=config["model"]["dropout"],
        num_timesteps=config["diffusion"]["num_timesteps"],
        num_classes=config["model"].get("num_classes", 0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    diffusion = DiffusionSampler(
        num_timesteps=config["diffusion"]["num_timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        schedule=config["diffusion"]["beta_schedule"],
    )
    dataset = SyntheticImageDataset(
        args.data_dir,
        image_size=config["model"]["image_size"],
        return_label=model.num_classes > 0,
    )
    if model.num_classes > 0:
        clean_image, clean_label = dataset[0]
        class_labels = torch.tensor([clean_label], dtype=torch.long)
    else:
        clean_image = dataset[0]
        class_labels = None
    clean = clean_image.unsqueeze(0)
    fixed_noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(7001))

    with torch.no_grad():
        with open(os.path.join(args.output_dir, "fixed_timestep_metrics.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "epsilon_mse", "x0_mse", "x0_pred_min", "x0_pred_max"])
            for timestep in [0, 100, 300, 600, 999]:
                t = torch.tensor([timestep])
                xt, noise = diffusion.process.q_sample(clean, t, fixed_noise)
                alpha_bar = diffusion.schedule.alphas_cumprod[timestep]
                alpha = torch.sqrt(alpha_bar)
                sigma = torch.sqrt(1 - alpha_bar)
                model_pred = model(xt, t, class_labels)
                prediction_type = config["diffusion"].get("prediction_type", "epsilon")
                if prediction_type == "v_prediction":
                    pred_x0 = alpha * xt - sigma * model_pred
                    pred_noise = sigma * xt + alpha * model_pred
                else:
                    pred_noise = model_pred
                    pred_x0 = (xt - sigma * pred_noise) / alpha
                writer.writerow([
                    timestep,
                    torch.mean((pred_noise - noise) ** 2).item(),
                    torch.mean((pred_x0 - clean) ** 2).item(),
                    pred_x0.min().item(),
                    pred_x0.max().item(),
                ])

        initial = torch.randn((1, 3, clean.shape[-2], clean.shape[-1]), generator=torch.Generator().manual_seed(9001))
        for steps in [50, 200, 1000]:
            for recompute in [False, True]:
                generated = diffusion.process.ddim_sample_loop(
                    model=model,
                    batch_size=1,
                    channels=3,
                    image_size=clean.shape[-1],
                    num_sampling_steps=steps,
                    device=torch.device("cpu"),
                    initial_noise=initial,
                    recompute_noise_after_clipping=recompute,
                    prediction_type=config["diffusion"].get("prediction_type", "epsilon"),
                    class_labels=class_labels,
                )[0]
                array = ((generated.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
                suffix = "recomputed" if recompute else "original"
                Image.fromarray(array).save(os.path.join(args.output_dir, f"ddim_{steps}_{suffix}.png"))

    print(f"Wrote fixed-timestep metrics and DDIM comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
