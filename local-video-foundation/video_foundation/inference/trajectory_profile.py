"""Small CPU model for smooth, event-compatible speed variation profiles."""

import math

import numpy as np
import torch
from torch import nn


class TrajectoryProfileNet(nn.Module):
    """Map normalized time and four profile controls to a positive speed scale."""

    def __init__(self, width=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


def make_profile_features(times, speed_style_id, acceleration, wave_amplitude, phase, cycles):
    """Build inputs for a profile; style IDs are 0=slow and 1=quick."""
    times = np.asarray(times, dtype=np.float32)
    style = 2.0 * float(speed_style_id) - 1.0
    return np.stack((
        times,
        np.full_like(times, style),
        np.full_like(times, acceleration),
        np.full_like(times, wave_amplitude),
        np.full_like(times, math.sin(phase)),
        np.full_like(times, math.cos(phase)),
        np.full_like(times, cycles),
    ), axis=1).astype(np.float32)


def target_profile_scale(times, speed_style_id, acceleration, wave_amplitude, phase, cycles):
    """Synthetic teacher profile: smooth ramp plus a bounded periodic variation."""
    times = np.asarray(times, dtype=np.float32)
    style = 2.0 * float(speed_style_id) - 1.0
    scale = (1.0 + 0.20 * acceleration * (2.0 * times - 1.0)
             + 0.14 * wave_amplitude * np.sin(2.0 * math.pi * cycles * times + phase)
             + 0.06 * style * np.sin(math.pi * times))
    return np.clip(scale, 0.55, 1.45).astype(np.float32)


def sample_profile_parameters(seed, speed_style_id):
    rng = np.random.default_rng(seed)
    return {
        "speed_style_id": int(speed_style_id),
        "acceleration": float(rng.uniform(-1.0, 1.0)),
        "wave_amplitude": float(rng.uniform(0.0, 1.0)),
        "phase": float(rng.uniform(-math.pi, math.pi)),
        "cycles": float(rng.uniform(0.65, 1.8)),
    }


def predict_profile(model, parameters, steps, threads=4):
    torch.set_num_threads(threads)
    times = (np.arange(steps, dtype=np.float32) + 0.5) / max(steps, 1)
    features = make_profile_features(times, **parameters)
    with torch.inference_mode():
        predicted = model(torch.from_numpy(features)).numpy()
    # A bounded positive scale prevents numerical artifacts from reversing direction.
    return np.clip(predicted, 0.55, 1.45).astype(np.float64), times
