"""Synthetic training teacher and explicit, bounded drawing controls."""

from dataclasses import asdict, dataclass

import numpy as np

PRESETS = ("fineliner", "pencil", "marker")


@dataclass(frozen=True)
class Controls:
    preset: str = "fineliner"
    angle: float = 25.0
    spacing: float = 9.0
    width: float = 1.2
    crosshatch: float = 0.6
    darkness: float = 0.85
    irregularity: float = 0.15

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ValueError(f"preset must be one of {PRESETS}")
        bounds = {"angle": (-90, 90), "spacing": (3, 24), "width": (0.4, 4),
                  "crosshatch": (0, 1), "darkness": (0.2, 1), "irregularity": (0, 1)}
        for name, (low, high) in bounds.items():
            if not low <= getattr(self, name) <= high:
                raise ValueError(f"{name} must be in [{low},{high}]")

    def to_dict(self):
        return asdict(self)

    def vector(self):
        angle = np.deg2rad(self.angle)
        return np.array([np.sin(angle), np.cos(angle), self.spacing / 24,
                         self.width / 4, self.crosshatch, self.darkness, self.irregularity,
                         *(float(self.preset == name) for name in PRESETS)], dtype=np.float32)


def variation(size, seed):
    rng = np.random.default_rng(seed)
    axis = np.linspace(-1, 1, size, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    smooth = np.zeros_like(x)
    for _ in range(5):
        a, b = rng.uniform(-6, 6, 2)
        smooth += np.sin(a * x + b * y + rng.uniform(0, 2 * np.pi)) / 5
    grain = rng.uniform(-1, 1, (size, size)).astype(np.float32)
    return smooth, grain


def _line(phase, width, size):
    # Phase is in cycles. Convert periodic distance to pixels through its gradient.
    gy, gx = np.gradient(phase)
    frequency = np.maximum(np.sqrt(gx * gx + gy * gy), 1e-5)
    distance = np.abs((phase + 0.5) % 1 - 0.5) / frequency
    return np.clip(width * size / 256 / 2 + 0.5 - distance, 0, 1)


def render_teacher(geometry, controls, noise):
    """Used for supervision/evaluation only, never to produce learned inference."""
    size = geometry["tone"].shape[0]
    smooth, grain = noise
    theta = np.deg2rad(controls.angle)
    u, v = geometry["u"], geometry["v"]
    jitter = smooth * controls.irregularity * 0.35
    density = (256 / 3) / controls.spacing
    first = (np.cos(theta) * u + np.sin(theta) * v) * density + jitter
    second = (np.cos(theta + 1.13) * u + np.sin(theta + 1.13) * v) * density - jitter
    width = controls.width * (2.6 if controls.preset == "marker" else 1)
    a = _line(first, width, size)
    b = _line(second, width * 0.85, size)
    tone = geometry["tone"]
    primary = np.clip((tone - 0.06) * 1.25, 0, 1)
    secondary = np.clip((tone - 0.38) * 2.0, 0, 1) * controls.crosshatch
    ink = 1 - (1 - a * primary) * (1 - b * secondary)
    if controls.preset == "pencil":
        ink *= np.clip(0.83 + grain * 0.24 + smooth * 0.10, 0.3, 1)
    elif controls.preset == "marker":
        ink = 1 - (1 - ink * 0.8) * (1 - tone * 0.14)
    ink = np.maximum(ink, geometry["contour"] * (0.8 + 0.1 * smooth * controls.irregularity))
    return np.clip(1 - controls.darkness * ink * geometry["mask"], 0, 1).astype(np.float32)


GEOMETRY_CHANNELS = ("mask", "object", "depth", "normal_x", "normal_y", "normal_z",
                     "contour", "tone", "shadow", "u", "v", "flow_cos2", "flow_sin2",
                     "material", "screen_x", "screen_y", "variation", "grain")
CONTROL_CHANNELS = ("angle_sin", "angle_cos", "spacing", "width", "crosshatch", "darkness",
                    "irregularity", "fineliner", "pencil", "marker")
CHANNEL_NAMES = GEOMETRY_CHANNELS + tuple(f"light_{a}" for a in "xyz") + CONTROL_CHANNELS
FEATURE_VERSION = 1


def make_features(geometry, controls, noise, material):
    size = geometry["tone"].shape[0]
    axis = np.linspace(-1, 1, size, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    channels = [geometry[k] for k in ("mask", "object", "depth")]
    channels += [geometry["normals"][..., i] for i in range(3)]
    channels += [geometry[k] for k in ("contour", "tone", "shadow", "u", "v")]
    channels += [geometry["flow"][..., i] for i in range(2)]
    channels += [np.full_like(x, material), x, y, *noise]
    channels += [np.full_like(x, value) for value in geometry["light"]]
    channels += [np.full_like(x, value) for value in controls.vector()]
    features = np.stack(channels).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite scene features")
    return features
