"""Tone, contour, and local orientation supervision in addition to raster error."""

import torch
from torch.nn import functional as F


def gradients(image):
    gx = F.pad(image[..., 1:] - image[..., :-1], (0, 1, 0, 0))
    gy = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
    return gx, gy


def drawing_loss(prediction, target, features, weights):
    error = (prediction - target).abs()
    raster = (error * (1 + 2 * (1 - target))).mean()
    contour_mask = features[:, 6:7]
    contour = (error * contour_mask).sum() / contour_mask.sum().clamp_min(1)
    tone = F.l1_loss(F.avg_pool2d(prediction, 8), F.avg_pool2d(target, 8))
    px, py = gradients(prediction)
    tx, ty = gradients(target)
    target_energy = tx.square() + ty.square()
    predicted_energy = px.square() + py.square()
    # Squared cosine treats both signs of a stroke direction as equivalent.
    cosine2 = (px * tx + py * ty).square() / (predicted_energy * target_energy + 1e-6)
    mask = (target_energy > 0.002).float() * (1 - contour_mask)
    orientation = ((1 - cosine2) * mask).sum() / mask.sum().clamp_min(1)
    components = {"raster": raster, "contour": contour, "tone": tone, "orientation": orientation}
    total = sum(weights.get(name, 0) * value for name, value in components.items())
    return total, {name: float(value.detach().cpu()) for name, value in components.items()}
