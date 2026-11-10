"""Compact conditional U-Net, initialized from scratch."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .drawing import CHANNEL_NAMES


class Block(nn.Module):
    def __init__(self, incoming, outgoing, conditioning):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.norm1 = nn.GroupNorm(math.gcd(8, outgoing), outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.norm2 = nn.GroupNorm(math.gcd(8, outgoing), outgoing)
        self.film = nn.Linear(conditioning, outgoing * 2)

    def forward(self, x, conditioning):
        x = F.silu(self.norm1(self.conv1(x)))
        scale, bias = self.film(conditioning).chunk(2, dim=1)
        x = self.norm2(self.conv2(x))
        return F.silu(x * (1 + scale[..., None, None]) + bias[..., None, None])


class SketchUNet(nn.Module):
    def __init__(self, base_channels=24, contour_prior=False):
        super().__init__()
        if base_channels < 4 or base_channels > 128:
            raise ValueError("base_channels must be between 4 and 128")
        b = base_channels
        self.contour_prior = contour_prior
        # Fixed Fourier encoding of coordinates is independent of drawing targets.
        self.register_buffer("frequencies", torch.tensor([1, 2, 4, 8], dtype=torch.float32))
        self.condition = nn.Sequential(nn.Linear(14, b * 4), nn.SiLU(), nn.Linear(b * 4, b * 4))
        self.encoder1 = Block(len(CHANNEL_NAMES) + 16, b, b * 4)
        self.encoder2 = Block(b, b * 2, b * 4)
        self.encoder3 = Block(b * 2, b * 4, b * 4)
        self.middle = Block(b * 4, b * 8, b * 4)
        self.decoder3 = Block(b * 12, b * 4, b * 4)
        self.decoder2 = Block(b * 6, b * 2, b * 4)
        self.decoder1 = Block(b * 3, b, b * 4)
        self.output = nn.Conv2d(b, 1, 1)
        # A mostly white initial page makes sparse ink errors visible from step one.
        nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.output.bias, 2.94)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != len(CHANNEL_NAMES) or x.shape[-1] % 8 or x.shape[-2] % 8:
            raise ValueError("Expected B×31×H×W features with dimensions divisible by eight")
        global_features = torch.cat([x[:, 13, 0, 0, None], x[:, 18:, 0, 0]], dim=1)
        condition = self.condition(global_features)
        phase = x[:, 9:11, None] * self.frequencies[None, None, :, None, None] * math.pi
        encoded = torch.cat([phase.sin(), phase.cos()], dim=2).flatten(1, 2)
        a = self.encoder1(torch.cat([x, encoded], dim=1), condition)
        b = self.encoder2(F.avg_pool2d(a, 2), condition)
        c = self.encoder3(F.avg_pool2d(b, 2), condition)
        d = self.middle(F.avg_pool2d(c, 2), condition)

        def up(value, skip, block):
            return block(torch.cat([F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1), condition)

        logits = self.output(up(up(up(d, c, self.decoder3), b, self.decoder2), a, self.decoder1))
        if self.contour_prior:
            logits = logits - 6 * x[:, 6:7] - x[:, 7:8]
        return logits.sigmoid()
