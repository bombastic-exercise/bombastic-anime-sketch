import time

import numpy as np
import torch

from .data import sample
from .drawing import CHANNEL_NAMES
from .io import write_json
from .losses import drawing_loss
from .model import SketchUNet
from .runtime import memory, select_device, synchronize
from .train import DEFAULTS


def benchmark(output, device="auto", size=64, batch_size=2, steps=5, base_channels=8):
    if min(steps, batch_size) < 1:
        raise ValueError("steps and batch_size must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = select_device(device)
    start = time.perf_counter()
    x, y, _ = sample({"shape": "helmet", "geometry_seed": 0, "condition_seed": 42}, size)
    generation_seconds = time.perf_counter() - start
    x = torch.from_numpy(np.repeat(x[None], batch_size, axis=0)).to(device)
    y = torch.from_numpy(np.repeat(y[None], batch_size, axis=0)).to(device)
    model = SketchUNet(base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    times, peak = [], {}
    for i in range(steps + 2):
        synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = drawing_loss(model(x), y, x, DEFAULTS["loss_weights"])
        loss.backward()
        optimizer.step()
        synchronize(device)
        elapsed = time.perf_counter() - start
        if i >= 2:
            times.append(elapsed)
        for key, value in memory(device).items():
            peak[key] = max(peak.get(key, 0), value)
    report = {"device": str(device), "torch": str(torch.__version__), "size": size,
              "channels": len(CHANNEL_NAMES), "batch_size": batch_size, "steps": steps,
              "parameters": sum(p.numel() for p in model.parameters()),
              "synthetic_sample_seconds": generation_seconds, "mean_training_step_seconds": float(np.mean(times)),
              "training_samples_per_second": batch_size / float(np.mean(times)),
              "sampled_peak_memory": peak,
              "note": "Two warm-up steps excluded. Training timing excludes data generation. MPS memory is sampled, not a true allocator high-water mark."}
    write_json(output, report)
    return report
