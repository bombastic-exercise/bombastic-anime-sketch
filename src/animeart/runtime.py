"""Device selection, checkpoint compatibility, and learned inference."""

import hashlib
import platform
import resource
from pathlib import Path

import numpy as np
import torch

from .drawing import CHANNEL_NAMES, FEATURE_VERSION, make_features, variation
from .model import SketchUNet
from .scene import render_geometry


def select_device(requested="auto"):
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this PyTorch build/runtime; select cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select cpu or mps")
    if requested not in ("cpu", "mps", "cuda"):
        raise ValueError("device must be auto, cpu, mps, or cuda")
    return torch.device(requested)


def memory(device):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024 ** 2 if platform.system() == "Darwin" else 1024)
    result = {"process_peak_rss_mb": rss_mb}
    if device.type == "mps":
        result["mps_current_allocated_mb"] = torch.mps.current_allocated_memory() / 1024 ** 2
        result["mps_driver_allocated_mb"] = torch.mps.driver_allocated_memory() / 1024 ** 2
    elif device.type == "cuda":
        result["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
    return result


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint, requested="auto"):
    device = select_device(requested)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("feature_version") != FEATURE_VERSION or state.get("channels") != list(CHANNEL_NAMES):
        raise ValueError("Checkpoint feature layout is incompatible with this version")
    model = SketchUNet(state["config"]["base_channels"],
                      contour_prior=state["config"].get("contour_prior", False))
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, state, device


def predict(model, device, scene, controls, seed, size):
    """No teacher call or target image is used by this function."""
    geometry = render_geometry(scene, size)
    features = make_features(geometry, controls, variation(size, seed), scene.material)
    with torch.inference_mode():
        result = model(torch.from_numpy(features[None]).to(device))[0, 0].cpu().numpy()
    return result, features, geometry


def stack_batch(dataset, indices, device):
    items = [dataset[int(i)] for i in indices]
    return (torch.from_numpy(np.stack([item[0] for item in items])).to(device),
            torch.from_numpy(np.stack([item[1] for item in items])).to(device))
