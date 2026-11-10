"""Run the learned figure renderer on a line guide extracted from an image.

The guide carries the composition and identity; this is image-conditioned
rendering, not character generation or a held-out evaluation.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from .drawing import Controls, make_features, variation
from .io import gallery, save_png, write_json
from .runtime import file_hash, load_model


def reference_features(source, size, controls, seed):
    image = Image.open(source).convert("L")
    original = np.asarray(image, dtype=np.float32) / 255
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    page = Image.new("L", (size, size), 255)
    page.paste(image, ((size - image.width) // 2, (size - image.height) // 2))

    # Remove uneven paper tone before measuring dark marks. Work at source
    # resolution so fine helmet and hair lines survive the downsampling.
    native = Image.fromarray(np.rint(original * 255).astype(np.uint8))
    background = np.asarray(native.filter(ImageFilter.GaussianBlur(radius=18)), dtype=np.float32) / 255
    native_guide = np.clip((background - original - 0.025) / 0.25, 0, 1)
    guide_image = Image.fromarray(np.rint(native_guide * 255).astype(np.uint8))
    guide_image.thumbnail((size, size), Image.Resampling.LANCZOS)
    guide_page = Image.new("L", (size, size), 0)
    guide_page.paste(guide_image, ((size - guide_image.width) // 2,
                                   (size - guide_image.height) // 2))
    guide = np.asarray(guide_page, dtype=np.float32) / 255
    source_page = np.asarray(page, dtype=np.float32) / 255

    support_image = guide_page.filter(ImageFilter.GaussianBlur(radius=4))
    support = np.clip(np.asarray(support_image, dtype=np.float32) / 255 * 10, 0, 1)
    axis = np.linspace(-1, 1, size, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    normals = np.zeros((size, size, 3), dtype=np.float32)
    normals[..., 2] = 1
    flow = np.zeros((size, size, 2), dtype=np.float32)
    flow[..., 0] = 1
    geometry = {"mask": support, "object": support, "depth": support * 0.5,
                "normals": normals, "contour": guide, "tone": support * 0.06,
                "shadow": np.zeros_like(guide), "u": x, "v": y, "flow": flow,
                "light": np.array([0, 0, 1], dtype=np.float32)}
    features = make_features(geometry, controls, variation(size, seed), material=0.3)
    return source_page, guide, features


def render_reference(checkpoint, source, output, device="auto", seed=12, size=256):
    """Export source, guide, raw prediction, and a contrast-adjusted view."""
    if size < 32 or size > 1024 or size % 8:
        raise ValueError("size must be a multiple of eight between 32 and 1024")
    model, state, device = load_model(checkpoint, device)
    controls = Controls(preset="fineliner", darkness=1, crosshatch=0)
    source_page, guide, features = reference_features(source, size, controls, seed)
    with torch.inference_mode():
        prediction = model(torch.from_numpy(features[None]).to(device))[0, 0].cpu().numpy()
    # A display adjustment removes weak gray background and restores dark ink.
    # Keep the raw prediction beside it so this step cannot be mistaken for learning.
    adjusted = 1 - np.clip((1 - prediction - 0.015) * 2.8, 0, 1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    images = []
    for label, pixels, name in (("source resized", source_page, "source.png"),
                                ("extracted line guide", 1 - guide, "guide.png"),
                                ("raw model output", prediction, "model-raw.png"),
                                ("model output, contrast adjusted", adjusted, "model.png")):
        path = output / name
        save_png(path, pixels)
        images.append((label, path))
    gallery(output / "index.html", [(f"Reference study · {Path(source).name}", images)])
    metadata = {"kind": "reference_conditioned_prediction", "source": str(Path(source).resolve()),
                "source_sha256": file_hash(source), "checkpoint": str(Path(checkpoint).resolve()),
                "checkpoint_sha256": file_hash(checkpoint), "step": state["step"],
                "size": size, "training_size": state["size"], "device": str(device), "seed": seed,
                "display_adjustment": "1 - clip((1 - raw_prediction - 0.015) * 2.8, 0, 1)",
                "note": "Composition and line structure come from the supplied image. This is not generated character art or held-out evaluation."}
    write_json(output / "metadata.json", metadata)
    return metadata
