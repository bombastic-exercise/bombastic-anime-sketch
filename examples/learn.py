"""Small, inspectable steps through AnimeArt's real learning pipeline.

Run from the repository root: .venv/bin/python examples/learn.py inspect
"""

import argparse
import json
from pathlib import Path

import torch

from animeart.data import SketchDataset
from animeart.drawing import CHANNEL_NAMES
from animeart.io import gallery, save_png
from animeart.losses import drawing_loss
from animeart.model import SketchUNet
from animeart.runtime import select_device
from animeart.train import DEFAULTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lesson", choices=("inspect", "step", "compare"))
    parser.add_argument("--dataset", default="data/smoke")
    parser.add_argument("--checkpoint", default="runs/overfit/last.pt")
    parser.add_argument("--output", default="runs/learning")
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    args = parser.parse_args()
    dataset = SketchDataset(args.dataset, split="train", cache_samples=4)
    features, target = dataset[0]
    if args.lesson == "inspect":
        print(json.dumps({"input_shape": list(features.shape), "target_shape": list(target.shape),
                          "input_dtype": str(features.dtype), "channels": list(CHANNEL_NAMES),
                          "target_range": [float(target.min()), float(target.max())]}, indent=2))
        return

    device = select_device(args.device)
    torch.manual_seed(42)
    x = torch.from_numpy(features[None]).to(device)
    y = torch.from_numpy(target[None]).to(device)
    if args.lesson == "step":
        model = SketchUNet(base_channels=8).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        prediction = model(x)
        loss, parts = drawing_loss(prediction, y, x, DEFAULTS["loss_weights"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(model.output.weight.grad.abs().mean().cpu())
        before = model.output.weight.detach().clone()
        optimizer.step()
        print(json.dumps({"loss_before_update": float(loss.detach().cpu()),
                          "loss_parts": parts, "output_weight_mean_absolute_gradient": gradient,
                          "output_weights_changed": bool((before != model.output.weight).any().item())}, indent=2))
        return

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["size"] != dataset.size:
        raise ValueError("Checkpoint and dataset image sizes differ")
    width = checkpoint["config"]["base_channels"]
    torch.manual_seed(checkpoint["config"]["seed"])
    prior = checkpoint["config"].get("contour_prior", False)
    before_model = SketchUNet(base_channels=width, contour_prior=prior).to(device).eval()
    after_model = SketchUNet(base_channels=width, contour_prior=prior).to(device).eval()
    after_model.load_state_dict(checkpoint["model"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for i in range(min(4, len(dataset))):
            sample_x, sample_y = dataset[i]
            tensor = torch.from_numpy(sample_x[None]).to(device)
            images = []
            for label, image in (("teacher target", sample_y[0]),
                                 ("random weights", before_model(tensor)[0, 0].cpu().numpy()),
                                 (f"after {checkpoint['step']} steps", after_model(tensor)[0, 0].cpu().numpy())):
                path = output / f"sample-{i}-{label.split()[0]}.png"
                save_png(path, image)
                images.append((label, path))
            rows.append((f"Training example {i + 1}", images))
    gallery(output / "index.html", rows)
    print(f"Open {output / 'index.html'}")


if __name__ == "__main__":
    main()
