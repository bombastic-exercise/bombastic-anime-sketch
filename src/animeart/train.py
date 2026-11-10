"""Step-based training, deterministic recipe sampling, and resumable state."""

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .data import SketchDataset
from .drawing import CHANNEL_NAMES, FEATURE_VERSION
from .io import write_json
from .losses import drawing_loss
from .model import SketchUNet
from .runtime import file_hash, memory, select_device, stack_batch, synchronize

DEFAULTS = {"steps": 10000, "batch_size": 4, "base_channels": 24, "learning_rate": 0.0003,
            "seed": 42, "device": "auto", "threads": 4, "log_every": 25, "save_every": 250,
            "validation_samples": 16, "cache_samples": 64,
            "loss_weights": {"raster": 1.0, "contour": 0.25, "tone": 0.5, "orientation": 0.1},
            "overfit": 0}


def validate(model, dataset, device, weights, count):
    model.eval()
    losses = []
    with torch.inference_mode():
        for i in range(min(count, len(dataset))):
            x, y = stack_batch(dataset, [i], device)
            loss, _ = drawing_loss(model(x), y, x, weights)
            losses.append(float(loss.cpu()))
    model.train()
    return float(np.mean(losses))


def train(config, resume=None):
    config = DEFAULTS | config
    for name in ("steps", "batch_size", "log_every", "save_every", "validation_samples", "threads"):
        if config[name] < 1:
            raise ValueError(f"{name} must be positive")
    if config["overfit"] < 0 or config["learning_rate"] <= 0:
        raise ValueError("overfit must be nonnegative and learning_rate positive")
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    device = select_device(config["device"])
    train_data = SketchDataset(config["dataset"], "train", config["cache_samples"])
    val_data = SketchDataset(config["dataset"], "val", 0)
    manifest_hash = file_hash(Path(config["dataset"]) / "manifest.json")
    model = SketchUNet(config["base_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    start_step, best_loss = 0, float("inf")
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["manifest_sha256"] != manifest_hash or state["feature_version"] != FEATURE_VERSION:
            raise ValueError("Resume requires the same dataset and feature version")
        for key in ("base_channels", "batch_size", "learning_rate", "seed", "overfit", "loss_weights"):
            if state["config"][key] != config[key]:
                raise ValueError(f"Cannot change {key} when resuming")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, best_loss = state["step"], state["best_validation_loss"]
        rng.bit_generator.state = state["sampler_rng"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "mps" and state.get("mps_rng") is not None:
            torch.mps.set_rng_state(state["mps_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    if start_step >= config["steps"]:
        raise ValueError("steps must exceed the checkpoint's completed step count")
    output = Path(config["output"])
    if not resume and (output / "last.pt").exists():
        raise ValueError(f"Run already exists at {output}; use --resume or another --output")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    weights = config["loss_weights"]
    population = min(config["overfit"], len(train_data)) if config["overfit"] else len(train_data)
    initial_overfit = validate(model, train_data, device, weights, population) if config["overfit"] else None
    train_started = time.perf_counter()
    sampled_peak = {}
    log = output / "metrics.jsonl"
    print(json.dumps({"device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
                      "size": train_data.size, "training_samples": population}), flush=True)

    def save_checkpoint(filename, step):
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                 "config": config, "feature_version": FEATURE_VERSION, "channels": list(CHANNEL_NAMES),
                 "size": train_data.size, "manifest_sha256": manifest_hash,
                 "best_validation_loss": best_loss, "sampler_rng": rng.bit_generator.state,
                 "torch_rng": torch.get_rng_state(),
                 "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        temporary = output / (filename + ".tmp")
        torch.save(state, temporary)
        temporary.replace(output / filename)

    model.train()
    for step in range(start_step + 1, config["steps"] + 1):
        step_started = time.perf_counter()
        indices = rng.integers(0, population, config["batch_size"])
        x, y = stack_batch(train_data, indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss, components = drawing_loss(model(x), y, x, weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}; last checkpoint is preserved")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        synchronize(device)
        elapsed = time.perf_counter() - step_started
        usage = memory(device)
        for name, value in usage.items():
            sampled_peak[name] = max(sampled_peak.get(name, 0), value)
        if step % config["log_every"] == 0 or step == 1:
            record = {"step": step, "loss": float(loss.detach().cpu()), **components,
                      "step_seconds": elapsed, "samples_per_second": config["batch_size"] / elapsed,
                      **usage}
            with log.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % config["save_every"] == 0 or step == config["steps"]:
            validation = validate(model, val_data, device, weights, config["validation_samples"])
            improved = validation < best_loss
            best_loss = min(best_loss, validation)
            save_checkpoint("last.pt", step)
            if improved:
                save_checkpoint("best.pt", step)
            with log.open("a") as handle:
                handle.write(json.dumps({"step": step, "validation_loss": validation}) + "\n")
            print(json.dumps({"step": step, "validation_loss": validation}), flush=True)
    report = {"completed_step": config["steps"], "device": str(device), "torch": str(torch.__version__),
              "python": platform.python_version(), "platform": platform.platform(),
              "parameters": sum(p.numel() for p in model.parameters()),
              "session_steps": config["steps"] - start_step,
              "session_seconds": time.perf_counter() - train_started,
              "best_validation_loss": best_loss, "sampled_peak_memory": sampled_peak,
              "manifest_sha256": manifest_hash}
    if config["overfit"]:
        final = validate(model, train_data, device, weights, population)
        report["overfit_diagnostic"] = {"samples": population, "session_initial_loss": initial_overfit,
                                        "final_loss": final, "halved_loss": final < initial_overfit * 0.5}
    write_json(output / "training-report.json", report)
    return report
