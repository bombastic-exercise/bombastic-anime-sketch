"""Held-out metrics, comparison sheets, and independent control/light sweeps."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .data import SketchDataset, recipe
from .drawing import Controls, render_teacher, variation
from .io import gallery, save_png, write_json
from .metrics import metrics
from .runtime import file_hash, load_model, predict
from .scene import Scene


def compare_case(model, device, scene, controls, seed, size, output, name):
    prediction, features, geometry = predict(model, device, scene, controls, seed, size)
    target = render_teacher(geometry, controls, variation(size, seed))
    images = []
    for label, pixels in [("teacher", target), ("learned", prediction), ("tone", 1 - geometry["tone"])]:
        path = output / f"{name}-{label}.png"
        save_png(path, pixels)
        images.append((label, path))
    return metrics(prediction, target, features), images


def control_sweeps(model, device, size, output):
    scene, controls, seed = Scene(), Controls(), 12
    rows, report = [], {}
    variations = {"spacing": [4, 10, 20], "angle": [-60, 0, 60], "width": [0.6, 1.5, 3.0],
                  "crosshatch": [0, 0.5, 1], "darkness": [0.3, 0.6, 1],
                  "irregularity": [0, 0.5, 1], "preset": ["fineliner", "pencil", "marker"]}
    for parameter, values in variations.items():
        values_report, figures = [], []
        for i, value in enumerate(values):
            measured, images = compare_case(model, device, scene, replace(controls, **{parameter: value}),
                                             seed, size, output, f"sweep-{parameter}-{i}")
            values_report.append({"value": value, **measured})
            figures += [(f"{parameter}={value} · {label}", path) for label, path in images[:2]]
        report[parameter] = values_report
        rows.append((f"Independent control: {parameter}", figures))
    for parameter, values in {"light_azimuth": [-90, 0, 90], "camera_yaw": [-70, 0, 70]}.items():
        figures, values_report = [], []
        for i, value in enumerate(values):
            measured, images = compare_case(model, device, replace(scene, **{parameter: value}), controls,
                                             seed, size, output, f"sweep-{parameter}-{i}")
            values_report.append({"value": value, **measured})
            figures += [(f"{parameter}={value} · {label}", path) for label, path in images]
        report[parameter] = values_report
        rows.append((f"Lighting/view stress test: {parameter}", figures))
    # Trends are reported as evidence, not silently treated as successful controls.
    trends = {}
    for name, direction in [("spacing", -1), ("width", 1), ("crosshatch", 1), ("darkness", 1)]:
        density = [row["predicted_ink_density"] for row in report[name]]
        teacher_density = [row["teacher_ink_density"] for row in report[name]]
        trends[name] = {"learned_expected_density_trend": bool(np.all(np.diff(density) * direction >= -0.005)),
                        "teacher_expected_density_trend": bool(np.all(np.diff(teacher_density) * direction >= -0.005))}
    return report, trends, rows


def evaluate(checkpoint, dataset, output, split="test", count=16, sweeps=False, device="auto"):
    if count < 1:
        raise ValueError("count must be positive")
    model, state, device = load_model(checkpoint, device)
    output = Path(output)
    if (output / "report.json").exists():
        raise ValueError("Evaluation directory already contains a report; choose a new output")
    output.mkdir(parents=True, exist_ok=True)
    data = SketchDataset(dataset, split, 0)
    if file_hash(Path(dataset) / "manifest.json") != state["manifest_sha256"]:
        raise ValueError("Use the checkpoint's dataset for a valid held-out evaluation")
    measured, rows = [], []
    # Cover the whole split instead of selecting only the first few geometry IDs.
    indices = np.linspace(0, len(data) - 1, min(count, len(data)), dtype=int)
    for i, index in enumerate(indices):
        record = data.records[index]
        scene, controls, seed = recipe(record)
        features, target = data[index]
        with torch.inference_mode():
            prediction = model(torch.from_numpy(features[None]).to(device))[0, 0].cpu().numpy()
        measured.append({"record_id": record["id"], "shape": record["shape"],
                         **metrics(prediction, target[0], features)})
        if i < 8:
            images = []
            for label, pixels in [("teacher", target[0]), ("learned", prediction), ("tone", 1 - features[7])]:
                path = output / f"heldout-{i:02d}-{label}.png"
                save_png(path, pixels)
                images.append((label, path))
            rows.append((f"Held-out {scene.shape} · record {record['id']}", images))
    keys = [key for key in measured[0] if key not in ("record_id", "shape")]
    aggregate = {key: float(np.mean([m[key] for m in measured if m[key] is not None]))
                 if any(m[key] is not None for m in measured) else None for key in keys}
    report = {"checkpoint_sha256": file_hash(checkpoint), "step": state["step"],
              "split": split, "samples": measured, "mean": aggregate,
              "metric_note": "Orientation uses coherent 16px patches away from contours, modulo 180 degrees; null means no reliable patches. Width is an ink-area/edge-contrast proxy biased by wash and overlap. Density and tone do not establish exact stroke measurements."}
    if sweeps:
        sweep_report, trends, sweep_rows = control_sweeps(model, device, data.size, output)
        report |= {"sweeps": sweep_report, "density_trends": trends}
        rows += sweep_rows
    write_json(output / "report.json", report)
    gallery(output / "index.html", rows)
    return report["mean"]
