from pathlib import Path

from .drawing import FEATURE_VERSION, render_teacher, variation
from .io import save_png, write_json
from .scene import render_geometry


def export_teacher(output, scene, controls, seed=12, size=256):
    geometry = render_geometry(scene, size)
    target = render_teacher(geometry, controls, variation(size, seed))
    save_png(output, target)
    write_json(Path(output).with_suffix(".json"), {"kind": "procedural_teacher", "size": size,
               "scene": scene.to_dict(), "controls": controls.to_dict(), "seed": seed,
               "feature_version": FEATURE_VERSION})


def export_prediction(checkpoint, output, scene, controls, seed=12, device="auto"):
    from .runtime import file_hash, load_model, predict

    model, state, device = load_model(checkpoint, device)
    image, _, _ = predict(model, device, scene, controls, seed, state["size"])
    save_png(output, image)
    write_json(Path(output).with_suffix(".json"), {"kind": "learned_prediction", "size": state["size"],
               "scene": scene.to_dict(), "controls": controls.to_dict(), "seed": seed,
               "feature_version": FEATURE_VERSION, "checkpoint": str(Path(checkpoint).resolve()),
               "checkpoint_sha256": file_hash(checkpoint), "step": state["step"], "device": str(device)})
