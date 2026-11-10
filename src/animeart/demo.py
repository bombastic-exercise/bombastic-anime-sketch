"""A local-only experiment viewer; a trained checkpoint is required."""

from pathlib import Path
from uuid import uuid4

import numpy as np

from .drawing import PRESETS, Controls
from .io import save_png, write_json
from .runtime import file_hash, load_model, predict
from .scene import SHAPES, Scene


def launch(checkpoint, device="auto", port=7860, output="runs/demo"):
    import gradio as gr

    model, state, device = load_model(checkpoint, device)
    checkpoint_hash = file_hash(checkpoint)
    output = Path(output)

    def draw(shape, geometry_seed, yaw, elevation, azimuth, light_elevation, material,
             preset, angle, spacing, width, crosshatch, darkness, irregularity, seed):
        scene = Scene(shape, int(geometry_seed), yaw, elevation, azimuth, light_elevation, material)
        controls = Controls(preset, angle, spacing, width, crosshatch, darkness, irregularity)
        image, _, _ = predict(model, device, scene, controls, int(seed), state["size"])
        path = output / f"sketch-{uuid4().hex}.png"
        save_png(path, image)
        metadata = path.with_suffix(".json")
        write_json(metadata, {"kind": "learned_prediction", "scene": scene.to_dict(),
                   "controls": controls.to_dict(), "seed": int(seed), "size": state["size"],
                   "checkpoint_sha256": checkpoint_hash, "step": state["step"], "device": str(device),
                   "feature_version": state["feature_version"]})
        return np.rint(image * 255).astype(np.uint8), str(path), str(metadata)

    with gr.Blocks(title="AnimeArt · sketch studies") as demo:
        gr.Markdown("# Sketch studies\nExplore learned mark-making on simple forms and armor studies. "
                    "Keep the seeds fixed to compare controls. Values are approximate learned targets.")
        with gr.Row():
            with gr.Column():
                shape = gr.Dropdown(list(SHAPES), value="helmet", label="Form")
                geometry_seed = gr.Number(value=0, precision=0, minimum=0, label="Geometry seed")
                with gr.Accordion("Camera and lighting", open=False):
                    yaw = gr.Slider(-90, 90, value=15, label="Camera yaw")
                    elevation = gr.Slider(5, 70, value=12, label="Camera elevation")
                    azimuth = gr.Slider(-180, 180, value=-45, label="Light azimuth")
                    light_elevation = gr.Slider(5, 85, value=45, label="Light elevation")
                    material = gr.Slider(0, 1, value=0.3, label="Metallic highlight strength")
                preset = gr.Dropdown(list(PRESETS), value="fineliner", label="Drawing medium")
                angle = gr.Slider(-90, 90, value=25, label="Hatch angle offset (degrees)")
                spacing = gr.Slider(3, 24, value=9, label="Hatch spacing (reference pixels at 256)")
                width = gr.Slider(0.4, 4, value=1.2, label="Mark width (reference pixels at 256)")
                crosshatch = gr.Slider(0, 1, value=0.6, label="Crosshatching in deep shadows")
                darkness = gr.Slider(0.2, 1, value=0.85, label="Ink darkness")
                irregularity = gr.Slider(0, 1, value=0.15, label="Mark irregularity")
                seed = gr.Number(value=12, precision=0, minimum=0, label="Variation seed")
                button = gr.Button("Draw study", variant="primary")
            with gr.Column():
                image = gr.Image(label="Learned sketch", image_mode="L", type="numpy", format="png")
                png = gr.File(label="Download PNG")
                metadata = gr.File(label="Download drawing settings")
        button.click(draw, [shape, geometry_seed, yaw, elevation, azimuth, light_elevation, material,
                           preset, angle, spacing, width, crosshatch, darkness, irregularity, seed],
                     [image, png, metadata], concurrency_limit=1)
    demo.queue().launch(server_name="127.0.0.1", server_port=port, share=False)
