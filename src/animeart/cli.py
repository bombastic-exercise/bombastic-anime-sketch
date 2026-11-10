import argparse
import json


def drawing_arguments(parser):
    parser.add_argument("--shape", choices=["sphere", "cylinder", "plate", "helmet", "arm"], default="helmet")
    parser.add_argument("--geometry-seed", type=int, default=0)
    parser.add_argument("--camera-yaw", type=float, default=15)
    parser.add_argument("--camera-elevation", type=float, default=12)
    parser.add_argument("--light-azimuth", type=float, default=-45)
    parser.add_argument("--light-elevation", type=float, default=45)
    parser.add_argument("--material", type=float, default=0.3)
    parser.add_argument("--preset", choices=["fineliner", "pencil", "marker"], default="fineliner")
    parser.add_argument("--angle", type=float, default=25)
    parser.add_argument("--spacing", type=float, default=9)
    parser.add_argument("--width", type=float, default=1.2)
    parser.add_argument("--crosshatch", type=float, default=0.6)
    parser.add_argument("--darkness", type=float, default=0.85)
    parser.add_argument("--irregularity", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--output", required=True)


def parser():
    root = argparse.ArgumentParser(description="From-scratch, geometry-conditioned sketch rendering")
    sub = root.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Write split recipes, previews, and optional cached samples")
    generate.add_argument("--output", default="data/pilot")
    generate.add_argument("--samples", type=int, default=10000)
    generate.add_argument("--geometries", type=int, default=500)
    generate.add_argument("--size", type=int, default=256)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--materialize", action="store_true")
    teacher = sub.add_parser("teacher", help="Export a procedural training target (not a learned drawing)")
    drawing_arguments(teacher)
    teacher.add_argument("--size", type=int, default=256)
    fit = sub.add_parser("train", help="Train from random initialization or resume a checkpoint")
    fit.add_argument("--config", default="configs/pilot.json")
    for name in ("steps", "batch-size", "base-channels", "overfit"):
        fit.add_argument(f"--{name}", type=int)
    for name in ("dataset", "output", "device"):
        fit.add_argument(f"--{name}")
    fit.add_argument("--resume")
    evaluate = sub.add_parser("evaluate", help="Export held-out comparisons and measured control sweeps")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--count", type=int, default=16)
    evaluate.add_argument("--sweeps", action="store_true")
    evaluate.add_argument("--device", default="auto")
    render = sub.add_parser("render", help="Export a learned prediction with reproducibility metadata")
    drawing_arguments(render)
    render.add_argument("--checkpoint", required=True)
    render.add_argument("--device", default="auto")
    bench = sub.add_parser("benchmark", help="Measure forward/backward time and sampled memory")
    bench.add_argument("--output", default="runs/benchmark.json")
    bench.add_argument("--device", default="auto")
    bench.add_argument("--size", type=int, default=64)
    bench.add_argument("--batch-size", type=int, default=2)
    bench.add_argument("--steps", type=int, default=5)
    bench.add_argument("--base-channels", type=int, default=8)
    demo = sub.add_parser("demo", help="Launch the local Gradio viewer for a trained checkpoint")
    demo.add_argument("--checkpoint", required=True)
    demo.add_argument("--device", default="auto")
    demo.add_argument("--port", type=int, default=7860)
    demo.add_argument("--output", default="runs/demo")
    return root


def main(argv=None):
    root = parser()
    args = vars(root.parse_args(argv))
    command = args.pop("command")
    try:
        result = None
        if command == "generate":
            from .data import generate_manifest
            manifest = generate_manifest(**args)
            result = {"samples": len(manifest["records"]), "size": manifest["size"]}
        elif command in ("teacher", "render"):
            from .drawing import Controls
            from .export import export_prediction, export_teacher
            from .scene import Scene
            scene_keys = ("shape", "geometry_seed", "camera_yaw", "camera_elevation", "light_azimuth", "light_elevation", "material")
            control_keys = ("preset", "angle", "spacing", "width", "crosshatch", "darkness", "irregularity")
            scene = Scene(**{key: args.pop(key) for key in scene_keys})
            controls = Controls(**{key: args.pop(key) for key in control_keys})
            (export_teacher if command == "teacher" else export_prediction)(scene=scene, controls=controls, **args)
            result = {"output": args["output"], "kind": command}
        elif command == "train":
            from pathlib import Path
            from .train import train
            config = json.loads(Path(args.pop("config")).read_text())
            resume = args.pop("resume")
            config.update({key: value for key, value in args.items() if value is not None})
            result = train(config, resume)
        elif command == "evaluate":
            from .evaluate import evaluate
            result = evaluate(**args)
        elif command == "benchmark":
            from .benchmark import benchmark
            result = benchmark(**args)
        elif command == "demo":
            from .demo import launch
            launch(**args)
        if result is not None:
            print(json.dumps(result, indent=2))
    except ModuleNotFoundError as error:
        root.exit(2, f"Missing dependency: {error.name}. Install with: python -m pip install -e '.[demo]'\n")
    except (ValueError, FileNotFoundError) as error:
        root.exit(2, f"{error}\n")
