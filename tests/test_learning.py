"""Run with PyTorch installed; skipped explicitly when it is unavailable."""

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class LearningTests(unittest.TestCase):
    def test_contour_prior_starts_with_white_paper_and_visible_outline(self):
        import torch
        from animeart.data import sample
        from animeart.model import SketchUNet

        features, _, _ = sample({"shape": "armored_heroine", "geometry_seed": 0,
                                 "condition_seed": 7}, 64)
        model = SketchUNet(4, contour_prior=True).eval()
        with torch.inference_mode():
            output = model(torch.from_numpy(features[None]))[0, 0].numpy()
        outline = features[6] > 0.5
        self.assertGreater(float(output[~outline].mean()), 0.85)
        self.assertLess(float(output[outline].mean()), 0.4)

    def test_forward_backward_and_loss(self):
        import torch
        from animeart.data import sample
        from animeart.losses import drawing_loss
        from animeart.model import SketchUNet
        from animeart.train import DEFAULTS

        torch.set_num_threads(2)
        x, y, _ = sample({"shape": "plate", "geometry_seed": 1, "condition_seed": 2}, 32)
        x, y = torch.from_numpy(x[None]), torch.from_numpy(y[None])
        model = SketchUNet(4)
        output = model(x)
        self.assertEqual(tuple(output.shape), (1, 1, 32, 32))
        loss, _ = drawing_loss(output, y, x, DEFAULTS["loss_weights"])
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_checkpoint_resume_and_learned_exports(self):
        import torch
        from animeart.data import generate_manifest
        from animeart.drawing import Controls
        from animeart.evaluate import evaluate
        from animeart.export import export_prediction
        from animeart.runtime import load_model
        from animeart.reference import render_reference
        from animeart.scene import Scene
        from animeart.train import train

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            generate_manifest(data, samples=30, geometries=10, size=32)
            config = {"dataset": str(data), "output": str(root / "split"), "device": "cpu",
                      "base_channels": 4, "batch_size": 1, "steps": 2, "threads": 2,
                      "log_every": 2, "save_every": 2, "validation_samples": 1}
            train(config)
            train(config | {"steps": 4}, root / "split/last.pt")
            train(config | {"steps": 4, "output": str(root / "whole")})
            _, resumed, _ = load_model(root / "split/last.pt", "cpu")
            _, whole, _ = load_model(root / "whole/last.pt", "cpu")
            for key in whole["model"]:
                torch.testing.assert_close(resumed["model"][key], whole["model"][key], rtol=0, atol=0)
            output = root / "learned.png"
            export_prediction(root / "split/last.pt", output, Scene(), Controls(), device="cpu")
            self.assertEqual(json.loads(output.with_suffix(".json").read_text())["kind"], "learned_prediction")
            reference = render_reference(root / "split/last.pt", output, root / "reference",
                                         device="cpu", size=32)
            self.assertEqual(reference["kind"], "reference_conditioned_prediction")
            self.assertTrue((root / "reference/model.png").exists())
            self.assertTrue((root / "reference/model-raw.png").exists())
            evaluate(root / "split/last.pt", data, root / "evaluation", count=1, device="cpu")
            self.assertTrue((root / "evaluation/report.json").exists())
            with self.assertRaises(ValueError):
                train(config | {"steps": 5, "batch_size": 2}, root / "split/last.pt")

    def test_mps_inference_and_repeatability(self):
        import torch
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        from animeart.drawing import Controls
        from animeart.model import SketchUNet
        from animeart.runtime import predict
        from animeart.scene import Scene

        model = SketchUNet(4).to("mps").eval()
        first = predict(model, torch.device("mps"), Scene(), Controls(), 12, 32)[0]
        second = predict(model, torch.device("mps"), Scene(), Controls(), 12, 32)[0]
        np.testing.assert_allclose(first, second, atol=1e-6)


class InferenceBoundaryTests(unittest.TestCase):
    def test_prediction_path_never_calls_teacher(self):
        source = Path(__file__).resolve().parents[1] / "src/animeart/runtime.py"
        tree = ast.parse(source.read_text())
        predict = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "predict")
        names = {node.id for node in ast.walk(predict) if isinstance(node, ast.Name)}
        self.assertNotIn("render_teacher", names)


if __name__ == "__main__":
    unittest.main()
