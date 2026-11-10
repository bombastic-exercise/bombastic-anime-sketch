import json
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from animeart.cli import main
from animeart.data import SketchDataset, generate_manifest, sample
from animeart.drawing import CHANNEL_NAMES, Controls, make_features, render_teacher, variation
from animeart.io import save_png
from animeart.metrics import metrics
from animeart.scene import SHAPES, Scene, render_geometry


class SyntheticTests(unittest.TestCase):
    def test_finite_geometry_and_unit_normals(self):
        for shape in SHAPES:
            with self.subTest(shape=shape):
                geometry = render_geometry(Scene(shape=shape), 64)
                for array in geometry.values():
                    self.assertTrue(np.isfinite(array).all())
                foreground = geometry["object"] > 0
                self.assertGreater(foreground.mean(), 0.02)
                np.testing.assert_allclose(np.linalg.norm(geometry["normals"][foreground], axis=-1), 1, atol=1e-5)
                self.assertTrue(np.all((geometry["tone"] >= 0) & (geometry["tone"] <= 1)))

    def test_shadow_moves_with_light_and_empty_paper_is_white(self):
        left = render_geometry(Scene(shape="sphere", light_azimuth=-65), 128)
        right = render_geometry(Scene(shape="sphere", light_azimuth=65), 128)
        cast_left = (left["object"] == 0) & (left["shadow"] > 0)
        cast_right = (right["object"] == 0) & (right["shadow"] > 0)
        self.assertGreater(cast_left.sum(), 10)
        self.assertGreater(cast_right.sum(), 10)
        self.assertGreater(np.abs(cast_left.astype(float) - cast_right).mean(), 0.01)
        x = np.indices(cast_left.shape)[1]
        self.assertGreater(float(x[cast_left].mean()), float(x[cast_right].mean()))
        self.assertTrue(np.all(left["tone"][(left["object"] == 0) & (left["shadow"] == 0)] == 0))

    def test_reproducible_samples_and_seed_variation(self):
        record = {"shape": "helmet", "geometry_seed": 3, "condition_seed": 5}
        first, second = sample(record, 64), sample(record, 64)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[0].shape, (len(CHANNEL_NAMES), 64, 64))
        self.assertEqual(first[1].shape, (1, 64, 64))
        self.assertFalse(np.array_equal(variation(64, 5)[0], variation(64, 6)[0]))

    def test_density_controls_and_marker_variation(self):
        geometry = render_geometry(Scene(shape="sphere"), 256)
        noise = variation(256, 7)
        controls = Controls(irregularity=0)
        mask = (geometry["object"] > 0) & (geometry["contour"] == 0)
        image = render_teacher(geometry, controls, noise)
        for key, low, high in [("darkness", 0.3, 1), ("width", 0.6, 3), ("crosshatch", 0, 1)]:
            lighter = render_teacher(geometry, replace(controls, **{key: low}), noise)
            darker = render_teacher(geometry, replace(controls, **{key: high}), noise)
            self.assertGreater(float(darker[mask].mean() * -1), float(lighter[mask].mean() * -1), key)
        dense = render_teacher(geometry, replace(controls, spacing=4), noise)
        sparse = render_teacher(geometry, replace(controls, spacing=20), noise)
        self.assertLess(float(dense[mask].mean()), float(sparse[mask].mean()))
        marker = render_teacher(geometry, replace(controls, preset="marker"), noise)
        self.assertGreater(float(np.abs(image - marker).mean()), 0.01)
        noisy = render_teacher(geometry, replace(controls, irregularity=1), variation(256, 8))
        self.assertGreater(float(np.abs(image - noisy).mean()), 0.005)

    def test_feature_controls_and_validation(self):
        geometry = render_geometry(Scene(), 32)
        features = make_features(geometry, Controls(), variation(32, 12), 0.3)
        self.assertEqual(features.shape[0], 31)
        self.assertEqual(len(set(CHANNEL_NAMES)), 31)
        with self.assertRaises(ValueError):
            Controls(spacing=0)
        with self.assertRaises(ValueError):
            Controls(darkness=float("nan"))
        with self.assertRaises(ValueError):
            Scene(light_elevation=0)
        with self.assertRaises(ValueError):
            render_geometry(Scene(), 33)

    def test_metrics_expose_blank_predictions_and_match_identical_targets(self):
        record = {"shape": "sphere", "geometry_seed": 3, "condition_seed": 5}
        features, target, _ = sample(record, 128)
        exact = metrics(target[0], target[0], features)
        self.assertEqual(exact["raster_mae"], 0)
        self.assertEqual(exact["shadow_tone_mae"], 0)
        self.assertAlmostEqual(exact["stroke_angle_error_degrees"], 0, places=4)
        blank = metrics(np.ones_like(target[0]), target[0], features)
        self.assertGreater(blank["raster_mae"], 0.01)
        self.assertEqual(blank["orientation_coverage"], 0)
        self.assertIsNone(blank["stroke_angle_error_degrees"])
        self.assertIsNone(blank["predicted_width_proxy_px"])

    def test_geometry_split_and_materialized_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            manifest = generate_manifest(output, samples=30, geometries=10, size=32, materialize=True)
            ids = {split: {r["geometry_id"] for r in manifest["records"] if r["split"] == split}
                   for split in ("train", "val", "test")}
            self.assertFalse(ids["train"] & ids["val"])
            self.assertFalse(ids["train"] & ids["test"])
            self.assertFalse(ids["test"] & ids["val"])
            dataset = SketchDataset(output, cache_samples=1)
            actual = dataset[0]
            expected = sample(dataset.records[0], dataset.size)
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            dataset[1]
            self.assertEqual(len(dataset.cache), 1)
            self.assertTrue((output / "preview/index.html").exists())
            with self.assertRaises(ValueError):
                generate_manifest(output)

    def test_cli_teacher_metadata_and_png(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "teacher.png"
            main(["teacher", "--output", str(output), "--size", "32", "--preset", "marker"])
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack("!II", output.read_bytes()[16:24]), (32, 32))
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["kind"], "procedural_teacher")
            self.assertEqual(metadata["controls"]["preset"], "marker")
            with self.assertRaises(ValueError):
                save_png(Path(temporary) / "bad.png", np.zeros((2, 2, 4)))


if __name__ == "__main__":
    unittest.main()
