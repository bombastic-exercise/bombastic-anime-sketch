"""Recipes split by geometry identity before lighting or drawing augmentation."""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .drawing import FEATURE_VERSION, Controls, PRESETS, make_features, render_teacher, variation
from .io import gallery, save_png, write_json
from .scene import SHAPES, Scene, render_geometry


def generate_manifest(output, samples=10000, geometries=500, size=256, seed=42, materialize=False):
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValueError(f"Dataset already exists: {output}; choose a new directory")
    if geometries < 10 or samples < geometries:
        raise ValueError("Use at least 10 geometries and at least one sample per geometry")
    # Validate dimensions before creating any output.
    if size < 32 or size > 1024 or size % 8:
        raise ValueError("size must be a multiple of 8 between 32 and 1024")
    rng = np.random.default_rng(seed)
    groups = rng.permutation(geometries)
    train_end, val_end = int(0.8 * geometries), int(0.9 * geometries)
    assigned = {int(group): ("train" if i < train_end else "val" if i < val_end else "test")
                for i, group in enumerate(groups)}
    records = [{"id": i, "geometry_id": i % geometries,
                "geometry_seed": seed * 100000 + i % geometries,
                "condition_seed": seed * 1000000 + i,
                "shape": SHAPES[(i % geometries) % len(SHAPES)],
                "split": assigned[i % geometries]} for i in range(samples)]
    manifest = {"version": 1, "feature_version": FEATURE_VERSION, "size": size,
                "seed": seed, "geometries": geometries, "records": records}
    write_json(output / "manifest.json", manifest)
    if materialize:
        for record in records:
            features, target, _ = sample(record, size)
            np.savez_compressed(output / f'{record["id"]:06d}.npz', features=features, target=target)
    preview_dataset(output)
    return manifest


def recipe(record):
    rng = np.random.default_rng(record["condition_seed"])
    scene = Scene(shape=record["shape"], geometry_seed=record["geometry_seed"],
                  camera_yaw=float(rng.uniform(-45, 45)), camera_elevation=float(rng.uniform(8, 30)),
                  light_azimuth=float(rng.uniform(-100, 100)), light_elevation=float(rng.uniform(15, 75)),
                  material=float(rng.uniform(0, 1)))
    controls = Controls(preset=str(rng.choice(PRESETS)), angle=float(rng.uniform(-75, 75)),
                        spacing=float(rng.uniform(4, 20)), width=float(rng.uniform(0.6, 2.2)),
                        crosshatch=float(rng.uniform(0, 1)), darkness=float(rng.uniform(0.5, 1)),
                        irregularity=float(rng.uniform(0, 0.7)))
    return scene, controls, record["condition_seed"]


def sample(record, size):
    scene, controls, seed = recipe(record)
    geometry = render_geometry(scene, size)
    noise = variation(size, seed)
    return (make_features(geometry, controls, noise, scene.material),
            render_teacher(geometry, controls, noise)[None],
            {"scene": scene.to_dict(), "controls": controls.to_dict(), "seed": seed})


class SketchDataset:
    # A plain indexing class also works with torch DataLoader, without importing torch here.
    def __init__(self, directory, split="train", cache_samples=64):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["version"] != 1 or self.manifest["feature_version"] != FEATURE_VERSION:
            raise ValueError("Unsupported dataset version; regenerate the dataset")
        self.size = self.manifest["size"]
        self.records = [r for r in self.manifest["records"] if r["split"] == split]
        if not self.records:
            raise ValueError(f"No records for split {split}")
        self.cache_samples = cache_samples
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        record = self.records[index]
        path = self.directory / f'{record["id"]:06d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as item:
                result = item["features"], item["target"]
        else:
            features, target, _ = sample(record, self.size)
            result = features, target
        if self.cache_samples > 0:
            self.cache[index] = result
            while len(self.cache) > self.cache_samples:
                self.cache.popitem(last=False)
        return result


def preview_dataset(directory, count=8):
    dataset = SketchDataset(directory, cache_samples=0)
    output = Path(directory) / "preview"
    rows = []
    for i in range(min(count, len(dataset))):
        features, target = dataset[i]
        images = []
        for label, pixels in [("teacher", target[0]), ("tone", 1 - features[7]),
                              ("normals", np.moveaxis(features[3:6], 0, -1) * 0.5 + 0.5),
                              ("cast shadows", 1 - features[8])]:
            path = output / f"{i:02d}-{label.replace(' ', '-')}.png"
            save_png(path, pixels)
            images.append((label, path))
        rows.append((f'{dataset.records[i]["shape"]} · sample {dataset.records[i]["id"]}', images))
    gallery(output / "index.html", rows)
