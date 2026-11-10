# AnimeArt

A PyTorch experiment that learns to render 2D sketches from geometry and drawing controls. It now includes articulated heroine and armored-heroine pose studies alongside the earlier simple forms. The drawing network starts from initialized weights; a figure-specific contour prior makes its first prediction resemble white paper with an outline.

This is a rendering research scaffold. Its new full-body figures are schematic pose targets, not finished anime art. It does not look up Malenia. [PoC.md](PoC.md) describes the original experiment; [MVP.md](MVP.md) describes the character-design workflow that follows it.

## Setup

For a guided introduction to tensors, gradients, checkpoints, and the first training run, follow [LEARN.md](LEARN.md). Its visible `TOPIX` flags mark concepts to read; `TODO` flags mark results to fill in after running the experiments.

Use Python 3.12 or 3.13. The initial supported training target is an Apple Silicon Mac with PyTorch MPS; CPU execution is also supported. Other GPU configurations have not been validated.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[demo]'
```

The CLI is available as `animeart` or `python -m animeart`. Run commands from the repository root so the example configuration paths resolve correctly. Without the optional demo, install with `python -m pip install -e .`.

## Current figure run and Malenia reference

The active experiment uses 120 procedural recipes at 128 pixels: 96 training examples, 12 validation examples, and 12 test examples. Geometry identities are split before lighting and drawing variation, so related recipes stay in one split. `configs/figures-smoke.json` trains 200 optimizer steps and checks four validation examples every 50 steps. It saves the checkpoint with the lowest validation loss. This is a pipeline check, not a quality claim.

The three images in `assets/` are visual references. They are not part of the training dataset. To run the current figure checkpoint on the line structure extracted from `malenia1.jpeg`:

```sh
animeart reference --checkpoint runs/figures-v3/best.pt --source assets/malenia1.jpeg --output runs/malenia1-reference --size 256
```

Open `runs/malenia1-reference/index.html` to compare the source, extracted line guide, raw prediction, and contrast-adjusted prediction. The image supplies Malenia's composition and detail; the model only renders from that guide. The checkpoint was trained at 128 pixels, so the 256-pixel reference run is exploratory. `metadata.json` records the source and checkpoint hashes plus the display adjustment.

The unit tests check geometry, data splits, deterministic rendering, model updates, checkpoint continuation, export, and reference input handling. They do not judge whether a drawing looks good. Validation comparisons use unseen procedural geometry; the test split is reserved for final assessment. Running `malenia1.jpeg` through the checkpoint is inference, not a held-out test, because its guide does not come from the procedural dataset.

## Inspect the training targets

For the new 2D pose path, generate a small dataset and inspect the two figure types and three joint poses:

```sh
animeart generate --output data/figures-v3 --samples 120 --geometries 30 --size 128 --shapes heroine armored_heroine
animeart teacher --shape armored_heroine --geometry-seed 1 --output runs/heroine-pose.png
animeart train --config configs/figures-smoke.json
animeart evaluate --checkpoint runs/figures-v3/best.pt --dataset data/figures-v3 --split val --output runs/figures-v3-evaluation
```

Open `data/figures-v3/preview/index.html` to inspect the targets. `contour_prior` in the figure config starts from white paper and the figure renderer's line guide. The model learns ink and shading from that guide; the 2D figure renderer supplies pose, face, hair, and armor lines. Existing datasets and checkpoints keep their earlier feature layout.

> **TOPIX · Read:** The shoulder–elbow–wrist and hip–knee–ankle chains in `src/animeart/figure.py`; tapered forearms are drawn from width profiles rather than ellipses.

> **TODO · Improve:** Replace these schematic targets with reviewed anime figure and armor drawings before a large training run. Check hand anatomy, face design, pose range, and armor construction against those references.

```sh
animeart teacher --shape helmet --preset fineliner --output runs/teacher-helmet.png
animeart teacher --shape arm --preset marker --crosshatch 0.8 --output runs/teacher-arm.png
animeart generate --output data/smoke --samples 120 --geometries 30 --size 64
```

Open `data/smoke/preview/index.html` to inspect targets, tones, normals, and shadows. Teacher exports are procedural examples, not model results. They include JSON settings alongside each PNG.

The current workspace already contains smoke and pilot manifests; skip their generation commands if present. A five-form, three-medium comparison is available at `runs/teacher-studies/index.html` and `runs/teacher-studies/contact-sheet.png`.

The original forms are `sphere`, `cylinder` (a rounded elongated volume), `plate`, `helmet`, and `arm`. Helmet and arm are simple ellipsoid compositions. The newer `heroine` and `armored_heroine` forms use a separate 2D joint and contour renderer.

## Measure and train

Start with the small benchmark and diagnostic before committing to a pilot run:

```sh
animeart benchmark --device auto --size 64 --base-channels 8 --output runs/benchmark-64.json
python -m unittest discover -s tests -v
animeart train --config configs/smoke.json
animeart train --config configs/overfit.json
```

`overfit.json` trains repeatedly on four examples. Inspect `runs/overfit/training-report.json`: it records whether the diagnostic loss halved. Inspect predictions as well; a falling loss alone does not prove useful hatching.

Evaluate and view the small run:

```sh
animeart evaluate --checkpoint runs/smoke/best.pt --dataset data/smoke --output runs/smoke-evaluation --sweeps
animeart demo --checkpoint runs/smoke/best.pt
```

Open `runs/smoke-evaluation/index.html` for teacher/prediction comparisons, and the local address printed by Gradio for the interactive viewer. Smoke training only checks the pipeline; 20 steps are not intended to produce good sketches.

For the 256 × 256 pilot, benchmark that configuration first:

```sh
animeart benchmark --size 256 --base-channels 24 --batch-size 4 --output runs/benchmark-256.json
animeart generate --output data/pilot --samples 10000 --geometries 500 --size 256
animeart train --config configs/pilot.json
animeart evaluate --checkpoint runs/pilot/best.pt --dataset data/pilot --output runs/pilot-evaluation --sweeps
```

The manifest stores 10,000 reproducible recipes, grouped into 500 geometry identities. Geometry is assigned to train/validation/test before lighting and drawing augmentation, with an 80/10/10 geometry split. Samples are rendered on demand with a bounded cache; `generate --materialize` can cache all arrays to disk, at substantial storage cost. Begin with the small dataset.

Training writes `last.pt`, `best.pt`, `config.json`, `metrics.jsonl`, and `training-report.json`. Validation chooses `best.pt`; reserve the test split for final assessment. Dataset generation and fresh training refuse to overwrite an existing dataset or checkpoint. For an interrupted run, increase the total step target and resume:

```sh
animeart train --config configs/pilot.json --resume runs/pilot/last.pt --steps 15000
```

Resume restores model, optimizer, sampler, and device RNG state, and checks dataset identity and training settings. Seeds support comparisons on the same backend; bitwise equivalence across CPU and MPS is not promised.

## Draw with a checkpoint

```sh
animeart render --checkpoint runs/pilot/best.pt --shape helmet --preset pencil --light-azimuth -60 --light-elevation 40 --angle 30 --spacing 8 --width 1.2 --crosshatch 0.7 --seed 12 --output runs/helmet.png
animeart demo --checkpoint runs/pilot/best.pt
```

The learned renderer receives geometry, lighting, explicit controls, and seeded variation. It never receives the target sketch at inference. Outputs include PNG and JSON with checkpoint hash, scene, controls, and seed. Keep geometry and variation seeds fixed when comparing settings. The demo listens on `127.0.0.1` and does not create a public share link.

| Control | Meaning |
| --- | --- |
| `angle` | Surface-coordinate hatch angle offset, −90° to 90°; not a constant screen angle |
| `spacing` | 3–24 reference pixels at 256 resolution |
| `width` | 0.4–4 reference pixels at 256; marker preset uses a broader footprint |
| `crosshatch` | Strength of the second hatch family in darker regions, 0–1 |
| `darkness` | Ink opacity target, 0.2–1 |
| `irregularity` | Seeded mark displacement and variation, 0–1 |
| `material` | Synthetic highlight strength, 0–1 |

These are learned controls with approximate behavior. Evaluation reports raster and contour error, shadow tone, local angle error with coverage, ink density, edge contrast, and a width proxy. Control sweeps report trends rather than assuming they succeed. A geometry-based teacher can only teach the appearance programmed into it; a curated drawing stage is needed to develop a personal style.

## Verification status

The synthetic pipeline has been executed at 64 and 256 pixels. Current artifacts include 64-pixel smoke and four-example overfit checkpoints, evaluations, and an MPS benchmark recorded on the local Mac. The overfit run reduced training loss, but held-out results are visibly blurry and evaluation records zero reliable stroke-angle coverage. These checkpoints are learning diagnostics, not usable character sketches. The synthetic teacher drawings also lack the anatomy and design detail needed for the character goal.

In this execution environment, 13 of 14 tests pass; the MPS test skips because MPS is unavailable here. Earlier local run reports record MPS training. `verification.json` is the historical pre-install record.

The 128-pixel figure run completed 200 CPU steps and reached best validation loss 0.0446. Its learned output follows the line guide on held-out recipes, but the source figures remain schematic. The three references in `assets/` show the desired detail: a winged helm, layered armor, flowing hair and cloth, stronger facial design, and deliberate line weight. The current pose guides do not contain that detail. Improving the guide and collecting reviewed target drawings comes before a larger training run. The figure artifacts live in ignored `data/figures-v3/` and `runs/figures-v3/` directories.
