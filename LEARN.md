# Learn deep learning with AnimeArt

This guide assumes you know basic Python but are new to deep learning. Run commands from the AnimeArt project directory. Each step uses the same data and model as the main application.

## 0. Get the environment ready

This project uses Python 3.12 in `.venv`. Your default `python3` may be a different version, so use the full path below. An internet connection to your Python package index is required for installation.

```sh
.venv/bin/python -m pip install -e '.[demo]'
.venv/bin/python -m pip check
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'MPS', torch.backends.mps.is_available())"
.venv/bin/python -m unittest discover -s tests -v
```

If pip reports that it cannot resolve the package index, retry from a network-enabled terminal. The environment needs NumPy and PyTorch before any lesson runs. `MPS True` means the Mac GPU is available; `False` still permits CPU lessons.

> **TOPIX · Read:** How a virtual environment keeps project packages separate from system Python.

## 1. Meet the training example

```sh
.venv/bin/python examples/learn.py inspect
```

The input is a tensor shaped `[31, 64, 64]`: 31 maps, each 64 by 64 pixels. Some maps describe shape, surface direction, lighting, or drawing controls. The target is a `[1, 64, 64]` teacher sketch. A batch adds a leading dimension, so the network sees `[batch, 31, 64, 64]`. Open `data/smoke/preview/index.html` to see example targets.

Try finding `tone`, `contour`, and `spacing` in the printed channel list. Which describe the object, and which ask for a drawing style?

> **TOPIX · Read:** Tensor shape as batch × channels × height × width.

## 2. Watch one update

```sh
.venv/bin/python examples/learn.py step
```

The model starts with random **weights**. It makes a prediction, then the **loss** measures how different that prediction is from the teacher sketch. `loss.backward()` computes **gradients**, which point to how weights affect loss. `optimizer.step()` changes the weights. The output reports a gradient and whether the final layer's weights changed. One step is only a mechanics demonstration; it will not make a useful drawing.

Try running the command twice. The numbers should match because it seeds the random starting weights.

> **TOPIX · Read:** Forward pass → loss → backward pass → optimizer step in `examples/learn.py`.

## 3. Check that the whole pipeline trains

```sh
.venv/bin/python -m animeart benchmark --device auto --size 64 --base-channels 8 --output runs/benchmark-64.json
.venv/bin/python -m animeart train --config configs/smoke.json
.venv/bin/python -m animeart evaluate --checkpoint runs/smoke/best.pt --dataset data/smoke --split val --output runs/smoke-evaluation
```

Open `runs/smoke-evaluation/index.html`. The 20-step smoke run checks that model updates, checkpoint saving, and evaluation work. It is too short to judge drawing quality. `benchmark-64.json` reports actual step time and memory on your machine; data generation and validation add time beyond that figure.

> **TODO · Record:** PyTorch version, MPS availability, test results, and 64-pixel benchmark after installation succeeds.

## 4. Ask the model to memorize four examples

```sh
.venv/bin/python -m animeart train --config configs/overfit.json
.venv/bin/python examples/learn.py compare --checkpoint runs/overfit/last.pt
```

Open `runs/learning/index.html`. For four training examples, it shows the teacher target, the model before learning, and the trained model. Read `runs/overfit/training-report.json`: `halved_loss` is a numeric diagnostic. Inspect whether the contours and hatch marks actually look closer to the targets. If the loss falls while the marks remain blurred, this is a model or loss problem to investigate before a larger run.

Training on these same examples repeatedly is **overfitting**. Here it is intentional: a model that cannot learn four examples is unlikely to learn thousands.

> **TODO · Record:** Whether the four-example loss halved and whether the strokes visibly improved.

## 5. Try unseen geometry and drawing controls

```sh
.venv/bin/python -m animeart evaluate --checkpoint runs/overfit/best.pt --dataset data/smoke --split val --output runs/overfit-evaluation --sweeps
.venv/bin/python -m animeart demo --checkpoint runs/overfit/best.pt
```

The validation split contains geometry identities kept out of training. Open `runs/overfit-evaluation/index.html` and compare it with the memorized examples. The existing held-out drawings are blurry, and the report records zero reliable stroke-angle coverage. The demo lets you inspect this failure, but its output is not a finished sketch. Hold geometry and variation seeds fixed while changing light, hatch spacing, or crosshatching. This difference between memorizing four examples and drawing unseen ones is **generalization**.

> **TOPIX · Read:** Training vs validation, checkpoint `best.pt` vs `last.pt`, and why low training loss is insufficient.

> **TODO · Improve:** Replace the synthetic drawing targets and primitive forms with examples that match the desired art direction, then require clear held-out strokes before a 10,000-step pilot.

The bigger character-design goal is in [MVP.md](MVP.md). This stage teaches the renderer to imitate synthetic sketch targets; it does not yet learn anatomy or a personal art style.

## 6. Study a 2D figure experiment

The figure path adds posed heroine and armored-heroine guides. It replaces ellipsoid arms with shoulder–elbow–wrist and hip–knee–ankle chains. The forearm outline follows a tapered width profile. The network's final bias starts the output near white; an explicit contour prior darkens the line guide before training. The network learns ink and shading around that guide. The generator supplies the anatomy and armor construction, so this exercise does not teach the network to invent a new body.

```sh
.venv/bin/python -m animeart generate --output data/figures-v3 --samples 120 --geometries 30 --size 128 --shapes heroine armored_heroine
.venv/bin/python -m animeart train --config configs/figures-smoke.json
.venv/bin/python -m animeart evaluate --checkpoint runs/figures-v3/best.pt --dataset data/figures-v3 --split val --output runs/figures-v3-evaluation
```

Open `data/figures-v3/preview/index.html` and `runs/figures-v3-evaluation/index.html`. The first shows the target drawings; the second shows what the trained model reproduces on unseen pose recipes. These are schematic anatomy studies, not finished anime drawings. Increasing training steps cannot add design detail absent from the target drawings.

> **TOPIX · Read:** Follow `contour_prior` from `configs/figures-smoke.json` into `SketchUNet.forward`; compare its first output with a trained checkpoint.

> **TODO · Art direction:** Study the three images in `assets/`. Extend the guide with their winged helm, layered plates, flowing hair and cloth, stronger facial design, and deliberate line weight. Add reviewed examples of other poses before scaling the dataset.

## 7. Try a real drawing as a guide

```sh
.venv/bin/python -m animeart reference --checkpoint runs/figures-v3/best.pt --source assets/malenia1.jpeg --output runs/malenia1-reference --size 256
```

Open `runs/malenia1-reference/index.html`. The guide is extracted from the existing drawing. The raw output shows what the network actually predicts; the last panel adjusts its contrast for viewing. The model has not learned to create Malenia or invent a new pose from this single image.

> **TOPIX · Read:** Distinguish a training target, a validation example, an input guide, and a reference image. They play different roles even when all are PNGs or JPEGs.
