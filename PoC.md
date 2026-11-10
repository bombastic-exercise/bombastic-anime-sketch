# Learned sketch rendering: proof of concept

## Question

Can a compact, randomly initialized network learn consistent contour and mark rendering, respond to explicit sketch controls, and preserve a known lighting arrangement on unseen forms?

The initial subject is rendering, not anatomy or character generation. Success here is a prerequisite for building a model with a recognizable drawing personality.

## Pipeline

1. Construct analytic ellipsoid scenes: simple volumes, curved plates, winged helmet studies, and jointed arm studies. Randomize dimensions, local rotations, orthographic camera, light direction, and highlight strength.
2. Ray-intersect camera rays and shadow rays. Produce masks, depth, normals, visible-part contours, tone, shadow visibility, local surface coordinates, and a direction field. A white receiver plane captures cast shadows. Lighting uses a single directional source, ambient contribution, and an approximate specular term; global illumination and soft shadows are outside this experiment.
3. A procedural teacher draws antialiased, surface-coordinate hatching. Darker tones strengthen the first hatch family; deep shadows enable the second. Pencil adds grain; marker broadens marks and introduces wash and overlap. Seeded variation is an explicit input.
4. Train a conditional U-Net from scratch. Predict a grayscale sketch directly; inference does not invoke the teacher. Fixed coordinate Fourier features help represent fine detail without embedding a hatch target.
5. Compare outputs against targets on geometry identities excluded from training. Sweep controls independently and inspect the resulting contact sheets.

The design borrows the connection between surface direction, mark density, and tone from [Real-Time Hatching](https://gfx.cs.princeton.edu/proj/hatching/). [Geometry-aware line drawing research](https://arxiv.org/abs/2203.12691) motivates preserving form, but its pretrained semantic/depth losses are not used here. This project is not a reproduction of either paper.

## Model and interfaces

`Scene` describes a procedural form, geometry seed, camera, light, and material. `Controls` describes medium, hatch angle/spacing/width, crosshatch strength, darkness, and irregularity.

Feature layout version 1 has 31 channels: masks (2), depth (1), normals (3), contour/tone/shadow (3), surface coordinates (2), flow (2), material (1), screen coordinates (2), variation/grain (2), light vector (3), and drawing controls (10). Geometry maps are available from the synthetic scene; obtaining equivalent maps from arbitrary character references is future work.

The U-Net has three downsampling levels, GroupNorm, SiLU, skip connections, bilinear upsampling, and global conditioning through affine feature modulation. Pilot base width is 24; smoke width is 8. Training uses float32 AdamW and clipped gradients. No pretrained encoder, rasterizer, diffusion model, or character model is loaded.

The loss combines ink-weighted raster L1, contour L1, pooled tone L1, and gradient-direction agreement modulo 180°. It is intentionally an experimental baseline. Dense fine marks may blur or average out; comparisons must expose that failure rather than treat low background error as success.

## Experiments and gates

| Stage | Evidence required before expansion |
| --- | --- |
| Teacher QA | Plausible normals and shadows; surface-following marks; white unshadowed paper; distinct media |
| Runtime benchmark | Measured forward/backward throughput and memory on the actual M3 Pro; no extrapolated training promise |
| Four-example overfit | Substantial loss reduction and recognizable reproduction of contours and marks |
| Held-out rendering | Unseen geometry comparison sheets; raster/contour/tone metrics; orientation error with reliable-patch coverage |
| Control behavior | Angle changes track the teacher; spacing, width, crosshatching, and darkness produce the intended density changes |
| Lighting behavior | Moving the light moves shadows and ink concentration while geometry stays fixed |
| Style development | Curated drawings introduce mark decisions beyond the synthetic teacher; human review confirms the intended personality |

The overfit report flags a halving of its initial loss as a diagnostic, not a visual-quality guarantee. Evaluation includes a white-image error baseline so sparse paper does not make a weak model look successful. The width estimate is an ink-area/edge-contrast proxy, biased by wash and overlap; it is not an exact stroke-width measurement. Angle measurements use coherent 16-pixel patches away from contours and report missing coverage explicitly.

The default training distribution uses camera yaw ±45°, elevation 8–30°, light azimuth ±100°, and light elevation 15–75°. Sweep sheets include camera yaw ±70° and control endpoints outside the sampled range to expose extrapolation failures. Current holdout geometry uses new dimensions and rotations from the same procedural families; it does not demonstrate generalization to human anatomy or unseen asset families.

## Artifacts and reproducibility

Dataset manifests retain complete recipe seeds and geometry split membership. Checkpoints retain model/optimizer state, RNG state, feature version, training configuration, and manifest hash. PNG exports retain scene, controls, seed, and checkpoint hash in JSON. A given teacher recipe is deterministic within the same numerical environment. GPU repeatability is tested with tolerance; cross-backend bitwise equality is not required.

Benchmarks separate data-generation time from optimizer-step time. MPS allocation is sampled and labeled accordingly; process peak RSS is also recorded. Cache size is configurable because a 256-pixel multi-channel sample is several megabytes.

The implementation includes training and evaluation commands, but no trained checkpoint is bundled. The implementation environment could not download PyTorch or Gradio. Recorded verification covers the synthetic pipeline; learning and demo execution remain unverified until dependencies are installed.

## After the pilot

If the model cannot overfit four examples, inspect losses and coordinate representation before scaling data. If it overfits but fails on held-out geometry, broaden geometry and mark examples. If it reproduces synthetic targets but lacks personality, move to curated artwork with style labels and paired or reviewed geometry annotations. Synthetic performance alone is not evidence of a learned personal drawing style.
