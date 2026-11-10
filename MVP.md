# Character sketch design roadmap

## Intended experience

Enter a character or design reference, review its defining attributes and sources, choose which attributes to retain, and compose a sketch with adjustable lighting and drawing technique. Anime, manga, and soulslike designs are the intended subjects. Malenia is the first reference case for testing the attribute workflow.

The current PoC implements the rendering foundation only. Source lookup, attribute review, composition, and character generation are later work.

## Source-backed character attributes

Prefer official character material, game text, official art books, and publisher/developer references. Use clearly identified secondary references when primary material is insufficient. Preserve the actual source URL or document location for each assertion; a model's memory is not evidence.

Keep facts, visual observations, and user choices distinct. A source can establish a name or prosthetic limb; visual references may establish helmet shape. Record the specific appearance/form so attributes from different phases, costumes, or adaptations are not silently combined. Wings, winged helmet, and metal arm must be reviewed in that context rather than assumed to be one canonical outfit.

The planned attribute record contains character identity, appearance/form, attribute text, evidence kind, source location, supporting excerpt or image region, review status, and any conflict note. An unavailable or contradictory source leaves the assertion unresolved. The user can approve, edit, omit, or intentionally combine attributes into a new design.

Reference retrieval must remain independent of drawing-model weights. Pretrained assistance may suggest candidate attributes or annotations, but it cannot independently certify their accuracy. Do not silently treat web-search snippets as verified facts, and do not automatically turn lookup results into training data.

## Composition and character rendering

Translate approved attributes into pose, proportions, silhouette, and part placement. Preserve the source references for review. A composition module must generate or accept the geometry and lighting maps required by the renderer; the current synthetic maps are not obtainable from arbitrary 2D images without additional work.

First support constrained upper-body armor studies with explicit part placement. Then evaluate anatomy, full-body poses, clothing, hair, wings, and occlusion. The composition module is a separate learned task; a rendering U-Net trained on ellipsoids does not solve it.

Keep lighting an explicit design choice and test whether hatch orientation and density remain consistent under changes in pose and illumination. Add editable outline/shading/hatching layers only after raster quality is useful; the current output contract is PNG plus reproducibility metadata.

## Developing a drawing personality

Collect a small, deliberately reviewed set of drawings with consistent mark-making. Label medium, contour economy, hatch behavior, line-weight variation, shadow grouping, and stylistic exaggeration. Begin with user-created or otherwise authorized training material and preserve provenance.

Use synthetic pretraining to establish controls, then continue training the project's drawing model on curated examples. The from-scratch requirement applies to initial drawing-model weights; continuing its training on curated data is expected. Compare against the synthetic checkpoint to distinguish new style from degraded geometry or lost control responsiveness.

Human review should assess silhouette clarity, material distinction, contour hierarchy, lighting consistency, and whether the output has the intended design personality. Later add stochastic composition or a generative drawing architecture if deterministic conditioned rendering limits variety. Those choices follow evidence from the pilot rather than being presumed solved.

## Release criteria for a character MVP

- A reviewer can trace every retained character attribute to evidence or an explicit user choice.
- Form/appearance conflicts are visible and can be resolved before drawing.
- The selected attributes are recognizable in a constrained composition.
- Light changes produce coherent shading, and drawing controls have measured behavior.
- A curated style appears consistently across held-out characters and poses without copying training compositions.
- Outputs retain enough source, configuration, and checkpoint information for review and regeneration.

No automatic character lookup, source claims about Malenia, pretrained character generation, or full-character quality is included in the initial rendering PoC.
