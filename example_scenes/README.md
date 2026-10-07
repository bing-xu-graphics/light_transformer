# Example scenes

This directory contains model-ready inputs and reference outputs for selected scenes shown in the paper.

The six included scenes correspond to the six rows of the paper's irradiance comparison:

- 1accfd5a
- 1b3f884d
- 2a3d1c70
- 2be44ba9
- 6576b487
- 8106d53

Each directory contains scene points, query points, and reference illumination. The first three also contain direct-illumination, prediction, and path-traced-reference images from the paper's full-render figure. These compact examples use 5,000 scene points and can be evaluated by the released model. The paper's training configuration used 20,000 scene points.

The [`54f6ceaa`](54f6ceaa) directory documents the complete teaser 3D scene, distributed separately as a GitHub Release asset because of its size.

Raw PBRT scenes can be 500 MB to 2 GB each. They are not included in this repository because they contain upstream generated assets. See [`dataset_helper`](../dataset_helper) for instructions on creating new scenes.

## Array layout

Scene points use 12 float channels: position, normal, albedo, and emissiveness.

Query points use 9 float channels: position, normal, and albedo.

Reference illumination is RGB.
