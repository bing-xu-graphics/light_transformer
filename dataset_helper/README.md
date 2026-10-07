# Dataset helper

This directory documents how we prepared procedural indoor scenes for the Light Transformer dataset. Third-party source code and assets are not included.

## Workflow

1. Generate indoor scenes with Infinigen.
2. Keep completed Blender scenes and discard failed generations.
3. Inspect and adjust scenes in Blender when needed.
4. Export the scenes to PBRT using `io_scene_pbrt`.
5. Process the exported scenes with the project rendering pipeline.

See [`scene_generation.md`](scene_generation.md) for the generation setup and an account of the original Blender-to-PBRT workflow.
