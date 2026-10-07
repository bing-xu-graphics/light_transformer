# Teaser 3D scene

Scene `54f6ceaa` is the dining-room scene used in the paper teaser. The complete PBRT scene is distributed as `teaser_scene_pbrt.zip` on the repository's GitHub Releases page because it exceeds GitHub's regular file-size limit.

The archive contains the final PBRT scene, referenced meshes and textures, a preview, and the Infinigen BSD 3-Clause license notice. It excludes generation logs, older PBRT revisions, Blender backup files, and machine-specific paths.

- File: `teaser_scene_pbrt.zip`
- Size: 194,398,009 bytes
- SHA-256: `4d0da8e077f2677d9c5e8b9ebcb455ef09da6af28b94ec718020462023c87c32`

Render from the extracted directory with a compatible PBRT installation:

```bash
pbrt scene.pbrt
```

This scene was procedurally generated with [Infinigen](https://github.com/princeton-vl/infinigen) `v1.12.1` and prepared for the Light Transformer project. See [`INFINIGEN_LICENSE.txt`](INFINIGEN_LICENSE.txt) and [`../../dataset_helper/THIRD_PARTY.md`](../../dataset_helper/THIRD_PARTY.md).
