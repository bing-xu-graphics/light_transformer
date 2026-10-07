"""Validate the local release assets without importing PyTorch."""

from __future__ import annotations

import ast
import hashlib
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENES = (
    "1accfd5a",
    "1b3f884d",
    "2a3d1c70",
    "2be44ba9",
    "6576b487",
    "8106d53",
)
FULL_RENDER_SCENES = ("1accfd5a", "1b3f884d", "2a3d1c70")
CHECKPOINT = ROOT / "weights" / "light_transformer_step350000.ckpt"
CHECKPOINT_SIZE = 502_984_850
CHECKPOINT_SHA256 = "d957187949f44c36415116436fe613e1061d940706b84565ef5b5e65fc56af7e"
TEASER_ARCHIVE = ROOT / "example_scenes" / "54f6ceaa" / "teaser_scene_pbrt.zip"
TEASER_ARCHIVE_SIZE = 194_398_009
TEASER_ARCHIVE_SHA256 = "4d0da8e077f2677d9c5e8b9ebcb455ef09da6af28b94ec718020462023c87c32"


def npy_header(path: Path) -> dict:
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"not a NumPy file: {path}")
        major, _ = stream.read(2)
        length_format = "<H" if major == 1 else "<I"
        header_length = struct.unpack(
            length_format, stream.read(struct.calcsize(length_format))
        )[0]
        return ast.literal_eval(stream.read(header_length).decode("latin1"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_shape(path: Path, shape: tuple[int, ...]) -> None:
    header = npy_header(path)
    if tuple(header["shape"]) != shape:
        raise ValueError(f"{path}: expected {shape}, found {header['shape']}")
    print(f"ok  {path.relative_to(ROOT)}  {shape}  {header['descr']}")


def main() -> None:
    for scene_id in SCENES:
        scene = ROOT / "example_scenes" / scene_id
        require_shape(scene / "scene_points.npy", (5_000, 12))
        require_shape(scene / "query_points.npy", (262_144, 9))
        require_shape(scene / "reference.npy", (262_144, 3))
    for scene_id in FULL_RENDER_SCENES:
        scene = ROOT / "example_scenes" / scene_id
        for image in (
            "direct_illumination.png",
            "prediction.png",
            "path_traced_reference.png",
        ):
            if not (scene / image).is_file():
                raise FileNotFoundError(scene / image)

    if CHECKPOINT.stat().st_size != CHECKPOINT_SIZE:
        raise ValueError(f"unexpected checkpoint size: {CHECKPOINT.stat().st_size}")
    checkpoint_hash = sha256(CHECKPOINT)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError(f"unexpected checkpoint SHA-256: {checkpoint_hash}")
    print(f"ok  {CHECKPOINT.relative_to(ROOT)}  {CHECKPOINT_SIZE} bytes")

    if TEASER_ARCHIVE.stat().st_size != TEASER_ARCHIVE_SIZE:
        raise ValueError(f"unexpected teaser archive size: {TEASER_ARCHIVE.stat().st_size}")
    teaser_hash = sha256(TEASER_ARCHIVE)
    if teaser_hash != TEASER_ARCHIVE_SHA256:
        raise ValueError(f"unexpected teaser archive SHA-256: {teaser_hash}")
    print(f"ok  {TEASER_ARCHIVE.relative_to(ROOT)}  {TEASER_ARCHIVE_SIZE} bytes")


if __name__ == "__main__":
    main()
