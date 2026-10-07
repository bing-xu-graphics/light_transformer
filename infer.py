import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from light_transformer.config import DATA_CONFIG
from light_transformer.model import Model


def load_model(checkpoint_path, device):
    model = Model(
        DATA_CONFIG=DATA_CONFIG,
        depth=6,
        dim=128,
        k=32,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model"]
    if next(iter(state)).startswith("module."):
        state = {name.removeprefix("module."): value for name, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("prediction.npy"))
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Light Transformer inference requires a CUDA-capable GPU")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    for filename in ("scene_points.npy", "query_points.npy"):
        if not (args.scene / filename).is_file():
            raise FileNotFoundError(args.scene / filename)

    device = torch.device("cuda")
    scene = torch.from_numpy(
        np.load(args.scene / "scene_points.npy", allow_pickle=False)
    ).float()[None].to(device)
    queries = torch.from_numpy(
        np.load(args.scene / "query_points.npy", allow_pickle=False)
    ).float()[None]

    scene[..., 3:6] = F.normalize(scene[..., 3:6], dim=-1)
    queries[..., 3:6] = F.normalize(queries[..., 3:6], dim=-1)

    model = load_model(args.checkpoint, device)
    scene_embedding, scene_positions = model.encode_scene(scene)
    predictions = []
    for start in range(0, queries.shape[1], args.chunk_size):
        query_chunk = queries[:, start : start + args.chunk_size].to(device)
        prediction = model.decode_queries(
            scene_embedding, scene_positions, query_chunk
        )
        predictions.append(prediction.cpu())

    output = torch.cat(predictions, dim=1)[0].numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)
    print(f"Saved {output.shape} to {args.output}")


if __name__ == "__main__":
    main()
