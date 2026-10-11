# -*- coding: utf-8 -*-
"""Export the Strands Decider vision tower to ONNX (the part ollaya's text graph does not have).

    .venv-ref\\Scripts\\python.exe export/export_vision.py

Builds ONLY the vision tower (Qwen3_5VisionModel, 24 blocks, 1024 wide, merging 2x2 patches into one 2048-d token) from
``config.json`` and loads its 333 tensors (``model.visual.*``, all in the last safetensors shard, bf16 -> fp32). The
language model is never loaded. One image per call (see export/vision_export_patch.py).

    pixel_values      float32 [patches, 3*2*16*16]
    bilinear_indices  int64   [4, patches]   corners of the bilinear interpolation of the learned position table
    bilinear_weights  float32 [4, patches]
    position_ids      int64   [patches, 2]   (row, col) of each patch, for the vision rotary embedding
    -> image_features float32 [patches / 4, 2048]

The last three depend on the image's grid size through data-dependent Python (``.tolist()`` and loops in
transformers' ``vision_utils``), which ``torch.export`` cannot trace. ``Qwen3_5VisionModel.forward`` accepts them
precomputed, so they are graph INPUTS here and strands_decide/vision.py computes them with numpy (verify/check_vision_inputs.py
compares that port with transformers' own functions).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "models" / "base"
OUT_DIR = ROOT / "models" / "onnx"


class VisionGraph(nn.Module):
    """pixel_values + image_grid_thw -> one embedding row per visual token."""

    def __init__(self, holder):
        super().__init__()
        self.holder = holder
        # only used for shapes/config inside forward; every value that depends on it is passed in precomputed
        self.register_buffer("dummy_grid", torch.tensor([[1, 16, 16]], dtype=torch.long), persistent=False)
        self.register_buffer("dummy_cu", torch.tensor([0, 256], dtype=torch.int32), persistent=False)

    def forward(self, pixel_values, bilinear_indices, bilinear_weights, position_ids):
        out = self.holder.visual(
            pixel_values, grid_thw=self.dummy_grid, interp_indices=bilinear_indices.transpose(0, 1),
            interp_weights=bilinear_weights.transpose(0, 1), position_ids=position_ids, cu_seqlens=self.dummy_cu,
            return_dict=True)
        return out.pooler_output


class Holder(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual


def load_vision():
    import json

    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    config = AutoConfig.from_pretrained(str(MODEL_DIR))
    visual = Qwen3_5VisionModel(config.vision_config).float().eval()
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    keys = [k for k in index if k.startswith("model.visual.")]
    shards = sorted({index[k] for k in keys})
    state = {}
    for shard in shards:
        with safe_open(str(MODEL_DIR / shard), framework="pt") as handle:
            for key in keys:
                if index[key] == shard:
                    state[key[len("model.visual."):]] = handle.get_tensor(key).float()
    missing, unexpected = visual.load_state_dict(state, strict=True), None
    print("loaded %d vision tensors from %s" % (len(state), shards), flush=True)
    return config, visual


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="vision_encoder.onnx")
    args = ap.parse_args()

    from export.vision_export_patch import apply as patch_vision

    config, visual = load_vision()
    holder = Holder(visual)
    inner = patch_vision(holder)
    vision = VisionGraph(inner).eval()
    vc = config.vision_config
    patch_dim = 3 * vc.temporal_patch_size * vc.patch_size ** 2
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids

    pixel_values = torch.randn(256, patch_dim)
    grid = torch.tensor([[1, 16, 16]], dtype=torch.long)
    side = int(vc.num_position_embeddings ** 0.5)
    bi, bw = get_vision_bilinear_indices_and_weights(grid, num_grid_per_side=side, spatial_merge_size=vc.spatial_merge_size)
    pos = get_vision_position_ids(grid, vc.spatial_merge_size)
    with torch.no_grad():
        probe = vision(pixel_values, bi, bw, pos)
    print("probe features:", tuple(probe.shape), flush=True)

    target = OUT_DIR / args.out
    started = time.time()
    torch.onnx.export(
        vision, (pixel_values, bi, bw, pos), str(target),
        input_names=["pixel_values", "bilinear_indices", "bilinear_weights", "position_ids"],
        output_names=["image_features"],
        dynamic_shapes={"pixel_values": {0: "patches"}, "bilinear_indices": {1: "patches"},
                        "bilinear_weights": {1: "patches"}, "position_ids": {0: "patches"}},
        opset_version=18, dynamo=True, external_data=True, optimize=False,
    )
    print("exported vision encoder in %.1f s -> %s" % (time.time() - started, target), flush=True)
    torch.save({"pixel_values": pixel_values, "bilinear_indices": bi, "bilinear_weights": bw,
                "position_ids": pos, "features": probe}, OUT_DIR / "vision_probe.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
