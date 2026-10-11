# -*- coding: utf-8 -*-
"""Fold the Strands LoRA adapter into Qwen3.5-2B-Base's weights, and read the pointer head. torch + safetensors.

    W' = W + (lora_alpha / r) * B @ A          (plain LoRA: no rslora, no DoRA, no bias; checked below)

The adapter was trained on the BARE text decoder, so its keys read ``base_model.model.layers.N.<module>.lora_A.weight``;
inside Qwen3_5Model the same decoder sits under ``language_model.layers.N``. The file holds 372 tensors = 186 modules x (A, B):
18 linear-attention layers x 8 modules (in_proj_qkv/z/a/b, out_proj, gate/up/down) + 6 full-attention layers x 7 modules
(q/k/v/o, gate/up/down). ``merge`` fails on any key that finds no weight and on any shape mismatch.
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict

import torch
from safetensors import safe_open

PREFIX = "base_model.model."


def adapter_pairs(adapter_dir: pathlib.Path):
    """({weight name: (A, B)}, scale). Weight names are the bare decoder's (``layers.N.mlp.up_proj.weight``).

    The deltas themselves are not formed here: B @ A is as large as the weight it patches, so the loader adds them one
    weight at a time while it streams the base checkpoint (a whole-model copy would not fit in 32 GB next to the model).
    """
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    for flag in ("use_rslora", "use_dora"):
        if cfg.get(flag):
            raise SystemExit("adapter uses %s, which this merge does not implement" % flag)
    if cfg.get("bias", "none") != "none" or cfg.get("fan_in_fan_out"):
        raise SystemExit("adapter has bias / fan_in_fan_out, which this merge does not implement")
    scale = float(cfg["lora_alpha"]) / float(cfg["r"])
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            if not key.startswith(PREFIX):
                raise SystemExit("unexpected adapter key %s" % key)
            module, _, rest = key[len(PREFIX):].rpartition(".lora_")       # layers.0.mlp.up_proj | A.weight
            pairs.setdefault(module + ".weight", {})[rest.split(".")[0]] = handle.get_tensor(key).float()
    return {name: (ab["A"], ab["B"]) for name, ab in pairs.items()}, scale


def head_arrays(path: pathlib.Path) -> Dict[str, "numpy.ndarray"]:  # noqa: F821
    """The pointer head's tensors as fp32 numpy arrays (the runtime reads them from head.npz)."""
    out = {}
    with safe_open(str(path), framework="np") as handle:
        for key in handle.keys():
            out[key] = handle.get_tensor(key).astype("float32")
    return out
