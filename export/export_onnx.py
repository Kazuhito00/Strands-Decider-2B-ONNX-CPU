# -*- coding: utf-8 -*-
"""Export the Strands Decider torso (Qwen3.5-2B-Base + merged LoRA) to ONNX: one prefill, hidden states out.

    .venv-ref\\Scripts\\python.exe export/export_onnx.py --seq-len 512

    input_ids [1, L], attention_mask [1, L], position_ids [3, 1, L], read_positions [R],
    image_features [N, 2048], image_slots [N]          ->  hidden [R, 2048]

``read_positions`` are the token positions the pointer head needs: the ``<answer>`` token first, then the last token of
each option's line. The head itself (LayerNorm + two 2048->256 projections) runs in numpy, fp32, in the runtime.

The sequence length L is fixed, not dynamic: the linear-attention layers loop over 64-token chunks in Python, which the
exporter unrolls at the traced length (a graph traced at 253 tokens silently ignores everything past chunk 4 on longer
inputs). Callers right-pad up to L; every read position precedes the padding and the model is causal, so the padding
cannot change the result (verify/check_padding.py). Export one graph per length you need.

Also writes models/onnx/head.npz and graph_meta.json. torch + transformers + onnx + onnxscript (.venv-ref).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export.merge_lora import adapter_pairs, head_arrays  # noqa: E402

BASE = ROOT / "models" / "base"
RELEASE = ROOT / "models" / "release"
CHUNK = 64


class TorsoGraph(nn.Module):
    """Prefill once; return the hidden states at the positions the head reads.

    ``image_features`` / ``image_slots`` scatter the vision encoder's output into the token embeddings (a no-op when
    both are empty), (the same layout as the other Qwen3.5 decision exports).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids, read_positions, image_features, image_slots):
        embeds = self.model.get_input_embeddings()(input_ids)
        embeds = embeds.index_copy(1, image_slots, image_features.unsqueeze(0).to(embeds.dtype))
        out = self.model(inputs_embeds=embeds, attention_mask=attention_mask, position_ids=position_ids,
                         use_cache=False, return_dict=True)
        return out.last_hidden_state[0].index_select(0, read_positions)


def load_merged_model(dtype=torch.float32):
    """Qwen3_5TextModel (the decoder; no vision tower, no lm_head) with the LoRA folded in.

    Built directly and filled one weight at a time from the base safetensors, adding the adapter's B @ A as each weight
    goes by, so the peak is the model itself (~7.6 GB in fp32) and not model + checkpoint + deltas.
    """
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    config = AutoConfig.from_pretrained(str(BASE))
    torch.set_default_dtype(torch.float32)
    model = Qwen3_5TextModel(config.text_config).eval()
    params = dict(model.named_parameters())
    pairs, scale = adapter_pairs(RELEASE / "lora")
    seen, merged = set(), 0
    prefix = "model.language_model."
    with torch.no_grad(), safe_open(str(BASE / "model.safetensors-00001-of-00001.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            if not key.startswith(prefix):
                continue                                   # vision tower, lm_head: not part of the torso graph
            name = key[len(prefix):]
            if name not in params:
                raise SystemExit("base weight %s has no parameter in the decoder" % name)
            w = handle.get_tensor(key).float()
            if name in pairs:
                a, b = pairs[name]
                w = w + (b @ a) * scale
                merged += 1
            params[name].copy_(w)
            seen.add(name)
    missing = [n for n in params if n not in seen]
    if missing:
        raise SystemExit("decoder parameters never loaded: %s" % missing[:5])
    if merged != len(pairs):
        raise SystemExit("merged %d of %d adapter modules" % (merged, len(pairs)))
    print("loaded %d tensors, merged %d LoRA modules" % (len(seen), merged), flush=True)
    return config, model.to(dtype)


def padded_inputs(ids, length: int, pad_id: int):
    real = len(ids)
    if real > length:
        raise ValueError("%d tokens exceeds the graph's fixed length %d" % (real, length))
    input_ids = torch.tensor([list(ids) + [pad_id] * (length - real)], dtype=torch.long)
    mask = torch.tensor([[1] * real + [0] * (length - real)], dtype=torch.long)
    positions = torch.arange(length, dtype=torch.long)[None, None, :].repeat(3, 1, 1)
    return input_ids, mask, positions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="models/onnx")
    ap.add_argument("--seq-len", type=int, default=512, help="fixed prompt length (a multiple of 64)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--no-graph", action="store_true", help="only write head.npz and graph_meta.json")
    args = ap.parse_args()
    if args.seq_len % CHUNK:
        raise SystemExit("--seq-len must be a multiple of %d" % CHUNK)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "head.npz", **head_arrays(RELEASE / "head.safetensors"))
    cfg = json.loads((RELEASE / "strands_decider_config.json").read_text(encoding="utf-8")) \
        if (RELEASE / "strands_decider_config.json").exists() else \
        json.loads((RELEASE / "hobson_config.json").read_text(encoding="utf-8"))
    meta = {"graph": None, "seq_len": args.seq_len, "dtype": args.dtype, "chunk": CHUNK,
            "pad_token_id": 248044, "max_length": cfg["max_length"], "temperature": cfg["temperature"],
            "temperature_by_kind": cfg["temperature_by_kind"], "ordinal_smoothing": cfg["ordinal_smoothing"],
            "release": json.loads((RELEASE / "provenance.json").read_text(encoding="utf-8"))["name"]}
    if args.no_graph:
        (out_dir / "graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return 0

    started = time.time()
    config, model = load_merged_model(getattr(torch, args.dtype))
    print("model ready in %.0f s" % (time.time() - started), flush=True)
    graph = TorsoGraph(model).eval()
    hidden = config.text_config.hidden_size
    pad = meta["pad_token_id"]
    feats0 = torch.zeros(0, hidden, dtype=next(graph.parameters()).dtype)
    slots0 = torch.zeros(0, dtype=torch.long)

    # 1. the merged torso (in torch) against the engine's own hidden states, before anything is exported
    golden = json.loads((ROOT / "artifacts" / "golden.json").read_text(encoding="utf-8"))
    hid = np.load(ROOT / "artifacts" / "golden_hidden.npz")
    worst = 0.0
    for case, qid in (("choice-basic", "team"), ("multi", "angry"), ("non-ascii", "team")):
        q = golden["cases"][case]["questions"][qid]
        ids = q["state_ids"] + q["question_ids"]
        input_ids, mask, positions = padded_inputs(ids, args.seq_len, pad)
        with torch.no_grad():
            got = graph(input_ids, mask, positions, torch.tensor(q["read_positions"]), feats0, slots0).float().numpy()
        diff = float(np.abs(got - hid["%s__%s" % (case, qid)]).max())
        scale = float(np.abs(hid["%s__%s" % (case, qid)]).max())
        worst = max(worst, diff)
        print("  torch torso vs engine  %-14s max|dh| %.2e   (scale %.1f)" % (case + "/" + qid, diff, scale), flush=True)
    if worst > 5e-2:
        raise SystemExit("the merged torso does not reproduce the engine's hidden states; not exporting")

    # 2. export
    q = golden["cases"]["choice-basic"]["questions"]["team"]
    ids = q["state_ids"] + q["question_ids"]
    input_ids, mask, positions = padded_inputs(ids, args.seq_len, pad)
    reads = torch.tensor(q["read_positions"], dtype=torch.long)
    feats = torch.zeros(4, hidden, dtype=next(graph.parameters()).dtype)
    slots = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    target = out_dir / ("torso_L%d.onnx" % args.seq_len)
    started = time.time()
    torch.onnx.export(
        graph, (input_ids, mask, positions, reads, feats, slots), str(target),
        input_names=["input_ids", "attention_mask", "position_ids", "read_positions", "image_features", "image_slots"],
        output_names=["hidden"],
        dynamic_shapes={"input_ids": {}, "attention_mask": {}, "position_ids": {}, "read_positions": {0: "reads"},
                        "image_features": {0: "visual"}, "image_slots": {0: "visual"}},
        opset_version=18, dynamo=True, external_data=True, optimize=False)
    print("exported in %.0f s -> %s" % (time.time() - started, target), flush=True)
    meta["graph"] = target.name
    (out_dir / ("graph_meta_L%d.json" % args.seq_len)).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
