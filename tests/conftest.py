# -*- coding: utf-8 -*-
"""Shared helpers. None of the tests needs the model: those that read models/onnx/tokenizer.json skip when it is absent."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                       # download_model.py, demo_inference_*.py and strands_decide from the checkout


def load_script(name: str, relative: str):
    """Import a script that is not a package (release/make_release.py)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def tiny_tokenizer_json(tmp_path_factory) -> str:
    """A small byte-level BPE in the layout Qwen's tokenizer.json declares (NFC, Split + ByteLevel, BPE, added tokens)."""
    from strands_decide.tokenizer import BYTE_TO_CHAR

    vocab = {BYTE_TO_CHAR[b]: b for b in range(256)}
    merges = [["t", "h"], ["th", "e"], ["a", "n"], ["an", "d"], ["Ġ", "the"]]
    for left, right in merges:
        vocab.setdefault(left + right, len(vocab))
    added = [{"id": len(vocab) + i, "content": c, "special": True, "normalized": False}
             for i, c in enumerate(["<|endoftext|>", "<|image_pad|>"])]
    spec = {
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|\\s+|[^\\s\\p{L}\\p{N}]+"}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False}]},
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
    }
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return str(path)
