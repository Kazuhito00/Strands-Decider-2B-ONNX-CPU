# -*- coding: utf-8 -*-
"""Put the inputs of the export under models/ (no weights are published in a runnable form for this model).

StrandsAgents/strands-decider-2B-hobson-v21 is NOT a full checkpoint. It is
  * a LoRA adapter (67 MB) for Qwen/Qwen3.5-2B-Base,
  * the pointer head (4 MB, fp32),
  * the tokenizer, the temperatures and the training configuration.
The base weights (4.5 GB) come from Qwen/Qwen3.5-2B-Base, at the revision the adapter was trained on.

    python export/prepare_model.py --list
    python export/prepare_model.py                  # download + verify
    python export/prepare_model.py --release v19    # the older release (HF repo ...-v19)

Then `export/export_onnx.py` merges the adapter into the base weights and writes the ONNX graph.
No git, no git-lfs: plain HTTPS from Hugging Face's resolve endpoint (standard library only), resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export.download import fetch_files  # noqa: E402

BASE_REPO = "Qwen/Qwen3.5-2B-Base"
#: provenance.json says the adapter was trained on "Hub main at training time"; main is still this commit.
BASE_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
RELEASES = {
    "v21": "StrandsAgents/strands-decider-2B-hobson-v21",     # the current release
    "v19": "StrandsAgents/strands-decider-2B-hobson-v19",
}
BASE_FILES = ["model.safetensors-00001-of-00001.safetensors", "model.safetensors.index.json", "config.json", "tokenizer.json", "tokenizer_config.json",
              "preprocessor_config.json", "vocab.json", "merges.txt", "LICENSE"]
RELEASE_FILES = ["head.safetensors", "lora/adapter_model.safetensors", "lora/adapter_config.json", "provenance.json",
                 "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE.md", "README.md",
                 "train_config.json"]
CONFIG_NAMES = ["strands_decider_config.json", "hobson_config.json"]     # v21 uses the first, v19 the second


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 24)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def lfs_oid(repo: str, revision: str, path: str) -> str:
    url = "https://huggingface.co/api/models/%s/tree/%s?recursive=true" % (repo, revision)
    request = urllib.request.Request(url, headers={"User-Agent": "strands-decide-onnx/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        for item in json.load(response):
            if item["path"] == path and item.get("lfs"):
                return item["lfs"]["oid"]
    raise SystemExit("%s@%s has no LFS file %s" % (repo, revision[:8], path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--release", default="v21", choices=sorted(RELEASES))
    args = parser.parse_args()
    repo = RELEASES[args.release]
    base_dir = ROOT / "models" / "base"
    rel_dir = ROOT / "models" / "release"

    if args.list:
        print("base     %s @ %s   (%s)" % (BASE_REPO, BASE_REVISION[:8], ", ".join(BASE_FILES[:1])))
        print("release  %s" % repo)
        for name in RELEASE_FILES + CONFIG_NAMES[:1 if args.release == "v21" else 2]:
            print("   ", name)
        print("\n~4.6 GB; lands under models/base and models/release")
        return 0

    print("release: %s" % repo)
    optional = [CONFIG_NAMES[1] if args.release == "v21" else CONFIG_NAMES[0]]
    fetch_files(repo, RELEASE_FILES + CONFIG_NAMES, rel_dir, optional=optional)
    print("base: %s @ %s" % (BASE_REPO, BASE_REVISION[:8]))
    fetch_files(BASE_REPO, BASE_FILES, base_dir, revision=BASE_REVISION)

    print("checking ...")
    prov = json.loads((rel_dir / "provenance.json").read_text(encoding="utf-8"))
    checks = [("adapter", rel_dir / "lora" / "adapter_model.safetensors", prov["adapter_sha256"]),
              ("head", rel_dir / "head.safetensors", prov["head_sha256"]),
              ("base weights", base_dir / BASE_FILES[0], lfs_oid(BASE_REPO, BASE_REVISION, BASE_FILES[0]))]
    bad = 0
    for name, path, want in checks:
        got = sha256_of(path)
        print("  %-14s %s" % (name, "ok" if got == want else "MISMATCH (got %s)" % got[:12]))
        bad += got != want
    if prov.get("base_model_revision") != BASE_REVISION:
        print("  note: the release says it was trained on base revision %s" % prov.get("base_model_revision"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
