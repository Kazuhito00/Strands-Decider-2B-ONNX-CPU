# -*- coding: utf-8 -*-
"""Stage a GitHub Release from models/: split what is too big, hash everything, write manifest.json.

    python release/make_release.py --dry-run                 # list the assets (names, sizes, parts); hashes nothing, writes nothing
    python release/make_release.py                           # models/ -> dist/release/
    python release/make_release.py --version v0.0.0 --max-part-gib 1.9
    python release/make_release.py --buckets 384 512 1280    # which torso graphs go in (default: those)
    python release/make_release.py --models path/to/models   # stage from another checkout's models/

GitHub allows at most 2 GiB per release asset (and 1000 assets per release), so a file larger than ``--max-part-gib`` is cut into
byte parts (``<name>.part000``, ...) that download_model.py joins again and checks against the whole file's sha256. Asset names are
flat, so ``onnx/torso_L384.onnx`` is published as ``onnx__torso_L384.onnx``; manifest.json keeps the real path.
Nothing is uploaded here. Upload dist/release/* yourself, for example:

    gh release create v0.0.0 dist/release/* --title "v0.0.0" --notes-file release/notes.md

What goes in is an allowlist, not "everything under models/onnx" (that folder also holds exports nobody should download):

* the torso graphs of the chosen buckets (``torso_L<N>.onnx``) and the data file(s) they read. The graphs share ONE weights file
  (``torso_L512.onnx.data``, 7.5 GB: ``torso_L384`` and ``torso_L1280`` point into it, byte-identical tensors), found by reading the
  external-data locations out of each graph. Graphs that read data files of more than one export are refused unless
  ``--allow-own-data`` is given: every such file would add 7.5 GB.
* ``vision_encoder.onnx`` (+ ``.data``), ``head.npz``, ``graph_meta_L<N>.json`` of the chosen buckets, ``tokenizer.json``,
  ``preprocessor_config.json``, ``provenance.json`` and the two licence texts (``LICENSE-STRANDS-DECIDER.md`` from models/release,
  ``LICENSE-QWEN`` from models/base).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHUNK = 1 << 20
DEFAULT_BUCKETS = (384, 512, 1280)
#: (source relative to models/, path inside models/onnx/ it is published under)
EXTRA_FILES = [
    ("onnx/head.npz", "head.npz"),
    ("onnx/tokenizer.json", "tokenizer.json"),
    ("onnx/preprocessor_config.json", "preprocessor_config.json"),
    ("release/provenance.json", "provenance.json"),
    ("release/LICENSE.md", "LICENSE-STRANDS-DECIDER.md"),
    ("base/LICENSE", "LICENSE-QWEN"),
]
#: the data file names this repository uses (a protobuf length byte sits right before each name, so no generic "word" pattern)
LOCATION = re.compile(rb"(?:torso_L\d+|vision_encoder)\.onnx\.data(?:_\d+)?")


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def external_locations(graph: pathlib.Path) -> list:
    """File names an ONNX graph reads its external data from (the ``location`` strings stored in the protobuf)."""
    found = set()
    with open(graph, "rb") as handle:
        tail = b""
        for block in iter(lambda: handle.read(CHUNK), b""):
            data = tail + block
            found.update(m.group(0).decode() for m in LOCATION.finditer(data))
            tail = data[-96:]
    return sorted(found)


def plan(models: pathlib.Path, buckets, allow_own_data: bool = False):
    """[(path under models/, source file, role, requires)] for everything the release holds. Raises SystemExit on a problem."""
    onnx_dir = models / "onnx"
    items, torso_data, vision_data = [], [], []

    def need(path: pathlib.Path) -> pathlib.Path:
        if not path.exists():
            raise SystemExit("missing %s (run the export or download_model.py first)" % path)
        return path

    for bucket in sorted(buckets):
        graph = need(onnx_dir / ("torso_L%d.onnx" % bucket))
        locs = external_locations(graph)
        torso_data += [loc for loc in locs if loc not in torso_data]
        items.append(("onnx/" + graph.name, graph, "graph", ["onnx/" + loc for loc in locs]))
        meta = need(onnx_dir / ("graph_meta_L%d.json" % bucket))
        items.append(("onnx/" + meta.name, meta, "other", []))
    if len({loc.rsplit(".data", 1)[0] for loc in torso_data}) > 1 and not allow_own_data:
        raise SystemExit("the chosen graphs read data files of more than one export (%s): that is a full 7.5 GB copy per extra "
                         "file. Share the weights first, drop the bucket, or pass --allow-own-data." % ", ".join(torso_data))
    vision = need(onnx_dir / "vision_encoder.onnx")
    vision_data = external_locations(vision)
    items.append(("onnx/vision_encoder.onnx", vision, "graph", ["onnx/" + loc for loc in vision_data]))
    for loc in torso_data + vision_data:
        items.append(("onnx/" + loc, need(onnx_dir / loc), "data", []))
    for rel, published in EXTRA_FILES:
        items.append(("onnx/" + published, need(models / rel), "other", []))
    return items


def link_or_copy(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)  # no second copy of a multi-GB file on the same drive
    except OSError:
        shutil.copy2(src, dst)


def split(src: pathlib.Path, out_dir: pathlib.Path, stem: str, part_bytes: int) -> list:
    parts = []
    with open(src, "rb") as handle:
        index = 0
        while True:
            name = "%s.part%03d" % (stem, index)
            digest, written = hashlib.sha256(), 0
            with open(out_dir / name, "wb") as out:
                while written < part_bytes:
                    block = handle.read(min(CHUNK, part_bytes - written))
                    if not block:
                        break
                    out.write(block)
                    digest.update(block)
                    written += len(block)
            if written == 0:
                (out_dir / name).unlink()
                break
            parts.append({"asset": name, "size": written, "sha256": digest.hexdigest()})
            index += 1
            if written < part_bytes:
                break
    return parts


def stage(models: pathlib.Path, out: pathlib.Path, buckets, limit: int, name: str, version: str,
          allow_own_data: bool = False) -> dict:
    """Write the assets and manifest.json into ``out`` and return the manifest."""
    items = plan(models, buckets, allow_own_data)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for rel, src, role, requires in items:
        asset = rel.replace("/", "__")
        size = src.stat().st_size
        entry = {"path": rel, "asset": asset, "size": size, "sha256": sha256_of(src), "parts": None, "role": role}
        if requires:
            entry["requires"] = requires
        if size > limit:
            entry["parts"] = split(src, out, asset, limit)
            print("split  %-44s %9.1f MB -> %d parts" % (rel, size / 1e6, len(entry["parts"])))
        else:
            link_or_copy(src, out / asset)
            print("asset  %-44s %9.1f MB" % (rel, size / 1e6))
        entries.append(entry)
    manifest = {"format": 1, "name": name, "version": version, "buckets": sorted(buckets), "files": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "SHA256SUMS").write_text("".join("%s  %s\n" % (e["sha256"], e["path"]) for e in entries), encoding="utf-8")
    too_big = [a.name for a in out.iterdir() if a.stat().st_size >= (2 << 30)]
    if too_big:
        raise SystemExit("asset(s) of 2 GiB or more: %s" % ", ".join(too_big))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=str(ROOT / "models"))
    parser.add_argument("--out", default=str(ROOT / "dist" / "release"))
    parser.add_argument("--name", default="strands-decider-2b-onnx")
    parser.add_argument("--version", default="v0.0.0")
    parser.add_argument("--buckets", type=int, nargs="+", default=list(DEFAULT_BUCKETS),
                        help="sequence-length buckets whose torso graphs go in (default: %s)" % " ".join(map(str, DEFAULT_BUCKETS)))
    parser.add_argument("--max-part-gib", type=float, default=1.9,
                        help="assets above this are split (GitHub's limit is 2 GiB)")
    parser.add_argument("--allow-own-data", action="store_true",
                        help="accept a graph that keeps its own copy of the weights")
    parser.add_argument("--dry-run", action="store_true", help="list the assets with sizes; no hashing, nothing written")
    args = parser.parse_args()

    models, out = pathlib.Path(args.models), pathlib.Path(args.out)
    limit = int(args.max_part_gib * (1 << 30))
    if args.dry_run:
        total = assets = 0
        for rel, src, role, requires in plan(models, args.buckets, args.allow_own_data):
            size = src.stat().st_size
            parts = -(-size // limit) if size > limit else 0
            total += size
            assets += parts or 1
            print("%-12s %-40s %14d bytes (%8.1f MB)%s%s" % (
                role, rel, size, size / 1e6, "  -> %d parts" % parts if parts else "",
                "  requires %s" % ", ".join(requires) if requires else ""))
        print("total %.2f GB in %d assets (+ manifest.json, SHA256SUMS)" % (total / 1e9, assets))
        return 0
    manifest = stage(models, out, args.buckets, limit, args.name, args.version, args.allow_own_data)
    print("buckets in the manifest: %s" % ", ".join(map(str, manifest["buckets"])))
    print("wrote %s: %d files, %d assets" % (out, len(manifest["files"]), len(list(out.iterdir()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
