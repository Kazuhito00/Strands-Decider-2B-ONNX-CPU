# -*- coding: utf-8 -*-
"""Make every sequence-length bucket read ONE shared weights file.

The decoder's weights do not depend on the traced length L: only the graph (the unrolled linear-attention chunk loop) does.
Exporting L=384, 512 and 1280 separately would write 7.5 GB of identical bytes three times. ONNX external data is just
(location, offset, length), so a graph can point at bytes that another graph's data file already holds. Nothing numeric
changes: a tensor is redirected only when its bytes are identical (sha256 and length) to one in the pool graph.

    python export/share_external_data.py models/onnx/torso_L384.onnx
        a fresh export: tensors that the pool graph (torso_L512.onnx) keeps in its data file are repointed to it,
        small constants go inline (ORT cannot shape-infer through an external reference), anything left over stays in
        torso_L384.onnx.data (nothing, normally; the file is then deleted)
    python export/share_external_data.py models/onnx/torso_L1280.onnx --pool-graph torso_L512.onnx

The pool is the data file of ``torso_L512.onnx`` (``torso_L512.onnx.data``): the first bucket that was exported. This is how
the graphs of the release (L=384, 512, 1280) are laid out. Do not delete the pool file. Needs the `onnx` package (the export
group); the runtime does not.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

MIN_SIZE = 1024          # smaller tensors stay inside the graph: shape inference reads them while loading


def _external(tensor) -> dict:
    return {e.key: e.value for e in tensor.external_data}


def _set_external(tensor, onnx, location: str, offset: int, length: int) -> None:
    del tensor.external_data[:]
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for key, value in (("location", location), ("offset", str(offset)), ("length", str(length))):
        entry = tensor.external_data.add()
        entry.key, entry.value = key, value


def _read(directory: pathlib.Path, info: dict) -> bytes:
    with open(directory / info["location"], "rb") as handle:
        handle.seek(int(info.get("offset", 0)))
        return handle.read(int(info["length"]))


def _tensors(model):
    from onnx.external_data_helper import _get_all_tensors

    return list(_get_all_tensors(model))


def build_pool_index(directory: pathlib.Path, pool_graph: pathlib.Path) -> dict:
    """(sha256, length) -> (location, offset) for every tensor the pool graph keeps in an external file."""
    import onnx

    index = {}
    model = onnx.load(str(pool_graph), load_external_data=False)
    for tensor in _tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        info = _external(tensor)
        key = (hashlib.sha256(_read(directory, info)).hexdigest(), int(info["length"]))
        index.setdefault(key, (info["location"], int(info.get("offset", 0))))
    return index


def share(graph: pathlib.Path, pool_graph: pathlib.Path) -> None:
    import onnx

    directory = graph.parent
    index = build_pool_index(directory, pool_graph)
    if not index:
        raise SystemExit("%s keeps no external data: nothing to share with" % pool_graph.name)
    print("pool: %d tensors" % len(index), flush=True)

    model = onnx.load(str(graph), load_external_data=False)
    inline = shared = kept = 0
    stash = []                                   # (tensor, bytes) not found in the pool
    for tensor in _tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        info = _external(tensor)
        data = _read(directory, info)
        if len(data) < MIN_SIZE:
            tensor.raw_data = data
            tensor.data_location = onnx.TensorProto.DEFAULT
            del tensor.external_data[:]
            inline += 1
            continue
        hit = index.get((hashlib.sha256(data).hexdigest(), len(data)))
        if hit:
            _set_external(tensor, onnx, hit[0], hit[1], len(data))
            shared += 1
        else:
            stash.append((tensor, data))
            kept += 1
    print("  %d shared with the pool, %d small kept inline, %d unique (%.1f MB)"
          % (shared, inline, kept, sum(len(d) for _, d in stash) / 1024 ** 2), flush=True)

    own = graph.name + ".data"
    if graph.resolve() == pool_graph.resolve():
        raise SystemExit("the graph is the pool graph")
    old = directory / own
    if stash:
        # unique bytes go to a temp name first: the source file may be the one being read
        with open(directory / (own + ".new"), "wb") as handle:
            offset = 0
            for tensor, data in stash:
                handle.write(data)
                _set_external(tensor, onnx, own, offset, len(data))
                offset += len(data)
    onnx.save(model, str(graph), save_as_external_data=False)
    if old.exists():
        old.unlink()
    if stash:
        (directory / (own + ".new")).rename(directory / own)
    print("%s: shares %d tensors with %s%s" % (graph.name, shared, pool_graph.name,
                                              "; own data %s" % own if stash else "; no data file of its own"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph", type=pathlib.Path)
    parser.add_argument("--pool-graph", default="torso_L512.onnx",
                        help="the graph whose data file is the pool, in the same folder (default: torso_L512.onnx)")
    args = parser.parse_args()
    graph = args.graph.resolve()
    pool = graph.parent / args.pool_graph
    for path in (graph, pool):
        if not path.exists():
            raise SystemExit("no such graph: %s" % path)
    share(graph, pool)
    return 0


if __name__ == "__main__":
    sys.exit(main())
