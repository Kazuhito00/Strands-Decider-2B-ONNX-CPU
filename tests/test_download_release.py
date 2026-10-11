# -*- coding: utf-8 -*-
"""download_model.py (repository choice, file selection, a full round trip from a local release) and release/make_release.py."""
import argparse
import functools
import hashlib
import http.server
import json
import pathlib
import threading

import pytest

import download_model as dl
from conftest import load_script


def _args(**kw):
    base = dict(repo=None, tag="latest", base_url=None, include=None, exclude=None, seq_len=None)
    base.update(kw)
    return argparse.Namespace(**base)


# ------------------------------------------------------------------------------------------------ repository

def test_repository_priority_argument_then_environment_then_default(monkeypatch):
    monkeypatch.delenv("STRANDS_ONNX_REPO", raising=False)
    assert dl.base_url(_args()) == "https://github.com/Kazuhito00/Strands-Decider-2B-ONNX-CPU/releases/latest/download"
    monkeypatch.setenv("STRANDS_ONNX_REPO", "someone/fork")
    assert dl.base_url(_args()) == "https://github.com/someone/fork/releases/latest/download"
    assert dl.base_url(_args(repo="other/repo", tag="v0.0.0")) == "https://github.com/other/repo/releases/download/v0.0.0"
    assert dl.base_url(_args(base_url="http://127.0.0.1:8000/")) == "http://127.0.0.1:8000"
    assert dl.DEFAULT_REPO == "Kazuhito00/Strands-Decider-2B-ONNX-CPU"


# ------------------------------------------------------------------------------------------------ selection

def _entries():
    def e(path, role="other", requires=None):
        out = {"path": path, "asset": path.replace("/", "__"), "size": 1, "sha256": "0", "parts": None, "role": role}
        if requires:
            out["requires"] = requires
        return out
    return [e("onnx/torso_L384.onnx", "graph", ["onnx/torso_L512.onnx.data"]),
            e("onnx/torso_L512.onnx", "graph", ["onnx/torso_L512.onnx.data"]),
            e("onnx/torso_L1280.onnx", "graph", ["onnx/torso_L512.onnx.data"]),
            e("onnx/vision_encoder.onnx", "graph", ["onnx/vision_encoder.onnx.data"]),
            e("onnx/torso_L512.onnx.data", "data"), e("onnx/vision_encoder.onnx.data", "data"),
            e("onnx/head.npz"), e("onnx/tokenizer.json")]


def _paths(**kw):
    return [f["path"] for f in dl.select(_entries(), _args(**kw))]


def test_default_selection_is_everything():
    assert len(_paths()) == 8


def test_seq_len_keeps_the_shared_weights():
    got = _paths(seq_len=[384])
    assert "onnx/torso_L384.onnx" in got and "onnx/torso_L512.onnx.data" in got
    assert "onnx/torso_L512.onnx" not in got and "onnx/torso_L1280.onnx" not in got
    assert "onnx/vision_encoder.onnx.data" in got


def test_excluding_a_graph_that_owns_the_name_never_drops_weights_still_needed():
    got = _paths(exclude=["torso_L512*"])                       # the glob matches the graph AND the data file's name
    assert "onnx/torso_L512.onnx" not in got
    assert "onnx/torso_L384.onnx" in got and "onnx/torso_L512.onnx.data" in got


def test_text_only_drops_the_vision_encoder_and_its_weights():
    got = _paths(exclude=["vision_encoder*"])
    assert not [p for p in got if "vision" in p] and "onnx/head.npz" in got


def test_nothing_needs_weights_when_no_graph_is_selected():
    got = _paths(seq_len=[4096])
    assert "onnx/torso_L512.onnx.data" not in got and "onnx/head.npz" in got


def test_include_globs():
    got = _paths(include=["*.json", "head*"])
    assert sorted(got) == ["onnx/head.npz", "onnx/tokenizer.json"]


def test_missing_required_file_is_reported():
    entries = [e for e in _entries() if e["path"] != "onnx/torso_L512.onnx.data"]
    with pytest.raises(SystemExit, match="torso_L512.onnx.data"):
        dl.select(entries, _args())


# ------------------------------------------------------------------------------------------------ release round trip

def _fake_models(root: pathlib.Path):
    onnx = root / "onnx"
    onnx.mkdir(parents=True)
    pool = bytes(range(256)) * 40                                          # 10240 bytes: split into parts in the test
    (onnx / "torso_L512.onnx.data").write_bytes(pool)
    (onnx / "vision_encoder.onnx.data").write_bytes(b"v" * 300)
    for bucket in (384, 512):                                              # both graphs read the SAME data file
        (onnx / ("torso_L%d.onnx" % bucket)).write_bytes(b"\x0a\x14graph" + b"torso_L512.onnx.data" + b"\x00" * 10)
        (onnx / ("graph_meta_L%d.json" % bucket)).write_text("{}", encoding="utf-8")
    (onnx / "vision_encoder.onnx").write_bytes(b"\x0avision_encoder.onnx.data\x00")
    for name in ("head.npz", "tokenizer.json", "preprocessor_config.json"):
        (onnx / name).write_bytes(name.encode())
    (root / "release").mkdir()
    (root / "release" / "provenance.json").write_text("{}", encoding="utf-8")
    (root / "release" / "LICENSE.md").write_text("apache", encoding="utf-8")
    (root / "base").mkdir()
    (root / "base" / "LICENSE").write_text("qwen", encoding="utf-8")
    return pool


def test_make_release_stages_parts_and_manifest(tmp_path):
    mk = load_script("make_release", "release/make_release.py")
    pool = _fake_models(tmp_path / "models")
    manifest = mk.stage(tmp_path / "models", tmp_path / "dist", [384, 512], 4000, "t", "v0.0.0")
    by_path = {f["path"]: f for f in manifest["files"]}
    assert manifest["buckets"] == [384, 512] and manifest["format"] == 1
    data = by_path["onnx/torso_L512.onnx.data"]
    assert [p["size"] for p in data["parts"]] == [4000, 4000, 2240] and data["sha256"] == hashlib.sha256(pool).hexdigest()
    assert by_path["onnx/torso_L384.onnx"]["requires"] == ["onnx/torso_L512.onnx.data"]
    assert by_path["onnx/vision_encoder.onnx"]["requires"] == ["onnx/vision_encoder.onnx.data"]
    assert "onnx/LICENSE-QWEN" in by_path and "onnx/LICENSE-STRANDS-DECIDER.md" in by_path
    assert (tmp_path / "dist" / "onnx__torso_L512.onnx.data.part002").stat().st_size == 2240


def test_make_release_refuses_a_second_copy_of_the_weights(tmp_path):
    mk = load_script("make_release", "release/make_release.py")
    _fake_models(tmp_path / "models")
    own = tmp_path / "models" / "onnx" / "torso_L384.onnx"
    own.write_bytes(b"graph" + b"torso_L384.onnx.data")
    (tmp_path / "models" / "onnx" / "torso_L384.onnx.data").write_bytes(b"x")
    with pytest.raises(SystemExit, match="more than one export"):
        mk.plan(tmp_path / "models", [384, 512])
    assert mk.plan(tmp_path / "models", [384, 512], allow_own_data=True)


def test_download_round_trip_from_a_local_release(tmp_path, monkeypatch):
    mk = load_script("make_release", "release/make_release.py")
    pool = _fake_models(tmp_path / "models")
    mk.stage(tmp_path / "models", tmp_path / "dist", [384, 512], 4000, "t", "v0.0.0")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path / "dist"))
    handler.log_message = lambda *a, **k: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = "http://127.0.0.1:%d" % server.server_address[1]
        out = tmp_path / "got"
        assert dl.main.__module__ == "download_model"
        monkeypatch.setattr("sys.argv", ["download_model.py", "--base-url", url, "--dest", str(out), "--seq-len", "384"])
        assert dl.main() == 0
        assert (out / "onnx" / "torso_L512.onnx.data").read_bytes() == pool               # four parts joined, sha256 checked
        assert (out / "onnx" / "torso_L384.onnx").exists() and not (out / "onnx" / "torso_L512.onnx").exists()
        assert not list(out.rglob("*.part")) and not list(out.rglob("*.dl"))
        monkeypatch.setattr("sys.argv", ["download_model.py", "--base-url", url, "--dest", str(out)])
        assert dl.main() == 0                                                              # a second run adds the rest
        assert (out / "onnx" / "torso_L512.onnx").exists()
        manifest = json.loads((tmp_path / "dist" / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["files"]) == len([p for p in out.rglob("*") if p.is_file()])
    finally:
        server.shutdown()
