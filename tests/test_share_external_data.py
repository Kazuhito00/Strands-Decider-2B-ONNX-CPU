# -*- coding: utf-8 -*-
"""export/share_external_data.py on two tiny graphs (needs the `onnx` package: the export group)."""
import pathlib

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from conftest import load_script  # noqa: E402


def _save(path: pathlib.Path, tensors: dict) -> None:
    inits = [numpy_helper.from_array(v, name=k) for k, v in tensors.items()]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])]
    nodes = [helper.make_node("Add", ["x", k], ["y_" + k]) for k in tensors if tensors[k].shape == (1,)]
    graph = helper.make_graph(nodes, "g", inputs, [], initializer=inits)
    onnx.save(helper.make_model(graph), str(path), save_as_external_data=True, all_tensors_to_one_file=True,
              location=path.name + ".data", size_threshold=0)


def test_share_repoints_identical_tensors_and_keeps_the_rest(tmp_path):
    mod = load_script("share_external_data", "export/share_external_data.py")
    rng = np.random.default_rng(0)
    w1, w2 = rng.normal(size=1000).astype(np.float32), rng.normal(size=2000).astype(np.float32)
    unique = rng.normal(size=1500).astype(np.float32)
    small = np.array([3.0], dtype=np.float32)
    _save(tmp_path / "torso_L512.onnx", {"w1": w1, "w2": w2})
    _save(tmp_path / "torso_L384.onnx", {"w1": w1, "w2": w2, "unique": unique, "small": small})
    pool_size = (tmp_path / "torso_L512.onnx.data").stat().st_size

    mod.share(tmp_path / "torso_L384.onnx", tmp_path / "torso_L512.onnx")

    model = onnx.load(str(tmp_path / "torso_L384.onnx"))              # follows the external locations
    got = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    np.testing.assert_array_equal(got["w1"], w1)
    np.testing.assert_array_equal(got["w2"], w2)
    np.testing.assert_array_equal(got["unique"], unique)
    np.testing.assert_array_equal(got["small"], small)
    by_name = {t.name: t for t in onnx.load(str(tmp_path / "torso_L384.onnx"), load_external_data=False).graph.initializer}
    loc = {n: {e.key: e.value for e in t.external_data}.get("location") for n, t in by_name.items()}
    assert loc["w1"] == loc["w2"] == "torso_L512.onnx.data"            # shared, not copied
    assert loc["unique"] == "torso_L384.onnx.data" and loc["small"] is None   # a leftover keeps a file; a small one is inline
    assert (tmp_path / "torso_L512.onnx.data").stat().st_size == pool_size    # the pool is untouched
    assert (tmp_path / "torso_L384.onnx.data").stat().st_size == unique.nbytes


def test_a_fully_shared_graph_needs_no_data_file(tmp_path):
    mod = load_script("share_external_data", "export/share_external_data.py")
    w = np.arange(1000, dtype=np.float32)
    _save(tmp_path / "torso_L512.onnx", {"w": w})
    _save(tmp_path / "torso_L1280.onnx", {"w": w})
    mod.share(tmp_path / "torso_L1280.onnx", tmp_path / "torso_L512.onnx")
    assert not (tmp_path / "torso_L1280.onnx.data").exists()
    got = numpy_helper.to_array(onnx.load(str(tmp_path / "torso_L1280.onnx")).graph.initializer[0])
    np.testing.assert_array_equal(got, w)
