# -*- coding: utf-8 -*-
"""Execution-provider choice: --device / --gpu / --require-gpu, and the session never silently pretends to be on a GPU."""
import argparse

import pytest

from strands_decide import add_device_args, providers_from_args, select_providers
from strands_decide import runtime

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"


def _parse(*argv):
    parser = argparse.ArgumentParser()
    add_device_args(parser)
    return parser.parse_args(list(argv))


def test_select_providers():
    assert select_providers("cpu", [CUDA, CPU]) == [CPU]
    assert select_providers("cuda", [CPU]) == [CUDA, CPU]                       # asked for explicitly, even if the build lacks it
    assert select_providers("auto", ["DmlExecutionProvider", CPU]) == ["DmlExecutionProvider", CPU]
    assert select_providers("auto", [CUDA, "DmlExecutionProvider", CPU]) == [CUDA, "DmlExecutionProvider", CPU]
    assert select_providers("auto", [CPU]) == [CPU]
    with pytest.raises(ValueError):
        select_providers("tpu")


def test_default_is_cpu_only():
    assert providers_from_args(_parse()) == (None, False)


def test_cuda_and_gpu_flags(monkeypatch):
    monkeypatch.setattr(runtime.ort, "get_available_providers", lambda: [CUDA, CPU])
    assert providers_from_args(_parse("--device", "cuda")) == ([CUDA, CPU], False)
    assert providers_from_args(_parse("--gpu")) == ([CUDA, CPU], False)
    assert providers_from_args(_parse("--require-gpu")) == ([CUDA, CPU], True)


def test_no_gpu_build_warns_or_fails(monkeypatch, capsys):
    monkeypatch.setattr(runtime.ort, "get_available_providers", lambda: [CPU])
    assert providers_from_args(_parse("--gpu"))[0] == [CPU]
    assert "no GPU provider" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="no GPU provider"):
        providers_from_args(_parse("--require-gpu"))


class _Session:
    def __init__(self, providers):
        self._p = providers

    def get_providers(self):
        return self._p


def _fake_ort(monkeypatch, behaviour):
    calls = []

    def factory(path, sess_options=None, providers=None, **kw):
        calls.append(list(providers))
        return behaviour(providers)

    monkeypatch.setattr(runtime.ort, "InferenceSession", factory)
    return calls


def test_gpu_session_is_returned_when_the_provider_holds(monkeypatch):
    calls = _fake_ort(monkeypatch, lambda p: _Session(list(p)))
    s = runtime._create_session("m.onnx", None, [CUDA, CPU])
    assert s.get_providers()[0] == CUDA and calls == [[CUDA, CPU]]


def test_a_dropped_provider_falls_back_loudly(monkeypatch, capsys):
    calls = _fake_ort(monkeypatch, lambda p: _Session([CPU]))                   # ORT dropped CUDA by itself
    s = runtime._create_session("m.onnx", None, [CUDA, CPU])
    assert s.get_providers() == [CPU] and calls[-1] == [CPU]
    assert "using the CPU" in capsys.readouterr().err


def test_out_of_memory_falls_back_with_a_warning(monkeypatch, capsys):
    def behaviour(p):
        if p[0] == CUDA:
            raise RuntimeError("CUDA failure 2: out of memory\n  details")
        return _Session([CPU])

    _fake_ort(monkeypatch, behaviour)
    s = runtime._create_session("m.onnx", None, [CUDA, CPU])
    assert s.get_providers() == [CPU]
    assert "out of memory" in capsys.readouterr().err


def test_require_gpu_raises_instead_of_falling_back(monkeypatch):
    _fake_ort(monkeypatch, lambda p: _Session([CPU]))
    with pytest.raises(RuntimeError, match="could not run m.onnx on the GPU"):
        runtime._create_session("m.onnx", None, [CUDA, CPU], require_gpu=True)

    def boom(p):
        raise RuntimeError("CUDA failure 2: out of memory")

    _fake_ort(monkeypatch, boom)
    with pytest.raises(RuntimeError, match="out of memory"):
        runtime._create_session("m.onnx", None, [CUDA, CPU], require_gpu=True)
    with pytest.raises(RuntimeError, match="no GPU provider was asked"):
        runtime._create_session("m.onnx", None, None, require_gpu=True)
