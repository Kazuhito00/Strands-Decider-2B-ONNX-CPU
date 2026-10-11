# -*- coding: utf-8 -*-
"""Strands Decider 2B typed decisions on onnxruntime + numpy.

One question is one forward pass of the torso graph (Qwen3.5-2B-Base with the LoRA folded in): token ids in, the hidden
states at the positions the pointer head reads out. The head (LayerNorm + two 2048->256 projections) and the temperature
run in numpy, fp32. The graphs have a FIXED length L (see export/export_onnx.py); a prompt is right-padded up to the
smallest L that holds it, which cannot change the result because the model is causal and every read position precedes
the padding.

Text is the default. Images (pictures inside <state>, read by Qwen3.5's own vision tower) need the vision encoder graph
and Pillow; see :mod:`strands_decide.image` (loaded only when images are passed).
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort

from .protocol import (PointerHead, Question, RenderedQuestion, answer_from_probs, option_token_indices, render_question,
                       render_state, softmax)
from .tokenizer import QwenBPE

__all__ = ["DEFAULT_SEQ_LEN", "Decider", "add_device_args", "fit_to_window", "gpu_providers", "providers_from_args",
           "select_providers"]

GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")


def gpu_providers() -> List[str]:
    """GPU execution providers available in this onnxruntime build, CPU last."""
    available = ort.get_available_providers()
    return [p for p in GPU_PROVIDERS if p in available] + ["CPUExecutionProvider"]


def select_providers(device: str = "cpu", available: Optional[Sequence[str]] = None) -> List[str]:
    """Execution providers for ``device``: ``cpu`` (default), ``cuda`` (CUDA, then CPU) or ``auto`` (CUDA / DirectML if this
    onnxruntime build has them, then CPU). ``cuda`` lists CUDAExecutionProvider even when the build lacks it: the session
    then warns (or fails with ``require_gpu``) instead of silently pretending to be on the GPU."""
    available = list(ort.get_available_providers() if available is None else available)
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "auto":
        return [p for p in GPU_PROVIDERS if p in available] + ["CPUExecutionProvider"]
    raise ValueError("device must be cpu, cuda or auto, got %r" % (device,))


def add_device_args(parser) -> None:
    """``--device cpu|cuda|auto`` (default cpu), ``--gpu`` (= ``--device auto``) and ``--require-gpu`` for a demo's parser."""
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu",
                        help="cpu (default); cuda = CUDAExecutionProvider then CPU; auto = CUDA / DirectML when this "
                             "onnxruntime build has it, else CPU")
    parser.add_argument("--gpu", action="store_true", help="same as --device auto")
    parser.add_argument("--require-gpu", action="store_true",
                        help="fail instead of falling back to the CPU when no GPU provider can be used")


def providers_from_args(args, available: Optional[Sequence[str]] = None) -> Tuple[Optional[List[str]], bool]:
    """(providers, require_gpu) for ``Decider(...)`` from the arguments added by :func:`add_device_args`."""
    device = "auto" if args.gpu and args.device == "cpu" else args.device
    if args.require_gpu and device == "cpu":
        device = "auto"
    providers = select_providers(device, available)
    if device != "cpu" and providers[0] == "CPUExecutionProvider":
        message = ("no GPU provider in this onnxruntime build (install onnxruntime-gpu or onnxruntime-directml "
                   "instead of onnxruntime)")
        if args.require_gpu:
            raise SystemExit("error: " + message)
        print("warning: %s; running on CPU" % message, file=sys.stderr)
    return (None if device == "cpu" else providers), bool(args.require_gpu)


def _create_session(path, options: ort.SessionOptions, providers: Optional[Sequence[str]] = None,
                    require_gpu: bool = False, **kwargs) -> ort.InferenceSession:
    """``providers`` None = CPU only. A GPU that cannot hold the graph (or lacks its DLLs) warns and falls back to the CPU,
    or raises RuntimeError with ``require_gpu``. The providers the session really got are checked, not assumed."""
    wanted = list(providers) if providers else ["CPUExecutionProvider"]
    name = pathlib.Path(str(path)).name
    if wanted[0] == "CPUExecutionProvider":
        if require_gpu:
            raise RuntimeError("require_gpu is set but no GPU provider was asked for")
        return ort.InferenceSession(str(path), sess_options=options, providers=wanted, **kwargs)
    try:
        session = ort.InferenceSession(str(path), sess_options=options, providers=wanted, **kwargs)
        if session.get_providers()[0] != "CPUExecutionProvider":
            return session
        reason = "the provider was dropped while loading (missing CUDA / cuDNN DLLs on PATH?)"
    except Exception as exc:
        reason = str(exc).splitlines()[0][:160] if str(exc) else type(exc).__name__
    if require_gpu:
        raise RuntimeError("%s could not run %s on the GPU (%s)" % (wanted[0], name, reason))
    print("warning: %s could not load %s (%s); using the CPU" % (wanted[0], name, reason), file=sys.stderr)
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"], **kwargs)


PAD = 248044               # <|endoftext|>
MAX_QUESTION_FRACTION = 0.75
CHUNK = 64
DEFAULT_SEQ_LEN = 384      # smallest torso graph used by default (buckets: 384, 512, 1024, 1280)


def fit_to_window(tokenizer: QwenBPE, max_length: int, state_text: str, rq: RenderedQuestion):
    """State ids, question ids and the question tokens' offsets, cut to the window the way the engine cuts them.

    The question gets the first claim on the window (front-truncated if it alone is too long, which keeps the options
    and the <answer> marker); the state is cut from the END to what is left.
    """
    q_ids, q_offsets = tokenizer.encode_with_offsets(rq.text)
    reserve = min(len(q_ids), max(1, int(max_length * MAX_QUESTION_FRACTION)))
    cut = max(0, len(q_ids) - reserve)
    q_ids, q_offsets = q_ids[cut:], q_offsets[cut:]
    state_budget = max(1, max_length - reserve)
    s_ids = tokenizer.encode(state_text)[:state_budget]
    return s_ids, q_ids, q_offsets


class Decider:
    """Load once, then :meth:`decide` / :meth:`predict`.

    >>> d = Decider("models/onnx")
    >>> d.decide("My card was charged twice.", {"team": choice("Which team?", {"billing": "charges", "tech": "bugs"})})
    """

    def __init__(self, model_dir: str | pathlib.Path = "models/onnx", intra_op_threads: Optional[int] = None,
                 load_vision: bool = False, providers: Optional[Sequence[str]] = None,
                 seq_len: int = DEFAULT_SEQ_LEN, require_gpu: bool = False) -> None:
        """``providers``: onnxruntime execution providers, e.g. ``select_providers("cuda")`` (default: CPU only).
        ``require_gpu``: raise RuntimeError instead of falling back to the CPU when a session cannot use the GPU.

        ``seq_len``: the smallest torso graph to use (a fixed-length graph per ``torso_L<N>.onnx``; default 384). The
        first request loads the smallest graph >= max(seq_len, the request); a longer request later replaces it by the
        smallest larger graph that fits, so only one 7.7 GB torso is in memory. Results do not depend on the bucket.
        """
        root = pathlib.Path(model_dir)
        self.root = root
        graphs = sorted(root.glob("torso_L*.onnx"), key=lambda p: int(p.stem.split("_L")[1]))
        if not graphs:
            raise FileNotFoundError("no torso_L*.onnx under %s; run `python export/prepare_model.py` then "
                                    "`export/export_onnx.py`" % root)
        on_disk = [int(p.stem.split("_L")[1]) for p in graphs]
        self.lengths = [L for L in on_disk if L >= seq_len]
        if not self.lengths:
            raise FileNotFoundError("no torso graph of at least %d tokens under %s (found %s); export a longer one with "
                                    "`export/export_onnx.py --seq-len N`" % (seq_len, root, on_disk))
        self.seq_len = self.lengths[0]
        meta = json.loads((root / ("graph_meta_L%d.json" % self.seq_len)).read_text(encoding="utf-8"))
        self.max_length = int(meta["max_length"])
        self.temperature = float(meta["temperature"])
        self.temperature_by_kind: Mapping[str, float] = meta["temperature_by_kind"]
        self.ordinal_smoothing = float(meta["ordinal_smoothing"])
        self.tokenizer = QwenBPE(str(root / "tokenizer.json"))
        head = np.load(root / "head.npz")
        self.head = PointerHead({k: head[k] for k in head.files})
        self._opts = ort.SessionOptions()
        self._opts.log_severity_level = 3          # the exporter leaves CastLike nodes ORT cannot fold; harmless
        if intra_op_threads:
            self._opts.intra_op_num_threads = intra_op_threads
        self._graphs = {int(p.stem.split("_L")[1]): p for p in graphs}
        self._providers = list(providers) if providers else None
        self._require_gpu = require_gpu
        self._sessions: Dict[int, ort.InferenceSession] = {}
        self.vision = None
        if load_vision:
            from .image import VisionEncoder
            self.vision = VisionEncoder(root, intra_op_threads, providers=self._providers, require_gpu=require_gpu)

    # ------------------------------------------------------------------ tokens

    @property
    def providers(self) -> List[str]:
        """All execution providers of the torso session, in priority order (what it really got, not what was asked)."""
        return self._session(next(iter(self._sessions), self.seq_len)).get_providers()

    @property
    def provider(self) -> str:
        """The execution provider the torso graph runs on (loads the first graph if none is loaded yet)."""
        return self._session(next(iter(self._sessions), self.seq_len)).get_providers()[0]

    def _session(self, length: int) -> ort.InferenceSession:
        if length not in self._sessions:
            self._sessions.clear()                  # one 7.7 GB torso at a time
            self._sessions[length] = _create_session(self._graphs[length], self._opts, self._providers, self._require_gpu)
        return self._sessions[length]

    def fit(self, state_text: str, rq: RenderedQuestion) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
        return fit_to_window(self.tokenizer, self.max_length, state_text, rq)

    # ----------------------------------------------------------------- running

    def hidden(self, ids: Sequence[int], read_positions: Sequence[int], image_features: Optional[np.ndarray] = None,
               image_slots: Optional[np.ndarray] = None, position_ids: Optional[np.ndarray] = None,
               extra_chunks: int = 0) -> np.ndarray:
        """Torso hidden states [R, 2048] at ``read_positions`` for one prompt."""
        n = len(ids)
        need = -(-n // CHUNK) * CHUNK + extra_chunks * CHUNK
        loaded = next(iter(self._sessions), None)
        if loaded is not None and loaded >= need:
            length = loaded                         # already in memory and big enough: do not swap it for a smaller one
        else:
            length = next((L for L in self.lengths if L >= need), None)
        if length is None:
            raise ValueError("a prompt of %d tokens needs a graph of at least %d; available: %s (export a longer one)"
                             % (n, need, self.lengths))
        input_ids = np.full((1, length), PAD, dtype=np.int64)
        input_ids[0, :n] = ids
        mask = np.zeros((1, length), dtype=np.int64)
        mask[0, :n] = 1
        if position_ids is None:
            positions = np.broadcast_to(np.arange(length, dtype=np.int64), (3, 1, length)).copy()
        else:
            positions = np.zeros((3, 1, length), dtype=np.int64)
            positions[:, 0, :n] = position_ids
            positions[:, 0, n:] = np.arange(n, length) + int(position_ids.max()) - n + 1   # padding: any increasing values
        hidden_size = 2048
        feats = np.zeros((0, hidden_size), dtype=np.float32) if image_features is None else image_features.astype(np.float32)
        slots = np.zeros((0,), dtype=np.int64) if image_slots is None else image_slots.astype(np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": mask, "position_ids": positions,
                 "read_positions": np.asarray(read_positions, dtype=np.int64),
                 "image_features": feats, "image_slots": slots}
        return self._session(length).run(["hidden"], feeds)[0]

    def probabilities(self, state_text: str, rq: RenderedQuestion, image_features: Optional[np.ndarray] = None,
                      image_grid_thw: Optional[np.ndarray] = None) -> Tuple[List[float], int]:
        s_ids, q_ids, q_offsets = self.fit(state_text, rq)
        options = [len(s_ids) + i for i in option_token_indices(q_offsets, rq.option_spans)]
        ids = s_ids + q_ids
        image: Dict[str, Any] = {}
        if image_features is not None and len(image_features):
            from .image import IMAGE_TOKEN_ID, position_ids
            slots = np.flatnonzero(np.asarray(s_ids) == IMAGE_TOKEN_ID)
            if len(slots) != len(image_features):
                raise ValueError("the images take more of the %d-token window than the state may use "
                                 "(%d of %d image tokens survive); send fewer or smaller images"
                                 % (self.max_length, len(slots), len(image_features)))
            image = {"image_features": image_features, "image_slots": slots,
                     "position_ids": position_ids(ids, image_grid_thw)}
        h = self.hidden(ids, [len(ids) - 1] + options, **image)
        logits = self.head.logits(h[0], h[1:])
        temp = float(self.temperature_by_kind.get(rq.kind, self.temperature))
        return softmax(logits.astype(np.float64) / max(temp, 1e-6)).tolist(), len(ids)

    # ------------------------------------------------------------------ public

    def decide(self, state: Any, questions: Mapping[str, Any], images: Sequence[Any] = ()) -> Dict[str, Dict[str, Any]]:
        return self.predict({"state": state, "questions": questions, "images": list(images)})["answers"]

    def predict(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """A System One request body (state + questions [+ images]) -> a response body."""
        specs = {qid: (q.spec() if isinstance(q, Question) else dict(q)) for qid, q in request["questions"].items()}
        if not specs:
            raise ValueError("at least one question is required")
        images = list(request.get("images") or [])
        extra: Dict[str, Any] = {}
        state_text = render_state(request["state"])
        if images:
            from .image import (VisionEncoder, expand_image_tokens, prepare_images,
                                render_image_state)
            if self.vision is None:
                self.vision = VisionEncoder(self.root, self._opts.intra_op_num_threads or None,
                                            providers=self._providers,
                                            require_gpu=self._require_gpu)
            prepared = prepare_images(images, self.vision.config)
            extra = {"image_features": self.vision.encode(prepared), "image_grid_thw": prepared.image_grid_thw}
            state_text = expand_image_tokens(render_image_state(state_text, len(images)), prepared.token_counts)
        answers: Dict[str, Dict[str, Any]] = {}
        tokens = 0
        for qid, spec in specs.items():
            rq = render_question(spec)
            probs, n = self.probabilities(state_text, rq, **extra)
            tokens += n
            answers[qid] = answer_from_probs(rq, probs, self.ordinal_smoothing)
        return {"model": request.get("model", "strands-decider-2b"), "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": len(specs)}}
