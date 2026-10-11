# -*- coding: utf-8 -*-
"""The request layout of Strands Decider and how its answers are read: a port of strands_decider's
``prompting.py`` / ``schema.py`` / the pointer-head arithmetic in ``modeling.py`` and ``infer.py`` (text and images).

One question is one prompt::

    <state>\\n{state}\\n</state>\\n
    <question type="choice">\\n{header}\\n{instructions}\\n<options>\\n1. name \\u2014 description\\n...\\n</options>\\n</question>\\n<answer>

The state and the question are tokenized SEPARATELY and the id lists concatenated (as upstream does). The pointer head
reads the hidden state of the LAST token of ``<answer>`` (the query) and of the last token of each option's line (the keys):

    logit_k = (k(LN(h_option_k)) . q(LN(h_answer))) / sqrt(256)      -> / temperature(kind) -> softmax over the K options

Nothing here needs torch or onnxruntime; ``numpy`` does the head.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Question", "RenderedQuestion", "answer_from_probs", "choice", "noul", "option_token_indices",
           "render_content", "render_question", "render_state", "score", "validate"]

NOUL_SLOT_LABELS = ("false", "true")
NOUL_DEFAULT_CRITERIA = {"false": "the statement does not hold for this state",
                         "true": "the statement holds for this state"}
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10
HEADERS = {"noul": "Decide whether the statement is true of the state.",
           "choice": "Select exactly one option.",
           "score": "Rate the state against the ordered levels below (lowest first)."}


@dataclass(frozen=True)
class Question:
    type: str
    instructions: Any = ""
    criteria: Any = None

    def spec(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            out["criteria"] = self.criteria
        return out


def choice(instructions: Any, criteria: Mapping[str, Any]) -> Question:
    return Question("choice", instructions, dict(criteria))


def score(instructions: Any, levels: Sequence[str]) -> Question:
    return Question("score", instructions, list(levels))


def noul(instructions: Any, criteria: Optional[Mapping[str, Any]] = None) -> Question:
    return Question("noul", instructions, dict(criteria) if criteria else None)


def render_content(content: Any) -> str:
    """A state or an instruction as text: strings are stripped, anything else is indented JSON (key order kept)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, indent=2, ensure_ascii=False, sort_keys=False)


@dataclass(frozen=True)
class RenderedQuestion:
    text: str
    slot_labels: Tuple[str, ...]
    slot_descriptions: Tuple[str, ...]
    kind: str
    option_spans: Tuple[Tuple[int, int], ...]   # character span of each option's line inside ``text``

    @property
    def n_slots(self) -> int:
        return len(self.slot_labels)


def _option_block(pairs: Sequence[Tuple[str, str]]):
    lines: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for i, (name, desc) in enumerate(pairs):
        desc = " ".join((desc or "").split())
        line = "%d. %s" % (i + 1, name) + (" — %s" % desc if desc else "")
        lines.append(line)
        spans.append((cursor, cursor + len(line)))
        cursor += len(line) + 1
    return "\n".join(lines), spans


def validate(question: Mapping[str, Any]) -> None:
    kind = question.get("type")
    if kind not in ("noul", "choice", "score"):
        raise ValueError("type must be noul, choice, or score")
    crit = question.get("criteria")
    if kind == "noul" and crit is not None and not set(crit).issubset({"true", "false"}):
        raise ValueError("noul criteria keys must be a subset of {'true', 'false'}")
    if kind == "choice" and not (2 <= len(crit or {}) <= MAX_CHOICE_OPTIONS):
        raise ValueError("choice requires between 2 and %d options" % MAX_CHOICE_OPTIONS)
    if kind == "score" and not (MIN_SCORE_LEVELS <= len(crit or []) <= MAX_SCORE_LEVELS):
        raise ValueError("score requires between %d and %d levels" % (MIN_SCORE_LEVELS, MAX_SCORE_LEVELS))


def render_question(question: Mapping[str, Any]) -> RenderedQuestion:
    validate(question)
    kind = question["type"]
    instructions = render_content(question.get("instructions"))
    if kind == "noul":
        crit = {**NOUL_DEFAULT_CRITERIA, **(question.get("criteria") or {})}
        pairs = [(label, render_content(crit[label])) for label in NOUL_SLOT_LABELS]
    elif kind == "choice":
        pairs = [(name, render_content(desc)) for name, desc in question["criteria"].items()]
    else:
        pairs = [(str(i), desc) for i, desc in enumerate(question["criteria"])]
    block, spans = _option_block(pairs)
    prefix = '<question type="%s">\n%s\n%s\n<options>\n' % (kind, HEADERS[kind], instructions)
    text = prefix + block + "\n</options>\n</question>\n<answer>"
    base = len(prefix)
    return RenderedQuestion(text=text, slot_labels=tuple(n for n, _ in pairs),
                            slot_descriptions=tuple((d or "").strip() for _, d in pairs), kind=kind,
                            option_spans=tuple((base + a, base + b) for a, b in spans))


def render_state(state: Any, image_placeholder: str = "") -> str:
    """The shared prefix. With images, their placeholder tokens go first inside <state> (before the text)."""
    body = render_content(state)
    if image_placeholder:
        body = image_placeholder + ("\n" + body if body else "")
    return "<state>\n%s\n</state>\n" % body


def option_token_indices(offsets: Sequence[Tuple[int, int]], spans: Sequence[Tuple[int, int]]) -> List[int]:
    """Index of the last token of each option's line (``offsets`` are the question tokens' character spans)."""
    out: List[int] = []
    for a, b in spans:
        last = -1
        for j, (lo, hi) in enumerate(offsets):
            if hi <= lo:
                continue
            if lo >= a and hi <= b:
                last = j
        if last < 0:
            raise ValueError("option span (%d,%d) has no tokens left; the prompt was truncated through its option list"
                             % (a, b))
        out.append(last)
    return out


# ------------------------------------------------------------------------------------------- the pointer head

class PointerHead:
    """LayerNorm(h) -> q / k projections (2048 -> 256) -> dot / sqrt(256).  fp32 numpy."""

    def __init__(self, weights: Mapping[str, np.ndarray], eps: float = 1e-5):
        self.eps = eps
        self.norm_w = weights["norm.weight"].astype(np.float32)
        self.norm_b = weights["norm.bias"].astype(np.float32)
        self.qw, self.qb = weights["q.weight"].astype(np.float32), weights["q.bias"].astype(np.float32)
        self.kw, self.kb = weights["k.weight"].astype(np.float32), weights["k.bias"].astype(np.float32)
        self.scale = np.float32(self.qw.shape[0] ** -0.5)

    def _norm(self, h: np.ndarray) -> np.ndarray:
        h = h.astype(np.float32)
        mean = h.mean(axis=-1, keepdims=True)
        var = ((h - mean) ** 2).mean(axis=-1, keepdims=True)
        return (h - mean) / np.sqrt(var + self.eps) * self.norm_w + self.norm_b

    def logits(self, answer: np.ndarray, options: np.ndarray) -> np.ndarray:
        """answer [d], options [K, d] -> raw logits [K]."""
        q = self._norm(answer) @ self.qw.T + self.qb
        k = self._norm(options) @ self.kw.T + self.kb
        return (k @ q) * self.scale


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ------------------------------------------------------------------------------------------- reading answers

def derive_confidence(p: Sequence[float]) -> float:
    n = len(p)
    if n <= 1:
        return 1.0
    return float(max(0.0, min(1.0, (n * max(p) - 1.0) / (n - 1.0))))


def derive_score_confidence(p: Sequence[float], ordinal_smoothing: float = 0.0) -> float:
    n = len(p)
    if n <= 1:
        return 1.0
    total = sum(p)
    if total <= 0:
        return 0.0
    q = [x / total for x in p]
    mean = sum(i * x for i, x in enumerate(q))
    sigma = sum(x * (i - mean) ** 2 for i, x in enumerate(q)) ** 0.5
    sigma_max = (n - 1) / 2.0
    floor = ordinal_smoothing ** 0.5 if ordinal_smoothing > 0 else 0.0
    if sigma_max <= floor:
        return 1.0 if sigma <= floor else 0.0
    return float(max(0.0, min(1.0, (sigma_max - sigma) / (sigma_max - floor))))


def answer_from_probs(rq: RenderedQuestion, probs: Sequence[float], ordinal_smoothing: float = 0.1) -> Dict[str, Any]:
    """The System One answer for one question, from the probabilities over its slots."""
    if rq.kind == "noul":
        return {"type": "noul", "noul": round(float(probs[rq.slot_labels.index("true")]), 4)}
    if rq.kind == "choice":
        by_name = {n: float(p) for n, p in zip(rq.slot_labels, probs)}
        best = max(by_name, key=lambda k: by_name[k])
        return {"type": "choice", "choice": best,
                "probabilities": {k: round(v, 4) for k, v in by_name.items()},
                "confidence": round(derive_confidence(list(by_name.values())), 4)}
    by_level = {n: float(p) for n, p in zip(rq.slot_labels, probs)}
    ordered = {str(i): by_level[str(i)] for i in range(len(by_level))}
    expected = sum(int(level) * p for level, p in by_level.items())
    legend = {level: desc for level, desc in sorted(zip(rq.slot_labels, rq.slot_descriptions), key=lambda kv: int(kv[0]))}
    return {"type": "score", "score": round(float(expected), 4), "legend": legend,
            "probabilities": {k: round(v, 4) for k, v in ordered.items()},
            "confidence": round(derive_score_confidence(list(ordered.values()), ordinal_smoothing), 4)}
