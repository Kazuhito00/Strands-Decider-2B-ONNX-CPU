# -*- coding: utf-8 -*-
"""Qwen's byte-level BPE tokenizer in pure Python.

Reads a HuggingFace ``tokenizer.json`` and reproduces
``tokenizers.Tokenizer.encode(text, add_special_tokens=False).ids`` so the
``tokenizers`` (Rust) dependency can be dropped. Only the pipeline this
checkpoint declares is implemented::

    added tokens -> NFC
                 -> Split(GPT-2 style regex, Isolated) -> ByteLevel(use_regex=false)
                 -> BPE (no dropout, no unk, no byte_fallback, ignore_merges=false)

Anything else a tokenizer.json might declare is rejected at load time rather
than silently ignored.

The split regex uses ``\\p{L}``/``\\p{N}``/``\\p{M}``/``\\s``, which Python's
``re`` cannot express; the classes come from the generated ``_uniprops`` table
so that no ``regex`` package is needed. See verify/check_tokenizer_fuzz.py.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple


__all__ = ["QwenBPE"]

from ._regex_translate import translate as _translate_pattern

def _byte_encoder() -> Dict[int, str]:
    """GPT-2's reversible byte -> printable-codepoint map (ByteLevel alphabet)."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    table = {b: chr(b) for b in printable}
    spare = 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + spare)
            spare += 1
    return table


BYTE_TO_CHAR = _byte_encoder()


class QwenBPE:
    """Encode text to token ids the way the Rust tokenizer would."""

    def __init__(self, tokenizer_json: str):
        with open(tokenizer_json, encoding="utf-8") as fh:
            spec = json.load(fh)

        model = spec["model"]
        self._split_pattern = self._check(spec, model)

        self.vocab: Dict[str, int] = model["vocab"]
        # Rank = position in the merges list; lower merges first. Newer files
        # store each merge as [a, b], older ones as the string "a b"; byte-level
        # pieces never contain a space, so splitting on it is unambiguous.
        self.ranks: Dict[Tuple[str, str], int] = {}
        for rank, merge in enumerate(model["merges"]):
            if isinstance(merge, str):
                parts = merge.split(" ")
                if len(parts) != 2:
                    raise ValueError("cannot parse merge %r" % (merge,))
                pair = (parts[0], parts[1])
            else:
                pair = (merge[0], merge[1])
            self.ranks[pair] = rank

        added = sorted(spec.get("added_tokens", []), key=lambda t: -len(t["content"]))
        self._added: Dict[str, int] = {t["content"]: t["id"] for t in added}
        self._raw_re = self._alternation(t for t in added if not t.get("normalized"))
        self._norm_re = self._alternation(t for t in added if t.get("normalized"))

        self._split_re = re.compile(_translate_pattern(self._split_pattern))
        self._bpe = lru_cache(maxsize=100_000)(self._bpe_uncached)

    # ---------------------------------------------------------- spec guards

    @staticmethod
    def _alternation(tokens):
        pattern = "|".join(re.escape(t["content"]) for t in tokens)
        return re.compile(pattern) if pattern else None

    @staticmethod
    def _check(spec, model) -> str:
        if model.get("type") != "BPE":
            raise ValueError("expected a BPE model, got %r" % (model.get("type"),))
        for flag in ("dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix"):
            if model.get(flag):
                raise ValueError("BPE option %s is not implemented" % flag)
        if model.get("byte_fallback") or model.get("ignore_merges"):
            raise ValueError("byte_fallback / ignore_merges are not implemented")
        norm = spec.get("normalizer")
        if not norm or norm.get("type") != "NFC":
            raise ValueError("expected an NFC normalizer, got %r" % (norm,))
        pre = spec.get("pre_tokenizer") or {}
        steps = pre.get("pretokenizers") if pre.get("type") == "Sequence" else [pre]
        kinds = [s.get("type") for s in steps if s]
        if kinds != ["Split", "ByteLevel"]:
            raise ValueError("unsupported pre_tokenizer sequence: %r" % (kinds,))
        if steps[0]["pattern"].get("Regex") is None or steps[0].get("behavior") != "Isolated":
            raise ValueError("unexpected Split step: %r" % (steps[0],))
        if steps[1].get("add_prefix_space") or steps[1].get("use_regex"):
            raise ValueError("unexpected ByteLevel step: %r" % (steps[1],))
        return steps[0]["pattern"]["Regex"]

    # ------------------------------------------------------------------ bpe

    def _bpe_uncached(self, word: str) -> Tuple[int, ...]:
        """Merge one pre-token (already byte-level encoded) into token ids."""
        parts = list(word)
        if len(parts) > 1:
            ranks = self.ranks
            while True:
                best, best_at = None, -1
                for i in range(len(parts) - 1):
                    rank = ranks.get((parts[i], parts[i + 1]))
                    if rank is not None and (best is None or rank < best):
                        best, best_at = rank, i
                if best is None:
                    break
                parts[best_at : best_at + 2] = [parts[best_at] + parts[best_at + 1]]
        vocab = self.vocab
        return tuple(vocab[p] for p in parts)

    # --------------------------------------------------------------- encode

    def _encode_plain(self, text: str) -> List[int]:
        out: List[int] = []
        for match in self._split_re.finditer(text):
            piece = match.group()
            byte_level = "".join(BYTE_TO_CHAR[b] for b in piece.encode("utf-8"))
            out.extend(self._bpe(byte_level))
        return out

    def _split_on(self, pattern, text: str):
        if pattern is None:
            yield text, None
            return
        pos = 0
        for m in pattern.finditer(text):
            if m.start() > pos:
                yield text[pos : m.start()], None
            yield m.group(), self._added[m.group()]
            pos = m.end()
        if pos < len(text):
            yield text[pos:], None

    def _encode_normalized(self, text: str) -> List[int]:
        out: List[int] = []
        for piece, tid in self._split_on(self._norm_re, text):
            if tid is not None:
                out.append(tid)
            else:
                out.extend(self._encode_plain(piece))
        return out

    def encode(self, text: str) -> List[int]:
        """Token ids for ``text``. Special tokens present in the text are honoured."""
        if not text:
            return []
        out: List[int] = []
        # Same two-pass structure as AddedVocabulary::extract_and_normalize:
        # normalized=false tokens are cut out of the raw text, and each
        # surviving piece is normalized on its own.
        for piece, tid in self._split_on(self._raw_re, text):
            if tid is not None:
                out.append(tid)
            else:
                out.extend(self._encode_normalized(unicodedata.normalize("NFC", piece)))
        return out

    # --------------------------------------------------- encode, with offsets

    def _split_on_positions(self, pattern, text: str):
        """Like _split_on, but yields (start, end, piece, added_token_id | None)."""
        if pattern is None:
            yield 0, len(text), text, None
            return
        pos = 0
        for m in pattern.finditer(text):
            if m.start() > pos:
                yield pos, m.start(), text[pos:m.start()], None
            yield m.start(), m.end(), m.group(), self._added[m.group()]
            pos = m.end()
        if pos < len(text):
            yield pos, len(text), text[pos:], None

    @staticmethod
    def _nfc_alignment(piece: str):
        """(normalized, map) where map[i] = (start, end) in ``piece`` of normalized char i.

        NFC only rewrites a base character together with the combining marks that follow it, so the text is cut
        into such clusters and each is normalized on its own. Returns map=None (identity) when nothing changes.
        """
        norm = unicodedata.normalize("NFC", piece)
        if norm == piece:
            return norm, None
        clusters, start = [], 0
        for i in range(1, len(piece) + 1):
            if i == len(piece) or unicodedata.combining(piece[i]) == 0:
                clusters.append((start, i))
                start = i
        out_chars, mapping = [], []
        for a, b in clusters:
            n = unicodedata.normalize("NFC", piece[a:b])
            for ch in n:
                out_chars.append(ch)
                mapping.append((a, b))
        if "".join(out_chars) != norm:      # a composition that crosses clusters (e.g. Hangul jamo): be coarse
            return norm, [(0, len(piece))] * len(norm)
        return norm, mapping

    def _plain_offsets(self, text: str, base: int, to_original):
        """ids and (start, end) offsets for text with no added tokens in it. ``base`` shifts positions."""
        ids: List[int] = []
        offs: List[Tuple[int, int]] = []
        for match in self._split_re.finditer(text):
            piece = match.group()
            raw = piece.encode("utf-8")
            byte_level = "".join(BYTE_TO_CHAR[b] for b in raw)
            # byte index -> the character it belongs to (a multi-byte character owns all of its bytes)
            owner, char_at = [], match.start()
            for ch in piece:
                n = len(ch.encode("utf-8"))
                owner.extend([char_at] * n)
                char_at += 1
            tokens = self._bpe(byte_level)
            cursor = 0
            # token lengths in bytes: look the token string up again (each byte-level char is one byte)
            inverse = self._id_to_piece
            for tid in tokens:
                length = len(inverse[tid])
                first, last = owner[cursor], owner[cursor + length - 1] + 1
                ids.append(tid)
                offs.append(to_original(first, last, base))
                cursor += length
        return ids, offs

    @property
    def _id_to_piece(self):
        table = getattr(self, "_id_to_piece_cache", None)
        if table is None:
            table = {i: p for p, i in self.vocab.items()}
            self._id_to_piece_cache = table
        return table

    def encode_with_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Token ids and the (start, end) character span of each token in ``text``.

        Offsets follow the Rust tokenizers: a token's span runs from the first character its bytes touch to the end of
        the last one; an added (special) token spans its own text.
        """
        ids: List[int] = []
        offs: List[Tuple[int, int]] = []
        if not text:
            return ids, offs
        for s0, e0, piece, tid in self._split_on_positions(self._raw_re, text):
            if tid is not None:
                ids.append(tid)
                offs.append((s0, e0))
                continue
            norm, mapping = self._nfc_alignment(piece)

            def to_original(a, b, base, mapping=mapping, s0=s0, piece=piece):
                if mapping is None:
                    return (s0 + a + base, s0 + b + base)
                return (s0 + mapping[a + base][0], s0 + mapping[b - 1 + base][1])

            for s1, e1, sub, tid2 in self._split_on_positions(self._norm_re, norm):
                if tid2 is not None:
                    ids.append(tid2)
                    offs.append(to_original(s1, e1, 0))
                else:
                    i, o = self._plain_offsets(sub, s1, to_original)
                    ids.extend(i)
                    offs.extend(o)
        return ids, offs

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]
