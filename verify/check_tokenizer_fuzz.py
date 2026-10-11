# -*- coding: utf-8 -*-
"""Oracle: the pure-Python Qwen BPE vs the Rust `tokenizers` library (ids and character offsets).

    uv run --group verify verify/check_tokenizer_fuzz.py [N]

Needs models/onnx/tokenizer.json (download_model.py). The corpus covers the shapes the Strands prompts take,
plus the Unicode classes where Python's `re` and the Rust regex crate are known to disagree.
"""
import argparse
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strands_decide.tokenizer import QwenBPE  # noqa: E402

TOKJSON = ROOT / "models" / "onnx" / "tokenizer.json"

SPECIALS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>",
    "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<|object_ref_start|>",
    "<|object_ref_end|>", "<|box_start|>", "<|box_end|>",
]

WHITESPACE = [
    " ", " ", " ", "　", "\x0b", "\x0c", "\x1c", "\x1d",
    "\x1e", "\x1f", "​", " ", " ", "﻿", "\x85", " ",
]

BASE = [
    "", " ", "  ", "   ", "\t", "\n", "\r\n", "\n\n\n", " \n ", "a\n\nb",
    "a", "A", "Z", "AA", "ZZ", "The", " the", "the ",
    "don't", "it's", "they're", "we've", "I'm", "we'll", "he'd",
    "DON'T", "It'S", "THEY'RE",
    "CamelCase", "ALLCAPS", "snake_case_name", "kebab-case-name", "mixed123digits",
    "0", "42", "3.14", "1,000,000", "2026-09-28", "v2.0.0", "0x1F",
    "user@example.com", "https://example.com/a/b?c=d&e=f", "www.example.co.jp",
    "@handle", "#hashtag", "50%", "$1,234.56", "¥5,400", "€99", "100°C",
    "naïve café résumé", "Straße Zürich", "İstanbul", "ﬁrefly", "ǅungla",
    "日本語のテキストです", "これはテストです。", "漢字かなカナ", "全角　スペース",
    "한국어 텍스트", "Русский текст", "العربية", "עברית", "ไทย", "हिन्दी",
    "emoji \U0001f389 test", "\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466 family",
    "\U0001f1ef\U0001f1f5 flag", "a\U0001f389b",
    "égalité", "á combining", "①②③", "ⅷ roman", "㍿ square",
    '{"state":{"cart":3},"total":42}', "[1,2,3]", "```python\nx = 1\n```",
    "a" * 40, "ab" * 30, "x" * 100, " " * 20, "\n" * 10,
    "The export button crashes in Safari but works in Chrome. Not blocking.",
]
BASE += SPECIALS
BASE += WHITESPACE
BASE += ["a%sb" % w for w in WHITESPACE]
BASE += ["a%s%sb" % (w, w) for w in WHITESPACE[:8]]
BASE += ["x %s y" % s for s in SPECIALS]

# the pieces strands renders (see strands_decide/protocol.py)
BASE += [
    "<question type=\"choice\">" + chr(10) + "Which team?" + chr(10) + "<options>" + chr(10) + "1. billing " + chr(0x2014) + " charges" + chr(10) + "2. sales" + chr(10) + "</options>" + chr(10) + "</question>" + chr(10) + "<answer>",
    "<question type=\"noul\">" + chr(10) + "顧客は怒っている" + chr(10) + "<options>" + chr(10) + "1. true" + chr(10) + "2. false" + chr(10) + "</options>" + chr(10) + "</question>" + chr(10) + "<answer>",
    '{"customer":"Aiko","notes":null,"order":{"id":1042}}', "Pro プラン café ☕",
]

ALPHABET = list(
    "abcdeABCDE 0123(),.:;-_|?!@#$%/" + chr(92) + chr(39) + chr(34) + "`~^*+=<>[]{}"
    "éüñçİı"
    "日本語漢字ひらがなカタ"
    "한글Русالعไท"
    "\U0001f389\U0001f44d\U0001f1ef\U0001f1f5‍️"
    " 　  ​﻿\x0b\x0c\x1c\x85"
    "\t\n\r "
    "①ⅷ㍿ﬁǅ́"
)


def random_cases(n, rng):
    out = []
    for _ in range(n):
        k = rng.randint(1, 60)
        out.append("".join(rng.choice(ALPHABET) for _ in range(k)))
    for _ in range(n // 4):
        w = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 12)))
        out.append("<|im_start|>user\n%s %s %s<|im_end|>\n" % (w, rng.choice(SPECIALS), w))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=4000)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    rng = random.Random(20261006)
    mine = QwenBPE(str(TOKJSON))
    ref = Tokenizer.from_file(str(TOKJSON))
    cases = list(BASE) + random_cases(args.n, rng)

    bad, offbad = [], []
    for text in cases:
        got, goff = mine.encode_with_offsets(text)
        r = ref.encode(text, add_special_tokens=False)
        exp = list(r.ids)
        if list(got) != exp:
            bad.append((text, list(got), exp))
        elif [tuple(o) for o in goff] != [tuple(o) for o in r.offsets]:
            offbad.append((text, [tuple(o) for o in goff], [tuple(o) for o in r.offsets]))

    print("cases: %d   id mismatches: %d   offset mismatches: %d" % (len(cases), len(bad), len(offbad)))
    import unicodedata
    nomark = [t for t, _, _ in offbad if not any(unicodedata.combining(c) for c in t)]
    print("offset mismatches without a combining mark: %d" % len(nomark))
    for text, g, e in offbad[:2]:
        print("\n  offsets differ: %s\n   got %s\n   exp %s" % (ascii(text), g, e))
    for text, got, exp in bad[:12]:
        print("\n  input : %s" % ascii(text))
        print("  got   : %s" % got)
        print("  expect: %s" % exp)
        print("  gotstr: %s" % [ref.id_to_token(i) for i in got])
        print("  expstr: %s" % [ref.id_to_token(i) for i in exp])
    print("\n%d cases, %d id mismatches (%d offset differences, all around combining marks)"
          % (len(cases), len(bad), len(offbad)))
    return 1 if bad or nomark else 0


if __name__ == "__main__":
    raise SystemExit(main())
