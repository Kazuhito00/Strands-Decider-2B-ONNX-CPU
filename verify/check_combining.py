# -*- coding: utf-8 -*-
"""Do HF's offsets (which differ from ours around combining marks) ever move an option's read position? Needs the verify group (tokenizers) and models/onnx/tokenizer.json."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strands_decide.protocol import option_token_indices, render_question  # noqa: E402
from strands_decide.tokenizer import QwenBPE  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

mine = QwenBPE(str(ROOT / "models" / "onnx" / "tokenizer.json"))
ref = Tokenizer.from_file(str(ROOT / "models" / "onnx" / "tokenizer.json"))
names = ["cafe" + chr(0x301), "a" + chr(0x301), "e" + chr(0x301) + "e" + chr(0x301), "na" + chr(0xef) + "ve", "ka" + chr(0x3099) + "ki",
         "x" + chr(0x301) + " y", "plain", chr(0x30ab) + chr(0x3099)]
bad = 0
for i in range(len(names)):
    crit = {names[i]: "d" + chr(0x301), names[(i + 1) % len(names)] + "!": "z"}
    rq = render_question({"type": "choice", "instructions": "q" + chr(0x301) + "?", "criteria": crit})
    ids, offs = mine.encode_with_offsets(rq.text)
    r = ref.encode(rq.text, add_special_tokens=False)
    a = option_token_indices(offs, rq.option_spans)
    b = option_token_indices([tuple(o) for o in r.offsets], rq.option_spans)
    bad += a != b
    print(ascii(names[i]), a, b, "ok" if a == b else "DIFFERS")
print("differences:", bad)
raise SystemExit(1 if bad else 0)
