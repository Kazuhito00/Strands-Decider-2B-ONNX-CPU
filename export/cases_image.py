# -*- coding: utf-8 -*-
"""Image requests the image oracle (export/make_golden_image.py) and the image gates (verify/) share.

Images are generated here (deterministic, no files to ship): a coloured-shapes scene, a "screen" with text and a red
error banner, a bar chart. Photos from artifacts/images (CC0 / public domain; see artifacts/images/LICENSES.json) are added
when they are present and recorded by file name.
"""
from __future__ import annotations

import io
import pathlib

from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL = ROOT / "artifacts" / "images"


def shapes() -> Image.Image:
    im = Image.new("RGB", (640, 480), (240, 240, 235))
    d = ImageDraw.Draw(im)
    d.ellipse((60, 60, 260, 260), fill=(200, 30, 30))
    d.rectangle((340, 120, 560, 300), fill=(30, 60, 200))
    d.polygon([(120, 440), (260, 320), (400, 440)], fill=(30, 160, 60))
    return im


def screen() -> Image.Image:
    im = Image.new("RGB", (800, 500), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 800, 60), fill=(30, 40, 90))
    d.text((20, 22), "Checkout - Pay now", fill=(255, 255, 255))
    d.rectangle((40, 100, 760, 190), fill=(255, 225, 225), outline=(200, 0, 0), width=3)
    d.text((60, 135), "Payment failed: card declined (error 402)", fill=(160, 0, 0))
    d.rectangle((300, 300, 500, 360), fill=(40, 120, 220))
    d.text((365, 325), "Retry", fill=(255, 255, 255))
    return im


def chart() -> Image.Image:
    im = Image.new("RGB", (600, 400), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for i, h in enumerate([80, 150, 230, 120]):
        d.rectangle((60 + i * 130, 360 - h, 150 + i * 130, 360), fill=(40 + 40 * i, 90, 200 - 40 * i))
    d.line((40, 360, 580, 360), fill=(0, 0, 0), width=2)
    return im


def png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def images():
    """name -> bytes (PNG/JPEG)."""
    out = {"shapes": png_bytes(shapes()), "screen": png_bytes(screen()), "chart": png_bytes(chart())}
    for name in ("cat", "stop_sign", "barchart", "bus_stop_ja"):
        p = REAL / (name + ".jpg")
        if p.exists():
            out[name] = p.read_bytes()
    return out


YESNO = {"type": "noul", "instructions": "Is there a red circle in the picture?"}
COLOR = {"type": "choice", "instructions": "Which colour is the largest rectangle?",
         "criteria": {"red": "", "blue": "", "green": "", "yellow": ""}}
SEVERE = {"type": "score", "instructions": "How severe is the error shown?",
          "criteria": ["no error", "warning", "blocking error"]}
NEXT = {"type": "choice", "instructions": "Which control should the agent use next?",
        "criteria": {"retry": "a retry button is shown", "back": "return to the cart", "none": "nothing to do"}}
ANIMAL = {"type": "choice", "instructions": "What animal is in the picture?",
          "criteria": {"cat": "", "dog": "", "horse": "", "bird": ""}}
JA = {"type": "choice", "instructions": "この画像に写っている色の組み合わせとして最も近いものはどれですか？",
      "criteria": {"赤と青": "", "緑と黄": "", "白と黒": ""}}

CASES = {
    "shapes-text": {"state": "", "images": ["shapes"], "questions": {"circle": YESNO, "color": COLOR, "ja": JA}},
    "screen-with-state": {"state": "Checkout page after the user tapped Pay.", "images": ["screen"],
                          "questions": {"severe": SEVERE, "next": NEXT}},
    "two-images": {"state": "Compare the two pictures.", "images": ["shapes", "chart"],
                   "questions": {"circle": YESNO}},
    "chart-only": {"state": "", "images": ["chart"],
                   "questions": {"rising": {"type": "noul", "instructions": "The bars rise from left to right"}}},
}
if (REAL / "cat.jpg").exists():
    CASES["photo-cat"] = {"state": "", "images": ["cat"], "questions": {"animal": ANIMAL}}
