# -*- coding: utf-8 -*-
"""Image input for Strands Decider: Pillow + numpy around the exported Qwen3.5 vision tower.

Everything here follows strands_decider's own ``vision.py`` (its main branch; 0.1.0 has no image path):

* the image is decoded upright (EXIF), transparency is read as white, and it is scaled DOWN (never up) so that its
  longer side is at most 448 px (196 tokens for a square image), then resized again by Qwen2VL's ``smart_resize``;
* images go inside ``<state>`` as ``<|vision_start|><|image_pad|>...<|vision_end|>``, one block per image, before the
  state text; each ``<|image_pad|>`` stands for one merged 2x2 patch;
* positions are Qwen3.5's M-RoPE: an image advances the rotary positions by its grid size, not its token count, and
  the text after it continues from there (``position_ids``).

The vision tower itself (``vision_encoder.onnx``) takes the patches plus three arrays that transformers derives from the
grid with data-dependent Python; they are computed here with numpy (``bilinear_inputs``, ``vision_position_ids``) and
verify/check_vision_inputs.py compares them with transformers' own functions.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
import pathlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

__all__ = ["ImageConfig", "PreparedImages", "VisionEncoder", "bilinear_inputs", "decode_image", "expand_image_tokens",
           "image_tokens_for_grid", "position_ids", "prepare_images", "render_image_state", "smart_resize",
           "vision_position_ids"]

ImageSource = Union[str, pathlib.Path, bytes, "Image.Image"]

VISION_START, IMAGE_PAD, VISION_END = "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"
IMAGE_TOKEN_ID = 248056          # <|image_pad|>
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")
MAX_IMAGES = 4
MAX_INPUT_PIXELS = 4096 * 4096


@dataclass(frozen=True)
class ImageConfig:
    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    min_pixels: int = 65536          # preprocessor_config.json: size.shortest_edge
    max_pixels: int = 16777216       # size.longest_edge
    long_side: int = 448             # Strands' fixed deployment setting: scale down so the longer side is at most this
    num_position_embeddings: int = 2304

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size

    @classmethod
    def from_pretrained(cls, model_dir, **overrides) -> "ImageConfig":
        path = pathlib.Path(model_dir) / "preprocessor_config.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        size = cfg.get("size") or {}
        kw = {"patch_size": cfg.get("patch_size", 16), "temporal_patch_size": cfg.get("temporal_patch_size", 2),
              "merge_size": cfg.get("merge_size", 2), "image_mean": tuple(cfg.get("image_mean", (0.5, 0.5, 0.5))),
              "image_std": tuple(cfg.get("image_std", (0.5, 0.5, 0.5)))}
        if "shortest_edge" in size:
            kw["min_pixels"] = int(size["shortest_edge"])
        if "longest_edge" in size:
            kw["max_pixels"] = int(size["longest_edge"])
        kw.update(overrides)
        return cls(**kw)


@dataclass
class PreparedImages:
    pixel_values: np.ndarray        # float32 [patches, 3*2*16*16]
    image_grid_thw: np.ndarray      # int64   [n_images, 3]

    @property
    def token_counts(self) -> List[int]:
        return image_tokens_for_grid(self.image_grid_thw, 2)


# --------------------------------------------------------------------------------------------- prompt

def render_image_state(state_text: str, n_images: int) -> str:
    """``render_state(state)`` with one unexpanded placeholder per image, before the text."""
    if not n_images:
        return state_text
    placeholders = "\n".join("%s%s%s" % (VISION_START, IMAGE_PAD, VISION_END) for _ in range(n_images))
    head, tail = "<state>\n", "\n</state>\n"
    body = state_text[len(head):-len(tail)]
    inner = "%s\n%s" % (placeholders, body) if body else placeholders
    return "%s%s%s" % (head, inner, tail)


def expand_image_tokens(prompt: str, tokens_per_image: Sequence[int]) -> str:
    parts = prompt.split(IMAGE_PAD)
    if len(parts) - 1 != len(tokens_per_image):
        raise ValueError("prompt holds %d image placeholders but %d token counts were given"
                         % (len(parts) - 1, len(tokens_per_image)))
    out = [parts[0]]
    for n, tail in zip(tokens_per_image, parts[1:]):
        out.append(IMAGE_PAD * int(n))
        out.append(tail)
    return "".join(out)


def image_tokens_for_grid(grid_thw: Iterable[Sequence[int]], merge_size: int) -> List[int]:
    m = merge_size * merge_size
    return [int(t) * int(h) * int(w) // m for t, h, w in grid_thw]


# --------------------------------------------------------------------------------------------- pixels

def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
    """transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200, got %s" % (max(height, width) / min(height, width)))
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def decode_image(source: ImageSource, max_pixels: int = MAX_INPUT_PIXELS) -> "Image.Image":
    """An upright RGB image from a path, bytes, a PIL image or a base64 / ``data:image`` string.

    Only PNG, JPEG, WebP and GIF are opened and the size is checked from the header, before any pixel is decoded.
    """
    if isinstance(source, Image.Image):
        img = source
    else:
        if isinstance(source, (bytes, bytearray)):
            raw = bytes(source)
        else:
            text = str(source)
            if text.startswith("data:image/") or (len(text) > 260 and not pathlib.Path(text).exists()):
                payload = re.sub(r"^data:image/[\w.+-]+;base64,", "", text.strip())
                try:
                    raw = base64.b64decode(payload, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError("image is not valid base64: %s" % exc) from exc
            else:
                raw = pathlib.Path(text).read_bytes()
        try:
            img = Image.open(io.BytesIO(raw), formats=IMAGE_FORMATS)
        except Exception as exc:
            raise ValueError("image could not be read as %s: %s" % ("/".join(IMAGE_FORMATS), exc)) from exc
    w, h = img.size
    if w * h > max_pixels:
        raise ValueError("image is %dx%d; at most %s pixels are accepted" % (w, h, format(max_pixels, ",")))
    try:
        img.load()
        img = ImageOps.exif_transpose(img)
    except Exception as exc:
        raise ValueError("image could not be decoded: %s" % exc) from exc
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return img.convert("RGB")


def fit_long_side(img: "Image.Image", long_side: int) -> "Image.Image":
    w, h = img.size
    if long_side and max(w, h) > long_side:
        s = long_side / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.BICUBIC)
    return img


def _patchify(pixels: np.ndarray, cfg: ImageConfig) -> Tuple[np.ndarray, int, int]:
    channel, height, width = pixels.shape
    patch, merge, temporal = cfg.patch_size, cfg.merge_size, cfg.temporal_patch_size
    grid_h, grid_w = height // patch, width // patch
    patches = pixels.reshape(channel, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
    patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)                       # gh, gw, mh, mw, C, ph, pw
    patches = np.broadcast_to(patches[:, :, :, :, :, None, :, :], (*patches.shape[:5], temporal, *patches.shape[5:]))
    return np.ascontiguousarray(patches.reshape(grid_h * grid_w, channel * temporal * patch * patch), dtype=np.float32), grid_h, grid_w


def prepare_images(sources: Sequence[ImageSource], cfg: ImageConfig | None = None) -> PreparedImages:
    """Decode, fit to the long-side cap, ``smart_resize``, normalise and patchify (Qwen2VLImageProcessorPil's recipe)."""
    cfg = cfg or ImageConfig()
    if len(sources) > MAX_IMAGES:
        raise ValueError("%d images; at most %d are taken" % (len(sources), MAX_IMAGES))
    mean = np.asarray(cfg.image_mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(cfg.image_std, dtype=np.float32).reshape(3, 1, 1)
    patches, grids = [], []
    for source in sources:
        img = fit_long_side(decode_image(source), cfg.long_side)
        h, w = smart_resize(img.height, img.width, cfg.factor, cfg.min_pixels, cfg.max_pixels)
        if (w, h) != img.size:
            img = img.resize((w, h), Image.Resampling.BICUBIC)
        px = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).astype(np.float32) * np.float32(1 / 255)
        px = (px - mean) / std
        flat, gh, gw = _patchify(px.astype(np.float32), cfg)
        patches.append(flat)
        grids.append([1, gh, gw])
    if not patches:
        dim = 3 * cfg.temporal_patch_size * cfg.patch_size ** 2
        return PreparedImages(np.zeros((0, dim), np.float32), np.zeros((0, 3), np.int64))
    return PreparedImages(np.concatenate(patches, axis=0), np.asarray(grids, dtype=np.int64))


# ------------------------------------------------------------------------- vision-tower side inputs

def _axis_taps(pos: np.ndarray, length: np.ndarray, side: int) -> Tuple[np.ndarray, np.ndarray]:
    """align_corners=True linear taps of a ``side``-long table at ``length`` evenly spaced points: (floor, ceil), weights."""
    denom = np.maximum(length - 1, 1)
    src = pos.astype(np.float64) * (side - 1) / denom
    src = np.where(length > 1, src, 0.0)
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, side - 1)
    frac = (src - lo).astype(np.float32)
    return np.stack([lo, hi], axis=1), np.stack([1.0 - frac, frac], axis=1)


def _block_order(h: int, w: int, merge: int) -> Tuple[np.ndarray, np.ndarray]:
    """(row, col) of every patch of an h x w grid in spatial-merge-block order."""
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    shape = (h // merge, merge, w // merge, merge)
    return (rows.reshape(shape).transpose(0, 2, 1, 3).reshape(-1),
            cols.reshape(shape).transpose(0, 2, 1, 3).reshape(-1))


def bilinear_inputs(grid_thw: np.ndarray, side: int, merge: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """``bilinear_indices`` int64 [4, patches] and ``bilinear_weights`` float32 [4, patches] (transformers'
    ``get_vision_bilinear_indices_and_weights``: align_corners=True, merge-block patch order)."""
    idx_all, w_all = [], []
    for t, h, w in np.asarray(grid_thw).tolist():
        rows, cols = _block_order(h, w, merge)
        r_taps, r_w = _axis_taps(rows, np.full(rows.shape, h), side)
        c_taps, c_w = _axis_taps(cols, np.full(cols.shape, w), side)
        idx = (r_taps[:, :, None] * side + c_taps[:, None, :]).reshape(-1, 4)
        wts = (r_w[:, :, None] * c_w[:, None, :]).reshape(-1, 4)
        idx_all.append(np.tile(idx, (t, 1)))
        w_all.append(np.tile(wts, (t, 1)))
    return (np.ascontiguousarray(np.concatenate(idx_all).T, dtype=np.int64),
            np.ascontiguousarray(np.concatenate(w_all).T, dtype=np.float32))


def vision_position_ids(grid_thw: np.ndarray, merge: int = 2) -> np.ndarray:
    """int64 [patches, 2]: (row, col) of every patch in merge-block order, for the vision rotary embedding."""
    out = []
    for t, h, w in np.asarray(grid_thw).tolist():
        rows, cols = _block_order(h, w, merge)
        out.append(np.tile(np.stack([rows, cols], axis=-1), (t, 1)))
    return np.concatenate(out).astype(np.int64)


class VisionEncoder:
    """``vision_encoder.onnx``: patches -> one 2048-d row per merged visual token (one image per call)."""

    def __init__(self, model_dir, intra_op_threads: int | None = None, config: ImageConfig | None = None,
                 providers: Sequence[str] | None = None, require_gpu: bool = False) -> None:
        from .runtime import _create_session

        root = pathlib.Path(model_dir)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if intra_op_threads:
            opts.intra_op_num_threads = intra_op_threads
        self.session = _create_session(root / "vision_encoder.onnx", opts, providers, require_gpu)
        pre = root / "preprocessor_config.json"
        self.config = config or (ImageConfig.from_pretrained(root) if pre.exists() else ImageConfig())

    def encode(self, prepared: PreparedImages) -> np.ndarray:
        """float32 [sum(tokens), 2048], images concatenated in order."""
        feats, start = [], 0
        side = int(round(self.config.num_position_embeddings ** 0.5))
        for grid in prepared.image_grid_thw:
            n = int(grid[0] * grid[1] * grid[2])
            g = grid[None, :]
            bi, bw = bilinear_inputs(g, side, self.config.merge_size)
            pos = vision_position_ids(g, self.config.merge_size)
            feats.append(self.session.run(["image_features"], {
                "pixel_values": prepared.pixel_values[start:start + n], "bilinear_indices": bi,
                "bilinear_weights": bw, "position_ids": pos})[0])
            start += n
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 2048), np.float32)


# --------------------------------------------------------------------------------------------- M-RoPE

def position_ids(input_ids: Sequence[int], image_grid_thw: np.ndarray, merge: int = 2,
                 image_token_id: int = IMAGE_TOKEN_ID) -> np.ndarray:
    """3-axis M-RoPE positions [3, n] (``Qwen3_5Model.get_rope_index``): text advances all axes together; an image lays
    its tokens on a (t, h, w) grid at the current cursor, then the cursor advances by max(h, w) // merge."""
    ids = np.asarray(input_ids, dtype=np.int64)
    is_img = ids == image_token_id
    grids = list(np.asarray(image_grid_thw).tolist()) if len(image_grid_thw) else []
    chunks, cursor, i, g = [], 0, 0, 0
    n = len(ids)
    while i < n:
        j = i
        while j < n and is_img[j] == is_img[i]:
            j += 1
        if not is_img[i]:
            chunks.append(np.broadcast_to(np.arange(j - i, dtype=np.int64), (3, j - i)) + cursor)
            cursor += j - i
        else:
            if g >= len(grids):
                raise ValueError("more image token runs than image grids")
            t, h, w = grids[g]
            g += 1
            hh, ww = h // merge, w // merge
            tt = np.zeros((t, hh, ww), dtype=np.int64)
            hs = np.broadcast_to(np.arange(hh, dtype=np.int64)[None, :, None] + cursor, (t, hh, ww))
            ws = np.broadcast_to(np.arange(ww, dtype=np.int64)[None, None, :] + cursor, (t, hh, ww))
            tt = tt + cursor + np.arange(t, dtype=np.int64)[:, None, None] * 0
            chunks.append(np.stack([tt, hs, ws]).reshape(3, -1))
            cursor += max(h, w) // merge
        i = j
    if g != len(grids):
        raise ValueError("%d image grids were never placed in the prompt" % (len(grids) - g))
    pos = np.concatenate(chunks, axis=1) if chunks else np.zeros((3, 0), np.int64)
    if pos.shape[1] != n:
        raise ValueError("position_ids length %d does not match %d input ids" % (pos.shape[1], n))
    return pos
