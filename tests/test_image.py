# -*- coding: utf-8 -*-
"""Image pre-processing and the vision-side inputs (strands_decide.image); needs Pillow only."""
import io

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from strands_decide.image import (IMAGE_TOKEN_ID, ImageConfig, bilinear_inputs, decode_image, expand_image_tokens,  # noqa: E402
                                  fit_long_side, image_tokens_for_grid, position_ids, prepare_images, render_image_state,
                                  smart_resize, vision_position_ids)


def test_smart_resize_rounds_to_the_factor_and_respects_bounds():
    assert smart_resize(448, 448, 32, 65536, 16777216) == (448, 448)
    assert smart_resize(100, 100, 32, 65536, 16777216) == (256, 256)           # grown to min_pixels
    h, w = smart_resize(5000, 5000, 32, 65536, 1_000_000)
    assert h % 32 == 0 and w % 32 == 0 and h * w <= 1_000_000
    with pytest.raises(ValueError):
        smart_resize(10, 3000, 32, 65536, 16777216)


def test_fit_long_side_only_shrinks():
    assert fit_long_side(Image.new("RGB", (1000, 500)), 448).size == (448, 224)
    assert fit_long_side(Image.new("RGB", (300, 200)), 448).size == (300, 200)


def test_prepare_images_shapes_and_token_counts():
    out = prepare_images([Image.new("RGB", (640, 480), (10, 200, 30)), Image.new("RGB", (448, 448))], ImageConfig())
    assert out.pixel_values.dtype == np.float32 and out.pixel_values.shape[1] == 3 * 2 * 16 * 16
    assert out.image_grid_thw.shape == (2, 3) and (out.image_grid_thw[:, 0] == 1).all()
    assert out.pixel_values.shape[0] == int(sum(g[1] * g[2] for g in out.image_grid_thw))
    assert out.token_counts == image_tokens_for_grid(out.image_grid_thw, 2)
    assert out.token_counts[1] == 196                                           # a 448 px square: 28 x 28 / 4
    assert -1.0 <= out.pixel_values.min() and out.pixel_values.max() <= 1.0


def test_prepare_images_limits():
    assert prepare_images([]).pixel_values.shape[0] == 0
    with pytest.raises(ValueError, match="at most 4"):
        prepare_images([Image.new("RGB", (64, 64))] * 5)


def test_decode_image_sources_and_errors(tmp_path):
    buf = io.BytesIO()
    Image.new("RGBA", (40, 30), (255, 0, 0, 0)).save(buf, "PNG")
    img = decode_image(buf.getvalue())                                          # transparent pixels become white
    assert img.mode == "RGB" and img.getpixel((0, 0)) == (255, 255, 255)
    path = tmp_path / "a.png"
    path.write_bytes(buf.getvalue())
    assert decode_image(str(path)).size == (40, 30)
    with pytest.raises(ValueError):
        decode_image(b"not an image")
    with pytest.raises(ValueError, match="at most"):
        decode_image(Image.new("RGB", (100, 100)), max_pixels=50)


def test_placeholders_expand_to_the_token_counts():
    prompt = render_image_state("<state>\nhi\n</state>\n", 2)
    assert prompt.count("<|image_pad|>") == 2 and prompt.index("hi") > prompt.index("<|vision_end|>")
    assert expand_image_tokens(prompt, [3, 5]).count("<|image_pad|>") == 8
    with pytest.raises(ValueError):
        expand_image_tokens(prompt, [3])
    assert render_image_state("<state>\nhi\n</state>\n", 0) == "<state>\nhi\n</state>\n"


def test_bilinear_taps_are_a_partition_of_unity():
    idx, wts = bilinear_inputs(np.array([[1, 8, 6]]), 48)
    assert idx.shape == wts.shape == (4, 48) and idx.dtype == np.int64 and wts.dtype == np.float32
    np.testing.assert_allclose(wts.sum(0), 1.0, atol=1e-6)
    assert idx.min() >= 0 and idx.max() < 48 * 48


def test_vision_position_ids_cover_the_grid_in_merge_block_order():
    pos = vision_position_ids(np.array([[1, 4, 4]]))
    assert pos.shape == (16, 2)
    assert pos[:4].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]                 # the first 2x2 merge block
    assert sorted(map(tuple, pos.tolist())) == [(r, c) for r in range(4) for c in range(4)]


def test_mrope_positions_text_only_advance_together():
    pos = position_ids([5, 6, 7, 8], np.zeros((0, 3), dtype=np.int64))
    assert pos.shape == (3, 4) and (pos == np.arange(4)).all()


def test_mrope_positions_with_an_image_use_three_axes():
    ids = [1] + [IMAGE_TOKEN_ID] * 4 + [2]
    pos = position_ids(ids, np.array([[1, 4, 4]]))
    assert pos.shape == (3, 6)
    assert pos[0, 1:5].tolist() == [1, 1, 1, 1]                                 # the temporal axis stays put inside the image
    assert pos[1, 1:5].tolist() == [1, 1, 2, 2] and pos[2, 1:5].tolist() == [1, 2, 1, 2]
    assert pos[:, 5].tolist() == [3, 3, 3]                                      # the text after it resumes past the image
