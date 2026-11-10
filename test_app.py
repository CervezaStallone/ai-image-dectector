#!/usr/bin/env python3
"""Minimal self-checks for app.py. Run: python test_app.py
No framework; asserts only. Covers the logic that broke before the fix.
"""
import os
import numpy as np
from PIL import Image

import app


def _solid(color=(120, 120, 120), size=(128, 128)):
    return Image.new("RGB", size, color)


def _noise(size=(128, 128)):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def test_ela_leaves_no_temp_file():
    before = set(os.listdir("."))
    app.ela_score(_noise())
    after = set(os.listdir("."))
    assert after == before, f"ela_score created files: {after - before}"
    assert not os.path.exists("temp_ela.jpg")


def test_chromatic_aberration_no_nan_on_flat_image():
    # Flat image -> channel edges are constant -> corrcoef would be NaN.
    val = app.chromatic_aberration_score(_solid())
    assert not np.isnan(val), "chromatic_aberration returned NaN on flat image"
    assert val == 1.0


def test_chromatic_aberration_finite_on_noise():
    val = app.chromatic_aberration_score(_noise())
    assert np.isfinite(val)
    assert -1.0 <= val <= 1.0


def test_metadata_keyword_word_boundary():
    # 'ai' must match as a word, not inside 'brain' / 'detail'.
    assert app._metadata_contains_keyword("made with ai tools", "ai")
    assert not app._metadata_contains_keyword("brain detail rain", "ai")
    assert app._metadata_contains_keyword("midJourney v6", "midjourney")


def test_load_model_none_without_checkpoint():
    # No checkpoint -> no model -> classifier excluded downstream.
    assert app.load_model(None) is None
    assert app.load_model("") is None


def test_metadata_missing_exif_is_safe():
    flag, data = app.check_metadata("test_app.py")  # not an image
    assert flag is False
    assert data is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
