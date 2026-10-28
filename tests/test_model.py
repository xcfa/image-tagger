import io

import numpy as np
import pytest
from PIL import Image

from conftest import TAGS_CSV, png
from tagger.model import ImageError, Tagger, decode


@pytest.fixture
def tagger(tmp_path, model_bytes):
    (tmp_path / "model.onnx").write_bytes(model_bytes)
    (tmp_path / "selected_tags.csv").write_text(TAGS_CSV, encoding="utf-8")
    return Tagger.load(str(tmp_path))


def test_the_picture_goes_in_square_on_white_as_bgr(tagger):
    batch = tagger.prepare(Image.new("RGBA", (16, 8), (255, 0, 0, 255)))

    assert batch.shape == (1, 8, 8, 3)
    assert batch.dtype == np.float32
    # Red in RGB is the last channel in BGR; the rows above and below the picture are white.
    assert batch[0, 4, 4].tolist() == [0.0, 0.0, 255.0]
    assert batch[0, 0, 4].tolist() == [255.0, 255.0, 255.0]


def test_transparency_is_white(tagger):
    batch = tagger.prepare(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
    assert np.all(batch == 255.0)


def test_tags_are_the_likeliest_rating_and_what_passes_the_thresholds(tagger):
    red = tagger.tag(png((255, 0, 0), (8, 8)), threshold=0.5, character_threshold=0.5)
    assert [(t["name"], t["category"]) for t in red] == [
        ("red_theme", "general"),
        ("ren_(oc)", "character"),
        ("explicit", "rating"),
    ]
    assert all(0 <= t["score"] <= 1 for t in red)

    # A character needs its own, higher threshold; the rating comes whatever its score.
    strict = tagger.tag(png((255, 0, 0), (8, 8)), threshold=0.5, character_threshold=0.99)
    assert [t["name"] for t in strict] == ["red_theme", "explicit"]

    white = tagger.tag(png((255, 255, 255), (8, 8)), threshold=0.5, character_threshold=0.5)
    assert [(t["name"], t["category"]) for t in white] == [("general", "rating")]


def test_only_pictures_of_the_four_formats_are_read():
    with pytest.raises(ImageError):
        decode(b"not an image")
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, "BMP")
    with pytest.raises(ImageError):
        decode(buffer.getvalue())
    assert decode(png((1, 2, 3))).size == (16, 10)
