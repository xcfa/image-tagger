"""A Waifu Diffusion tagger (SmilingWolf's wd-v1-4-* and wd-*-tagger-v3) on onnxruntime.

The models share a shape: one NHWC input of a square BGR picture as 0..255 floats (448 px for most), one output of
scores 0..1 per row of `selected_tags.csv` (`tag_id,name,category,count`; category 9 — rating, 4 — character, the rest
general).
"""

from __future__ import annotations

import csv
import io
import os
from typing import Protocol

import numpy as np
from PIL import Image

# The decoders tried, whatever the Content-Type says: no other Pillow plugin (EPS would call Ghostscript) sees the bytes.
FORMATS = ["JPEG", "PNG", "WEBP", "GIF"]
MAX_PIXELS = 50_000_000
# Pillow only warns between MAX_IMAGE_PIXELS and twice that; the size is checked by hand below, this stops the worst.
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

CATEGORIES = {9: "rating", 4: "character"}


class ImageError(ValueError):
    """The bytes are not a picture this service reads (or it is too large): the client's fault, 422."""


class Session(Protocol):
    """onnxruntime's InferenceSession, as far as the tagger uses it."""

    def get_inputs(self): ...

    def get_outputs(self): ...

    def run(self, output_names, input_feed): ...


class Tagger:
    def __init__(self, session: Session, tags: list[tuple[str, str]]) -> None:
        self._session = session
        self._input = session.get_inputs()[0]
        self.size = int(self._input.shape[1])
        self.tags = tags
        outputs = session.get_outputs()[0].shape[-1]
        if isinstance(outputs, int) and outputs != len(tags):
            raise ValueError(f"The model scores {outputs} tags, selected_tags.csv lists {len(tags)}.")

    @classmethod
    def load(cls, folder: str, threads: int = 0) -> Tagger:
        import onnxruntime as ort  # here, so the rest (and its tests) does not need it

        options = ort.SessionOptions()
        if threads > 0:
            options.intra_op_num_threads = threads
        session = ort.InferenceSession(os.path.join(folder, "model.onnx"), options, providers=["CPUExecutionProvider"])
        return cls(session, read_tags(os.path.join(folder, "selected_tags.csv")))

    def prepare(self, image: Image.Image) -> np.ndarray:
        """What the models were trained on: the picture on a white square, `size` pixels, BGR, 0..255 floats."""
        image = image.convert("RGBA")
        canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
        side = max(image.size)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        square = square.resize((self.size, self.size), Image.BICUBIC)
        array = np.asarray(square, dtype=np.float32)[:, :, ::-1]
        return np.ascontiguousarray(array)[np.newaxis, ...]

    def tag(self, data: bytes, threshold: float, character_threshold: float) -> list[dict]:
        """The tags of the image, best first: the most likely rating always, characters at or above
        `character_threshold`, the rest at or above `threshold`."""
        batch = self.prepare(decode(data))
        scores = self._session.run(None, {self._input.name: batch})[0][0]
        result = []
        ratings = [
            (float(score), name) for (name, category), score in zip(self.tags, scores, strict=True) if category == "rating"
        ]
        if ratings:
            score, name = max(ratings)
            result.append({"name": name, "category": "rating", "score": round(score, 3)})
        for (name, category), score in zip(self.tags, scores, strict=True):
            limit = character_threshold if category == "character" else threshold
            if category != "rating" and score >= limit:
                result.append({"name": name, "category": category, "score": round(float(score), 3)})
        return sorted(result, key=lambda t: (-t["score"], t["name"]))


def decode(data: bytes) -> Image.Image:
    """The first frame of a JPEG, PNG, WebP or GIF of at most MAX_PIXELS, loaded."""
    try:
        with Image.open(io.BytesIO(data), formats=FORMATS) as image:
            if image.width * image.height > MAX_PIXELS:
                raise ImageError(f"The image is {image.width}x{image.height}, more than {MAX_PIXELS} pixels.")
            image.seek(0)
            image.load()
            return image.copy()
    except ImageError:
        raise
    except Exception as error:  # noqa: BLE001 — a decoder of untrusted bytes fails in many ways (EOFError, struct.error…)
        # The details (library messages) go to the log, not to the client.
        print(f"decode: {type(error).__name__}: {error}"[:2000], flush=True)
        raise ImageError("The image cannot be decoded.") from error


def read_tags(path: str) -> list[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as file:
        return [(row["name"], CATEGORIES.get(int(row["category"]), "general")) for row in csv.DictReader(file)]
