"""Shared bits: a tiny real ONNX model shaped like the WD taggers, test images, a fake Hugging Face hub."""

from __future__ import annotations

import hashlib
import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

# The model's rows: rating general/explicit, a general tag, a character. Scores follow the image's mean colour:
# white → general rating, bright red → "red_theme" and the character.
TAGS_CSV = "tag_id,name,category,count\n1,general,9,100\n2,explicit,9,100\n3,red_theme,0,100\n4,ren_(oc),4,100\n"


def tiny_model(size: int = 8) -> bytes:
    """NHWC [1, size, size, 3] BGR 0..255 → sigmoid(mean colour · W + b), 4 scores: what the server expects of a WD model."""
    from onnx import TensorProto, helper, numpy_helper

    # Inputs are BGR: channel 2 is red. White (255,255,255) → general high; red (B=0,G=0,R=255) → red_theme, ren.
    weights = np.array(
        [
            [0.02, -0.02, -0.02, -0.02],  # blue
            [0.02, -0.02, -0.02, -0.02],  # green
            [-0.01, 0.0, 0.04, 0.04],
        ],  # red
        dtype=np.float32,
    )
    bias = np.array([-2.0, -4.0, -6.0, -6.5], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["input"], ["mean"], axes=[1, 2], keepdims=0),
            helper.make_node("MatMul", ["mean", "w"], ["z"]),
            helper.make_node("Add", ["z", "b"], ["zb"]),
            helper.make_node("Sigmoid", ["zb"], ["output"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", size, size, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 4])],
        initializer=[numpy_helper.from_array(weights, "w"), numpy_helper.from_array(bias, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString()


def png(colour: tuple[int, int, int, int] | tuple[int, int, int], size: tuple[int, int] = (16, 10)) -> bytes:
    mode = "RGBA" if len(colour) == 4 else "RGB"
    buffer = io.BytesIO()
    Image.new(mode, size, colour).save(buffer, "PNG")
    return buffer.getvalue()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()  # noqa: S324


class FakeHub:
    """Serves `files` (repo path → bytes) the way huggingface.co does: a large file (model.onnx) answers the hub's
    302 with X-Linked-Etag (its sha256) and X-Linked-Size to a storage URL on another host (127.0.0.1 vs localhost);
    a small one answers 200 with its git blob hash as ETag. Records every request with its Authorization."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.requests: list[tuple[str, str, str, str | None]] = []  # (host, method, path, authorization)
        self.status: dict[str, int] = {}  # path → status to answer instead
        self.fail_get: set[str] = set()  # paths whose HEAD answers but whose GET is 404
        self.moved: dict[str, str] = {}  # path → another path on the hub: 307, relative, no ETag (a renamed repo)
        self.corrupt = False
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def _serve(self, head: bool) -> None:
                host = self.headers.get("Host", "")
                hub.requests.append((host.split(":")[0], self.command, self.path, self.headers.get("Authorization")))
                if self.path.startswith("/storage/"):
                    data = hub.files[self.path[len("/storage") :]]
                    if hub.corrupt:
                        data = data[:-1] + bytes([data[-1] ^ 1])
                    return self._send(200, data, {}, head)
                if self.path in hub.moved:
                    return self._send(307, b"", {"Location": hub.moved[self.path]}, head)
                if not head and self.path in hub.fail_get:
                    return self._send(404, b"gone", {}, head)
                if self.path in hub.status:
                    return self._send(hub.status[self.path], b'{"error":"no"}', {}, head)
                data = hub.files.get(self.path)
                if data is None:
                    return self._send(404, b"not found", {}, head)
                if self.path.endswith(".onnx"):
                    storage = f"http://127.0.0.1:{hub.port}/storage{self.path}"
                    return self._send(
                        302,
                        b"",
                        {
                            "Location": storage,
                            "X-Linked-Etag": f'"{hashlib.sha256(data).hexdigest()}"',
                            "X-Linked-Size": str(len(data)),
                            "ETag": '"some-other-etag"',
                        },
                        head,
                    )
                return self._send(200, data, {"ETag": f'"{git_blob_sha1(data)}"'}, head)

            def _send(self, status: int, data: bytes, headers: dict[str, str], head: bool) -> None:
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if not head:
                    self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802
                self._serve(False)

            def do_HEAD(self) -> None:  # noqa: N802
                self._serve(True)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self.endpoint = f"http://localhost:{self.port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def add_model(self, repo: str, model: bytes, tags: str = TAGS_CSV, revision: str = "main") -> None:
        self.files[f"/{repo}/resolve/{revision}/model.onnx"] = model
        self.files[f"/{repo}/resolve/{revision}/selected_tags.csv"] = tags.encode()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def hub():
    fake = FakeHub()
    yield fake
    fake.close()


@pytest.fixture(scope="session")
def model_bytes() -> bytes:
    return tiny_model()
