import json
import threading
import urllib.error
import urllib.request

import pytest

from conftest import TAGS_CSV, png
from tagger.config import ModelSpec, Settings
from tagger.model import Tagger
from tagger.preferences import Preferences
from tagger.server import State, serve

KEY = "test-key-0123456789"


@pytest.fixture
def app(tmp_path, model_bytes):
    """A running server with two models, "wd" ready and "eva" still loading."""
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "model.onnx").write_bytes(model_bytes)
    (tmp_path / "m" / "selected_tags.csv").write_text(TAGS_CSV, encoding="utf-8")
    settings = Settings(
        models=[ModelSpec("wd", "SmilingWolf/wd-test"), ModelSpec("eva", "SmilingWolf/eva-test")],
        models_dir=str(tmp_path),
        port=0,
        api_key=KEY,
        threshold=0.5,
        character_threshold=0.5,
    )
    state = State(specs=settings.models)
    state.ready("wd", Tagger.load(str(tmp_path / "m")))
    server = serve(settings, state, Preferences(str(tmp_path), settings.threshold, settings.character_threshold))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    class App:
        url = f"http://127.0.0.1:{server.server_address[1]}"

        def __init__(self) -> None:
            self.state = state
            self.folder = tmp_path

        def call(self, method, path, body=None, headers=None, key=KEY):
            request = urllib.request.Request(self.url + path, data=body, method=method, headers=headers or {})
            if key:
                request.add_header("Authorization", f"Bearer {key}")
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    return response.status, response.headers, response.read()
            except urllib.error.HTTPError as error:
                return error.code, error.headers, error.read()

        def json(self, method, path, body=None, headers=None, key=KEY):
            status, _, data = self.call(method, path, body, headers, key)
            return status, json.loads(data)

    yield App()
    server.shutdown()
    server.server_close()


RED = png((255, 0, 0), (8, 8))
IMAGE = {"Content-Type": "image/png"}


def test_tags_come_from_the_first_model_or_the_one_named(app):
    status, body = app.json("POST", "/tag", RED, IMAGE)
    assert status == 200
    assert body["model"] == "wd"
    assert [t["name"] for t in body["tags"]] == ["red_theme", "ren_(oc)", "explicit"]
    assert (body["threshold"], body["characterThreshold"]) == (0.5, 0.5)

    assert app.json("POST", "/", RED, IMAGE)[1]["model"] == "wd"
    assert app.json("POST", "/tag/WD", RED, IMAGE)[1]["model"] == "wd"


def test_a_request_may_name_its_thresholds(app):
    status, body = app.json("POST", "/tag", RED, {**IMAGE, "X-Threshold": "0.999", "X-Character-Threshold": "0.6"})
    assert status == 200
    assert [t["name"] for t in body["tags"]] == ["ren_(oc)", "explicit"]
    assert (body["threshold"], body["characterThreshold"]) == (0.999, 0.6)


def test_every_call_but_health_and_the_page_needs_the_key(app):
    for method, path, body, headers in [
        ("POST", "/tag", RED, IMAGE),
        ("GET", "/models", None, None),
        ("GET", "/settings", None, None),
        ("PUT", "/settings", b"{}", {"Content-Type": "application/json"}),
    ]:
        assert app.call(method, path, body, headers, key=None)[0] == 401
        assert app.call(method, path, body, headers, key="wrong-key-0123456789")[0] == 401

    status, headers, page = app.call("GET", "/", key=None)
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"Image tagger" in page
    assert app.call("GET", "/health", key=None)[0] == 503  # eva is not ready yet


def test_a_model_still_loading_answers_503_and_health_follows(app):
    status, body = app.json("POST", "/tag/eva", RED, IMAGE)
    assert status == 503
    assert "still being downloaded" in body["error"]

    status, body = app.json("GET", "/models")
    assert [(m["name"], m["ready"], m["default"]) for m in body["models"]] == [("wd", True, True), ("eva", False, False)]
    assert body["models"][0]["size"] == 8

    app.state.ready("eva", app.state.taggers["wd"])
    assert app.json("GET", "/health", key=None) == (200, {"status": "ready"})


@pytest.mark.parametrize(
    "path, body, headers, status",
    [
        ("/tag/clip", RED, IMAGE, 404),
        ("/tag", RED, {"Content-Type": "image/bmp"}, 415),
        ("/tag", b"not an image", IMAGE, 422),
        ("/tag", b"", IMAGE, 400),
        ("/other", RED, IMAGE, 404),
    ],
)
def test_bad_requests_are_refused_with_a_reason(app, path, body, headers, status):
    code, answer = app.json("POST", path, body, headers)
    assert code == status
    assert answer["error"]


def test_the_thresholds_are_saved_and_used(app, tmp_path):
    assert app.json("GET", "/settings") == (200, {"threshold": 0.5, "characterThreshold": 0.5})

    status, body = app.json(
        "PUT", "/settings", json.dumps({"characterThreshold": 0.99}).encode(), {"Content-Type": "application/json"}
    )
    assert (status, body) == (200, {"threshold": 0.5, "characterThreshold": 0.99})
    assert [t["name"] for t in app.json("POST", "/tag", RED, IMAGE)[1]["tags"]] == ["red_theme", "explicit"]

    # They live in the models folder: a new container reads them back.
    assert Preferences(str(tmp_path), 0.35, 0.85).get() == {"threshold": 0.5, "characterThreshold": 0.99}

    for bad in [b"[1]", b"not json", b'{"threshold": 2}', b'{"threshold": true}', b'{"color": 1}']:
        assert app.call("PUT", "/settings", bad, {"Content-Type": "application/json"})[0] == 400
