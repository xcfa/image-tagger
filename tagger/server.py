"""The HTTP API (Shizuka's image tagger protocol; any client may use it) and the page.

Every call but `GET /health` and the page (`GET /`) needs `Authorization: Bearer <TAGGER_API_KEY>` (else 401).

- `POST /tag` — the first model; `POST /tag/<name>` — that one. The body is the image (Content-Type
  image/jpeg|png|webp|gif); `X-Threshold` and `X-Character-Threshold` (0..1) — the lowest scores kept, else the saved
  settings. 200: `{"model", "threshold", "characterThreshold", "tags": [{"name", "category", "score"}, …]}`, best
  first (the most likely rating always). 400/411/413 — a bad request, 415 — not an image type, 422 — the image cannot
  be decoded, 404 — no such model, 503 — the models are still being downloaded or loaded, 500 — the model failed.
- `GET /models` — the models, their repositories and state.
- `GET /settings`, `PUT /settings` `{"threshold", "characterThreshold"}` — the saved thresholds.
- `GET /health` — 200 when every model is ready, 503 before (Docker's healthcheck).
"""

from __future__ import annotations

import hmac
import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import unquote, urlsplit

from .config import ModelSpec, Settings
from .model import ImageError, Tagger
from .preferences import Preferences

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_BYTES = 20 * 1024 * 1024
MAX_JSON = 16 * 1024
PAGE = resources.files("tagger").joinpath("static/index.html").read_bytes()


@dataclass
class State:
    """The models and how far their start went; the loader fills it while the server already answers."""

    specs: list[ModelSpec]
    taggers: dict[str, Tagger] = field(default_factory=dict)
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def ready(self, name: str, tagger: Tagger) -> None:
        with self.lock:
            self.taggers[name] = tagger

    def failed(self, error: str) -> None:
        with self.lock:
            self.error = error

    @property
    def all_ready(self) -> bool:
        return len(self.taggers) == len(self.specs)


def make_handler(settings: Settings, state: State, preferences: Preferences) -> type[BaseHTTPRequestHandler]:
    workers = threading.BoundedSemaphore(settings.workers)
    by_name = {spec.name: spec for spec in state.specs}
    default = state.specs[0].name
    expected = f"Bearer {settings.api_key}".encode()

    class Handler(BaseHTTPRequestHandler):
        # A client that stops sending mid-body does not hold a thread for ever.
        timeout = 60
        server_version = "image-tagger/1"
        sys_version = ""

        def log_message(self, format: str, *args) -> None:  # noqa: A002 — http.server's name
            pass  # no line per request; failures are printed where they happen

        # ── answers ──────────────────────────────────────────────────────

        def _send(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def _answer(self, status: int, body: dict) -> None:
            self._send(status, json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _authorized(self) -> bool:
            if hmac.compare_digest(self.headers.get("Authorization", "").encode("utf-8"), expected):
                return True
            self._answer(401, {"error": "Wrong or missing key (Authorization: Bearer …)."})
            return False

        def _path(self) -> str:
            return urlsplit(self.path).path.rstrip("/") or "/"

        def _body(self, limit: int) -> bytes | None:
            """The request's body, at most `limit` bytes; None — an error was answered."""
            try:
                length = int(self.headers.get("Content-Length") or "")
            except ValueError:
                self._answer(411, {"error": "Content-Length is required."})
                return None
            if length <= 0:
                self._answer(400, {"error": "The body is empty."})
                return None
            if length > limit:
                self._answer(413, {"error": f"The body is larger than {limit // (1024 * 1024) or 1} MB."})
                return None
            data = self.rfile.read(length)
            if len(data) != length:
                self._answer(400, {"error": "The body was cut off."})
                return None
            return data

        # ── routes ───────────────────────────────────────────────────────

        def do_GET(self) -> None:  # noqa: N802 (http.server's name)
            path = self._path()
            if path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if path == "/health":
                if state.all_ready:
                    return self._answer(200, {"status": "ready"})
                return self._answer(503, {"status": "failed" if state.error else "starting", "error": state.error})
            if not self._authorized():
                return None
            if path == "/models":
                return self._answer(200, {"models": [self._model(spec) for spec in state.specs]})
            if path == "/settings":
                return self._answer(200, preferences.get())
            return self._answer(404, {"error": "Not found."})

        do_HEAD = do_GET  # noqa: N815

        def do_PUT(self) -> None:  # noqa: N802
            if not self._authorized():
                return None
            if self._path() != "/settings":
                return self._answer(404, {"error": "Not found."})
            data = self._body(MAX_JSON)
            if data is None:
                return None
            try:
                changes = json.loads(data)
                if not isinstance(changes, dict):
                    raise ValueError("The body must be a JSON object.")
                return self._answer(200, preferences.update(changes))
            except ValueError as error:
                return self._answer(400, {"error": str(error) if not isinstance(error, json.JSONDecodeError) else "Not JSON."})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return None
            path = self._path()
            if path in ("/", "/tag"):
                name = default
            elif path.startswith("/tag/"):
                name = unquote(path[len("/tag/") :]).lower()
            else:
                return self._answer(404, {"error": "Not found."})
            if name not in by_name:
                return self._answer(404, {"error": f"No model {name!r}; there are: {', '.join(by_name)}."})
            content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if content_type not in IMAGE_TYPES:
                return self._answer(415, {"error": f"Not an image: {content_type or 'no Content-Type'}."})
            image = self._body(MAX_BYTES)
            if image is None:
                return None
            tagger = state.taggers.get(name)
            if tagger is None:
                return self._answer(503, {"error": state.error or f"The model {name!r} is still being downloaded or loaded."})
            saved = preferences.get()
            threshold = _threshold(self.headers.get("X-Threshold"), saved["threshold"])
            character = _threshold(self.headers.get("X-Character-Threshold"), saved["characterThreshold"])
            try:
                with workers:
                    tags = tagger.tag(image, threshold, character)
            except ImageError as error:
                return self._answer(422, {"error": str(error)})
            except Exception as error:  # noqa: BLE001 — the model failed: an answer, not a dropped connection
                print(f"{name}: {type(error).__name__}: {error}"[:2000], flush=True)
                return self._answer(500, {"error": "The model failed on this image."})
            return self._answer(200, {"model": name, "threshold": threshold, "characterThreshold": character, "tags": tags})

        def _model(self, spec: ModelSpec) -> dict:
            tagger = state.taggers.get(spec.name)
            return {
                "name": spec.name,
                "repo": spec.repo_id,
                "revision": spec.revision,
                "default": spec.name == default,
                "ready": tagger is not None,
                **({"size": tagger.size, "tags": len(tagger.tags)} if tagger else {}),
            }

    return Handler


def _threshold(header: str | None, fallback: float) -> float:
    """A header's threshold, kept within 0.01..1 (0 would list every tag); missing or not a number — `fallback`."""
    try:
        value = float(header) if header else fallback
    except ValueError:
        value = fallback
    if value != value:  # NaN
        value = fallback
    return min(max(value, 0.01), 1.0)


def serve(settings: Settings, state: State, preferences: Preferences) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", settings.port), make_handler(settings, state, preferences))  # noqa: S104
    server.daemon_threads = True
    return server
