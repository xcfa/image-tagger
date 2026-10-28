"""Gets a model's files from Hugging Face into the models folder, once.

A model is `model.onnx` and `selected_tags.csv` of a repository (`resolve/<revision>/<file>`, the way the hub serves
them). They are written next to a `.part` file and renamed when whole and checked: a large (LFS) file against the
sha256 the hub names in `X-Linked-Etag`, a small one against its git blob hash (`ETag`). A folder that already holds
both files is used as is, so a mounted volume survives the container; `update` asks the hub whether they changed.
The token goes only to the hub itself: a redirect to the file storage (another host) is followed without it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from .config import ModelSpec

FILES = ("model.onnx", "selected_tags.csv")
MARKER = ".source.json"
ATTEMPTS = 3
CHUNK = 1024 * 1024

Log = Callable[[str], None]


class DownloadError(RuntimeError):
    """The model could not be got: the service does not start (or keeps what it has, when updating)."""


@dataclass(frozen=True)
class Remote:
    """What the hub says of a file before it is fetched."""

    etag: str | None
    size: int | None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001 (urllib's signature)
        return None


class _StripAuthOnHostChange(urllib.request.HTTPRedirectHandler):
    """Follows redirects, but the token stays with the host it was given to."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


class Downloader:
    def __init__(self, endpoint: str, token: str | None, models_dir: str, log: Log = print, timeout: float = 60) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._token = token
        self._dir = models_dir
        self._log = log
        self._timeout = timeout
        self._head = urllib.request.build_opener(_NoRedirect)
        self._get = urllib.request.build_opener(_StripAuthOnHostChange)

    def folder(self, model: ModelSpec) -> str:
        return os.path.join(self._dir, *model.folder.split("/"))

    def ensure(self, model: ModelSpec, update: bool = False) -> str:
        """The folder with the model's files, fetched when missing (or changed on the hub, with `update`)."""
        folder = self.folder(model)
        os.makedirs(folder, exist_ok=True)
        source = self._read_marker(folder)
        same_source = source.get("repo") == model.repo_id and source.get("revision") == model.revision
        have = same_source and all(os.path.isfile(os.path.join(folder, name)) for name in FILES)
        if have and not update:
            self._log(f"{model.name}: {model.repo_id}@{model.revision} is in {folder}")
            return folder

        etags = dict(source.get("etags") or {}) if same_source else {}
        for name in FILES:
            path = os.path.join(folder, name)
            try:
                remote = self._remote(model, name)
            except DownloadError:
                if have:  # updating, and the hub cannot be reached: what is there still works
                    self._log(f"{model.name}: cannot ask the hub about {name}, keeping the copy in {folder}")
                    continue
                raise
            if os.path.isfile(path) and remote.etag and etags.get(name) == remote.etag:
                continue
            self._fetch(model, name, path, remote)
            etags[name] = remote.etag
            self._write_marker(folder, {"repo": model.repo_id, "revision": model.revision, "etags": etags})
        self._write_marker(folder, {"repo": model.repo_id, "revision": model.revision, "etags": etags})
        return folder

    # ── the hub ──────────────────────────────────────────────────────────

    def _url(self, model: ModelSpec, name: str) -> str:
        repo = "/".join(urllib.parse.quote(part, safe="") for part in model.repo_id.split("/"))
        revision = urllib.parse.quote(model.revision, safe="")
        return f"{self._endpoint}/{repo}/resolve/{revision}/{name}"

    def _request(self, url: str, method: str = "GET") -> urllib.request.Request:
        # HF_ENDPOINT is checked to be http(s) at start; nothing else builds these addresses.
        request = urllib.request.Request(url, method=method, headers={"User-Agent": "image-tagger/1"})  # noqa: S310
        if self._token:
            request.add_header("Authorization", f"Bearer {self._token}")
        return request

    def _remote(self, model: ModelSpec, name: str) -> Remote:
        """HEAD without following the redirect: the hub's own answer names the file's hash and size."""
        url = self._url(model, name)
        try:
            response = self._head.open(self._request(url, "HEAD"), timeout=self._timeout)
            headers = response.headers
            response.close()
        except urllib.error.HTTPError as error:
            if error.code not in (301, 302, 303, 307, 308):
                raise DownloadError(self._explain(model, name, error.code)) from error
            headers = error.headers
        except (urllib.error.URLError, OSError) as error:
            raise DownloadError(f"{model.name}: cannot reach {self._endpoint}: {getattr(error, 'reason', error)}") from error
        etag = headers.get("X-Linked-Etag") or headers.get("ETag")
        size = headers.get("X-Linked-Size") or (headers.get("Content-Length") if not headers.get("Location") else None)
        etag = etag.strip().removeprefix("W/").strip('"') if etag else None
        return Remote(etag=etag or None, size=int(size) if size and size.isdigit() else None)

    def _explain(self, model: ModelSpec, name: str, status: int) -> str:
        where = f"{model.repo_id}@{model.revision}/{name}"
        if status == 401:
            return f"{model.name}: {where} needs a token (HF_TOKEN) or the token is wrong (401)."
        if status == 403:
            return f"{model.name}: {where} is gated — accept its terms on Hugging Face with the token's account (403)."
        if status == 404:
            return f"{model.name}: {where} does not exist (404): check TAGGER_MODELS."
        return f"{model.name}: the hub answered {status} for {where}."

    def _fetch(self, model: ModelSpec, name: str, path: str, remote: Remote) -> None:
        part = path + ".part"
        for attempt in range(1, ATTEMPTS + 1):
            try:
                self._download(model, name, part, remote)
                os.replace(part, path)
                return
            except DownloadError:
                _remove(part)
                raise
            except (urllib.error.URLError, OSError, TimeoutError) as error:
                _remove(part)
                if attempt == ATTEMPTS:
                    raise DownloadError(f"{model.name}: downloading {name} failed: {getattr(error, 'reason', error)}") from error
                self._log(f"{model.name}: downloading {name} failed ({getattr(error, 'reason', error)}), again in {2**attempt} s")
                time.sleep(2**attempt)

    def _download(self, model: ModelSpec, name: str, part: str, remote: Remote) -> None:
        try:
            response = self._get.open(self._request(self._url(model, name)), timeout=self._timeout)
        except urllib.error.HTTPError as error:
            if error.code >= 500:
                raise
            raise DownloadError(self._explain(model, name, error.code)) from error
        total = remote.size or _int(response.headers.get("Content-Length"))
        sha256, sha1 = hashlib.sha256(), hashlib.sha1()  # noqa: S324 — git's blob hash, not security
        if total is not None:
            sha1.update(f"blob {total}\0".encode())
        self._log(f"{model.name}: downloading {name}" + (f" ({total / 1e6:.0f} MB)" if total else ""))
        done, shown, started = 0, -1, time.monotonic()
        with response, open(part, "wb") as file:
            while chunk := response.read(CHUNK):
                file.write(chunk)
                sha256.update(chunk)
                sha1.update(chunk)
                done += len(chunk)
                if total and (percent := done * 100 // total // 10 * 10) > shown:
                    shown = percent
                    self._log(f"{model.name}: {name} {percent}% ({time.monotonic() - started:.0f} s)")
            file.flush()
            os.fsync(file.fileno())
        if total is not None and done != total:
            raise OSError(f"got {done} bytes of {total}")
        self._verify(model, name, remote.etag, sha256.hexdigest(), sha1.hexdigest() if total is not None else None)

    def _verify(self, model: ModelSpec, name: str, etag: str | None, sha256: str, sha1: str | None) -> None:
        if not etag:
            return
        if re.fullmatch(r"[0-9a-f]{64}", etag) and etag != sha256:
            raise DownloadError(f"{model.name}: {name} came broken: sha256 {sha256}, the hub says {etag}.")
        if re.fullmatch(r"[0-9a-f]{40}", etag) and sha1 is not None and etag != sha1:
            raise DownloadError(f"{model.name}: {name} came broken: git hash {sha1}, the hub says {etag}.")

    # ── the marker ───────────────────────────────────────────────────────

    @staticmethod
    def _read_marker(folder: str) -> dict:
        try:
            with open(os.path.join(folder, MARKER), encoding="utf-8") as file:
                value = json.load(file)
                return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _write_marker(folder: str, value: dict) -> None:
        path = os.path.join(folder, MARKER)
        with open(path + ".part", "w", encoding="utf-8") as file:
            json.dump(value, file, indent=2)
        os.replace(path + ".part", path)


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _int(value: str | None) -> int | None:
    return int(value) if value and value.isdigit() else None
