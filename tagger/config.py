"""Settings from the environment. Every secret also reads from `<NAME>_FILE` (Docker secrets)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import quote

DEFAULT_OWNER = "SmilingWolf"
DEFAULT_MODELS = "SmilingWolf/wd-swinv2-tagger-v3"

# name=owner/repo@revision, any part but the repository optional.
_MODEL = re.compile(
    r"^(?:(?P<name>[A-Za-z0-9][A-Za-z0-9._-]{0,63})=)?"
    r"(?:(?P<owner>[A-Za-z0-9][A-Za-z0-9._-]*)/)?(?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:@(?P<revision>[A-Za-z0-9._/-]+))?$"
)


class ConfigError(ValueError):
    """A setting that cannot be used: the service does not start."""


@dataclass(frozen=True)
class ModelSpec:
    """A model on Hugging Face: `name` is how a request picks it (`POST /tag/<name>`)."""

    name: str
    repo_id: str
    revision: str = "main"

    @property
    def folder(self) -> str:
        """Where it lives under the models folder: `owner/repo`, `owner/repo@revision` for a revision other than main
        (two revisions of one repository must not take turns overwriting one folder)."""
        if self.revision == "main":
            return self.repo_id
        return f"{self.repo_id}@{quote(self.revision, safe='')}"


@dataclass(frozen=True)
class Settings:
    models: list[ModelSpec]
    models_dir: str = "/models"
    hf_endpoint: str = "https://huggingface.co"
    hf_token: str | None = None
    update: bool = False
    port: int = 8000
    api_key: str = ""
    threshold: float = 0.35
    character_threshold: float = 0.85
    workers: int = 2
    threads: int = 0


def parse_models(value: str) -> list[ModelSpec]:
    """`TAGGER_MODELS`: comma- or space-separated `[name=][owner/]repo[@revision]`; the owner defaults to SmilingWolf,
    the name to the repository's (lower case), the revision to main. The first one answers `POST /tag`."""
    models: list[ModelSpec] = []
    for item in re.split(r"[\s,]+", value.strip()):
        if not item:
            continue
        match = _MODEL.match(item)
        if not match:
            raise ConfigError(f"TAGGER_MODELS: {item!r} is not [name=][owner/]repo[@revision].")
        repo_id = f"{match['owner'] or DEFAULT_OWNER}/{match['repo']}"
        name = (match["name"] or match["repo"]).lower()
        if any(m.name == name for m in models):
            raise ConfigError(f"TAGGER_MODELS: the name {name!r} is used twice; give one of them its own (name=repo).")
        models.append(ModelSpec(name=name, repo_id=repo_id, revision=match["revision"] or "main"))
    if not models:
        raise ConfigError("TAGGER_MODELS names no model.")
    return models


def _secret(env: dict[str, str], name: str) -> str:
    path = env.get(f"{name}_FILE")
    if path:
        try:
            with open(path, encoding="utf-8") as file:
                return file.read().strip()
        except OSError as error:
            raise ConfigError(f"{name}_FILE: cannot read {path}: {error.strerror}.") from error
    return env.get(name, "").strip()


def _number(env: dict[str, str], name: str, default: float, low: float, high: float, kind=float):
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = kind(raw)
    except ValueError as error:
        raise ConfigError(f"{name} must be a number, not {raw!r}.") from error
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}.")
    return value


def _flag(env: dict[str, str], name: str) -> bool:
    raw = env.get(name, "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    raise ConfigError(f"{name} must be true or false, not {raw!r}.")


def _api_key(env: dict[str, str]) -> str:
    key = _secret(env, "TAGGER_API_KEY")
    if not key:
        raise ConfigError("TAGGER_API_KEY is required: every call but /health and the page asks for it (Bearer).")
    if len(key) < 16:
        raise ConfigError("TAGGER_API_KEY must be at least 16 characters.")
    if any(ord(c) < 33 for c in key):
        raise ConfigError("TAGGER_API_KEY must not hold spaces or control characters.")
    return key


def load(env: dict[str, str] | None = None) -> Settings:
    env = dict(os.environ if env is None else env)
    endpoint = env.get("HF_ENDPOINT", "").strip().rstrip("/") or "https://huggingface.co"
    if not endpoint.startswith(("https://", "http://")):
        raise ConfigError("HF_ENDPOINT must be an http(s) address.")
    token = _secret(env, "HF_TOKEN") or None
    if token and any(ord(c) < 32 for c in token):
        raise ConfigError("HF_TOKEN must not hold control characters.")
    return Settings(
        models=parse_models(env.get("TAGGER_MODELS", "") or DEFAULT_MODELS),
        models_dir=env.get("TAGGER_MODELS_DIR", "").strip() or "/models",
        hf_endpoint=endpoint,
        hf_token=token,
        update=_flag(env, "TAGGER_UPDATE"),
        port=_number(env, "TAGGER_PORT", 8000, 1, 65535, int),
        api_key=_api_key(env),
        threshold=_number(env, "TAGGER_THRESHOLD", 0.35, 0.0, 1.0),
        character_threshold=_number(env, "TAGGER_CHARACTER_THRESHOLD", 0.85, 0.0, 1.0),
        workers=_number(env, "TAGGER_WORKERS", 2, 1, 64, int),
        threads=_number(env, "TAGGER_THREADS", 0, 0, 256, int),
    )
