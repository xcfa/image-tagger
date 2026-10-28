"""Start: read the settings, answer HTTP at once (503 until ready), get each model from Hugging Face into the models
folder (once: a mounted volume keeps them), load it. A model that cannot be got or loaded stops the service with the
reason in the log; a container restart policy tries again."""

from __future__ import annotations

import signal
import sys
import threading

from . import config
from .download import Downloader, DownloadError
from .model import Tagger
from .preferences import Preferences
from .server import State, serve


def log(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    try:
        settings = config.load()
    except config.ConfigError as error:
        log(f"settings: {error}")
        return 2

    state = State(specs=settings.models)
    preferences = Preferences(settings.models_dir, settings.threshold, settings.character_threshold)
    server = serve(settings, state, preferences)
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    log(f"listening on :{settings.port}; models: {', '.join(f'{m.name} ({m.repo_id}@{m.revision})' for m in settings.models)}")
    if not settings.hf_token:
        log("no HF_TOKEN: public models only, at the hub's anonymous rate")

    downloader = Downloader(settings.hf_endpoint, settings.hf_token, settings.models_dir, log)
    for spec in settings.models:
        try:
            folder = downloader.ensure(spec, update=settings.update)
            tagger = Tagger.load(folder, settings.threads)
        except DownloadError as error:
            state.failed(str(error))
            log(str(error))
            return 1
        except Exception as error:  # noqa: BLE001 — onnxruntime or the CSV: say which model and why
            state.failed(f"{spec.name}: cannot load the model: {error}")
            log(state.error or "")
            return 1
        state.ready(spec.name, tagger)
        log(f"{spec.name}: ready ({len(tagger.tags)} tags, {tagger.size} px)")

    log("ready")
    threading.Event().wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
