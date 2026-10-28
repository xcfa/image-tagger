# image-tagger

A Waifu Diffusion tagger (SmilingWolf's wd-* ONNX models) behind HTTP, in Docker. Python 3.13, stdlib HTTP server,
numpy + Pillow + onnxruntime only (no web framework, no huggingface_hub: the download is urllib). README.md (Russian) is
the user documentation and the API contract; code comments are English (say why, not what).

## Map
- `tagger/config.py` — settings from the environment (`TAGGER_*`, `HF_*`, `*_FILE` secrets), `TAGGER_MODELS` parsing
- `tagger/download.py` — `Downloader`: files from `<HF_ENDPOINT>/<repo>/resolve/<rev>/<file>` into
  `<models>/<owner>/<repo>/`, hash check, `.source.json` marker, the token never sent to another host
- `tagger/model.py` — `Tagger`: preprocessing (white square, BGR, NHWC), thresholds; `decode` (4 formats, 50 MP)
- `tagger/preferences.py` — thresholds saved by the page (`<models>/settings.json`)
- `tagger/server.py` — routes, key check, `State` (models loading in the background); `tagger/static/index.html` — the page
- `tagger/__main__.py` — start: serve at once (503), download, load; exit 1 on failure, 2 on bad settings
- `tests/` (pytest) — `conftest.py`: a tiny real ONNX model, `FakeHub`; `test_main.py` runs `python -m tagger`

## Rules
- The wire is Shizuka's tagger protocol (`X-Threshold`, `{"tags": [...]}`, 415/422 per image): keep it compatible; a
  change goes to README «API», and to Shizuka (`tools/image-tagger`, its docs) when it touches what Shizuka sends or reads.
- Every call but `GET /health` and `GET /` needs the key; the key comparison is constant-time.
- Never skip or loosen a test. `ruff check . && ruff format --check . && python -m pytest -q` before a push.
- Package versions are pinned in `requirements*.txt`; the image installs `requirements.txt` only.
