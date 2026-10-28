"""The service as Docker runs it: `python -m tagger` against a fake hub, from an empty models folder and again."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from conftest import png

KEY = "main-test-key-0123456789"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(hub, models_dir, port):
    env = {
        **os.environ,
        "TAGGER_API_KEY": KEY,
        "TAGGER_PORT": str(port),
        "TAGGER_MODELS": "wd=SmilingWolf/wd-test-tagger-v3",
        "TAGGER_MODELS_DIR": str(models_dir),
        "HF_ENDPOINT": hub.endpoint,
        "HF_TOKEN": "hf_token",
        "PYTHONUNBUFFERED": "1",
    }
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.Popen(
        [sys.executable, "-m", "tagger"], cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


def wait_ready(port, process, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"exited {process.returncode}: {process.stdout.read()}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.2)
    raise AssertionError("not ready in time")


def tag(port):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/tag",
        data=png((255, 0, 0), (8, 8)),
        headers={"Content-Type": "image/png", "Authorization": f"Bearer {KEY}", "X-Threshold": "0.5"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def test_it_downloads_once_then_starts_from_the_volume(hub, tmp_path, model_bytes):
    hub.add_model("SmilingWolf/wd-test-tagger-v3", model_bytes)
    for run in range(2):
        port = free_port()
        process = start(hub, tmp_path, port)
        try:
            wait_ready(port, process)
            assert "red_theme" in [t["name"] for t in tag(port)["tags"]]
        finally:
            process.terminate()
            process.wait(10)
        if run == 0:
            fetched = len(hub.requests)
            assert fetched > 0
    assert len(hub.requests) == fetched  # the second start asked the hub nothing


def test_a_missing_model_stops_the_service_with_the_reason(hub, tmp_path):
    process = start(hub, tmp_path, free_port())
    out, _ = process.communicate(timeout=30)
    assert process.returncode == 1
    assert "does not exist (404)" in out


def test_no_key_no_start(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TAGGER_")}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run([sys.executable, "-m", "tagger"], cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "TAGGER_API_KEY is required" in result.stdout


def test_a_models_folder_it_cannot_write_says_so(hub, tmp_path, model_bytes):
    hub.add_model("SmilingWolf/wd-test-tagger-v3", model_bytes)
    blocked = tmp_path / "models"
    blocked.write_text("a file where the folder should be")
    process = start(hub, blocked, free_port())
    out, _ = process.communicate(timeout=30)
    assert process.returncode == 1
    assert f"cannot write to {blocked}" in out
