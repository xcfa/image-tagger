import json
import os

import pytest

from conftest import TAGS_CSV
from tagger.config import ModelSpec
from tagger.download import MARKER, Downloader, DownloadError

SPEC = ModelSpec("wd", "SmilingWolf/wd-test-tagger-v3")


def downloader(hub, tmp_path, token="hf_secret"):
    return Downloader(hub.endpoint, token, str(tmp_path), log=lambda _: None, timeout=5)


def test_the_files_come_once_into_the_models_folder(hub, tmp_path, model_bytes):
    hub.add_model(SPEC.repo_id, model_bytes)

    folder = downloader(hub, tmp_path).ensure(SPEC)

    assert folder == str(tmp_path / "SmilingWolf" / "wd-test-tagger-v3")
    assert open(os.path.join(folder, "model.onnx"), "rb").read() == model_bytes
    assert open(os.path.join(folder, "selected_tags.csv"), encoding="utf-8").read() == TAGS_CSV
    assert not [name for name in os.listdir(folder) if name.endswith(".part")]
    marker = json.load(open(os.path.join(folder, MARKER), encoding="utf-8"))
    assert (marker["repo"], marker["revision"]) == (SPEC.repo_id, "main")

    # A second start (a new container on the same volume) asks the hub nothing.
    asked = len(hub.requests)
    assert downloader(hub, tmp_path).ensure(SPEC) == folder
    assert len(hub.requests) == asked


def test_the_token_goes_to_the_hub_and_not_to_the_storage(hub, tmp_path, model_bytes):
    hub.add_model(SPEC.repo_id, model_bytes)
    downloader(hub, tmp_path).ensure(SPEC)

    to_hub = [auth for host, _, path, auth in hub.requests if host == "localhost"]
    to_storage = [auth for host, _, path, auth in hub.requests if path.startswith("/storage/")]
    assert to_hub and set(to_hub) == {"Bearer hf_secret"}
    assert to_storage and set(to_storage) == {None}


def test_without_a_token_nothing_is_sent(hub, tmp_path, model_bytes):
    hub.add_model(SPEC.repo_id, model_bytes)
    downloader(hub, tmp_path, token=None).ensure(SPEC)
    assert {auth for *_, auth in hub.requests} == {None}


def test_update_fetches_only_what_changed(hub, tmp_path, model_bytes):
    hub.add_model(SPEC.repo_id, model_bytes)
    folder = downloader(hub, tmp_path).ensure(SPEC)

    hub.files[f"/{SPEC.repo_id}/resolve/main/selected_tags.csv"] = (TAGS_CSV + "5,blue_theme,0,1\n").encode()
    hub.requests.clear()
    downloader(hub, tmp_path).ensure(SPEC, update=True)

    gets = [path for _, method, path, _ in hub.requests if method == "GET"]
    assert gets == [f"/{SPEC.repo_id}/resolve/main/selected_tags.csv"]
    assert "blue_theme" in open(os.path.join(folder, "selected_tags.csv"), encoding="utf-8").read()


def test_update_keeps_the_copy_when_the_hub_is_away(hub, tmp_path, model_bytes):
    hub.add_model(SPEC.repo_id, model_bytes)
    folder = downloader(hub, tmp_path).ensure(SPEC)
    hub.close()

    assert downloader(hub, tmp_path).ensure(SPEC, update=True) == folder


def test_another_revision_is_fetched_again(hub, tmp_path, model_bytes):
    hub.add_model(SPEC.repo_id, model_bytes)
    hub.add_model(SPEC.repo_id, model_bytes, revision="v2")
    downloader(hub, tmp_path).ensure(SPEC)
    hub.requests.clear()

    downloader(hub, tmp_path).ensure(ModelSpec("wd", SPEC.repo_id, "v2"))

    assert any("/resolve/v2/model.onnx" in path for _, method, path, _ in hub.requests if method == "GET")


def test_a_broken_download_leaves_nothing(hub, tmp_path, model_bytes):
    hub.add_model(SPEC.repo_id, model_bytes)
    hub.corrupt = True

    with pytest.raises(DownloadError, match="came broken"):
        downloader(hub, tmp_path).ensure(SPEC)
    folder = tmp_path / "SmilingWolf" / "wd-test-tagger-v3"
    assert not (folder / "model.onnx").exists()
    assert not [name for name in os.listdir(folder) if name.endswith(".part")]


@pytest.mark.parametrize("status, message", [(401, "needs a token"), (403, "gated"), (404, "does not exist")])
def test_the_hub_refusing_says_why(hub, tmp_path, model_bytes, status, message):
    hub.add_model(SPEC.repo_id, model_bytes)
    hub.status[f"/{SPEC.repo_id}/resolve/main/model.onnx"] = status
    with pytest.raises(DownloadError, match=message):
        downloader(hub, tmp_path).ensure(SPEC)


def test_an_unreachable_hub_is_a_clear_error(tmp_path):
    with pytest.raises(DownloadError, match="cannot reach"):
        Downloader("http://127.0.0.1:9", None, str(tmp_path), log=lambda _: None, timeout=2).ensure(SPEC)
