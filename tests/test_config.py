import pytest

from tagger.config import ConfigError, ModelSpec, load, parse_models

KEY = "k" * 20


def test_models_take_an_owner_a_name_and_a_revision_or_their_defaults():
    assert parse_models("wd-swinv2-tagger-v3, eva=SmilingWolf/wd-eva02-large-tagger-v3@v1.0  other/My-Tagger") == [
        ModelSpec("wd-swinv2-tagger-v3", "SmilingWolf/wd-swinv2-tagger-v3", "main"),
        ModelSpec("eva", "SmilingWolf/wd-eva02-large-tagger-v3", "v1.0"),
        ModelSpec("my-tagger", "other/My-Tagger", "main"),
    ]


@pytest.mark.parametrize("value", ["", " , ", "a/b/c", "na!me=x", "x x=y", "wd, SmilingWolf/wd"])
def test_a_bad_model_list_stops_the_start(value):
    with pytest.raises(ConfigError):
        parse_models(value)


def test_defaults_and_secrets_from_files(tmp_path):
    key_file = tmp_path / "key"
    key_file.write_text(KEY + "\n")
    token_file = tmp_path / "token"
    token_file.write_text("hf_abc\n")

    settings = load({"TAGGER_API_KEY_FILE": str(key_file), "HF_TOKEN_FILE": str(token_file)})

    assert settings.api_key == KEY
    assert settings.hf_token == "hf_abc"
    assert [m.repo_id for m in settings.models] == ["SmilingWolf/wd-swinv2-tagger-v3"]
    assert (settings.models_dir, settings.port, settings.threshold, settings.character_threshold) == ("/models", 8000, 0.35, 0.85)
    assert settings.update is False


@pytest.mark.parametrize(
    "env, message",
    [
        ({}, "TAGGER_API_KEY is required"),
        ({"TAGGER_API_KEY": "short"}, "at least 16"),
        ({"TAGGER_API_KEY": KEY, "TAGGER_THRESHOLD": "2"}, "TAGGER_THRESHOLD"),
        ({"TAGGER_API_KEY": KEY, "TAGGER_PORT": "x"}, "TAGGER_PORT"),
        ({"TAGGER_API_KEY": KEY, "TAGGER_UPDATE": "maybe"}, "TAGGER_UPDATE"),
        ({"TAGGER_API_KEY": KEY, "HF_ENDPOINT": "ftp://x"}, "HF_ENDPOINT"),
        ({"TAGGER_API_KEY_FILE": "/no/such/file"}, "cannot read"),
    ],
)
def test_bad_settings_say_which(env, message):
    with pytest.raises(ConfigError, match=message):
        load(env)
