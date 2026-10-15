"""Startup configuration: required keys, finite limits, consistency, no silent typos."""

from pathlib import Path

import pytest

from app.config import Config, ConfigError

KEY = "k" * 20


def test_defaults_with_one_key():
    cfg = Config.from_env({"GIPFORMER_API_KEYS": KEY})
    assert cfg.api_keys == (KEY,) and cfg.whisper_alias is False
    assert cfg.queue_max_buffered_samples == int(cfg.queue_max_buffered_seconds * 16000)
    assert KEY not in repr(cfg)


def test_parses_typed_values():
    cfg = Config.from_env({
        "GIPFORMER_API_KEYS": f" {KEY} , {'j' * 20} ", "GIPFORMER_WHISPER_ALIAS": "true", "GIPFORMER_QUEUE_MAX_JOBS": "8",
        "GIPFORMER_MAX_CHUNK_SECONDS": "20.5", "GIPFORMER_MODELS_DIR": "/srv/models", "GIPFORMER_LOG_LEVEL": "debug",
        "GIPFORMER_SPOOL_DIR": "/var/spool/gipformer",
    })
    assert cfg.api_keys == (KEY, "j" * 20) and cfg.whisper_alias is True and cfg.queue_max_jobs == 8
    assert cfg.max_chunk_seconds == 20.5 and cfg.models_dir == Path("/srv/models") and cfg.log_level == "DEBUG"
    assert cfg.max_chunk_samples == 328000 and cfg.spool_dir == Path("/var/spool/gipformer")


def test_api_keys_file(tmp_path):
    f = tmp_path / "keys"
    f.write_text(f"# comment\n{KEY}\n\n{'j' * 20}\n")
    assert Config.from_env({"GIPFORMER_API_KEYS_FILE": str(f)}).api_keys == (KEY, "j" * 20)
    with pytest.raises(ConfigError):
        Config.from_env({"GIPFORMER_API_KEYS_FILE": str(tmp_path / "missing")})


@pytest.mark.parametrize("env", [
    {},  # no public default key
    {"GIPFORMER_API_KEYS": " , "},
    {"GIPFORMER_API_KEYS": "short"},
    {"GIPFORMER_API_KEYS": "é" * 20},
    {"GIPFORMER_API_KEYS": "界" * 20},
    {"GIPFORMER_API_KEYS": "a" * 19 + "\u007f"},
    {"GIPFORMER_API_KEYS": f"{KEY},{KEY}"},
    {"GIPFORMER_API_KEYS": KEY, "GIPFORMER_API_KEYS_FILE": "/x"},
    {"GIPFORMER_API_KEYS": KEY, "HF_TOKEN": KEY},  # gateway key must not be the HF token
    {"GIPFORMER_API_KEYS": KEY, "GIPFORMER_QUEUE_MAX_JOB": "4"},  # typo
    {"GIPFORMER_API_KEYS": KEY, "QUAN_QUEUE_MAX_JOBS": "4"},  # leftover pre-rename variable
    {"GIPFORMER_API_KEYS": KEY, "QUAN_GPU": "0"},
    {"GIPFORMER_API_KEYS": KEY, "GIPFORMER_QUEUE_MAX_JOBS": "many"},
    {"GIPFORMER_API_KEYS": KEY, "GIPFORMER_QUEUE_MAX_JOBS": "0"},
    {"GIPFORMER_API_KEYS": KEY, "GIPFORMER_MAX_AUDIO_SECONDS": "inf"},
    {"GIPFORMER_API_KEYS": KEY, "GIPFORMER_MAX_AUDIO_SECONDS": "nan"},
    {"GIPFORMER_API_KEYS": KEY, "GIPFORMER_DECODE_TIMEOUT_SECONDS": "-1"},
    {"GIPFORMER_API_KEYS": KEY, "GIPFORMER_WHISPER_ALIAS": "maybe"},
    {"GIPFORMER_API_KEYS": KEY, "GIPFORMER_LOG_LEVEL": "TRACE"},
    {"GIPFORMER_API_KEYS": KEY, "GIPFORMER_MAX_CHUNK_SECONDS": "60", "GIPFORMER_QUEUE_MAX_BUFFERED_SECONDS": "30"},
    {"GIPFORMER_API_KEYS": KEY, "GIPFORMER_MAX_TURN_SECONDS": "1000", "GIPFORMER_QUEUE_MAX_BUFFERED_SECONDS": "900"},
    {"GIPFORMER_API_KEYS": KEY, "GIPFORMER_QUEUE_MAX_JOBS_PER_LANE": "10", "GIPFORMER_QUEUE_MAX_JOBS": "5"},
    {"GIPFORMER_API_KEYS": KEY, "GIPFORMER_MAX_SESSIONS": "40"},
    {"GIPFORMER_API_KEYS": KEY, "GIPFORMER_MAX_CHUNK_SECONDS": "0.5"},  # splitter needs room above the short-clip floor
    {"GIPFORMER_API_KEYS": KEY, "GIPFORMER_MAX_DECODED_SECONDS": "100"},  # a max-length file could never be held
    {"GIPFORMER_API_KEYS": KEY, "GIPFORMER_MAX_UPLOAD_BYTES": "1000", "GIPFORMER_MAX_SPOOL_BYTES": "999"},
])
def test_invalid_config_fails_startup(env):
    with pytest.raises(ConfigError) as e:
        Config.from_env(env)
    assert KEY not in str(e.value)
