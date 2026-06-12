from __future__ import annotations

from types import SimpleNamespace

import pytest
from zonos2.server import api_server
from zonos2.tts.prompt import (
    BYTE_TEXT_VOCAB_SIZE,
    speaking_rate_token_id,
    text_to_byte_ids,
    tokens_to_prompt_tokens,
)
from zonos2.utils.hf import cached_load_checkpoint_config


class _DummyModelConfig:
    speaking_rate_num_buckets = 4
    speaking_rate_buckets = ("0-10", "10-15", "15-20", "20+")


class _DummyConfig:
    model_config = _DummyModelConfig()
    tts_speaking_rate_num_buckets = 4
    tts_speaking_rate_buckets = _DummyModelConfig.speaking_rate_buckets


class _NoRateModelConfig:
    speaking_rate_num_buckets = 0
    speaking_rate_buckets = ()


class _NoRateConfig:
    model_config = _NoRateModelConfig()
    tts_speaking_rate_num_buckets = 0
    tts_speaking_rate_buckets = ()


class _MaxTokenConfig:
    max_seq_len = 4096


def test_speaking_rate_token_id_uses_training_offset():
    assert (
        speaking_rate_token_id(text_vocab=469, speaking_rate_num_buckets=21, speaking_rate_bucket=2)
        == 450
    )


def test_tokens_to_prompt_tokens_prepends_speaking_rate_frame():
    prompt = tokens_to_prompt_tokens(
        [7, 8],
        n_codebooks=2,
        audio_pad_id=99,
        text_vocab=469,
        speaking_rate_num_buckets=21,
        speaking_rate_bucket=2,
    )

    assert prompt == [
        [99, 99, 450],
        [99, 99, 7],
        [99, 99, 8],
    ]


def test_tokens_to_prompt_tokens_rejects_invalid_bucket():
    with pytest.raises(ValueError, match="speaking_rate_bucket"):
        tokens_to_prompt_tokens(
            [7],
            n_codebooks=2,
            audio_pad_id=99,
            text_vocab=469,
            speaking_rate_num_buckets=21,
            speaking_rate_bucket=21,
        )


def test_api_resolves_configured_speaking_rate_to_bucket():
    bucket = api_server._resolve_speaking_rate_bucket(
        _DummyConfig(),
        speaking_rate=16.0,
        speaking_rate_enabled=True,
    )

    assert bucket == 2


def test_api_resolves_exact_speaking_rate_bucket():
    assert api_server._resolve_speaking_rate_bucket(
        _DummyConfig(),
        speaking_rate_bucket=1,
        speaking_rate_enabled=True,
    ) == 1


def test_api_resolves_speed_around_neutral_rate():
    assert (
        api_server._resolve_speaking_rate_bucket(
            _DummyConfig(),
            speed=1.0,
            speaking_rate_enabled=True,
        )
        == 2
    )
    assert (
        api_server._resolve_speaking_rate_bucket(
            _DummyConfig(),
            speed=0.5,
            speaking_rate_enabled=True,
        )
        == 0
    )


def test_api_can_disable_speaking_rate_conditioning():
    assert api_server._resolve_speaking_rate_bucket(_DummyConfig(), speaking_rate_bucket=1) is None
    assert (
        api_server._resolve_speaking_rate_bucket(
            _DummyConfig(),
            speaking_rate_bucket=1,
            speaking_rate_enabled=False,
        )
        is None
    )
    assert (
        api_server._resolve_speaking_rate_bucket(
            _DummyConfig(),
            speed=1.0,
            speaking_rate_enabled=False,
        )
        is None
    )


def test_api_rejects_multiple_speaking_rate_controls():
    with pytest.raises(ValueError, match="only one"):
        api_server._resolve_speaking_rate_bucket(
            _DummyConfig(),
            speaking_rate_bucket=1,
            speed=1.0,
            speaking_rate_enabled=True,
        )


def test_api_ignores_speed_for_models_without_speaking_rate():
    assert (
        api_server._resolve_speaking_rate_bucket(
            _NoRateConfig(),
            speed=1.0,
            speaking_rate_enabled=True,
        )
        is None
    )


def test_api_rejects_explicit_rate_for_models_without_speaking_rate():
    with pytest.raises(ValueError, match="does not support"):
        api_server._resolve_speaking_rate_bucket(
            _NoRateConfig(),
            speaking_rate_bucket=0,
            speaking_rate_enabled=True,
        )


def test_field_was_set_handles_pydantic_v1_and_v2_names():
    assert api_server._field_was_set(SimpleNamespace(model_fields_set={"speed"}), "speed")
    assert api_server._field_was_set(SimpleNamespace(__fields_set__={"speed"}), "speed")


def test_tts_request_defaults_byte_mode_on_rate_bin_off():
    tts_req = api_server.TTSGenerateRequest(text="Hello")
    assert not hasattr(tts_req, "byte_tokenize_all")
    assert tts_req.language == "en_us"
    assert tts_req.text_normalization is True
    assert tts_req.speaking_rate_enabled is False
    assert tts_req.max_tokens is None

    speech_req = api_server.OpenAISpeechRequest(model="zonos2", input="Hello")
    assert not hasattr(speech_req, "byte_tokenize_all")
    assert speech_req.speaking_rate_enabled is False


def test_tts_request_language_normalization():
    assert api_server._normalize_tts_request_language("EN-US") == "en_us"
    assert api_server._normalize_tts_request_language("cmn") == "cmn"
    with pytest.raises(ValueError, match="Unsupported language"):
        api_server._normalize_tts_request_language("klingon")


def test_byte_tokenizer_uses_training_byte_offset():
    assert BYTE_TEXT_VOCAB_SIZE == 448
    assert text_to_byte_ids("Az") == [2, 192 + ord("A"), 192 + ord("z"), 3]


def test_tts_max_tokens_default_resolves_to_model_max():
    assert api_server._resolve_tts_max_tokens(_MaxTokenConfig(), None) == 4096
    assert api_server._resolve_tts_max_tokens(_MaxTokenConfig(), 9999) == 4096
    assert api_server._resolve_tts_max_tokens(_MaxTokenConfig(), 256) == 256

    with pytest.raises(ValueError, match="positive"):
        api_server._resolve_tts_max_tokens(_MaxTokenConfig(), 0)


def test_zonos2_sidecar_loads_speaking_rate_conditioning(tmp_path):
    (tmp_path / "config.yaml").write_text(
        """
model:
  n_layers: 1
  dim: 64
  head_dim: 32
  ffn_dim_multiplier: 1.0
  max_seqlen: 512
  text_vocab: 999
data:
  speaking_rate_enabled: true
  speaking_rate_buckets: ["0-10", "10+"]
""",
        encoding="utf-8",
    )

    cfg = cached_load_checkpoint_config(str(tmp_path))

    assert cfg.speaking_rate_num_buckets == 2


def test_zonos2_sidecar_preserves_checkpoint_text_vocab(tmp_path):
    (tmp_path / "config.yaml").write_text(
        """
model:
  n_layers: 1
  dim: 64
  head_dim: 32
  ffn_dim_multiplier: 1.0
  max_seqlen: 512
  text_vocab: 469
  speaking_rate_num_buckets: 2
data:
  speaking_rate_enabled: true
  speaking_rate_buckets: ["0-10", "10+"]
""",
        encoding="utf-8",
    )

    cfg = cached_load_checkpoint_config(str(tmp_path))

    assert cfg.text_vocab == 469
    assert cfg.speaking_rate_num_buckets == 2
