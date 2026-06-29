"""Weights-backed offline<->server parity (loads the real model; GPU).

Skipped unless ZONOS2_TEST_MODEL points at a checkpoint (e.g. "Zyphra/ZONOS2").
Run with:

    ZONOS2_TEST_MODEL=Zyphra/ZONOS2 CUDA_VISIBLE_DEVICES=2 \
        PYTHONPATH=.../ZONOS2-offline-paritiy/python \
        python -m pytest -o addopts="" tests/misc/test_offline_parity_gpu.py -q

Asserts the offline path builds the *same* model input_ids the server (tokenizer
worker + scheduler) builds, and that normalization / voice cloning actually take
effect end to end. Everything downstream of the front end is shared code, so
token-exact input == token-exact output by construction.
"""

from __future__ import annotations

import os

import pytest
import torch

MODEL = os.environ.get("ZONOS2_TEST_MODEL")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not MODEL, reason="set ZONOS2_TEST_MODEL to run (loads weights)"),
]

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VOICE = os.path.join(REPO, "default_voices", "AmericanFemale.mp3")
TEXT = "Call me at 123 Main Street."
SEED = 12345


@pytest.fixture(scope="module")
def tts():
    from zonos2.tts.llm import TTSLLM

    return TTSLLM(model_path=MODEL)


def _offline_input(tts, speaker_embedding, clean_bg=False, accurate=True):
    """Final model input the offline path feeds the scheduler."""
    from zonos2.message.tts import TTSSamplingParams, TTSUserMsg

    ids = tts._tokenize_one(TEXT, language="en_us", text_normalization=True)
    if speaker_embedding is None:
        return ids
    msg = TTSUserMsg(
        uid=0,
        input_ids=ids,
        sampling_params=TTSSamplingParams(),
        speaker_embedding=speaker_embedding,
        clean_speaker_background=clean_bg,
        accurate_mode=accurate,
    )
    return tts._with_speaker_frames(ids, msg, msg.speaker_token_position)


def _server_input(tts, speaker_embedding, clean_bg=False, accurate=True):
    """Final model input the server (tokenizer worker + scheduler) feeds.

    tokenizer/server.py: normalize -> build(text, quality_buckets=default), then
    prepend the speaker slot (position 0) when an embedding is present; the
    scheduler adds the background/accurate markers.
    """
    from zonos2.message.tts import TTSSamplingParams, TTSUserMsg

    norm = tts._text_normalizer.normalize(TEXT, "en_us")
    default_quality = tts._resolve_quality_buckets(None)
    ids = tts._prompt_builder.build(norm, quality_buckets=default_quality)
    pos = -1
    if speaker_embedding is not None:
        slot = tts._prompt_builder.speaker_slot(dtype=ids.dtype, device=ids.device)
        ids = torch.cat([slot, ids], dim=0)
        pos = 0
    if speaker_embedding is None:
        return ids
    msg = TTSUserMsg(
        uid=0,
        input_ids=ids,
        sampling_params=TTSSamplingParams(),
        speaker_embedding=speaker_embedding,
        clean_speaker_background=clean_bg,
        accurate_mode=accurate,
    )
    return tts._with_speaker_frames(ids, msg, pos)


def test_text_normalization_rewrites_digits(tts):
    norm = tts._text_normalizer.normalize(TEXT, "en_us")
    assert norm != TEXT and "123" not in norm


def test_embed_speaker_file_dim(tts):
    emb = tts.embed_speaker_file(VOICE)
    assert emb.numel() == tts.speaker_embedding_dim
    assert emb.dtype == torch.float32


@pytest.mark.parametrize(
    "clone, clean_bg, accurate",
    [
        (False, False, True),   # no voice
        (True, False, True),    # cloned, noisy background, accurate
        (True, True, True),     # cloned, clean background
        (True, False, False),   # cloned, expressive
    ],
)
def test_front_end_input_ids_match_server(tts, clone, clean_bg, accurate):
    emb = tts.embed_speaker_file(VOICE) if clone else None
    a = _offline_input(tts, emb, clean_bg=clean_bg, accurate=accurate)
    b = _server_input(tts, emb, clean_bg=clean_bg, accurate=accurate)
    assert a.shape == b.shape
    assert torch.equal(a, b)


def test_normalization_changes_generated_tokens(tts):
    from zonos2.message.tts import TTSSamplingParams

    norm = tts.generate_one(TEXT, TTSSamplingParams(seed=SEED), text_normalization=True)
    raw = tts.generate_one(TEXT, TTSSamplingParams(seed=SEED), text_normalization=False)
    assert norm["audio_tokens"] != raw["audio_tokens"]


def test_voice_cloning_changes_tokens_and_produces_audio(tts):
    from zonos2.message.tts import TTSSamplingParams

    emb = tts.embed_speaker_file(VOICE)
    plain = tts.generate_one(TEXT, TTSSamplingParams(seed=SEED))
    clone = tts.generate_one(TEXT, TTSSamplingParams(seed=SEED), speaker_embedding=emb)
    assert clone["audio_tokens"] != plain["audio_tokens"]
    assert clone["audio"] and len(clone["audio"]) > 0


# --------------------------------------------------------------------------- #
# Parameter-resolution helpers (server parity; need the real model config)
# --------------------------------------------------------------------------- #
def test_resolve_max_tokens_clamps(tts):
    limit = max(1, int(tts.engine.max_seq_len))
    assert tts.resolve_max_tokens(None) == limit
    assert tts.resolve_max_tokens(10) == min(10, limit)
    assert tts.resolve_max_tokens(10**9) == limit
    with pytest.raises(ValueError):
        tts.resolve_max_tokens(0)


def test_resolve_speaking_rate_bucket(tts):
    n = tts.speaking_rate_num_buckets
    if n <= 0:
        pytest.skip("model has no speaking-rate buckets")
    assert tts.resolve_speaking_rate_bucket(speaking_rate_bucket=0) == 0
    bucket = tts.resolve_speaking_rate_bucket(speed=1.0)
    assert isinstance(bucket, int) and 0 <= bucket < n
    with pytest.raises(ValueError):
        tts.resolve_speaking_rate_bucket(speaking_rate_bucket=0, speed=1.0)


def test_resolve_quality_buckets(tts):
    if sum(tts.quality_bucket_counts) <= 0:
        pytest.skip("model has no quality buckets")
    out = tts.resolve_quality_buckets(quality_values={tts.quality_features[0]: 0.0})
    assert isinstance(out, list)
    assert len(out) == len(tts.quality_features)
