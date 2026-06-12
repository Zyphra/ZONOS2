from __future__ import annotations

import pytest
from zonos2.tts.prompt import (
    accurate_mode_token_id,
    conditioned_text_vocab_size,
    quality_token_id,
    speaker_background_token_id,
    text_to_prompt_tokens,
)

# Layout of the released checkpoint: 448 byte vocab + 8 speaking-rate buckets
# + 60 quality buckets (12, 12, 12, 8, 8, 8) + 2 background markers
# + 1 accurate-mode marker = 519.
TEXT_VOCAB = 519
RATE_BUCKETS = 8
QUALITY_COUNTS = (12, 12, 12, 8, 8, 8)
BACKGROUND_BUCKETS = 2
ACCURATE_BUCKETS = 1


def test_conditioned_text_vocab_size_matches_release_model():
    assert (
        conditioned_text_vocab_size(
            RATE_BUCKETS, sum(QUALITY_COUNTS), BACKGROUND_BUCKETS, ACCURATE_BUCKETS
        )
        == TEXT_VOCAB
    )


def test_conditioning_token_ids_use_training_layout():
    assert (
        quality_token_id(
            TEXT_VOCAB, RATE_BUCKETS, QUALITY_COUNTS, 0, 0, BACKGROUND_BUCKETS, ACCURATE_BUCKETS
        )
        == 456
    )
    assert (
        quality_token_id(
            TEXT_VOCAB, RATE_BUCKETS, QUALITY_COUNTS, 5, 7, BACKGROUND_BUCKETS, ACCURATE_BUCKETS
        )
        == 515
    )
    assert (
        speaker_background_token_id(
            TEXT_VOCAB, RATE_BUCKETS, QUALITY_COUNTS, True, BACKGROUND_BUCKETS, ACCURATE_BUCKETS
        )
        == 516
    )
    assert (
        speaker_background_token_id(
            TEXT_VOCAB, RATE_BUCKETS, QUALITY_COUNTS, False, BACKGROUND_BUCKETS, ACCURATE_BUCKETS
        )
        == 517
    )
    assert (
        accurate_mode_token_id(
            TEXT_VOCAB, RATE_BUCKETS, QUALITY_COUNTS, BACKGROUND_BUCKETS, ACCURATE_BUCKETS
        )
        == 518
    )


def test_quality_rows_prepended_in_feature_order():
    prompt = text_to_prompt_tokens(
        "hi",
        n_codebooks=2,
        audio_pad_id=1025,
        text_vocab=TEXT_VOCAB,
        speaking_rate_num_buckets=RATE_BUCKETS,
        speaking_rate_bucket=3,
        quality_bucket_counts=QUALITY_COUNTS,
        quality_buckets=[None, 1, None, None, None, 4],
        speaker_background_num_buckets=BACKGROUND_BUCKETS,
        accurate_mode_num_buckets=ACCURATE_BUCKETS,
    )
    text_col = [row[-1] for row in prompt]
    assert text_col[0] == 448 + 3  # speaking-rate token
    assert text_col[1] == 456 + 12 + 1  # estimated_snr bucket 1
    assert text_col[2] == 456 + 12 + 12 + 12 + 8 + 8 + 4  # trailing_silence_s bucket 4
    # BOS follows the conditioning rows.
    assert text_col[3] == 2


def test_quality_bucket_out_of_range_rejected():
    with pytest.raises(ValueError):
        text_to_prompt_tokens(
            "hi",
            text_vocab=TEXT_VOCAB,
            speaking_rate_num_buckets=RATE_BUCKETS,
            quality_bucket_counts=QUALITY_COUNTS,
            quality_buckets=[12, None, None, None, None, None],
            speaker_background_num_buckets=BACKGROUND_BUCKETS,
            accurate_mode_num_buckets=ACCURATE_BUCKETS,
        )
