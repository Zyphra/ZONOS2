from __future__ import annotations

import pytest
import torch
from zonos2.tts.longform import (
    build_continuation_prompt,
    context_chunks,
    split_text,
    split_text_sentences,
    split_text_words,
    trim_chunk_codes,
    windowed_context_indices,
)
from zonos2.tts.prompt import TTSPromptBuilder, TTSPromptConfig

N_CODEBOOKS = 9
AUDIO_PAD_ID = 1025
TEXT_VOCAB = 519


def _builder() -> TTSPromptBuilder:
    return TTSPromptBuilder(
        TTSPromptConfig(
            n_codebooks=N_CODEBOOKS,
            audio_pad_id=AUDIO_PAD_ID,
            text_vocab=TEXT_VOCAB,
            prepend_silence=True,
        )
    )


def test_split_respects_max_chars_and_words():
    text = "one two three four five six seven eight nine ten eleven twelve"
    chunks = split_text_words(text, max_chars=20)
    assert chunks  # non-empty
    assert all(len(c) <= 20 for c in chunks)
    # No word is split across chunks; whitespace collapses to single spaces.
    assert " ".join(chunks) == " ".join(text.split())


def test_split_oversized_word_becomes_own_chunk():
    word = "a" * 50
    chunks = split_text_words(f"hi {word} bye", max_chars=10)
    assert word in chunks


def test_split_single_chunk_when_short():
    assert split_text_words("hello world", max_chars=150) == ["hello world"]


def test_sentence_split_keeps_sentences_whole():
    text = "First short one. Second short one. Third short one. Fourth one here."
    chunks = split_text_sentences(text, max_chars=40)
    assert all(len(c) <= 40 for c in chunks)
    # Each chunk is composed of whole sentences (ends with sentence punctuation).
    assert all(c.rstrip().endswith((".", "!", "?")) for c in chunks)
    # Words and order preserved.
    assert " ".join(chunks).split() == text.split()


def test_sentence_split_packs_multiple_sentences():
    text = "A b. C d. E f."
    # All fit within 40 chars -> single chunk packing all sentences.
    assert split_text_sentences(text, max_chars=40) == ["A b. C d. E f."]


def test_sentence_split_falls_back_to_words_for_long_sentence():
    long_sentence = "word " * 30  # ~150 chars, no sentence breaks
    chunks = split_text_sentences(long_sentence.strip(), max_chars=40)
    assert len(chunks) > 1
    assert all(len(c) <= 40 for c in chunks)


def test_split_text_dispatch():
    text = "Alpha beta. Gamma delta."
    assert split_text(text, 40, "word") == split_text_words(text, 40)
    assert split_text(text, 40, "sentence") == split_text_sentences(text, 40)
    with pytest.raises(ValueError):
        split_text(text, 40, "paragraph")


def test_context_chunks_default_window_is_one():
    # window_chunks=2 -> one chunk of context (the original behavior)
    assert context_chunks(2) == 1


def test_context_chunks_window_one_disables():
    assert context_chunks(1) == 0


def test_context_chunks_larger_window():
    assert context_chunks(3) == 2
    assert context_chunks(4) == 3


def test_context_chunks_rejects_bad_window():
    with pytest.raises(ValueError):
        context_chunks(0)


def test_trim_slices_at_eos_and_drops_trailing():
    frames = [[i] * N_CODEBOOKS for i in range(10)]
    # eos_frame=8 keeps 0..7, drop_trailing_frames=2 removes 6,7 -> keep 0..5
    trimmed = trim_chunk_codes(frames, 8, drop_trailing_frames=2)
    assert trimmed == [[i] * N_CODEBOOKS for i in range(6)]


def test_trim_handles_none_eos():
    frames = [[i] * N_CODEBOOKS for i in range(5)]
    assert trim_chunk_codes(frames, None) == frames


def test_trim_is_identical_for_prefix_and_kept():
    frames = [[i] * N_CODEBOOKS for i in range(20)]
    a = trim_chunk_codes(frames, 15, drop_trailing_frames=3)
    b = trim_chunk_codes(frames, 15, drop_trailing_frames=3)
    assert a == b


def test_continuation_prompt_shape_and_columns():
    builder = _builder()
    prev_codes = [[7] * N_CODEBOOKS for _ in range(4)]
    text_rows = builder.build_text_prompt("hello world today")
    prompt = build_continuation_prompt(
        builder,
        "hello world",
        "today",
        prev_codes,
        text_vocab=TEXT_VOCAB,
    )
    assert prompt.shape == (text_rows.shape[0] + len(prev_codes), N_CODEBOOKS + 1)
    # The audio-prefix rows are the last len(prev_codes) rows.
    audio_rows = prompt[-len(prev_codes):]
    assert torch.all(audio_rows[:, :N_CODEBOOKS] == 7)
    assert torch.all(audio_rows[:, N_CODEBOOKS] == TEXT_VOCAB)


def test_continuation_prompt_with_speaker_slot():
    builder = _builder()
    slot = builder.speaker_slot()
    prev_codes = [[3] * N_CODEBOOKS for _ in range(2)]
    text_rows = builder.build_text_prompt("a b c")
    prompt = build_continuation_prompt(
        builder,
        "a b",
        "c",
        prev_codes,
        text_vocab=TEXT_VOCAB,
        speaker_slot=slot,
    )
    assert prompt.shape == (1 + text_rows.shape[0] + len(prev_codes), N_CODEBOOKS + 1)
    assert torch.equal(prompt[0], slot[0])


def test_continuation_prompt_empty_prefix():
    builder = _builder()
    text_rows = builder.build_text_prompt("a b c")
    prompt = build_continuation_prompt(
        builder, "a b", "c", [], text_vocab=TEXT_VOCAB
    )
    assert prompt.shape == (text_rows.shape[0], N_CODEBOOKS + 1)


def test_windowed_indices_first_continuation_is_just_prev():
    # chunk 1: rolling tail of 1; anchor (0) is the tail, no separate prepend.
    assert windowed_context_indices(1, context_chunks=1, pin_anchor=True) == [0]
    assert windowed_context_indices(1, context_chunks=1, pin_anchor=False) == [0]


def test_windowed_indices_pins_anchor_and_evicts_middle():
    # chunk 5, tail of 1: pin chunk 0 + the immediately preceding chunk 4.
    assert windowed_context_indices(5, context_chunks=1, pin_anchor=True) == [0, 4]
    # Without pinning: just the rolling tail.
    assert windowed_context_indices(5, context_chunks=1, pin_anchor=False) == [4]


def test_windowed_indices_always_end_with_prev_chunk():
    # Join invariant: the window ends with cur_index-1 so the next chunk
    # continues from the immediately preceding chunk's tail.
    for cur in range(1, 8):
        for ctx in (1, 2, 3):
            idx = windowed_context_indices(cur, context_chunks=ctx, pin_anchor=True)
            assert idx[-1] == cur - 1
            assert idx == sorted(idx)
            assert len(idx) == len(set(idx))  # no duplicate chunk


def test_windowed_indices_no_anchor_dup_while_in_tail():
    # While chunk 0 is still inside the rolling tail it is not also prepended.
    assert windowed_context_indices(2, context_chunks=2, pin_anchor=True) == [0, 1]
    assert windowed_context_indices(3, context_chunks=2, pin_anchor=True) == [0, 1, 2]
    assert windowed_context_indices(4, context_chunks=2, pin_anchor=True) == [0, 2, 3]


def test_windowed_indices_pin_only_when_no_recent_window():
    # context_chunks=0 (window_chunks=1) + pin -> anchor only.
    assert windowed_context_indices(3, context_chunks=0, pin_anchor=True) == [0]
    # No pin, no recent window -> fresh (empty).
    assert windowed_context_indices(3, context_chunks=0, pin_anchor=False) == []


def test_windowed_indices_first_chunk_is_empty():
    assert windowed_context_indices(0, context_chunks=1, pin_anchor=True) == []
