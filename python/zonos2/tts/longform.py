"""Helpers for long-form (windowed teacher-forced) TTS generation.

Long text is split into word-bounded chunks. Each chunk is synthesized as a
literal autoregressive continuation of earlier audio codes, which are fed back as
an acoustic prefix (VALL-E style). Because a chunk continues the exact frames
placed immediately before it, the per-chunk sheared outputs concatenate into one
continuous sheared stream, so a single shear_up + decode reconstructs correct
cross-codebook delay at every internal boundary.

Two windowing strategies are supported:

* Rolling window (legacy): the prefix is the last ``window_chunks - 1`` chunks'
  trimmed audio. The only acoustic reference is recent, so timbre can drift over
  long passages (see :func:`context_chunks`).
* Pinned-anchor window: the first accepted chunk is pinned (whole) into every
  prefix alongside a rolling tail of the last ``window_chunks - 1`` chunks, with
  the middle evicted (see :func:`windowed_context_indices`). This re-grounds
  every chunk on the same high-quality reference, preventing drift, while keeping
  the context bounded.

Both strategies feed only WHOLE chunks: each chunk's text is the complete
transcript of its audio, so the acoustic prefix stays aligned to the prompt text.
In both cases the window ends with the immediately preceding chunk's exact kept
tail, otherwise the concatenated stream stops being contiguous at the join.

These helpers are deliberately model-agnostic: callers pass the live model's
``text_vocab`` and already-resolved conditioning, so the same code drives both the
offline ``TTSLLM`` API and the server tokenizer worker.
"""

from __future__ import annotations

import re
from typing import List, Sequence

import torch

from .prompt import TTSPromptBuilder

# Split after ASCII sentence enders followed by whitespace, and immediately after
# CJK full-width enders (which usually have no trailing space). Optional closing
# quotes/brackets stay with the sentence.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?…])["\'”’)\]]*\s+|(?<=[。！？])')


def context_chunks(window_chunks: int) -> int:
    """Number of previous chunks to carry as teacher-forced context.

    ``window_chunks`` is the total number of chunks fed to the model per step:
    the one new chunk being generated plus the preceding context chunks. So the
    teacher-forced context is ``window_chunks - 1``.

    ``window_chunks=2`` (the default) carries one chunk of context (the original
    behavior); ``window_chunks=1`` disables teacher forcing (each chunk generated
    independently); ``window_chunks=3`` carries two chunks of context.
    """
    if window_chunks < 1:
        raise ValueError("window_chunks must be >= 1.")
    return window_chunks - 1


def split_text_words(text: str, max_chars: int = 150) -> List[str]:
    """Split text into chunks of at most ``max_chars``, never splitting a word.

    Greedily packs whitespace-separated words. A single word longer than
    ``max_chars`` becomes its own (oversized) chunk. ``" ".join(chunks)`` is
    equal to the input with runs of whitespace collapsed to single spaces.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")

    words = text.split()
    chunks: List[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


def split_text_sentences(text: str, max_chars: int = 150) -> List[str]:
    """Split text into chunks of at most ``max_chars``, preferring sentence boundaries.

    Greedily packs whole sentences up to ``max_chars``. A single sentence longer
    than ``max_chars`` falls back to word-level splitting (see
    :func:`split_text_words`), so chunk boundaries land on sentence ends whenever
    possible and never split a word.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            # Sentence too long to ever fit: flush, then word-split it.
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_text_words(sentence, max_chars))
        elif not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def split_text(text: str, max_chars: int = 150, mode: str = "word") -> List[str]:
    """Split text into <=``max_chars`` chunks using the given mode.

    ``mode="word"`` packs words greedily; ``mode="sentence"`` packs whole
    sentences (falling back to word splitting for over-long sentences).
    """
    if mode == "word":
        return split_text_words(text, max_chars)
    if mode == "sentence":
        return split_text_sentences(text, max_chars)
    raise ValueError(f"Unknown split mode {mode!r}; expected 'word' or 'sentence'.")


def trim_chunk_codes(
    frames: Sequence[Sequence[int]],
    eos_frame: int | None,
    *,
    drop_trailing_frames: int = 0,
) -> List[List[int]]:
    """Trim a chunk's generated frames to its content.

    ``frames`` are the raw sheared output frames (``[cb0..cb_{n-1}]`` each).
    Slices to ``eos_frame`` (or keeps everything when ``eos_frame is None``, i.e.
    the model hit ``max_tokens`` without emitting EOA), then drops a few extra
    trailing frames to remove conditioned trailing silence before a continuation.

    The SAME returned list must be used both as the kept audio segment AND as the
    next chunk's prefix, otherwise the concatenated stream stops being contiguous.
    """
    end = len(frames) if eos_frame is None else max(0, int(eos_frame))
    end = max(0, min(end, len(frames)) - max(0, int(drop_trailing_frames)))
    return [list(frame) for frame in frames[:end]]


def windowed_context_indices(
    cur_index: int,
    *,
    context_chunks: int,
    pin_anchor: bool,
) -> List[int]:
    """Indices of prior chunks to feed as the window for chunk ``cur_index``.

    Returns a rolling tail of the last ``context_chunks`` chunks. When
    ``pin_anchor`` is set and chunk 0 has already scrolled out of that tail, the
    fixed anchor (chunk 0) is prepended and the middle is evicted.

    Only WHOLE chunks are ever returned: every chunk's text is the complete
    transcript of its audio, so the acoustic prefix stays aligned to the prompt
    text (a truncated chunk would leave un-realized text that the model would try
    to finish, re-uttering it on every step). The returned indices always end
    with ``cur_index - 1`` (when non-empty), so the next chunk continues from the
    immediately preceding chunk's tail and the concatenated stream stays seamless.

    Examples (``context_chunks=1``, ``pin_anchor=True``): chunk 1 -> ``[0]``;
    chunk 2 -> ``[0, 1]``; chunk 5 -> ``[0, 4]`` (chunks 1-3 evicted).
    """
    recent_start = max(0, cur_index - context_chunks)
    idx = list(range(recent_start, cur_index))
    if pin_anchor and recent_start > 0:
        return [0] + idx
    return idx


def build_continuation_prompt(
    prompt_builder: TTSPromptBuilder,
    context_text: str,
    cur_text: str,
    context_codes: Sequence[Sequence[int]],
    *,
    text_vocab: int,
    speaking_rate_bucket: int | None = None,
    quality_buckets: Sequence[int | None] | None = None,
    speaker_slot: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a pre-tokenized continuation prompt for a non-first chunk.

    Layout (rows of width ``n_codebooks + 1``):
      [optional speaker slot]
      text rows for ``context_text + " " + cur_text``  (BOS..EOS, no silence)
      audio-prefix rows = each frame of ``context_codes`` + text-padding col

    ``context_text`` is the concatenation of the previous ``context`` chunks'
    text and ``context_codes`` is the concatenation of those chunks' trimmed
    audio frames (see :func:`context_chunks`). Including the context text makes
    the teacher-forced audio align to visible text so the model continues into
    ``cur_text``. ``quality_buckets`` is the already-resolved per-feature list
    (pass trailing-silence disabled for non-final chunks).
    """
    text_rows = prompt_builder.build_text_prompt(
        f"{context_text} {cur_text}",
        speaking_rate_bucket=speaking_rate_bucket,
        quality_buckets=quality_buckets,
    )

    parts: List[torch.Tensor] = []
    if speaker_slot is not None:
        parts.append(speaker_slot.to(dtype=text_rows.dtype))
    parts.append(text_rows)

    if len(context_codes) > 0:
        audio = torch.tensor(
            [list(frame) for frame in context_codes], dtype=text_rows.dtype
        )
        text_col = torch.full((audio.shape[0], 1), int(text_vocab), dtype=text_rows.dtype)
        parts.append(torch.cat([audio, text_col], dim=1))

    return torch.cat(parts, dim=0)
