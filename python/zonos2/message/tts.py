from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from .utils import deserialize_type, serialize_type


@dataclass
class TTSSamplingParams:
    """Sampling parameters for TTS generation."""

    temperature: float = 1.15
    topk: int = 106
    top_p: float = 0.0
    min_p: float = 0.18
    max_tokens: int = 1024
    ignore_eos: bool = False
    repetition_window: int = 50
    repetition_penalty: float = 1.2
    repetition_codebooks: int = 8
    seed: int | None = None
    # Speaker-embedding classifier-free guidance scale. 1.0 disables guidance
    # (pure conditional). >1.0 pushes generation toward the target speaker by
    # running a paired unconditional (no speaker embedding) sequence and
    # combining logits: guided = uncond + cfg_scale * (cond - uncond).
    cfg_scale: float = 1.0
    # Acoustic-prefix classifier-free guidance scale for long-form continuation
    # chunks. 1.0 disables guidance. Any other value (including <1 / negative)
    # runs a paired branch that drops the acoustic prefix and context text
    # (keeping the speaker) and combines logits the same way:
    # guided = uncond + prefix_cfg_scale * (cond - uncond). >1 sharpens
    # continuity across chunk joins; <1 / negative downweights or opposes the
    # prefix. Has no effect on the first chunk (which has no prefix).
    prefix_cfg_scale: float = 1.0


@dataclass
class BaseTTSTokenizerMsg:
    """Base class for TTS tokenizer messages."""

    @staticmethod
    def encoder(msg: BaseTTSTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTTSTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTTSTokenizerMsg(BaseTTSTokenizerMsg):
    """Batch of TTS tokenizer messages."""

    data: List[BaseTTSTokenizerMsg]


@dataclass
class TTSTokenizeMsg(BaseTTSTokenizerMsg):
    """Request to build TTS prompt frames from text (Frontend -> Tokenizer)."""

    uid: int
    text: str
    sampling_params: TTSSamplingParams
    # Language code used for text normalization (e.g. "en_us").
    language: str = "en_us"
    # Run written->spoken text normalization before tokenization.
    text_normalization: bool = True
    # Optional speaker embedding for voice cloning; shape: (speaker_embedding_dim,)
    speaker_embedding: torch.Tensor | None = None
    # Token position in the prompt sequence where speaker embedding is injected.
    speaker_token_position: int = -1
    # Whether the speaker embedding should be marked as having a clean background.
    clean_speaker_background: bool = False
    # Whether to condition generation on the accurate-mode marker token
    # (off = expressive mode).
    accurate_mode: bool = True
    # Optional speaking-rate conditioning bucket. The tokenizer turns this into
    # one text-column token before normal text.
    speaking_rate_bucket: int | None = None
    # Optional per-feature quality bucket indices (aligned with the model's
    # configured quality features). Each becomes one text-column token.
    quality_buckets: List[int | None] | None = None
    # Long-form (windowed teacher-forced continuation) generation. None = auto:
    # engage when the normalized text exceeds long_form_chunk_chars. True/False
    # force it on/off. Text is split into <=long_form_chunk_chars-char chunks
    # (the new text per step); long_form_window_chunks is the total number of
    # chunks fed per step, so window-1 previous chunks are teacher-forced.
    # window_chunks=2 keeps one chunk of context; =1 disables teacher forcing.
    long_form: bool | None = None
    long_form_chunk_chars: int = 150
    long_form_window_chunks: int = 2
    # Pin the whole first chunk into every continuation prefix (alongside the
    # rolling window of long_form_window_chunks-1 recent chunks), evicting the
    # middle. Prevents timbre drift over long passages. Only whole chunks are fed
    # so the acoustic prefix stays aligned to the prompt text.
    long_form_pin_anchor: bool = True
    # How to split text into chunks: "word" (greedy word packing) or "sentence"
    # (pack whole sentences, falling back to words for over-long sentences).
    long_form_split_mode: str = "word"


@dataclass
class TTSDetokenizeMsg(BaseTTSTokenizerMsg):
    """TTS output frame to vocoder (Scheduler -> Tokenizer).

    Contains audio codes for one generated frame.
    """

    uid: int
    audio_codes: List[int]  # [cb0, cb1, ..., cb8] for one frame
    finished: bool
    eos_frame: int | None = None


@dataclass
class BaseTTSBackendMsg:
    """Base class for TTS backend messages."""

    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseTTSBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTTSBackendMsg(BaseTTSBackendMsg):
    """Batch of TTS backend messages."""

    data: List[BaseTTSBackendMsg]


@dataclass
class TTSUserMsg(BaseTTSBackendMsg):
    """Prompt-tokenized TTS request to scheduler (Tokenizer -> Scheduler).

    Contains 2D token tensor in unpacked format.
    """

    uid: int
    input_ids: torch.Tensor  # 2D tensor (seq_len, frame_width)
    sampling_params: TTSSamplingParams
    # Optional speaker embedding for voice cloning; shape: (speaker_embedding_dim,)
    speaker_embedding: torch.Tensor | None = None
    # Token position in the prompt sequence where speaker embedding is injected.
    speaker_token_position: int = -1
    # Whether the speaker embedding should be marked as having a clean background.
    clean_speaker_background: bool = False
    # Whether to condition generation on the accurate-mode marker token.
    accurate_mode: bool = True
    # Optional prefix-stripped prompt for prefix classifier-free guidance. When
    # set (and sampling_params.prefix_cfg_scale != 1.0), the scheduler builds the
    # unconditional CFG twin from this prompt (the "fresh chunk" prompt without
    # the acoustic prefix / context text) instead of a copy of input_ids, keeping
    # the speaker embedding on the twin. 2D tensor (seq_len, frame_width).
    cfg_uncond_input_ids: torch.Tensor | None = None


@dataclass
class BaseTTSFrontendMsg:
    """Base class for TTS frontend messages."""

    @staticmethod
    def encoder(msg: BaseTTSFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTTSFrontendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTTSFrontendMsg(BaseTTSFrontendMsg):
    """Batch of TTS frontend messages."""

    data: List[BaseTTSFrontendMsg]


@dataclass
class TTSAudioReply(BaseTTSFrontendMsg):
    """Audio chunk reply to frontend (Tokenizer -> Frontend).

    Contains PCM audio data for streaming playback.
    """

    uid: int
    audio_data: bytes  # PCM audio chunk (float32, 44.1kHz)
    finished: bool
    sample_rate: int = 44100
