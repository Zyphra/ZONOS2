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
