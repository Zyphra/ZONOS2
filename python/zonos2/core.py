from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from zonos2.attention import BaseAttnBackend, BaseAttnMetadata
    from zonos2.kvcache import BaseCacheHandle


@dataclass
class Context:
    page_size: int
    attn_backend: BaseAttnBackend
    _batch: TTSBatch | None = field(default=None, init=False)

    @property
    def batch(self) -> TTSBatch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: TTSBatch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX


# =============================================================================
# TTS-specific data structures
# =============================================================================


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


@dataclass(eq=False)
class TTSReq:
    """Request class for TTS generation with 2D token format.

    Tokens are in unpacked format: [cb0, cb1, ..., cb8, text_token] per frame.
    """

    input_ids: torch.Tensor  # 2D CPU tensor (seq_len, frame_width)
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: TTSSamplingParams
    cache_handle: BaseCacheHandle
    n_codebooks: int = 9
    eoa_id: int = 1024
    eos_frame: int = -1  # Aligned frame where EOS first appeared (-1 = not seen)
    eos_countdown: int = -1  # Steps remaining after EOS (-1 = not in countdown)
    total_generated: int = 0  # Total frames generated (for logging)
    rng: torch.Generator | None = None  # Per-request RNG for deterministic sampling
    speaker_embedding: torch.Tensor | None = None  # 1D CPU float32 tensor
    speaker_token_position: int = -1  # Injection position within the prompt sequence

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        assert self.input_ids.dim() == 2, "TTS input_ids must be 2D (seq_len, frame_width)"
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

        if self.speaker_embedding is not None:
            emb = self.speaker_embedding
            if emb.dim() == 2 and emb.shape[0] == 1:
                emb = emb.squeeze(0)
            if emb.dim() != 1:
                raise ValueError(
                    f"speaker_embedding must be 1D or (1, D), got shape {tuple(emb.shape)}"
                )
            self.speaker_embedding = emb.to(dtype=torch.float32, device="cpu")

        if self.speaker_token_position < 0:
            # Training convention: reserved speaker slot is at prompt position 0.
            self.speaker_token_position = 0
        if self.speaker_token_position >= self.device_len:
            self.speaker_token_position = 0

    @property
    def frame_width(self) -> int:
        """Number of elements per frame (n_codebooks + extras)."""
        return self.input_ids.shape[-1]

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    @property
    def num_completion_tokens(self) -> int:
        """Number of generated tokens (frames)."""
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1
        self.total_generated += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        """Append a single frame (unpacked token) to input_ids."""
        assert next_token.dim() == 1, "next_token must be 1D (frame_width,)"
        self.input_ids = torch.cat([self.input_ids, next_token.unsqueeze(0)], dim=0)

    def can_decode(self) -> bool:
        return self.remain_len > 0 and self.eos_countdown != 0

    def check_eos(self, audio_codes: List[int]) -> bool:
        """Check for EOS and update countdown state.

        Args:
            audio_codes: List of audio codebook values for one frame

        Returns:
            True if sequence is finished (countdown reached 0)
        """
        if self.sampling_params.ignore_eos:
            return False

        # Match Zonos2 reference inference: any sampled EOA codebook starts the
        # delayed stop countdown. The aligned frame is shifted back by the
        # highest EOA codebook index and clamped at zero.
        # Use total_generated because this request only sees one decode frame at a time.
        if self.eos_frame < 0:
            step = self.total_generated - 1
            eos_cols = [c == self.eoa_id for c in audio_codes[: self.n_codebooks]]
            if any(eos_cols):
                # First EOS: compute aligned frame
                max_eos_cb = max(i for i, is_eos in enumerate(eos_cols) if is_eos)
                self.eos_frame = max(0, step - max_eos_cb)
                self.eos_countdown = self.n_codebooks + 1

        # Decrement countdown
        if self.eos_countdown > 0:
            self.eos_countdown -= 1
            if self.eos_countdown == 0:
                return True

        return False

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len}, eos_frame={self.eos_frame})"
        )


@dataclass
class TTSBatch:
    """Batch of TTS requests with 2D token format."""

    reqs: List[TTSReq]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)  # (total_tokens, frame_width)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[TTSReq] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)
    # Optional per-batch speaker conditioning data (set by TTS scheduler).
    speaker_emb_values: torch.Tensor | None = field(default=None, init=False)
    speaker_token_positions: torch.Tensor | None = field(default=None, init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)
