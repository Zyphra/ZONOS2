from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from zonos2.message.tts import TTSSamplingParams


@dataclass
class TTSSequence:
    """Sequence tracking for TTS generation with EOS detection.

    Token format: Each token is a list of [cb0, cb1, ..., cb8, text_token]
    (unpacked format with n_codebooks audio codes + 1 text token)

    EOS Detection:
    - When EOS token (eoa_id) is detected in any delayed codebook,
      we record the aligned eos_frame and start a countdown.
    - The countdown allows n_codebooks + 1 additional steps for frame alignment.
    - Sequence is finished when countdown reaches 0 after EOS detection.
    """

    prompt_ids: List[List[int]]
    sampling_params: TTSSamplingParams
    seq_id: int = 0
    token_ids: List[List[int]] = field(default_factory=list)
    eos_frame: Optional[int] = None
    eos_countdown: int = 0
    n_codebooks: int = 9
    eoa_id: int = 1024

    def __post_init__(self):
        # Initialize token_ids with prompt
        self.token_ids = list(self.prompt_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self) -> List[List[int]]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def last_token(self) -> List[int]:
        return self.token_ids[-1] if self.token_ids else []

    @property
    def is_finished(self) -> bool:
        """Check if sequence is finished.

        A sequence is finished when:
        1. EOS has been detected (eos_frame is set)
        2. The countdown has reached 0
        3. Or max_tokens limit is reached
        """
        if self.num_completion_tokens >= self.sampling_params.max_tokens:
            return True
        if self.sampling_params.ignore_eos:
            return False
        return self.eos_frame is not None and self.eos_countdown <= 0

    def append_token(self, token: List[int]) -> None:
        """Append a token and check for EOS."""
        self.token_ids.append(token)
        self._check_eos(token)

    def _check_eos(self, token: List[int]) -> None:
        """Check if EOS is detected in the delayed audio codebooks."""
        if self.eos_frame is None:
            eos_cols = [token[i] == self.eoa_id for i in range(self.n_codebooks)]
            if any(eos_cols):
                step = self.num_completion_tokens - 1
                max_eos_cb = max(i for i, is_eos in enumerate(eos_cols) if is_eos)
                self.eos_frame = max(0, step - max_eos_cb)
                self.eos_countdown = self.n_codebooks + 1

        if self.eos_frame is not None and self.eos_countdown > 0:
            self.eos_countdown -= 1

    def __len__(self) -> int:
        return len(self.token_ids)


_seq_counter = 0


def create_sequence(
    prompt_ids: List[List[int]],
    sampling_params: TTSSamplingParams,
    n_codebooks: int = 9,
    eoa_id: int = 1024,
) -> TTSSequence:
    """Factory function to create a new TTS sequence with unique ID."""
    global _seq_counter
    seq = TTSSequence(
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
        seq_id=_seq_counter,
        n_codebooks=n_codebooks,
        eoa_id=eoa_id,
    )
    _seq_counter += 1
    return seq


def reset_seq_counter() -> None:
    """Reset the sequence counter (useful for testing)."""
    global _seq_counter
    _seq_counter = 0
