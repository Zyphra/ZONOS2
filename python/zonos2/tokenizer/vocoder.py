"""TTS vocoder manager using DAC for audio decoding."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from zonos2.message import TTSDetokenizeMsg

# Global DAC model cache
_dac_model = None


def _get_dac():
    """Lazy load and cache the DAC 44kHz model."""
    global _dac_model
    if _dac_model is None:
        import dac as dac_module

        _dac_model = (
            dac_module.DAC.load(dac_module.utils.download(model_type="44khz"))
            .eval()
            .to("cuda")
        )
    return _dac_model


def shear_up(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Remove delay: column j shifted up by j rows.

    This is the inverse of shear() - it removes the delay pattern applied
    during generation to align all codebook outputs for decoding.

    Args:
        x: Input tensor of shape (..., H, W) where H is frames and W is codebooks
        pad_id: Padding value to fill empty positions

    Returns:
        De-sheared tensor of same shape
    """
    H, W = x.shape[-2:]
    out = x.new_full(x.shape, pad_id)
    for j in range(W):
        if H > j:
            out[..., : H - j, j] = x[..., j:, j]
    return out


def decode_dac(codes: torch.Tensor) -> torch.Tensor:
    """Decode audio codes using DAC.

    Args:
        codes: Tensor of shape (batch, seq_len, n_codebooks) with audio codes

    Returns:
        Audio tensor of shape (batch, num_samples) at 44.1kHz
    """
    dac = _get_dac()
    # Clamp codes to valid range
    codes = torch.clamp(codes, max=1023)
    # DAC expects (batch, codebooks, seq)
    codes = codes.permute(0, 2, 1)

    with torch.no_grad(), torch.inference_mode():
        z = dac.quantizer.from_codes(codes)[0]
        audio = dac.decode(z).float().squeeze(1).cpu()

    return audio


class TTSVocoderManager:
    """Converts audio codebook tokens to PCM audio using DAC.

    Uses streaming-compatible incremental decoding that correctly handles
    the shear pattern in multi-codebook audio. The model outputs tokens with
    a delay pattern where codebook j is delayed by j frames. shear_up removes
    this delay, but requires enough context (n_codebooks frames) to work
    correctly.

    For streaming: accumulates frames, waits for enough context, then decodes
    incrementally. At end of stream, decodes remaining frames with padding.
    """

    def __init__(
        self,
        n_codebooks: int = 9,
        audio_pad_id: int = 1025,
        min_decode_chunk: int = 16,
        sample_rate: int = 44100,
        overlap_frames: int = 4,
        hop_length: int = 512,
    ):
        """Initialize the vocoder manager.

        Args:
            n_codebooks: Number of audio codebooks
            audio_pad_id: Audio padding token ID (used in shear_up)
            min_decode_chunk: Minimum frames to decode at once (for efficiency)
            sample_rate: Output sample rate (44100 for DAC)
            overlap_frames: Number of frames to overlap between chunks for OLA
            hop_length: DAC audio samples per codebook frame
        """
        assert min_decode_chunk > overlap_frames, (
            f"min_decode_chunk ({min_decode_chunk}) must be > "
            f"overlap_frames ({overlap_frames})"
        )
        self.n_codebooks = n_codebooks
        self.audio_pad_id = audio_pad_id
        self.min_decode_chunk = min_decode_chunk
        self.sample_rate = sample_rate
        self.overlap_frames = overlap_frames
        self.hop_length = hop_length

        # Per-request state
        self._frame_buffers: Dict[int, List[List[int]]] = {}
        self._decoded_counts: Dict[int, int] = {}
        self._eos_frames: Dict[int, int] = {}
        self._overlap_tails: Dict[int, np.ndarray] = {}

        # Crossfade window cache
        self._window_cache: Dict[int, np.ndarray] = {}

    def decode_frames(self, msgs: List[TTSDetokenizeMsg]) -> List[bytes]:
        """Decode audio frames to PCM chunks with streaming support.

        Accumulates frames and decodes incrementally when enough context is
        available. The shear pattern requires n_codebooks-1 extra frames of
        context to correctly decode output frames.

        With N accumulated frames, can decode output frames 0 to N-9 completely.
        At end of stream, decodes remaining pre-EOS frames. Tail frames
        generated for delay alignment are kept as context but are sliced away
        before DAC decode.

        Args:
            msgs: List of TTS detokenize messages

        Returns:
            List of PCM audio bytes (float32, 44.1kHz), one per message
        """
        results = []
        for msg in msgs:
            uid = msg.uid

            # Initialize state if needed
            if uid not in self._frame_buffers:
                self._frame_buffers[uid] = []
                self._decoded_counts[uid] = 0
            if msg.eos_frame is not None:
                self._eos_frames[uid] = max(0, int(msg.eos_frame))

            # Accumulate frame
            self._frame_buffers[uid].append(msg.audio_codes)

            # Calculate how many complete frames we can decode
            total = len(self._frame_buffers[uid])
            # With N frames, can decode output frames 0 to N-(n_codebooks-1)-1
            complete_frames = total - (self.n_codebooks - 1)
            if uid in self._eos_frames:
                complete_frames = min(complete_frames, self._eos_frames[uid])
            new_decodable = complete_frames - self._decoded_counts[uid]

            should_decode = False
            if msg.finished:
                # End of stream: decode everything before EOS, then flush any
                # withheld overlap tail from the last non-final chunk.
                should_decode = (
                    complete_frames > self._decoded_counts[uid]
                    or uid in self._overlap_tails
                )
            elif new_decodable >= self.min_decode_chunk:
                # Streaming: batch decode when we have enough new frames
                should_decode = True

            if should_decode:
                audio = self._decode_incremental(uid, msg.finished)
                results.append(audio)
            else:
                results.append(b"")

            # Cleanup finished requests
            if msg.finished:
                self._frame_buffers.pop(uid, None)
                self._decoded_counts.pop(uid, None)
                self._eos_frames.pop(uid, None)
                self._overlap_tails.pop(uid, None)

        return results

    def _get_crossfade_window(self, length: int) -> np.ndarray:
        """Get a cached raised-cosine fade-in window of the given sample length."""
        if length not in self._window_cache:
            self._window_cache[length] = (
                0.5 * (1.0 - np.cos(np.linspace(0, np.pi, length)))
            ).astype(np.float32)
        return self._window_cache[length]

    def _decode_incremental(self, uid: int, is_final: bool) -> bytes:
        """Decode new frames incrementally with OLA crossfading.

        Re-decodes ``overlap_frames`` previously-decoded frames together with
        the new frames so DAC has continuous context.  The overlapping audio
        region is crossfaded with the withheld tail from the previous chunk.

        Each non-final chunk withholds its last ``overlap_frames * hop_length``
        samples (stored in ``_overlap_tails``).  These are crossfaded with the
        next chunk's head and output at that time.  The final chunk outputs all
        remaining audio including the tail.

        Args:
            uid: Request user ID
            is_final: Whether this is the final decode (end of stream)

        Returns:
            PCM audio bytes (float32)
        """
        buffer = self._frame_buffers[uid]
        decoded = self._decoded_counts[uid]
        total = len(buffer)

        if is_final:
            target_frames = total
        else:
            target_frames = total - (self.n_codebooks - 1)
        if uid in self._eos_frames:
            target_frames = min(target_frames, self._eos_frames[uid])

        if target_frames <= decoded:
            if is_final and uid in self._overlap_tails:
                return self._overlap_tails.pop(uid).tobytes()
            return b""

        # --- Determine overlap with previous chunk ---
        overlap = min(self.overlap_frames, decoded)

        # --- Raw slice including overlap context for shear_up ---
        decode_start = decoded - overlap
        raw_end = min(target_frames + self.n_codebooks - 1, total)
        raw_slice = buffer[decode_start:raw_end]

        # --- shear_up and take output frames ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        codes = torch.tensor(raw_slice, dtype=torch.int64, device=device)
        codes = shear_up(codes, self.audio_pad_id)
        output_count = target_frames - decode_start  # overlap + new frames
        codes = codes[:output_count].unsqueeze(0)

        # --- DAC decode ---
        audio = decode_dac(codes)
        audio_np = audio[0].numpy().astype(np.float32)

        # --- Crossfade overlap region with previous tail ---
        if overlap > 0 and uid in self._overlap_tails:
            prev_tail = self._overlap_tails[uid]
            overlap_samples = min(
                overlap * self.hop_length, len(prev_tail), len(audio_np)
            )
            if overlap_samples > 0:
                fade_in = self._get_crossfade_window(overlap_samples)
                audio_np[:overlap_samples] = (
                    (1.0 - fade_in) * prev_tail[-overlap_samples:]
                    + fade_in * audio_np[:overlap_samples]
                )

        # --- Withhold tail for next chunk's crossfade (unless final) ---
        if not is_final and self.overlap_frames > 0:
            tail_samples = min(self.overlap_frames * self.hop_length, len(audio_np))
            self._overlap_tails[uid] = audio_np[-tail_samples:].copy()
            output_audio = audio_np[:-tail_samples] if tail_samples < len(audio_np) else np.array([], dtype=np.float32)
        else:
            output_audio = audio_np
            self._overlap_tails.pop(uid, None)

        # --- Advance decoded count by new frames only ---
        self._decoded_counts[uid] = target_frames

        return output_audio.tobytes()

    def decode_all(
        self,
        codes: torch.Tensor,
        apply_shear_up: bool = True,
    ) -> torch.Tensor:
        """Decode a complete sequence of audio codes.

        Args:
            codes: Tensor of shape (batch, seq_len, n_codebooks)
            apply_shear_up: Whether to apply shear_up before decoding

        Returns:
            Audio tensor of shape (batch, num_samples)
        """
        if apply_shear_up:
            codes = shear_up(codes, self.audio_pad_id)

        codes = codes.to("cuda")
        return decode_dac(codes)

    def reset(self, uid: int | None = None):
        """Reset frame buffers and decode state.

        Args:
            uid: If provided, reset only this request's state. Otherwise reset all.
        """
        if uid is not None:
            self._frame_buffers.pop(uid, None)
            self._decoded_counts.pop(uid, None)
            self._eos_frames.pop(uid, None)
            self._overlap_tails.pop(uid, None)
        else:
            self._frame_buffers.clear()
            self._decoded_counts.clear()
            self._eos_frames.clear()
            self._overlap_tails.clear()
