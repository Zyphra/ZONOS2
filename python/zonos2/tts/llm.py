"""Offline TTSLLM class for batch TTS generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from zonos2.distributed import DistributedInfo
from zonos2.message import (
    BaseTTSBackendMsg,
    BatchTTSTokenizerMsg,
    TTSDetokenizeMsg,
    TTSUserMsg,
)
from zonos2.message.tts import TTSSamplingParams
from zonos2.scheduler import SchedulerConfig
from zonos2.scheduler.scheduler import TTSScheduler
from zonos2.tokenizer.vocoder import TTSVocoderManager, shear_up

from .prompt import TTSPromptBuilder, TTSPromptConfig


class RequestAllFinished(Exception):
    """Raised when all requests are finished."""

    pass


# Default quality conditioning, matching the server: trailing silence 0.25-0.5s.
DEFAULT_QUALITY_BUCKETS = {"trailing_silence_s": 3}


@dataclass
class TTSRequestStatus:
    """Status tracking for a TTS request."""

    uid: int
    input_ids: torch.Tensor  # 2D (seq_len, frame_width)
    output_frames: List[List[int]]  # List of audio code frames
    eos_frame: int | None = None
    finished: bool = False


class TTSLLM(TTSScheduler):
    """TTS-specific LLM interface for offline audio generation.

    This class provides a simplified interface for TTS generation with:
    - Text or pre-tokenized prompt input
    - Multi-codebook sampling with repetition penalty and top-k/top-p/min-p
    - EOS detection with frame alignment
    - Batch generation support
    - Optional audio decoding with DAC

    Example usage:
        tts = TTSLLM(model_path="/path/to/model")

        # Generate from text
        results = tts.generate(["Hello world"], TTSSamplingParams())

        # Results contain audio tokens and optionally decoded audio
        for r in results:
            print(r["audio_tokens"])  # List of frames
            if r["audio"]:
                # r["audio"] is PCM bytes at 44.1kHz
                pass
    """

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        n_codebooks: int | None = None,
        codebook_size: int | None = None,
        text_vocab: int | None = None,
        eoa_id: int | None = None,
        audio_pad_id: int | None = None,
        decode_audio: bool = True,
        **kwargs,
    ):
        """Initialize TTSLLM.

        Args:
            model_path: Path to the model checkpoint
            dtype: Data type for model (default: bfloat16)
            n_codebooks: Number of audio codebooks (optional, auto-detected from model)
            codebook_size: Size of each codebook vocabulary (optional, auto-detected from model)
            text_vocab: Text vocabulary size (optional, auto-detected from model)
            eoa_id: End-of-audio token ID (optional, auto-detected from model)
            audio_pad_id: Audio padding token ID (optional, auto-detected from model)
            decode_audio: Whether to decode audio using DAC (default: True)
            **kwargs: Additional arguments passed to scheduler
        """
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)

        # Override TTS settings only if explicitly provided (otherwise use model config values)
        if n_codebooks is not None:
            self.n_codebooks = n_codebooks
        if codebook_size is not None:
            self.codebook_size = codebook_size
        if eoa_id is not None:
            self.eoa_id = eoa_id
        if audio_pad_id is not None:
            self.audio_pad_id = audio_pad_id
        if text_vocab is not None:
            self.text_vocab = text_vocab

        self.decode_audio = decode_audio

        # Request tracking
        self.pending_requests: List[Tuple[torch.Tensor, TTSSamplingParams]] = []
        self.status_map: Dict[int, TTSRequestStatus] = {}
        self.counter = 0

        # Optional vocoder for audio decoding
        if decode_audio:
            self._vocoder = TTSVocoderManager(
                n_codebooks=n_codebooks,
                audio_pad_id=audio_pad_id,
            )
        else:
            self._vocoder = None
        self._prompt_builder = TTSPromptBuilder(
            TTSPromptConfig(
                n_codebooks=self.n_codebooks,
                audio_pad_id=self.audio_pad_id,
                text_vocab=self.text_vocab,
                speaking_rate_num_buckets=self.speaking_rate_num_buckets,
                quality_bucket_counts=tuple(self.quality_bucket_counts),
                speaker_background_num_buckets=self.speaker_background_num_buckets,
                accurate_mode_num_buckets=self.accurate_mode_num_buckets,
                # Match the server prompt format: trained prompts end with a
                # short silence prefix before audio generation begins.
                prepend_silence=True,
            )
        )

    def _resolve_quality_buckets(
        self, quality_buckets: Dict[str, int | None] | List[int | None] | None
    ) -> List[int | None] | None:
        """Map a quality bucket dict/list onto the model's feature order.

        None applies the default conditioning (same as the server); pass an
        empty dict or list to disable quality tokens entirely.
        """
        if not self.quality_bucket_counts or sum(self.quality_bucket_counts) == 0:
            return None
        if quality_buckets is None:
            quality_buckets = DEFAULT_QUALITY_BUCKETS
        if isinstance(quality_buckets, dict):
            return [quality_buckets.get(feature) for feature in self.quality_features]
        resolved = list(quality_buckets)[: len(self.quality_features)]
        resolved += [None] * (len(self.quality_features) - len(resolved))
        return resolved

    def _tokenize_one(
        self,
        prompt: str | List[List[int]],
        speaking_rate_bucket: int | None = None,
        quality_buckets: Dict[str, int | None] | List[int | None] | None = None,
    ) -> torch.Tensor:
        """Convert a prompt to 2D token tensor.

        Args:
            prompt: Text string or pre-tokenized unpacked tokens

        Returns:
            2D tensor of shape (seq_len, frame_width)
        """
        if isinstance(prompt, str):
            return self._prompt_builder.build(
                prompt,
                speaking_rate_bucket=speaking_rate_bucket,
                quality_buckets=self._resolve_quality_buckets(quality_buckets),
            )
        else:
            prompt_ids = prompt

        return torch.tensor(prompt_ids, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseTTSBackendMsg]:
        """Receive messages from the pending queue."""
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()

        results: List[BaseTTSBackendMsg] = []
        added, sum_input_len = 0, 0

        for i, (input_ids, sampling_params) in enumerate(self.pending_requests):
            if sum_input_len >= self.prefill_budget:
                break

            input_len = len(input_ids)
            sum_input_len += input_len
            uid = self.counter + added
            added += 1

            results.append(
                TTSUserMsg(
                    uid=uid,
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                )
            )

            self.status_map[uid] = TTSRequestStatus(
                uid=i,  # Map back to original index
                input_ids=input_ids,
                output_frames=[],
            )

        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: BatchTTSTokenizerMsg) -> None:
        """Process results from the scheduler."""
        for msg in reply.data:
            assert isinstance(msg, TTSDetokenizeMsg)
            status = self.status_map[msg.uid]

            if not msg.finished:
                status.output_frames.append(msg.audio_codes)
            else:
                status.finished = True
                status.eos_frame = msg.eos_frame

    def generate(
        self,
        prompts: List[str] | List[List[List[int]]],
        sampling_params: TTSSamplingParams | List[TTSSamplingParams],
        decode_audio: bool | None = None,
        speaking_rate_bucket: int | List[int | None] | None = None,
        quality_buckets: Dict[str, int | None] | List[int | None] | None = None,
    ) -> List[Dict]:
        """Generate audio tokens for a batch of prompts.

        Args:
            prompts: List of text strings or pre-tokenized prompts
            sampling_params: Sampling parameters (single or per-prompt)
            decode_audio: Override instance setting for audio decoding
            speaking_rate_bucket: Optional bucket index, or one bucket per prompt.
            quality_buckets: Quality bucket indices, keyed by feature name or as
                a list in the model's feature order. None applies the default
                conditioning; pass {} to disable quality tokens.

        Returns:
            List of dicts with:
                - "audio_tokens": List of generated frames [[cb0, ..., cb8], ...]
                - "eos_frame": Frame index where EOS was detected (or None)
                - "audio": PCM audio bytes if decode_audio=True, else None
                - "sample_rate": Audio sample rate (44100)
        """
        # Reset state
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0

        # Normalize sampling_params
        if isinstance(sampling_params, TTSSamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        if isinstance(speaking_rate_bucket, list):
            speaking_rate_buckets = speaking_rate_bucket
        else:
            speaking_rate_buckets = [speaking_rate_bucket] * len(prompts)

        # Tokenize and queue all requests
        for prompt, sp, rate_bucket in zip(
            prompts,
            sampling_params,
            speaking_rate_buckets,
        ):
            input_ids = self._tokenize_one(
                prompt,
                speaking_rate_bucket=rate_bucket,
                quality_buckets=quality_buckets,
            )
            self.pending_requests.append((input_ids, sp))

        # Run generation
        try:
            self.run_forever()
        except RequestAllFinished:
            pass

        # Determine audio decoding
        should_decode = decode_audio if decode_audio is not None else self.decode_audio

        # Collect results in order
        results = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            audio_tokens = status.output_frames

            # Decode audio if requested
            audio_bytes = None
            if should_decode and audio_tokens and self._vocoder:
                # Convert to tensor, align delayed codebooks, then drop EOS and
                # post-EOS frames before DAC decode.
                codes = torch.tensor(audio_tokens, dtype=torch.int64, device="cuda")
                codes = shear_up(codes, self.audio_pad_id)
                if status.eos_frame is not None:
                    codes = codes[: max(0, status.eos_frame)]
                if codes.numel() == 0:
                    results.append(
                        {
                            "audio_tokens": audio_tokens,
                            "eos_frame": status.eos_frame,
                            "audio": b"",
                            "sample_rate": 44100,
                        }
                    )
                    continue
                codes = codes.unsqueeze(0)  # Add batch dim

                # Decode with vocoder
                audio = self._vocoder.decode_all(codes, apply_shear_up=False)
                audio_bytes = audio[0].numpy().astype("float32").tobytes()

            results.append(
                {
                    "audio_tokens": audio_tokens,
                    "eos_frame": status.eos_frame,
                    "audio": audio_bytes,
                    "sample_rate": 44100,
                }
            )

        return results

    def generate_one(
        self,
        prompt: str | List[List[int]],
        sampling_params: TTSSamplingParams | None = None,
        decode_audio: bool | None = None,
        speaking_rate_bucket: int | None = None,
        quality_buckets: Dict[str, int | None] | List[int | None] | None = None,
    ) -> Dict:
        """Generate audio for a single prompt.

        Convenience method that wraps generate() for single inputs.

        Args:
            prompt: Text string or pre-tokenized prompt
            sampling_params: Sampling parameters (default: TTSSamplingParams())
            decode_audio: Override instance setting for audio decoding
            speaking_rate_bucket: Optional speaking-rate conditioning bucket.
            quality_buckets: Quality bucket indices by feature name; None applies
                the default conditioning, {} disables quality tokens.

        Returns:
            Dict with audio_tokens, eos_frame, audio, sample_rate
        """
        if sampling_params is None:
            sampling_params = TTSSamplingParams()

        results = self.generate(
            [prompt],
            sampling_params,
            decode_audio=decode_audio,
            speaking_rate_bucket=speaking_rate_bucket,
            quality_buckets=quality_buckets,
        )
        return results[0]

    def save_audio(
        self,
        audio_bytes: bytes,
        path: str,
        sample_rate: int = 44100,
    ) -> None:
        """Save PCM audio bytes to a WAV file.

        Args:
            audio_bytes: PCM audio data (float32)
            path: Output file path
            sample_rate: Audio sample rate
        """
        import wave

        import numpy as np

        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        # Convert to int16 for WAV
        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)  # 16-bit
            f.setframerate(sample_rate)
            f.writeframes(audio_int16.tobytes())
