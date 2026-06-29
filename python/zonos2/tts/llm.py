"""Offline TTSLLM class for batch TTS generation."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List

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


@dataclass
class PendingRequest:
    """A tokenized prompt queued for offline generation.

    Mirrors the per-request fields the server forwards on a ``TTSUserMsg`` so the
    offline path conditions identically (text already tokenized into ``input_ids``).
    """

    input_ids: torch.Tensor  # 2D (seq_len, frame_width)
    sampling_params: TTSSamplingParams
    speaker_embedding: torch.Tensor | None = None
    clean_speaker_background: bool = False
    accurate_mode: bool = True


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

        # Text normalization, gated exactly like the server tokenizer worker so the
        # offline path produces the same tokens (e.g. "123" -> "one hundred ...").
        from zonos2.tokenizer.textnorm import TTSTextNormalizer, normalization_enabled

        self._text_normalizer = TTSTextNormalizer() if normalization_enabled() else None
        if self._text_normalizer is not None:
            # Compile the English grammars off the hot path, matching the server.
            threading.Thread(
                target=self._text_normalizer.warmup, args=(["en"],), daemon=True
            ).start()

        # Speaker encoder is loaded lazily on first embed_speaker* call.
        self._speaker_embedder = None

        # Request tracking
        self.pending_requests: List[PendingRequest] = []
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

    @staticmethod
    def _broadcast_per_prompt(value, n: int) -> List:
        """Expand a scalar to one value per prompt, or pass a per-prompt list through.

        A plain ``list`` is treated as per-prompt; everything else (including a
        single ``torch.Tensor`` speaker embedding) is broadcast to every prompt.
        """
        if isinstance(value, list):
            if len(value) != n:
                raise ValueError(
                    f"Expected {n} per-prompt values, got {len(value)}."
                )
            return value
        return [value] * n

    def embed_speaker(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Compute a speaker embedding from a waveform, matching the server.

        Mirrors the server's _compute_speaker_embedding_from_waveform: runs the
        release speaker encoder and returns the float32 CPU embedding whose size
        matches the model's speaker_embedding_dim. The result can be passed
        directly as the ``speaker_embedding`` argument to generate().
        """
        if not self.speaker_enabled or self.speaker_embedding_dim <= 0:
            raise ValueError("Current model does not support speaker conditioning.")

        if self._speaker_embedder is None:
            import os

            from zonos2.models.speaker_cloning import Qwen3SpeakerEmbedding

            # Load the encoder on CPU by default (same env var the server uses) so
            # it does not permanently occupy GPU memory next to the TTS model.
            device = os.getenv("ZONOS2_SPEAKER_EMBEDDER_DEVICE", "cpu")
            self._speaker_embedder = Qwen3SpeakerEmbedding(device=device)

        with torch.inference_mode():
            output = self._speaker_embedder(wav, sample_rate)

        if isinstance(output, tuple):
            candidates = [t.squeeze(0).to(dtype=torch.float32, device="cpu") for t in output]
        else:
            candidates = [output.squeeze(0).to(dtype=torch.float32, device="cpu")]

        for candidate in candidates:
            if candidate.numel() == self.speaker_embedding_dim:
                return candidate.contiguous()

        produced = ", ".join(str(c.numel()) for c in candidates)
        raise ValueError(
            f"Reference embedding dimension mismatch. Model expects "
            f"{self.speaker_embedding_dim}, but speaker encoder produced {produced}."
        )

    def embed_speaker_file(self, path: str) -> torch.Tensor:
        """Load an audio file and compute its speaker embedding (see embed_speaker).

        Decodes via the same ffmpeg path the server uses
        (``_decode_audio_bytes``) so the resulting embedding matches the server's
        for the same file; falls back to torchaudio if the server module (and its
        ffmpeg dependency) is unavailable.
        """
        with open(path, "rb") as f:
            audio_bytes = f.read()
        try:
            from zonos2.tts.audio import _decode_audio_bytes

            wav, sample_rate = _decode_audio_bytes(audio_bytes)
        except Exception:
            import torchaudio

            wav, sample_rate = torchaudio.load(path)
        return self.embed_speaker(wav, sample_rate)

    def _server_config_adapter(self):
        """Minimal stand-in for the server's ServerArgs.

        The server resolution helpers only read ``model_config``, ``max_seq_len``
        and optional ``tts_*`` overrides (getattr-defaulted), so this lets the
        offline path reuse them verbatim for guaranteed parity.
        """
        from types import SimpleNamespace

        return SimpleNamespace(
            model_config=self.engine.model_config,
            max_seq_len=self.engine.max_seq_len,
        )

    def resolve_speaking_rate_bucket(
        self,
        *,
        speaking_rate_bucket: int | None = None,
        speaking_rate: float | None = None,
        speed: float | None = None,
    ) -> int | None:
        """Resolve a speaking-rate bucket from a bucket index, a bytes/sec rate, or
        a speed multiplier, using the server's exact bucketing logic.

        Provide at most one of the three (matching the server).
        """
        from zonos2.tts.conditioning import _resolve_speaking_rate_bucket

        return _resolve_speaking_rate_bucket(
            self._server_config_adapter(),
            speaking_rate_bucket=speaking_rate_bucket,
            speaking_rate=speaking_rate,
            speed=speed,
            speaking_rate_enabled=True,
        )

    def resolve_quality_buckets(
        self,
        *,
        quality_buckets=None,
        quality_values=None,
    ) -> List[int | None] | None:
        """Resolve per-feature quality bucket indices from bucket indices or raw
        metric values, using the server's exact bucketing logic.

        Provide at most one of ``quality_buckets`` / ``quality_values``.
        """
        from zonos2.tts.conditioning import _resolve_quality_buckets

        return _resolve_quality_buckets(
            self._server_config_adapter(),
            quality_buckets=quality_buckets,
            quality_values=quality_values,
            quality_enabled=True,
        )

    def resolve_max_tokens(self, requested: int | None) -> int:
        """Clamp a requested max_tokens to the model limit (server parity)."""
        from zonos2.tts.conditioning import _resolve_tts_max_tokens

        return _resolve_tts_max_tokens(self._server_config_adapter(), requested)

    def _tokenize_one(
        self,
        prompt: str | List[List[int]],
        speaking_rate_bucket: int | None = None,
        quality_buckets: Dict[str, int | None] | List[int | None] | None = None,
        language: str = "en_us",
        text_normalization: bool = True,
    ) -> torch.Tensor:
        """Convert a prompt to 2D token tensor.

        Args:
            prompt: Text string or pre-tokenized unpacked tokens
            language: Server language code used for text normalization.
            text_normalization: Apply language-aware text normalization (matching
                the server) before tokenizing string prompts.

        Returns:
            2D tensor of shape (seq_len, frame_width)
        """
        if isinstance(prompt, str):
            text = prompt
            if text_normalization and self._text_normalizer is not None:
                text = self._text_normalizer.normalize(text, language)
            return self._prompt_builder.build(
                text,
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

        for i, req in enumerate(self.pending_requests):
            if sum_input_len >= self.prefill_budget:
                break

            input_len = len(req.input_ids)
            sum_input_len += input_len
            uid = self.counter + added
            added += 1

            results.append(
                TTSUserMsg(
                    uid=uid,
                    input_ids=req.input_ids,
                    sampling_params=req.sampling_params,
                    # Speaker conditioning, identical to the server's TTSUserMsg.
                    # The shared scheduler injects the speaker slot + background /
                    # accurate-mode markers when an embedding is present, so the
                    # offline path only needs to pass these fields through.
                    speaker_embedding=req.speaker_embedding,
                    clean_speaker_background=req.clean_speaker_background,
                    accurate_mode=req.accurate_mode,
                )
            )

            self.status_map[uid] = TTSRequestStatus(
                uid=i,  # Map back to original index
                input_ids=req.input_ids,
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
        speaking_rate: float | None = None,
        speed: float | None = None,
        quality_buckets: Dict[str, int | None] | List[int | None] | None = None,
        quality_values: Dict[str, float | None] | List[float | None] | None = None,
        max_tokens: int | None = None,
        language: str = "en_us",
        text_normalization: bool = True,
        speaker_embedding: torch.Tensor | List[torch.Tensor | None] | None = None,
        clean_speaker_background: bool | List[bool] = False,
        accurate_mode: bool | List[bool] = True,
    ) -> List[Dict]:
        """Generate audio tokens for a batch of prompts.

        Args:
            prompts: List of text strings or pre-tokenized prompts
            sampling_params: Sampling parameters (single or per-prompt)
            decode_audio: Override instance setting for audio decoding
            speaking_rate_bucket: Optional bucket index, or one bucket per prompt.
            speaking_rate: Target speaking rate in bytes/sec; resolved to a bucket
                applied to all prompts (server parity). Mutually exclusive with
                speaking_rate_bucket / speed.
            speed: Speed multiplier (1.0 = neutral); resolved to a bucket applied
                to all prompts. Mutually exclusive with the two above.
            quality_buckets: Quality bucket indices, keyed by feature name or as
                a list in the model's feature order. None applies the default
                conditioning; pass {} to disable quality tokens.
            quality_values: Raw per-feature quality metric values, resolved to
                bucket indices (server parity). Mutually exclusive with
                quality_buckets.
            max_tokens: Per-request decode cap; clamped to the model limit and
                applied to every sampling_params (server parity).
            language: Server language code (e.g. "en_us") for text normalization.
            text_normalization: Apply language-aware text normalization, matching
                the server default.
            speaker_embedding: Speaker embedding tensor for voice cloning (single,
                applied to all prompts, or one per prompt; None disables cloning).
                Use embed_speaker()/embed_speaker_file() to compute one from audio.
            clean_speaker_background: Mark the speaker embedding as clean-background
                (single or per-prompt), matching the server flag.
            accurate_mode: Accurate (on) vs expressive (off) mode, matching the
                server default of True (single or per-prompt).

        Returns:
            List of dicts with:
                - "audio_tokens": List of generated frames [[cb0, ..., cb8], ...]
                - "eos_frame": Frame index where EOS was detected (or None)
                - "audio": PCM audio bytes if decode_audio=True, else None
                - "sample_rate": Audio sample rate (44100)
        """
        # Validate the language code up front, matching the server's
        # _normalize_tts_request_language behavior.
        if text_normalization:
            from zonos2.tokenizer.textnorm import SERVER_TO_NEMO_LANG

            normalized_lang = str(language or "").strip().lower().replace("-", "_")
            if normalized_lang not in SERVER_TO_NEMO_LANG:
                supported = ", ".join(SERVER_TO_NEMO_LANG)
                raise ValueError(
                    f"Unsupported language code: {language!r}. Supported: {supported}."
                )
            language = normalized_lang

        # Resolve continuous speaking-rate / quality controls to buckets using the
        # server's exact logic (mutually exclusive sources, as on the server).
        if sum(v is not None for v in (speaking_rate_bucket, speaking_rate, speed)) > 1:
            raise ValueError(
                "Provide only one of speaking_rate_bucket, speaking_rate, or speed."
            )
        if speaking_rate is not None or speed is not None:
            speaking_rate_bucket = self.resolve_speaking_rate_bucket(
                speaking_rate=speaking_rate, speed=speed
            )
        if quality_values is not None:
            if quality_buckets is not None:
                raise ValueError("Provide only one of quality_buckets or quality_values.")
            quality_buckets = self.resolve_quality_buckets(quality_values=quality_values)

        # Reset state
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0

        # Normalize per-prompt arguments
        if isinstance(sampling_params, TTSSamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        # Clamp max_tokens to the model limit and apply to every request.
        if max_tokens is not None:
            from dataclasses import replace

            clamped = self.resolve_max_tokens(max_tokens)
            sampling_params = [replace(sp, max_tokens=clamped) for sp in sampling_params]
        if isinstance(speaking_rate_bucket, list):
            speaking_rate_buckets = speaking_rate_bucket
        else:
            speaking_rate_buckets = [speaking_rate_bucket] * len(prompts)
        speaker_embeddings = self._broadcast_per_prompt(speaker_embedding, len(prompts))
        clean_backgrounds = self._broadcast_per_prompt(
            clean_speaker_background, len(prompts)
        )
        accurate_modes = self._broadcast_per_prompt(accurate_mode, len(prompts))

        # Tokenize and queue all requests
        for prompt, sp, rate_bucket, spk_emb, clean_bg, acc_mode in zip(
            prompts,
            sampling_params,
            speaking_rate_buckets,
            speaker_embeddings,
            clean_backgrounds,
            accurate_modes,
        ):
            input_ids = self._tokenize_one(
                prompt,
                speaking_rate_bucket=rate_bucket,
                quality_buckets=quality_buckets,
                language=language,
                text_normalization=text_normalization,
            )
            self.pending_requests.append(
                PendingRequest(
                    input_ids=input_ids,
                    sampling_params=sp,
                    speaker_embedding=spk_emb,
                    clean_speaker_background=bool(clean_bg),
                    accurate_mode=bool(acc_mode),
                )
            )

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
        speaking_rate: float | None = None,
        speed: float | None = None,
        quality_buckets: Dict[str, int | None] | List[int | None] | None = None,
        quality_values: Dict[str, float | None] | List[float | None] | None = None,
        max_tokens: int | None = None,
        language: str = "en_us",
        text_normalization: bool = True,
        speaker_embedding: torch.Tensor | None = None,
        clean_speaker_background: bool = False,
        accurate_mode: bool = True,
    ) -> Dict:
        """Generate audio for a single prompt.

        Convenience method that wraps generate() for single inputs. See generate()
        for the full argument semantics (speaking_rate/speed, quality_values,
        max_tokens, etc.).

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
            speaking_rate=speaking_rate,
            speed=speed,
            quality_buckets=quality_buckets,
            quality_values=quality_values,
            max_tokens=max_tokens,
            language=language,
            text_normalization=text_normalization,
            speaker_embedding=speaker_embedding,
            clean_speaker_background=clean_speaker_background,
            accurate_mode=accurate_mode,
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
