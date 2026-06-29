"""Reference-audio decoding (ffmpeg -> PCM WAV -> mono float tensor).

Shared by the HTTP server and the offline :class:`~zonos2.tts.llm.TTSLLM`; has no
FastAPI / server dependency.
"""

from __future__ import annotations

import io
import subprocess
import wave

import torch

from zonos2.utils import init_logger

logger = init_logger(__name__, "tts-audio")

def _decode_wav_bytes(wav_bytes: bytes) -> tuple[torch.Tensor, int]:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            pcm = wav_file.readframes(n_frames)
    except Exception as exc:  # pragma: no cover - branch depends on invalid user payload
        raise ValueError("Reference audio must be a valid PCM WAV file.") from exc

    if len(pcm) == 0:
        raise ValueError("Reference audio is empty.")

    pcm_view = memoryview(bytearray(pcm))

    if sample_width == 1:
        audio = torch.frombuffer(pcm_view, dtype=torch.uint8).to(torch.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = torch.frombuffer(pcm_view, dtype=torch.int16).to(torch.float32)
        audio = audio / 32768.0
    elif sample_width == 4:
        audio = torch.frombuffer(pcm_view, dtype=torch.int32).to(torch.float32)
        audio = audio / 2147483648.0
    else:
        raise ValueError("Unsupported WAV bit depth. Use 8/16/32-bit PCM WAV.")

    if channels > 1:
        audio = audio.view(-1, channels).transpose(0, 1).contiguous()
    else:
        audio = audio.view(1, -1)

    return audio, sample_rate


def _transcode_audio_bytes_to_wav(audio_bytes: bytes) -> bytes:
    if not audio_bytes:
        raise ValueError("Reference file is empty.")

    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                "pipe:1",
            ],
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            "Reference file must contain an audio stream supported by ffmpeg."
        ) from exc

    if proc.returncode != 0 or not proc.stdout:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        suffix = f" ffmpeg said: {stderr}" if stderr else ""
        raise ValueError(
            "Reference file must contain an audio stream supported by ffmpeg."
            + suffix
        )

    logger.debug("Converted reference media to PCM WAV via ffmpeg for speaker embedding.")
    return proc.stdout


def _decode_audio_bytes(audio_bytes: bytes) -> tuple[torch.Tensor, int]:
    return _decode_wav_bytes(_transcode_audio_bytes_to_wav(audio_bytes))
