from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import math
import os
import re
import subprocess
import time
import uuid
import wave
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from zonos2.message import (
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchTTSFrontendMsg,
    TTSAudioReply,
    TTSSamplingParams,
    TTSTokenizeMsg,
)
from zonos2.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger

from .args import ServerArgs

_UI_HTML = Path(__file__).resolve().parent.parent.parent.parent / "tts_ui.html"

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None
_SPEAKER_EMBEDDERS = {}
_SPEAKER_EMBEDDER_LOCK = asyncio.Lock()
_SPEAKER_EMBEDDER_DEVICE = os.getenv("ZONOS2_SPEAKER_EMBEDDER_DEVICE", "cpu")
_SESSION_SPEAKER_CACHE: Dict[str, Dict[str, "CachedSpeakerReference"]] = {}
_SESSION_SPEAKER_CACHE_LOCK = asyncio.Lock()
_DEFAULT_SPEAKER_CACHE: Dict[Tuple[str, int], Dict[str, "DefaultSpeakerReference"]] = {}
_DEFAULT_SPEAKER_CACHE_LOCK = asyncio.Lock()
_SPEAKING_RATE_FPS = 86.0 * (44070.0 / 44000.0)
_DEFAULT_SPEAKING_RATE_BYTES_PER_SECOND = 15.0
_SPEAKING_RATE_CLOSED_BUCKET_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$")
_SPEAKING_RATE_OPEN_BUCKET_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*\+\s*$")
_QUALITY_NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_QUALITY_EXACT_BUCKET_RE = re.compile(rf"^\s*({_QUALITY_NUMBER_RE})\s*$")
_QUALITY_CLOSED_BUCKET_RE = re.compile(
    rf"^\s*({_QUALITY_NUMBER_RE})\s*-\s*({_QUALITY_NUMBER_RE})\s*$"
)
_QUALITY_OPEN_BUCKET_RE = re.compile(rf"^\s*({_QUALITY_NUMBER_RE})\s*\+\s*$")
_QUALITY_METRIC_FIELDS = (
    "lufs",
    "estimated_snr",
    "max_pause",
    "estimated_bandlimit_hz",
    "leading_silence_s",
    "trailing_silence_s",
)
_DEFAULT_QUALITY_BUCKETS = {"trailing_silence_s": 3}
_DEFAULT_VOICE_AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}
_DEFAULT_VOICE_EMBEDDING_EXTENSIONS = {".npy", ".npz"}


@dataclass
class CachedSpeakerReference:
    speaker_id: str
    label: str
    source_type: Literal["audio", "embedding_file"]
    embedding: torch.Tensor
    created_at: float
    original_name: str | None = None
    audio_bytes: bytes | None = None
    audio_media_type: str | None = None


@dataclass
class DefaultSpeakerReference:
    speaker_id: str
    label: str
    source_type: Literal["audio", "embedding_file"]
    path: Path
    mtime: float
    original_name: str
    embedding: torch.Tensor | None = None
    audio_bytes: bytes | None = None
    audio_media_type: str | None = None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[TTSAudioReply]:
    if isinstance(msg, BatchTTSFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, TTSAudioReply)
            result.append(reply)
        return result
    if isinstance(msg, TTSAudioReply):
        return [msg]
    raise TypeError(f"Unexpected frontend message type: {type(msg).__name__}")


def _model_supports_speaker(config: ServerArgs) -> bool:
    return bool(getattr(config.model_config, "speaker_enabled", False))


def _model_speaker_dim(config: ServerArgs) -> int:
    return int(getattr(config.model_config, "speaker_embedding_dim", 128))


def _model_speaking_rate_num_buckets(config: ServerArgs) -> int:
    model_value = int(getattr(config.model_config, "speaking_rate_num_buckets", 0) or 0)
    server_value = int(getattr(config, "tts_speaking_rate_num_buckets", 0) or 0)
    return model_value or server_value


def _model_speaking_rate_buckets(config: ServerArgs) -> list[str]:
    raw = getattr(config.model_config, "speaking_rate_buckets", None)
    if not raw:
        raw = getattr(config, "tts_speaking_rate_buckets", ())
    return [str(item) for item in (raw or ())]


def _model_tts_max_tokens(config: ServerArgs) -> int:
    return max(1, int(config.max_seq_len))


def _resolve_tts_max_tokens(config: ServerArgs, requested: int | None) -> int:
    model_max = _model_tts_max_tokens(config)
    if requested is None:
        return model_max
    requested = int(requested)
    if requested <= 0:
        raise ValueError("max_tokens must be positive.")
    return min(requested, model_max)


def _field_was_set(model: BaseModel, field_name: str) -> bool:
    fields_set = getattr(model, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(model, "__fields_set__", set())
    return field_name in fields_set


def _parse_speaking_rate_bucket(spec: str) -> tuple[float, float | None]:
    closed = _SPEAKING_RATE_CLOSED_BUCKET_RE.match(str(spec))
    if closed is not None:
        return float(closed.group(1)), float(closed.group(2))

    open_ended = _SPEAKING_RATE_OPEN_BUCKET_RE.match(str(spec))
    if open_ended is not None:
        return float(open_ended.group(1)), None

    raise ValueError(f"Invalid speaking-rate bucket {spec!r}; expected ranges like '0-3' or '60+'.")


def _speaking_rate_bucket_ranges(config: ServerArgs) -> list[tuple[float, float | None]]:
    ranges = [_parse_speaking_rate_bucket(spec) for spec in _model_speaking_rate_buckets(config)]
    if not ranges:
        return ranges

    first_low, _ = ranges[0]
    if not math.isclose(first_low, 0.0, abs_tol=1e-9):
        raise ValueError("speaking-rate buckets must start at 0.")

    previous_high: float | None = None
    for idx, (low, high) in enumerate(ranges):
        if low < 0.0:
            raise ValueError("speaking-rate buckets must use non-negative ranges.")
        if high is not None and high <= low:
            raise ValueError(f"speaking-rate bucket {idx} has an empty or inverted range.")
        if previous_high is None and idx > 0:
            raise ValueError(
                "speaking-rate buckets cannot define ranges after an open-ended bucket."
            )
        if previous_high is not None and not math.isclose(low, previous_high, abs_tol=1e-9):
            raise ValueError("speaking-rate buckets must be contiguous and ordered.")
        previous_high = high

    if ranges[-1][1] is not None:
        raise ValueError("speaking-rate buckets must end with an open-ended range like '60+'.")
    return ranges


def _speaking_rate_bucket_for_rate(
    rate_bytes_per_second: float,
    *,
    num_buckets: int,
    ranges: list[tuple[float, float | None]],
) -> int:
    if rate_bytes_per_second <= 0:
        raise ValueError("speaking_rate must be positive.")

    if ranges:
        for idx, (_, high) in enumerate(ranges):
            if high is None or (
                rate_bytes_per_second < high
                and not math.isclose(rate_bytes_per_second, high, rel_tol=1e-12, abs_tol=1e-9)
            ):
                return idx
        return len(ranges) - 1

    rate_bytes_per_frame = rate_bytes_per_second / _SPEAKING_RATE_FPS
    bucket = int(rate_bytes_per_frame * num_buckets)
    return min(max(bucket, 0), num_buckets - 1)


def _neutral_speaking_rate_bytes_per_second(
    ranges: list[tuple[float, float | None]],
) -> float:
    if not ranges:
        return _DEFAULT_SPEAKING_RATE_BYTES_PER_SECOND

    low, high = ranges[len(ranges) // 2]
    if high is None:
        return max(low, _DEFAULT_SPEAKING_RATE_BYTES_PER_SECOND)
    return (low + high) / 2.0


def _resolve_speaking_rate_bucket(
    config: ServerArgs,
    *,
    speaking_rate_bucket: int | None = None,
    speaking_rate: float | None = None,
    speed: float | None = None,
    speaking_rate_enabled: bool = False,
) -> int | None:
    if not speaking_rate_enabled:
        return None

    supplied = [
        speaking_rate_bucket is not None,
        speaking_rate is not None,
        speed is not None,
    ]
    if sum(supplied) == 0:
        return None
    if sum(supplied) > 1:
        raise ValueError("Provide only one of speaking_rate_bucket, speaking_rate, or speed.")

    num_buckets = _model_speaking_rate_num_buckets(config)
    if num_buckets <= 0:
        if speed is not None and speaking_rate_bucket is None and speaking_rate is None:
            return None
        raise ValueError("Current model does not support speaking-rate conditioning.")

    if speaking_rate_bucket is not None:
        bucket = int(speaking_rate_bucket)
        if bucket < 0 or bucket >= num_buckets:
            raise ValueError(
                f"speaking_rate_bucket must be in [0, {num_buckets - 1}], got {bucket}."
            )
        return bucket

    ranges = _speaking_rate_bucket_ranges(config)
    if ranges and len(ranges) != num_buckets:
        raise ValueError(
            f"Model has {num_buckets} speaking-rate buckets, but config defines {len(ranges)} ranges."
        )

    if speaking_rate is not None:
        return _speaking_rate_bucket_for_rate(
            float(speaking_rate),
            num_buckets=num_buckets,
            ranges=ranges,
        )

    assert speed is not None
    speed_value = float(speed)
    if speed_value <= 0:
        raise ValueError("speed must be positive.")
    return _speaking_rate_bucket_for_rate(
        _neutral_speaking_rate_bytes_per_second(ranges) * speed_value,
        num_buckets=num_buckets,
        ranges=ranges,
    )


def _normalize_tts_request_language(language: str) -> str:
    from zonos2.tokenizer.textnorm import SERVER_TO_NEMO_LANG

    normalized = str(language or "").strip().lower().replace("-", "_")
    if normalized not in SERVER_TO_NEMO_LANG:
        supported = ", ".join(SERVER_TO_NEMO_LANG)
        raise ValueError(
            f"Unsupported language code: {language!r}. Supported: {supported}."
        )
    return normalized


def _model_speaker_background_token_enabled(config: ServerArgs) -> bool:
    return bool(
        getattr(config.model_config, "speaker_background_token_enabled", False)
        or getattr(config, "tts_speaker_background_token_enabled", False)
    )


def _model_accurate_mode_token_enabled(config: ServerArgs) -> bool:
    return _model_speaker_background_token_enabled(config) and bool(
        getattr(config.model_config, "accurate_mode_token_enabled", False)
        or getattr(config, "tts_accurate_mode_token_enabled", False)
    )


def _model_quality_features(config: ServerArgs) -> list[str]:
    raw = getattr(config.model_config, "quality_features", None)
    if not raw:
        model_buckets = getattr(config.model_config, "quality_buckets", None) or {}
        raw = model_buckets.keys()
    if not raw:
        raw = getattr(config, "tts_quality_features", ())
    if not raw:
        server_buckets = getattr(config, "tts_quality_buckets", {}) or {}
        raw = server_buckets.keys()
    if raw is None:
        raw = _QUALITY_METRIC_FIELDS
    if isinstance(raw, Mapping) or hasattr(raw, "items"):
        return [str(feature) for feature, enabled in raw.items() if bool(enabled)]
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in (raw or ())]


def _model_quality_buckets(config: ServerArgs) -> dict[str, list[str]]:
    raw = getattr(config.model_config, "quality_buckets", None)
    if not raw:
        raw = getattr(config, "tts_quality_buckets", {})
    features = _model_quality_features(config)
    return {
        feature: [str(item) for item in ((raw or {}).get(feature, ()) or ())]
        for feature in features
    }


def _model_quality_bucket_counts(config: ServerArgs) -> list[int]:
    buckets = _model_quality_buckets(config)
    return [len(buckets.get(feature, ())) for feature in _model_quality_features(config)]


def _model_quality_num_buckets(config: ServerArgs) -> int:
    model_value = int(getattr(config.model_config, "quality_num_buckets", 0) or 0)
    server_value = int(getattr(config, "tts_quality_num_buckets", 0) or 0)
    return model_value or server_value or sum(_model_quality_bucket_counts(config))


def _model_quality_dropout(config: ServerArgs) -> float | dict[str, float] | None:
    raw = getattr(config.model_config, "quality_dropout", None)
    if raw is None:
        raw = getattr(config, "tts_quality_dropout", None)
    if raw is None:
        return None
    if isinstance(raw, Mapping) or hasattr(raw, "items"):
        return {str(feature): float(dropout) for feature, dropout in raw.items()}
    return float(raw)


def _model_quality_dropout_by_feature(config: ServerArgs) -> dict[str, float]:
    raw = _model_quality_dropout(config)
    features = _model_quality_features(config)
    if raw is None:
        return dict.fromkeys(features, 0.0)
    if isinstance(raw, dict):
        default = float(raw.get("default", 0.0))
        return {feature: float(raw.get(feature, default)) for feature in features}
    return {feature: float(raw) for feature in features}


def _parse_quality_bucket(spec: str) -> tuple[str, float, float | None]:
    value = str(spec)
    exact = _QUALITY_EXACT_BUCKET_RE.match(value)
    if exact is not None:
        return "exact", float(exact.group(1)), None

    closed = _QUALITY_CLOSED_BUCKET_RE.match(value)
    if closed is not None:
        return "range", float(closed.group(1)), float(closed.group(2))

    open_ended = _QUALITY_OPEN_BUCKET_RE.match(value)
    if open_ended is not None:
        return "range", float(open_ended.group(1)), None

    raise ValueError(
        f"Invalid quality bucket {spec!r}; expected exact values like '0', "
        "ranges like '-30--25', or open-ended ranges like '22050+'."
    )


def _quality_bucket_specs(
    config: ServerArgs, feature: str
) -> list[tuple[str, float, float | None]]:
    raw = _model_quality_buckets(config).get(feature, ())
    specs = [_parse_quality_bucket(spec) for spec in raw]
    for idx, (kind, low, high) in enumerate(specs):
        if not math.isfinite(low):
            raise ValueError(f"quality_buckets.{feature} must use finite bucket values.")
        if kind == "range":
            if high is not None and not math.isfinite(high):
                raise ValueError(f"quality_buckets.{feature} must use finite bucket values.")
            if high is not None and high <= low:
                raise ValueError(
                    f"quality_buckets.{feature} has an empty or inverted range at index {idx}."
                )
    return specs


def _quality_bucket_for_value(value: float, config: ServerArgs, feature: str) -> int | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None

    specs = _quality_bucket_specs(config, feature)
    if not specs:
        return None

    for idx, (kind, low, _) in enumerate(specs):
        if kind == "exact" and math.isclose(value, low, rel_tol=1e-12, abs_tol=1e-9):
            return idx

    range_indexes = [idx for idx, (kind, _, _) in enumerate(specs) if kind == "range"]
    if not range_indexes:
        return None

    for idx in range_indexes:
        _, low, high = specs[idx]
        if high is None:
            if value >= low:
                return idx
        elif idx == range_indexes[-1]:
            if low <= value <= high:
                return idx
        elif low <= value < high:
            return idx

    _, first_low, _ = specs[range_indexes[0]]
    if value < first_low:
        return range_indexes[0]
    return range_indexes[-1]


def _quality_control_to_feature_list(value: Any, features: list[str]) -> list[Any]:
    if value is None:
        return [None] * len(features)
    if isinstance(value, dict):
        return [value.get(feature) for feature in features]
    if isinstance(value, (list, tuple)):
        return [value[idx] if idx < len(value) else None for idx in range(len(features))]
    raise ValueError("quality_buckets and quality_values must be a list or feature-name object.")


def _resolve_quality_buckets(
    config: ServerArgs,
    *,
    quality_buckets: Any = None,
    quality_values: Any = None,
    quality_enabled: bool = False,
) -> list[int | None] | None:
    if not quality_enabled:
        return None

    if quality_buckets is None and quality_values is None:
        return None
    if quality_buckets is not None and quality_values is not None:
        raise ValueError("Provide only one of quality_buckets or quality_values.")

    features = _model_quality_features(config)
    counts = _model_quality_bucket_counts(config)
    if not features or _model_quality_num_buckets(config) <= 0 or sum(counts) <= 0:
        raise ValueError("Current model does not support quality conditioning.")

    if any(count <= 0 for count in counts):
        raise ValueError("Every configured quality feature must define at least one bucket.")

    if quality_buckets is not None:
        raw_buckets = _quality_control_to_feature_list(quality_buckets, features)
        resolved: list[int | None] = []
        for feature, count, raw_bucket in zip(features, counts, raw_buckets, strict=True):
            if raw_bucket is None:
                resolved.append(None)
                continue
            bucket = int(raw_bucket)
            if bucket < 0 or bucket >= count:
                raise ValueError(
                    f"quality_buckets.{feature} must be in [0, {count - 1}], got {bucket}."
                )
            resolved.append(bucket)
        return resolved

    raw_values = _quality_control_to_feature_list(quality_values, features)
    return [
        _quality_bucket_for_value(raw_value, config, feature) if raw_value is not None else None
        for feature, raw_value in zip(features, raw_values, strict=True)
    ]


def _default_quality_buckets(config: ServerArgs) -> list[int | None] | None:
    """Resolve the default quality conditioning for models that support it."""
    if _model_quality_num_buckets(config) <= 0:
        return None
    try:
        return _resolve_quality_buckets(
            config, quality_buckets=dict(_DEFAULT_QUALITY_BUCKETS), quality_enabled=True
        )
    except ValueError:
        return None


def _apply_fade_out_pcm(
    audio_bytes: bytes, fade_out_ms: float, sample_rate: int = 44100
) -> bytes:
    """Apply a cosine fade-out to the tail of float32 PCM bytes."""
    if fade_out_ms <= 0 or not audio_bytes:
        return audio_bytes
    samples = np.frombuffer(audio_bytes, dtype=np.float32).copy()
    n = min(len(samples), int(sample_rate * fade_out_ms / 1000.0))
    if n <= 0:
        return audio_bytes
    fade = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32)))
    samples[-n:] *= fade
    return samples.tobytes()


def _normalize_session_id(session_id: str | None) -> str | None:
    if session_id is None:
        return None
    normalized = session_id.strip()
    return normalized or None


def _require_session_id(session_id: str | None) -> str:
    normalized = _normalize_session_id(session_id)
    if normalized is None:
        raise ValueError("Cached speaker operations require the X-TTS-Session-ID header.")
    return normalized


def _default_speaker_label(
    explicit_label: str | None,
    file_name: str | None,
    *,
    fallback: str,
) -> str:
    if explicit_label and explicit_label.strip():
        return explicit_label.strip()
    if file_name:
        stem = Path(file_name).stem.strip()
        if stem:
            return stem
    return fallback


def _slerp_embeddings(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    v0 = v0.detach().to(dtype=torch.float32, device="cpu").contiguous().view(-1)
    v1 = v1.detach().to(dtype=torch.float32, device="cpu").contiguous().view(-1)
    if v0.shape != v1.shape:
        raise ValueError(
            f"Cannot blend speaker embeddings with different shapes {tuple(v0.shape)} and {tuple(v1.shape)}."
        )

    blend = float(t)
    if blend < 0.0 or blend > 1.0:
        raise ValueError("speaker_blend_t must be between 0.0 and 1.0.")

    v0_norm = torch.linalg.vector_norm(v0) + 1e-8
    v1_norm = torch.linalg.vector_norm(v1) + 1e-8
    v0_n = v0 / v0_norm
    v1_n = v1 / v1_norm
    omega = torch.acos(torch.clamp(torch.dot(v0_n, v1_n), -1.0, 1.0))
    if float(omega) < 1e-6:
        return ((1.0 - blend) * v0 + blend * v1).contiguous()

    sin_omega = torch.sin(omega)
    mixed = (
        (torch.sin((1.0 - blend) * omega) / sin_omega) * v0
        + (torch.sin(blend * omega) / sin_omega) * v1
    )
    return mixed.to(dtype=torch.float32, device="cpu").contiguous()


def _serialize_cached_speaker(entry: CachedSpeakerReference) -> dict:
    return {
        "id": entry.speaker_id,
        "label": entry.label,
        "source_type": entry.source_type,
        "original_name": entry.original_name,
        "dimension": int(entry.embedding.numel()),
        "created_at": entry.created_at,
        "has_preview": bool(entry.audio_bytes),
    }


def _serialize_default_speaker(entry: DefaultSpeakerReference, expected_dim: int) -> dict:
    return {
        "id": entry.speaker_id,
        "label": entry.label,
        "source_type": entry.source_type,
        "scope": "default",
        "is_default": True,
        "original_name": entry.original_name,
        "dimension": expected_dim,
        "created_at": entry.mtime,
        "has_preview": entry.source_type == "audio",
    }


def _default_voice_root(config: ServerArgs) -> Path | None:
    raw = getattr(config, "tts_default_voices_dir", None)
    if not raw:
        return None
    return Path(str(raw)).expanduser()


def _default_voice_source_type(path: Path) -> Literal["audio", "embedding_file"] | None:
    suffix = path.suffix.lower()
    if suffix in _DEFAULT_VOICE_EMBEDDING_EXTENSIONS:
        return "embedding_file"
    if suffix in _DEFAULT_VOICE_AUDIO_EXTENSIONS:
        return "audio"
    return None


def _default_voice_id(root: Path, path: Path) -> str:
    rel = path.relative_to(root).as_posix()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
    return f"default_{digest}"


def _default_voice_label(path: Path) -> str:
    label = path.stem.replace("_", " ").replace("-", " ").strip()
    return label or path.name


async def _list_default_speakers(config: ServerArgs) -> list[DefaultSpeakerReference]:
    root = _default_voice_root(config)
    if root is None:
        return []
    if not root.exists() or not root.is_dir():
        logger.warning("TTS default voices directory does not exist or is not a directory: %s", root)
        return []

    root = root.resolve()
    expected_dim = _model_speaker_dim(config)
    cache_key = (str(root), expected_dim)

    async with _DEFAULT_SPEAKER_CACHE_LOCK:
        previous = _DEFAULT_SPEAKER_CACHE.get(cache_key, {})
        current: dict[str, DefaultSpeakerReference] = {}
        for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
            source_type = _default_voice_source_type(path)
            if source_type is None:
                continue

            resolved_path = path.resolve()
            try:
                stat = resolved_path.stat()
            except OSError:
                continue

            speaker_id = _default_voice_id(root, path)
            existing = previous.get(speaker_id)
            if (
                existing is not None
                and existing.path == resolved_path
                and existing.mtime == stat.st_mtime
                and existing.source_type == source_type
            ):
                current[speaker_id] = existing
                continue

            current[speaker_id] = DefaultSpeakerReference(
                speaker_id=speaker_id,
                label=_default_voice_label(resolved_path),
                source_type=source_type,
                path=resolved_path,
                mtime=stat.st_mtime,
                original_name=resolved_path.name,
            )

        _DEFAULT_SPEAKER_CACHE[cache_key] = current
        return sorted(current.values(), key=lambda entry: entry.label.lower())


async def _get_default_speaker(
    config: ServerArgs,
    speaker_id: str,
) -> DefaultSpeakerReference | None:
    speakers = await _list_default_speakers(config)
    return next((entry for entry in speakers if entry.speaker_id == speaker_id), None)


def _read_default_voice_file(entry: DefaultSpeakerReference) -> bytes:
    try:
        return entry.path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Failed to read default voice '{entry.label}'.") from exc


async def _default_speaker_preview(entry: DefaultSpeakerReference) -> tuple[bytes, str]:
    if entry.source_type != "audio":
        raise ValueError("Default embedding speakers do not include preview audio.")
    if entry.audio_bytes is None:
        entry.audio_bytes = _transcode_audio_bytes_to_wav(_read_default_voice_file(entry))
        entry.audio_media_type = "audio/wav"
    return entry.audio_bytes, entry.audio_media_type or "audio/wav"


async def _load_default_speaker_embedding(
    config: ServerArgs,
    entry: DefaultSpeakerReference,
) -> torch.Tensor:
    expected_dim = _model_speaker_dim(config)
    if entry.embedding is None:
        if entry.source_type == "embedding_file":
            entry.embedding = _load_embedding_vector(
                _read_default_voice_file(entry),
                expected_dim=expected_dim,
                file_name=entry.original_name,
            )
        else:
            wav_bytes, _ = await _default_speaker_preview(entry)
            wav, sample_rate = _decode_wav_bytes(wav_bytes)
            entry.embedding = await _compute_speaker_embedding_from_waveform(
                wav,
                sample_rate,
                expected_dim=expected_dim,
            )

    return entry.embedding.detach().to(dtype=torch.float32, device="cpu").contiguous().clone()


async def _cache_speaker_reference(
    *,
    session_id: str,
    label: str,
    source_type: Literal["audio", "embedding_file"],
    embedding: torch.Tensor,
    original_name: str | None = None,
    audio_bytes: bytes | None = None,
    audio_media_type: str | None = None,
) -> CachedSpeakerReference:
    entry = CachedSpeakerReference(
        speaker_id=f"spk_{uuid.uuid4().hex[:12]}",
        label=label,
        source_type=source_type,
        embedding=embedding.detach().to(dtype=torch.float32, device="cpu").contiguous().clone(),
        created_at=time.time(),
        original_name=original_name,
        audio_bytes=audio_bytes,
        audio_media_type=audio_media_type,
    )
    async with _SESSION_SPEAKER_CACHE_LOCK:
        session_cache = _SESSION_SPEAKER_CACHE.setdefault(session_id, {})
        session_cache[entry.speaker_id] = entry
    return entry


async def _list_cached_speakers(session_id: str) -> list[CachedSpeakerReference]:
    async with _SESSION_SPEAKER_CACHE_LOCK:
        session_cache = _SESSION_SPEAKER_CACHE.get(session_id, {})
        return sorted(
            session_cache.values(),
            key=lambda item: item.created_at,
            reverse=True,
        )


async def _get_cached_speaker(session_id: str, speaker_id: str) -> CachedSpeakerReference:
    async with _SESSION_SPEAKER_CACHE_LOCK:
        session_cache = _SESSION_SPEAKER_CACHE.get(session_id, {})
        entry = session_cache.get(speaker_id)
    if entry is None:
        raise ValueError(f"Unknown cached speaker id '{speaker_id}'.")
    return entry


def _decode_base64_blob(data: str, field_name: str) -> bytes:
    payload = data.strip()
    if payload.startswith("data:"):
        if "," not in payload:
            raise ValueError(f"Invalid data URL for {field_name}.")
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:  # pragma: no cover - branch depends on invalid user payload
        raise ValueError(f"{field_name} must be valid base64-encoded data.") from exc


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


def _load_embedding_vector(
    embedding_bytes: bytes,
    *,
    expected_dim: int,
    file_name: str | None = None,
) -> torch.Tensor:
    try:
        loaded = np.load(io.BytesIO(embedding_bytes), allow_pickle=False)
    except Exception as exc:
        file_desc = file_name or "Speaker embedding file"
        raise ValueError(
            f"{file_desc} must be a valid .npy or .npz file saved without pickle."
        ) from exc

    arr = None
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "emb" in loaded.files:
                arr = loaded["emb"]
            elif len(loaded.files) == 1:
                arr = loaded[loaded.files[0]]
            else:
                raise ValueError(
                    "Embedding archive must contain an 'emb' array or exactly one array."
                )
        finally:
            loaded.close()
    else:
        arr = loaded

    if isinstance(arr, np.ndarray) and arr.dtype.names:
        if "emb" in arr.dtype.names:
            arr = arr["emb"]
        elif len(arr.dtype.names) == 1:
            arr = arr[arr.dtype.names[0]]
        else:
            raise ValueError("Structured embedding arrays must contain an 'emb' field.")

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        vector = arr
    elif arr.ndim == 2:
        if arr.shape[0] == 0:
            raise ValueError("Speaker embedding file is empty.")
        # Multi-row embedding files come from segmented exports; average them into one speaker vector.
        vector = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
    else:
        raise ValueError(
            f"Unsupported embedding array shape {tuple(arr.shape)}. Expected (D,) or (N, D)."
        )

    vector = np.ascontiguousarray(vector, dtype=np.float32)
    if vector.shape[-1] != expected_dim:
        raise ValueError(
            f"Reference embedding dimension mismatch. Model expects {expected_dim}, "
            f"but loaded file has {vector.shape[-1]}."
        )

    return torch.from_numpy(vector).to(dtype=torch.float32, device="cpu")


async def _get_speaker_embedder(expected_dim: int):
    global _SPEAKER_EMBEDDERS
    if expected_dim != 2048:
        raise RuntimeError(
            f"The release speaker encoder only supports 2048D embeddings; model expects {expected_dim}D."
        )
    backend_key = "qwen3"
    if backend_key in _SPEAKER_EMBEDDERS:
        return _SPEAKER_EMBEDDERS[backend_key]

    async with _SPEAKER_EMBEDDER_LOCK:
        if backend_key in _SPEAKER_EMBEDDERS:
            return _SPEAKER_EMBEDDERS[backend_key]
        try:
            from zonos2.models.speaker_cloning import Qwen3SpeakerEmbedding

            embedder = Qwen3SpeakerEmbedding(device=_SPEAKER_EMBEDDER_DEVICE)
        except Exception as exc:  # pragma: no cover - depends on environment deps
            raise RuntimeError(
                "Speaker embedding backend failed to load. Ensure torchaudio, "
                "transformers, and huggingface_hub are installed and that the "
                f"Qwen3 speaker model is accessible. Original error: {type(exc).__name__}: {exc}"
            ) from exc
        _SPEAKER_EMBEDDERS[backend_key] = embedder
    return _SPEAKER_EMBEDDERS[backend_key]


async def _compute_speaker_embedding_from_audio(
    speaker_audio_base64: str,
    expected_dim: int,
) -> torch.Tensor:
    audio_bytes = _decode_base64_blob(speaker_audio_base64, "speaker_audio_base64")
    wav, sample_rate = _decode_audio_bytes(audio_bytes)
    return await _compute_speaker_embedding_from_waveform(
        wav,
        sample_rate,
        expected_dim=expected_dim,
    )


async def _compute_speaker_embedding_from_waveform(
    wav: torch.Tensor,
    sample_rate: int,
    *,
    expected_dim: int,
) -> torch.Tensor:
    embedder = await _get_speaker_embedder(expected_dim)
    with torch.inference_mode():
        output = embedder(wav, sample_rate)

    if isinstance(output, tuple):
        candidates = [
            tensor.squeeze(0).to(dtype=torch.float32, device="cpu")
            for tensor in output
        ]
    else:
        candidates = [output.squeeze(0).to(dtype=torch.float32, device="cpu")]

    for candidate in candidates:
        if candidate.numel() == expected_dim:
            return candidate

    raise ValueError(
        f"Reference embedding dimension mismatch. Model expects {expected_dim}, "
        f"but speaker encoder produced {', '.join(str(c.numel()) for c in candidates)}."
    )


def _compute_speaker_embedding_from_file(
    speaker_embedding_base64: str,
    expected_dim: int,
    speaker_embedding_name: str | None = None,
) -> torch.Tensor:
    embedding_bytes = _decode_base64_blob(
        speaker_embedding_base64,
        "speaker_embedding_base64",
    )
    return _load_embedding_vector(
        embedding_bytes,
        expected_dim=expected_dim,
        file_name=speaker_embedding_name,
    )


async def _resolve_cached_speaker_embedding(
    *,
    config: ServerArgs,
    session_id: str | None,
    speaker_id: str,
) -> torch.Tensor | None:
    if not _model_supports_speaker(config):
        logger.info(
            "Cached speaker reference provided, but current model does not support speaker embeddings; ignoring."
        )
        return None

    default_entry = await _get_default_speaker(config, speaker_id)
    if default_entry is not None:
        return await _load_default_speaker_embedding(config, default_entry)

    resolved_session_id = _require_session_id(session_id)
    entry = await _get_cached_speaker(resolved_session_id, speaker_id)
    expected_dim = _model_speaker_dim(config)
    actual_dim = int(entry.embedding.numel())
    if actual_dim != expected_dim:
        raise ValueError(
            f"Cached speaker '{entry.label}' has {actual_dim} dimensions, but the current model expects {expected_dim}."
        )
    return entry.embedding.detach().to(dtype=torch.float32, device="cpu").contiguous().clone()


async def _resolve_speaker_embedding(
    *,
    config: ServerArgs,
    session_id: str | None = None,
    speaker_audio_base64: str | None = None,
    speaker_audio_name: str | None = None,
    speaker_embedding_base64: str | None = None,
    speaker_embedding_name: str | None = None,
    speaker_embedding_id: str | None = None,
    speaker_blend_embedding_id_a: str | None = None,
    speaker_blend_embedding_id_b: str | None = None,
    speaker_blend_t: float | None = None,
    legacy_speaker_wav_base64: str | None = None,
) -> torch.Tensor | None:
    speaker_audio_base64 = speaker_audio_base64 or legacy_speaker_wav_base64
    direct_source_count = sum(
        bool(value) for value in (speaker_audio_base64, speaker_embedding_base64)
    )
    if direct_source_count > 1:
        raise ValueError(
            "Provide either a reference audio file or a saved embedding file, not both."
        )

    has_direct_source = direct_source_count == 1
    has_cached_source = bool(speaker_embedding_id)
    has_blended_source = bool(speaker_blend_embedding_id_a or speaker_blend_embedding_id_b)
    selected_modes = sum(bool(mode) for mode in (has_direct_source, has_cached_source, has_blended_source))
    if selected_modes == 0:
        return None
    if selected_modes > 1:
        raise ValueError(
            "Provide speaker conditioning via upload, one cached speaker id, or two cached ids for blending."
        )

    if has_blended_source:
        if not speaker_blend_embedding_id_a or not speaker_blend_embedding_id_b:
            raise ValueError("Provide both cached speaker ids to blend between two speakers.")
        speaker_a = await _resolve_cached_speaker_embedding(
            config=config,
            session_id=session_id,
            speaker_id=speaker_blend_embedding_id_a,
        )
        speaker_b = await _resolve_cached_speaker_embedding(
            config=config,
            session_id=session_id,
            speaker_id=speaker_blend_embedding_id_b,
        )
        if speaker_a is None or speaker_b is None:
            return None
        return _slerp_embeddings(
            speaker_a,
            speaker_b,
            0.5 if speaker_blend_t is None else speaker_blend_t,
        )

    if has_cached_source:
        assert speaker_embedding_id is not None
        return await _resolve_cached_speaker_embedding(
            config=config,
            session_id=session_id,
            speaker_id=speaker_embedding_id,
        )

    if not _model_supports_speaker(config):
        logger.info(
            "Speaker reference provided, but current model does not support speaker embeddings; ignoring."
        )
        return None

    expected_dim = _model_speaker_dim(config)
    if speaker_embedding_base64:
        return _compute_speaker_embedding_from_file(
            speaker_embedding_base64,
            expected_dim=expected_dim,
            speaker_embedding_name=speaker_embedding_name,
        )
    assert speaker_audio_base64 is not None
    return await _compute_speaker_embedding_from_audio(
        speaker_audio_base64,
        expected_dim=expected_dim,
    )


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


# TTS Request Models
class TTSGenerateRequest(BaseModel):
    """Request model for TTS generation."""

    text: str
    language: str = "en_us"
    text_normalization: bool = True
    temperature: float = 1.15
    topk: int = 106
    top_p: float = 0.0
    min_p: float = 0.18
    max_tokens: int | None = None
    fade_out_ms: float = 0.0
    repetition_window: int = 50
    repetition_penalty: float = 1.2
    repetition_codebooks: int = 8
    seed: int | None = None
    speaking_rate_enabled: bool = False
    speed: float | None = None
    speaking_rate: float | None = None
    speaking_rate_bucket: int | None = None
    quality_enabled: bool = True
    # Per-feature quality bucket indices (dict keyed by feature or list in
    # feature order). quality_values takes raw metric values instead.
    quality_buckets: Dict[str, int | None] | List[int | None] | None = Field(
        default_factory=lambda: dict(_DEFAULT_QUALITY_BUCKETS)
    )
    quality_values: Dict[str, float | None] | List[float | None] | None = None
    # Mark the speaker embedding as having a clean background.
    clean_speaker_background: bool = False
    # Accurate mode (on) vs expressive mode (off).
    accurate_mode: bool = True
    stream: bool = True
    speaker_audio_base64: str | None = None
    speaker_audio_name: str | None = None
    speaker_embedding_base64: str | None = None
    speaker_embedding_name: str | None = None
    speaker_embedding_id: str | None = None
    speaker_blend_embedding_id_a: str | None = None
    speaker_blend_embedding_id_b: str | None = None
    speaker_blend_t: float | None = None
    speaker_wav_base64: str | None = None


class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible TTS request."""

    model: str
    input: str
    voice: str = "alloy"
    response_format: str = "pcm"
    speaking_rate_enabled: bool = False
    speed: float = 1.0
    speaking_rate: float | None = None
    speaking_rate_bucket: int | None = None
    repetition_window: int = 0
    repetition_penalty: float = 1.0
    repetition_codebooks: int = -1
    speaker_audio_base64: str | None = None
    speaker_audio_name: str | None = None
    speaker_embedding_base64: str | None = None
    speaker_embedding_name: str | None = None
    speaker_embedding_id: str | None = None
    speaker_blend_embedding_id_a: str | None = None
    speaker_blend_embedding_id_b: str | None = None
    speaker_blend_t: float | None = None
    speaker_wav_base64: str | None = None


class TTSSpeakerCacheRequest(BaseModel):
    label: str | None = None
    speaker_audio_base64: str | None = None
    speaker_audio_name: str | None = None
    speaker_embedding_base64: str | None = None
    speaker_embedding_name: str | None = None
    speaker_wav_base64: str | None = None


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[TTSAudioReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
    start_time_map: Dict[int, float] = field(default_factory=dict)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        self.start_time_map[uid] = time.perf_counter()
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    # Replies can keep arriving after a client abort; drop them.
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]

        while True:
            await event.wait()
            event.clear()

            pending = self.ack_map.get(uid)
            if pending is None:
                # The request was aborted while we were waiting.
                return
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack
            if ack and ack.finished:
                break

        self.ack_map.pop(uid, None)
        self.event_map.pop(uid, None)

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        self.start_time_map.pop(uid, None)
        logger.warning("Aborting request for user %s", uid)

    async def stream_tts_audio(self, uid: int, fade_out_ms: float = 0.0):
        """Stream TTS audio chunks as raw PCM bytes.

        When fade_out_ms > 0, the last fade_out_ms of audio is withheld until
        the stream finishes so a cosine fade-out can be applied to the tail.
        """
        t_start = self.start_time_map.pop(uid, None)
        first_byte = True
        total_audio_bytes = 0
        chunk_count = 0
        fade_bytes = int(44100 * max(0.0, fade_out_ms) / 1000.0) * 4
        held = b""
        async for ack in self.wait_for_ack(uid):
            if isinstance(ack, TTSAudioReply):
                if ack.audio_data:
                    if first_byte and t_start is not None:
                        logger.info(
                            "uid=%d TTS TTFB=%.3fms", uid,
                            (time.perf_counter() - t_start) * 1000,
                        )
                        first_byte = False
                    total_audio_bytes += len(ack.audio_data)
                    chunk_count += 1
                    if fade_bytes > 0:
                        held += ack.audio_data
                        if len(held) > fade_bytes:
                            yield held[:-fade_bytes]
                            held = held[-fade_bytes:]
                    else:
                        yield ack.audio_data
                if ack.finished:
                    break
        if held:
            yield _apply_fade_out_pcm(held, fade_out_ms)
        if t_start is not None:
            e2e_s = time.perf_counter() - t_start
            audio_s = total_audio_bytes / (44100 * 4)  # float32 @ 44.1kHz
            logger.info(
                "uid=%d TTS completed: E2E=%.3fms, audio=%.2fs, chunks=%d, RTF=%.2fx",
                uid, e2e_s * 1000, audio_s, chunk_count,
                audio_s / e2e_s if e2e_s > 0 else 0,
            )

    async def wait_for_tts_audio(self, uid: int, fade_out_ms: float = 0.0):
        """Collect all TTS audio data and return as single bytes object."""
        t_start = self.start_time_map.pop(uid, None)
        audio_data = b""
        async for ack in self.wait_for_ack(uid):
            if isinstance(ack, TTSAudioReply):
                if ack.audio_data:
                    audio_data += ack.audio_data
                if ack.finished:
                    break
        audio_data = _apply_fade_out_pcm(audio_data, fade_out_ms)
        if t_start is not None:
            e2e_s = time.perf_counter() - t_start
            audio_s = len(audio_data) / (44100 * 4)
            logger.info(
                "uid=%d TTS completed (non-stream): E2E=%.3fms, audio=%.2fs, RTF=%.2fx",
                uid, e2e_s * 1000, audio_s,
                audio_s / e2e_s if e2e_s > 0 else 0,
            )
        return audio_data

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="Zonos2 API Server", version="0.0.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_ui():
    """Serve the TTS web UI."""
    if _UI_HTML.exists():
        return FileResponse(
            _UI_HTML,
            media_type="text/html",
            headers={"Cache-Control": "no-store, max-age=0"},
        )
    return {"error": "tts_ui.html not found"}


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


# =============================================================================
# TTS Endpoints
# =============================================================================


@app.post("/v1/audio/speech")
async def create_speech(
    req: OpenAISpeechRequest,
    x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID"),
):
    """OpenAI-compatible speech endpoint."""
    logger.debug("Received TTS speech request: %s", req.input[:50])
    state = get_global_state()
    try:
        speaker_embedding = await _resolve_speaker_embedding(
            config=state.config,
            session_id=x_tts_session_id,
            speaker_audio_base64=req.speaker_audio_base64,
            speaker_audio_name=req.speaker_audio_name,
            speaker_embedding_base64=req.speaker_embedding_base64,
            speaker_embedding_name=req.speaker_embedding_name,
            speaker_embedding_id=req.speaker_embedding_id,
            speaker_blend_embedding_id_a=req.speaker_blend_embedding_id_a,
            speaker_blend_embedding_id_b=req.speaker_blend_embedding_id_b,
            speaker_blend_t=req.speaker_blend_t,
            legacy_speaker_wav_base64=req.speaker_wav_base64,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        speaking_rate_bucket = _resolve_speaking_rate_bucket(
            state.config,
            speaking_rate_bucket=req.speaking_rate_bucket,
            speaking_rate=req.speaking_rate,
            speed=req.speed if _field_was_set(req, "speed") else None,
            speaking_rate_enabled=req.speaking_rate_enabled,
        )
        max_tokens = _model_tts_max_tokens(state.config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    uid = state.new_user()

    await state.send_one(
        TTSTokenizeMsg(
            uid=uid,
            text=req.input,
            sampling_params=TTSSamplingParams(
                max_tokens=max_tokens,
                repetition_window=req.repetition_window,
                repetition_penalty=req.repetition_penalty,
                repetition_codebooks=req.repetition_codebooks,
            ),
            speaker_embedding=speaker_embedding,
            speaking_rate_bucket=speaking_rate_bucket,
            quality_buckets=_default_quality_buckets(state.config),
        )
    )

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_tts_audio(uid),
        media_type="audio/pcm",
        headers={
            "X-Audio-Sample-Rate": "44100",
            "X-Audio-Channels": "1",
            "X-Audio-Format": "float32",
        },
        background=BackgroundTask(_abort),
    )


@app.post("/tts/generate")
async def tts_generate(
    req: TTSGenerateRequest,
    x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID"),
):
    """Simple TTS generation endpoint."""
    logger.debug("Received TTS generate request: %s", req.text[:50])
    state = get_global_state()
    try:
        speaker_embedding = await _resolve_speaker_embedding(
            config=state.config,
            session_id=x_tts_session_id,
            speaker_audio_base64=req.speaker_audio_base64,
            speaker_audio_name=req.speaker_audio_name,
            speaker_embedding_base64=req.speaker_embedding_base64,
            speaker_embedding_name=req.speaker_embedding_name,
            speaker_embedding_id=req.speaker_embedding_id,
            speaker_blend_embedding_id_a=req.speaker_blend_embedding_id_a,
            speaker_blend_embedding_id_b=req.speaker_blend_embedding_id_b,
            speaker_blend_t=req.speaker_blend_t,
            legacy_speaker_wav_base64=req.speaker_wav_base64,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        language = _normalize_tts_request_language(req.language)
        speaking_rate_bucket = _resolve_speaking_rate_bucket(
            state.config,
            speaking_rate_bucket=req.speaking_rate_bucket,
            speaking_rate=req.speaking_rate,
            speed=req.speed,
            speaking_rate_enabled=req.speaking_rate_enabled,
        )
        quality_buckets = _resolve_quality_buckets(
            state.config,
            quality_buckets=req.quality_buckets,
            quality_values=req.quality_values,
            quality_enabled=req.quality_enabled and _model_quality_num_buckets(state.config) > 0,
        )
        max_tokens = _resolve_tts_max_tokens(state.config, req.max_tokens)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    uid = state.new_user()

    await state.send_one(
        TTSTokenizeMsg(
            uid=uid,
            text=req.text,
            language=language,
            text_normalization=req.text_normalization,
            sampling_params=TTSSamplingParams(
                temperature=req.temperature,
                topk=req.topk,
                top_p=req.top_p,
                min_p=req.min_p,
                max_tokens=max_tokens,
                repetition_window=req.repetition_window,
                repetition_penalty=req.repetition_penalty,
                repetition_codebooks=req.repetition_codebooks,
                seed=req.seed,
            ),
            speaker_embedding=speaker_embedding,
            clean_speaker_background=req.clean_speaker_background,
            accurate_mode=req.accurate_mode,
            speaking_rate_bucket=speaking_rate_bucket,
            quality_buckets=quality_buckets,
        )
    )

    async def _abort():
        await state.abort_user(uid)

    if req.stream:
        return StreamingResponse(
            state.stream_tts_audio(uid, fade_out_ms=req.fade_out_ms),
            media_type="audio/pcm",
            headers={
                "X-Audio-Sample-Rate": "44100",
                "X-Audio-Channels": "1",
                "X-Audio-Format": "float32",
            },
            background=BackgroundTask(_abort),
        )
    else:
        # Collect full audio then return
        audio_data = await state.wait_for_tts_audio(uid, fade_out_ms=req.fade_out_ms)
        return Response(
            content=audio_data,
            media_type="audio/pcm",
            headers={
                "X-Audio-Sample-Rate": "44100",
                "X-Audio-Channels": "1",
                "X-Audio-Format": "float32",
            },
        )


@app.post("/tts/speakers")
async def tts_cache_speaker(
    req: TTSSpeakerCacheRequest,
    x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID"),
):
    """Cache a speaker embedding for later selection in the current session."""
    state = get_global_state()
    try:
        session_id = _require_session_id(x_tts_session_id)
        if not _model_supports_speaker(state.config):
            raise ValueError("Current model does not support speaker embeddings.")

        speaker_audio_base64 = req.speaker_audio_base64 or req.speaker_wav_base64
        direct_source_count = sum(
            bool(value)
            for value in (speaker_audio_base64, req.speaker_embedding_base64)
        )
        if direct_source_count == 0:
            raise ValueError("Provide an audio file or saved embedding file to cache.")
        if direct_source_count > 1:
            raise ValueError(
                "Provide either a reference audio file or a saved embedding file, not both."
            )

        expected_dim = _model_speaker_dim(state.config)
        if speaker_audio_base64:
            audio_name = req.speaker_audio_name
            audio_bytes = _decode_base64_blob(speaker_audio_base64, "speaker_audio_base64")
            preview_wav_bytes = _transcode_audio_bytes_to_wav(audio_bytes)
            wav, sample_rate = _decode_wav_bytes(preview_wav_bytes)
            embedding = await _compute_speaker_embedding_from_waveform(
                wav,
                sample_rate,
                expected_dim=expected_dim,
            )
            entry = await _cache_speaker_reference(
                session_id=session_id,
                label=_default_speaker_label(req.label, audio_name, fallback="Speaker audio"),
                source_type="audio",
                embedding=embedding,
                original_name=audio_name,
                audio_bytes=preview_wav_bytes,
                audio_media_type="audio/wav",
            )
        else:
            embedding_name = req.speaker_embedding_name
            embedding = _compute_speaker_embedding_from_file(
                req.speaker_embedding_base64,
                expected_dim=expected_dim,
                speaker_embedding_name=embedding_name,
            )
            entry = await _cache_speaker_reference(
                session_id=session_id,
                label=_default_speaker_label(req.label, embedding_name, fallback="Speaker embedding"),
                source_type="embedding_file",
                embedding=embedding,
                original_name=embedding_name,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _serialize_cached_speaker(entry)


@app.get("/tts/speakers")
async def tts_list_speakers(
    x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID"),
):
    """List default disk speakers and cached speakers for the current session."""
    state = get_global_state()
    expected_dim = _model_speaker_dim(state.config)
    default_entries = await _list_default_speakers(state.config)

    entries: list[CachedSpeakerReference] = []
    session_id = _normalize_session_id(x_tts_session_id)
    if session_id is not None:
        entries = await _list_cached_speakers(session_id)

    return {
        "speakers": [
            *[_serialize_default_speaker(entry, expected_dim) for entry in default_entries],
            *[_serialize_cached_speaker(entry) for entry in entries],
        ]
    }


@app.get("/tts/speakers/{speaker_id}/preview")
async def tts_preview_speaker(
    speaker_id: str,
    x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID"),
):
    """Return the cached reference audio for a speaker, when available."""
    state = get_global_state()
    try:
        default_entry = await _get_default_speaker(state.config, speaker_id)
        if default_entry is not None:
            audio_bytes, media_type = await _default_speaker_preview(default_entry)
            return Response(content=audio_bytes, media_type=media_type)

        session_id = _require_session_id(x_tts_session_id)
        entry = await _get_cached_speaker(session_id, speaker_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not entry.audio_bytes:
        raise HTTPException(status_code=404, detail="Cached speaker preview audio is not available.")

    return Response(
        content=entry.audio_bytes,
        media_type=entry.audio_media_type or "audio/wav",
    )


@app.get("/tts/capabilities")
async def tts_capabilities():
    """Return TTS feature flags for the currently loaded model."""
    from zonos2.tokenizer.textnorm import SERVER_TO_NEMO_LANG, normalization_enabled

    state = get_global_state()
    speaking_rate_num_buckets = _model_speaking_rate_num_buckets(state.config)
    quality_num_buckets = _model_quality_num_buckets(state.config)
    return {
        "text_normalization_enabled": normalization_enabled(),
        "text_norm_languages": list(SERVER_TO_NEMO_LANG),
        "speaker_enabled": _model_supports_speaker(state.config),
        "speaker_embedding_dim": _model_speaker_dim(state.config),
        "n_codebooks": int(getattr(state.config.model_config, "n_codebooks", 9)),
        "max_tokens": _model_tts_max_tokens(state.config),
        "speaking_rate_enabled": speaking_rate_num_buckets > 0,
        "speaking_rate_num_buckets": speaking_rate_num_buckets,
        "speaking_rate_buckets": _model_speaking_rate_buckets(state.config),
        "quality_enabled": quality_num_buckets > 0,
        "quality_num_buckets": quality_num_buckets,
        "quality_features": _model_quality_features(state.config),
        "quality_buckets": _model_quality_buckets(state.config),
        "quality_dropout_by_feature": _model_quality_dropout_by_feature(state.config),
        "default_quality_buckets": dict(_DEFAULT_QUALITY_BUCKETS),
        "speaker_background_token_enabled": _model_speaker_background_token_enabled(
            state.config
        ),
        "accurate_mode_token_enabled": _model_accurate_mode_token_enabled(state.config),
        "speaker_audio_upload": True,
        "speaker_embedding_upload": True,
        "speaker_embedding_cache": True,
        "speaker_embedding_blend": True,
        "default_voices_enabled": bool(_default_voice_root(state.config)),
    }


async def tts_shell():
    """Interactive TTS shell for generating and playing audio."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter

    commands = [
        "/exit",
        "/save",
        "/play",
        "/temperature",
        "/topk",
        "/repwindow",
        "/reppenalty",
        "/repcodebooks",
    ]
    completer = WordCompleter(commands)
    session = PromptSession("TTS> ", completer=completer)

    # TTS settings
    temperature = 1.15
    topk = 106
    repetition_window = 50
    repetition_penalty = 1.2
    repetition_codebooks = 8
    last_audio: bytes | None = None

    # Try to import sounddevice for audio playback
    try:
        import numpy as np
        import sounddevice as sd

        can_play_audio = True
    except ImportError:
        can_play_audio = False
        print("Note: Install 'sounddevice' to enable audio playback")

    print("TTS Shell - Enter text to generate speech")
    print(
        "Commands: /exit, /save <filename>, /play, /temperature <val>, "
        "/topk <val>, /repwindow <frames>, /reppenalty <val>, "
        "/repcodebooks <count>"
    )
    print()

    try:
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue

            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                elif cmd.startswith("/save"):
                    parts = cmd.split(maxsplit=1)
                    if len(parts) < 2:
                        print("Usage: /save <filename.wav>")
                        continue
                    filename = parts[1]
                    if last_audio is None:
                        print("No audio to save. Generate some speech first.")
                        continue
                    _save_wav(last_audio, filename)
                    print(f"Saved to {filename}")
                    continue
                elif cmd == "/play":
                    if not can_play_audio:
                        print("Audio playback not available. Install 'sounddevice'.")
                        continue
                    if last_audio is None:
                        print("No audio to play. Generate some speech first.")
                        continue
                    audio = np.frombuffer(last_audio, dtype=np.float32)
                    sd.play(audio, samplerate=44100)
                    sd.wait()
                    continue
                elif cmd.startswith("/temperature"):
                    parts = cmd.split()
                    if len(parts) < 2:
                        print(f"Current temperature: {temperature}")
                        continue
                    temperature = float(parts[1])
                    print(f"Temperature set to {temperature}")
                    continue
                elif cmd.startswith("/topk"):
                    parts = cmd.split()
                    if len(parts) < 2:
                        print(f"Current topk: {topk}")
                        continue
                    topk = int(parts[1])
                    print(f"Top-k set to {topk}")
                    continue
                elif cmd.startswith("/repwindow"):
                    parts = cmd.split()
                    if len(parts) < 2:
                        print(f"Current repetition window: {repetition_window}")
                        continue
                    repetition_window = max(int(parts[1]), 0)
                    print(f"Repetition window set to {repetition_window}")
                    continue
                elif cmd.startswith("/reppenalty"):
                    parts = cmd.split()
                    if len(parts) < 2:
                        print(f"Current repetition penalty: {repetition_penalty}")
                        continue
                    repetition_penalty = max(float(parts[1]), 1.0)
                    print(f"Repetition penalty set to {repetition_penalty}")
                    continue
                elif cmd.startswith("/repcodebooks"):
                    parts = cmd.split()
                    if len(parts) < 2:
                        print(f"Current repetition codebooks: {repetition_codebooks}")
                        continue
                    repetition_codebooks = int(parts[1])
                    print(f"Repetition codebooks set to {repetition_codebooks}")
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    continue

            # Generate TTS
            state = get_global_state()
            uid = state.new_user()

            await state.send_one(
                TTSTokenizeMsg(
                    uid=uid,
                    text=cmd,
                    sampling_params=TTSSamplingParams(
                        temperature=temperature,
                        topk=topk,
                        repetition_window=repetition_window,
                        repetition_penalty=repetition_penalty,
                        repetition_codebooks=repetition_codebooks,
                    ),
                )
            )

            # Collect audio
            print("Generating...", end="", flush=True)
            audio_chunks = []
            async for ack in state.wait_for_ack(uid):
                if isinstance(ack, TTSAudioReply):
                    if ack.audio_data:
                        audio_chunks.append(ack.audio_data)
                        print(".", end="", flush=True)
                    if ack.finished:
                        break

            if audio_chunks:
                last_audio = b"".join(audio_chunks)
                duration = len(last_audio) / (44100 * 4)  # float32 = 4 bytes
                print(f" Done! ({duration:.2f}s)")

                # Auto-play if sounddevice is available
                if can_play_audio:
                    audio = np.frombuffer(last_audio, dtype=np.float32)
                    sd.play(audio, samplerate=44100)
                    sd.wait()
            else:
                print(" No audio generated")

    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting TTS shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def _save_wav(audio_bytes: bytes, path: str, sample_rate: int = 44100) -> None:
    """Save PCM audio bytes to a WAV file."""
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


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(tts_shell())
