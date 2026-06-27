"""Emotion control via sliders on the speaker-embedding space.

ZONOS2 conditions TTS on a raw speaker embedding (2048-D for the release
checkpoint). Emotion control works by *adding* precomputed "emotion direction"
vectors to that embedding before it is handed to the model:

    emb' = base + strength * (sum_e slider_e * dir_e + valence * v + arousal * a)

The direction vectors live in the same space as the speaker embedding and are
built offline by ``scripts/build_emotion_directions.py`` as the mean shift
between an emotion and neutral (``dir_e = mean(emotion) - mean(neutral)``), so a
slider value of ``1.0`` reproduces the *average* emotional shift seen in the
training data and larger values exaggerate it.

This module is intentionally dependency-light (torch + numpy + json) and does
no GPU work: applying emotion is pure CPU vector math, so it is shared by the
HTTP server and any offline caller. The only GPU-touching helper, ``embed_wav``,
lazily imports the speaker encoder and is used by the build script.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"

# Names treated as the dimensional affect axes rather than discrete emotions.
AXIS_NAMES = ("valence", "arousal")

CALIBRATION_NAME = "calibration.json"


@dataclass
class EmotionCalibration:
    """Per-speaker, per-emotion strength chosen offline to maximise the
    emotion-recogniser response (see scripts/calibrate_emotion_strength.py).

    ``strength(speaker_key, emotion)`` falls back per-emotion to ``default``
    and finally to ``global_default`` for unknown speakers/emotions.
    """

    by_speaker: Dict[str, Dict[str, float]] = field(default_factory=dict)
    default: Dict[str, float] = field(default_factory=dict)
    global_default: float = 1.0

    def strength(self, speaker_key: str | None, emotion: str) -> float:
        if speaker_key:
            s = self.by_speaker.get(speaker_key, {}).get(emotion)
            if s is not None:
                return float(s)
        s = self.default.get(emotion)
        if s is not None:
            return float(s)
        return float(self.global_default)

    @classmethod
    def load(cls, directory: str | Path) -> "EmotionCalibration | None":
        path = Path(directory).expanduser() / CALIBRATION_NAME
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        return cls(
            by_speaker={k: {e: float(v) for e, v in d.items()}
                        for k, d in data.get("by_speaker", {}).items()},
            default={e: float(v) for e, v in data.get("default", {}).items()},
            global_default=float(data.get("global_default", 1.0)),
        )


@dataclass
class EmotionDirections:
    """Loaded emotion direction vectors for one speaker-embedding space.

    Each direction is a 1-D float32 CPU tensor with ``dim`` elements, stored as
    the mean-difference vector (not unit-normalised).

    ``space`` selects where the direction lives and is applied:

    * ``"raw"`` -- the 2048-D speaker-embedding space the encoder produces.
      ``dim == input_dim``; the delta is added directly to the embedding.
    * ``"lda"`` -- the model's post-LDA speaker space (e.g. 1024-D for ZONOS2).
      Directions have ``dim == lda_dim``; ``apply_emotion`` projects the raw
      embedding through the model's LDA (``W x + b``), adds the delta there
      (so renormalisation happens in the space the model actually consumes),
      then maps back to a raw embedding via the LDA pseudo-inverse so the
      unchanged model re-applies LDA and recovers the injected vector.
    """

    dim: int
    named: Dict[str, torch.Tensor] = field(default_factory=dict)
    axes: Dict[str, torch.Tensor] = field(default_factory=dict)
    ref_base_norm: float | None = None
    space: str = "raw"
    input_dim: int | None = None  # speaker-embedding dim fed to the model
    # LDA matrices (only for space == "lda")
    lda_weight: torch.Tensor | None = None  # (lda_dim, input_dim)
    lda_bias: torch.Tensor | None = None    # (lda_dim,)
    lda_pinv: torch.Tensor | None = None    # (input_dim, lda_dim)

    @property
    def emotion_names(self) -> list[str]:
        return list(self.named.keys())

    @property
    def axis_names(self) -> list[str]:
        return list(self.axes.keys())

    @property
    def expected_input_dim(self) -> int:
        return int(self.input_dim if self.input_dim is not None else self.dim)

    def is_empty(self) -> bool:
        return not self.named and not self.axes

    @classmethod
    def load(cls, directory: str | Path) -> "EmotionDirections | None":
        """Load directions from a directory containing ``manifest.json``.

        Returns ``None`` when the directory or manifest is missing so callers
        can treat emotion control as simply unavailable.
        """
        root = Path(directory).expanduser()
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.is_file():
            return None

        manifest = json.loads(manifest_path.read_text())
        dim = int(manifest["dim"])
        space = str(manifest.get("space", "raw"))
        input_dim = int(manifest.get("input_dim", dim))
        ref_base_norm = manifest.get("ref_base_norm")
        named: Dict[str, torch.Tensor] = {}
        axes: Dict[str, torch.Tensor] = {}

        for name, entry in manifest.get("directions", {}).items():
            vec = _load_vector(root / entry["file"], expected_dim=dim, name=name)
            if str(entry.get("kind")) == "axis" or name in AXIS_NAMES:
                axes[name] = vec
            else:
                named[name] = vec

        lda_weight = lda_bias = lda_pinv = None
        if space == "lda":
            lda = manifest.get("lda", {})
            lda_weight = _load_matrix(root / lda["weight"])
            lda_bias = _load_matrix(root / lda["bias"])
            lda_pinv = _load_matrix(root / lda["pinv"])

        directions = cls(
            dim=dim,
            named=named,
            axes=axes,
            ref_base_norm=None if ref_base_norm is None else float(ref_base_norm),
            space=space,
            input_dim=input_dim,
            lda_weight=lda_weight,
            lda_bias=lda_bias,
            lda_pinv=lda_pinv,
        )
        if directions.is_empty():
            logger.warning("Emotion directions manifest at %s contains no directions.", manifest_path)
        return directions


def _load_vector(path: Path, *, expected_dim: int, name: str) -> torch.Tensor:
    arr = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected_dim:
        raise ValueError(
            f"Emotion direction '{name}' has dim {arr.shape[0]}, expected {expected_dim} ({path})."
        )
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=torch.float32, device="cpu")


def _load_matrix(path: Path) -> torch.Tensor:
    arr = np.asarray(np.load(path), dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=torch.float32, device="cpu")


def apply_emotion(
    base_embedding: torch.Tensor,
    *,
    directions: EmotionDirections | None,
    sliders: Mapping[str, float] | None = None,
    valence: float = 0.0,
    arousal: float = 0.0,
    strength: float = 1.0,
    preserve_norm: bool = True,
    strict: bool = False,
) -> torch.Tensor:
    """Return ``base_embedding`` nudged along the requested emotion directions.

    Args:
        base_embedding: Speaker embedding, shape ``(D,)`` or ``(1, D)``.
        directions: Loaded :class:`EmotionDirections`, or ``None`` (no-op).
        sliders: Mapping of named-emotion -> weight (typically in ``[-1, 1]``).
        valence: Weight along the valence axis (pleasant <-> unpleasant).
        arousal: Weight along the arousal axis (calm <-> excited).
        strength: Global gain applied to the combined delta.
        preserve_norm: Rescale the result to the base embedding's L2 norm so the
            injected vector keeps the magnitude the model was trained on.
        strict: Raise on unknown slider names instead of skipping them.

    The result is a contiguous 1-D float32 CPU tensor. When ``directions`` is
    ``None`` or nothing is requested, the base embedding is returned unchanged
    (as a 1-D float32 CPU tensor).
    """
    base = base_embedding.detach().to(dtype=torch.float32, device="cpu").reshape(-1).contiguous()
    if directions is None or directions.is_empty():
        return base
    if directions.space == "proj":
        # Post-projection directions are applied inside the model; callers use
        # emotion_hidden_delta() and pass the delta separately. Leave base as-is.
        return base

    if base.numel() != directions.expected_input_dim:
        raise ValueError(
            f"Speaker embedding dim {base.numel()} does not match emotion "
            f"directions input dim {directions.expected_input_dim}."
        )

    delta, requested = _combine_delta(directions, sliders, valence, arousal, strict)
    if not requested or float(strength) == 0.0:
        return base

    if directions.space == "lda":
        return _apply_lda(base, delta, float(strength), preserve_norm, directions)

    # Raw space: add the delta directly to the speaker embedding.
    emb = base + float(strength) * delta
    if preserve_norm:
        emb = _rescale_to(emb, torch.linalg.vector_norm(base))
    return emb.contiguous()


def _combine_delta(
    directions: EmotionDirections,
    sliders: Mapping[str, float] | None,
    valence: float,
    arousal: float,
    strict: bool,
) -> tuple[torch.Tensor, bool]:
    """Accumulate the weighted sum of requested direction vectors."""
    delta = torch.zeros(directions.dim, dtype=torch.float32)
    requested = False

    for name, weight in (sliders or {}).items():
        w = float(weight)
        if w == 0.0:
            continue
        vec = directions.named.get(name)
        if vec is None:
            vec = directions.axes.get(name)
        if vec is None:
            msg = f"Unknown emotion direction '{name}'. Available: {directions.emotion_names + directions.axis_names}."
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            continue
        delta += w * vec
        requested = True

    for axis_name, axis_weight in (("valence", valence), ("arousal", arousal)):
        w = float(axis_weight)
        if w == 0.0:
            continue
        vec = directions.axes.get(axis_name)
        if vec is None:
            if strict:
                raise ValueError(f"Emotion axis '{axis_name}' is not available.")
            continue
        delta += w * vec
        requested = True

    return delta, requested


def emotion_hidden_delta(
    directions: EmotionDirections | None,
    *,
    sliders: Mapping[str, float] | None = None,
    valence: float = 0.0,
    arousal: float = 0.0,
    strength: float = 1.0,
    strict: bool = False,
) -> torch.Tensor | None:
    """Combined hidden-space emotion delta for ``space="proj"`` directions.

    Returns the ``strength * Σ weight·direction`` vector (shape ``(hidden,)``)
    to be added to the projected speaker hidden state inside the model, or
    ``None`` when nothing is requested or directions are not post-projection.
    """
    if directions is None or directions.is_empty() or directions.space != "proj":
        return None
    delta, requested = _combine_delta(directions, sliders, valence, arousal, strict)
    if not requested or float(strength) == 0.0:
        return None
    return (float(strength) * delta).contiguous()


def _rescale_to(vec: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    cur = torch.linalg.vector_norm(vec)
    if float(cur) > 1e-8:
        return vec * (target_norm / cur)
    return vec


def _apply_lda(
    base: torch.Tensor,
    delta_lda: torch.Tensor,
    strength: float,
    preserve_norm: bool,
    directions: "EmotionDirections",
) -> torch.Tensor:
    """Inject the emotion delta in the model's post-LDA space.

    Projects the raw embedding through the model's LDA (``W x + b``), adds the
    delta there (renormalising in that space so the vector the model consumes
    keeps a stable magnitude), then maps back to a raw embedding via the LDA
    pseudo-inverse. The unchanged model re-applies LDA and recovers the
    injected vector (``W·W⁺ = I``).
    """
    W, b, Wp = directions.lda_weight, directions.lda_bias, directions.lda_pinv
    lda_base = W @ base + b                      # (lda_dim,)
    lda_new = lda_base + strength * delta_lda
    if preserve_norm:
        lda_new = _rescale_to(lda_new, torch.linalg.vector_norm(lda_base))
    raw_out = Wp @ (lda_new - b)                 # (input_dim,); model recovers lda_new
    return raw_out.contiguous()


def embed_wav(
    wav: torch.Tensor,
    sample_rate: int,
    *,
    embedder=None,
    device: str | None = None,
) -> torch.Tensor:
    """Encode a waveform into a pooled speaker embedding (GPU step).

    Lazily constructs a :class:`Qwen3SpeakerEmbedding` when ``embedder`` is not
    supplied. Pools any time dimension into a single vector and returns a 1-D
    float32 CPU tensor. Used by the offline direction-building script.
    """
    if embedder is None:
        from zonos2.models.speaker_cloning import Qwen3SpeakerEmbedding

        kwargs = {} if device is None else {"device": device}
        embedder = Qwen3SpeakerEmbedding(**kwargs)

    with torch.inference_mode():
        out = embedder(wav, sample_rate)

    if isinstance(out, (tuple, list)):
        out = out[0]
    out = out.squeeze(0).to(dtype=torch.float32, device="cpu")
    if out.ndim == 2:  # (seq, D) -> mean-pool over time
        out = out.mean(dim=0)
    return out.reshape(-1).contiguous()
