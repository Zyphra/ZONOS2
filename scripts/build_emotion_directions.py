#!/usr/bin/env python
"""Build emotion direction vectors in the speaker-embedding space.

Encodes an emotion-labelled audio corpus with the ZONOS2 speaker encoder and
derives, for each emotion, the mean shift relative to *neutral*:

    dir_e = mean(emb | emotion=e) - mean(emb | emotion=neutral)

When per-speaker neutral references are available the shift is computed per
speaker first and then averaged, which cancels speaker identity and isolates
the emotional component. Valence/arousal axes are synthesised from the named
directions using a standard affect layout. Outputs are written as
``<name>.npy`` plus a ``manifest.json`` consumable by
``zonos2.tts.emotion.EmotionDirections``.

Input can be given either as:

  * ``--dir ROOT --layout emotion``          ROOT/<emotion>/**/*.wav
  * ``--dir ROOT --layout speaker_emotion``  ROOT/<speaker>/<emotion>/**/*.wav  (e.g. ESD)
  * ``--manifest manifest.json``             {"emotion": [wavs]} or
                                             {"emotion": {"speaker": [wavs]}}

Example (GPU). ``--space proj`` projects the directions into the model's hidden
space, which is the set the server applies at inference (the shipped
``emotion_directions/`` was built this way):

    python scripts/build_emotion_directions.py \
        --dir /path/to/ESD/en --layout speaker_emotion \
        --space proj --lda-checkpoint /path/to/model/weights \
        --out emotion_directions --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from zonos2.tts.emotion import embed_wav

logger = logging.getLogger("build_emotion_directions")

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".webm"}

# Standard affect layout. Signs are applied to each named direction (which is
# already relative to neutral) to synthesise the valence/arousal axes. Only
# emotions actually present contribute.
DEFAULT_VALENCE_SIGNS = {
    "happy": 1.0, "amused": 1.0, "content": 1.0, "calm": 1.0, "excited": 0.5,
    "neutral": 0.0, "surprise": 0.0,
    "sad": -1.0, "angry": -1.0, "fear": -1.0, "fearful": -1.0, "disgust": -1.0,
}
DEFAULT_AROUSAL_SIGNS = {
    "angry": 1.0, "happy": 1.0, "surprise": 1.0, "surprised": 1.0, "fear": 1.0,
    "fearful": 1.0, "excited": 1.0,
    "neutral": -0.25, "disgust": 0.0,
    "sad": -1.0, "calm": -1.0, "content": -0.5, "sleepy": -1.0,
}


def _norm_label(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _gather_from_dir(root: Path, layout: str, pattern: str) -> Dict[str, Dict[str, List[Path]]]:
    """Return {emotion: {speaker: [wavs]}}."""
    out: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    if layout == "emotion":
        for emo_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            emo = _norm_label(emo_dir.name)
            for wav in sorted(emo_dir.glob(pattern)):
                if wav.suffix.lower() in AUDIO_EXTS:
                    out[emo]["_all"].append(wav)
    elif layout == "speaker_emotion":
        for spk_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            spk = spk_dir.name
            for emo_dir in sorted(p for p in spk_dir.iterdir() if p.is_dir()):
                emo = _norm_label(emo_dir.name)
                for wav in sorted(emo_dir.glob(pattern)):
                    if wav.suffix.lower() in AUDIO_EXTS:
                        out[emo][spk].append(wav)
    else:
        raise ValueError(f"Unknown layout '{layout}'.")
    return out


def _gather_from_manifest(path: Path) -> Dict[str, Dict[str, List[Path]]]:
    data = json.loads(path.read_text())
    out: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    for emo_raw, value in data.items():
        emo = _norm_label(emo_raw)
        if isinstance(value, dict):
            for spk, wavs in value.items():
                out[emo][spk].extend(Path(w) for w in wavs)
        else:
            out[emo]["_all"].extend(Path(w) for w in value)
    return out


def _load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """Load audio as a (channels, samples) float32 tensor + sample rate.

    Uses soundfile (libsndfile) which handles wav/flac/ogg without the
    TorchCodec backend that recent torchaudio.load requires.
    """
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, channels)
    wav = torch.from_numpy(np.ascontiguousarray(data.T))  # (channels, frames)
    return wav, int(sr)


def _load_state_dict(checkpoint: Path) -> dict:
    path = str(checkpoint)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_lda(checkpoint: Path) -> dict:
    """Load the model's affine LDA projection (weight, bias) + pseudo-inverse."""
    sd = _load_state_dict(checkpoint)
    W = sd["speaker_lda_projection.weight"].float()  # (lda_dim, input_dim)
    b = sd["speaker_lda_projection.bias"].float()     # (lda_dim,)
    Wp = torch.linalg.pinv(W)                          # (input_dim, lda_dim)
    return {"weight": W, "bias": b, "pinv": Wp}


def _load_speaker_chain(checkpoint: Path) -> dict:
    """Load LDA + speaker_projection so embeddings can be mapped to hidden space.

    hidden = A (W x + b_W) + b_A  =  speaker_projection(LDA(x)).
    """
    sd = _load_state_dict(checkpoint)
    W = sd["speaker_lda_projection.weight"].float()
    bW = sd["speaker_lda_projection.bias"].float()
    A = sd["speaker_projection.weight"].float()        # (hidden, lda_dim)
    bA = sd["speaker_projection.bias"].float()         # (hidden,)
    return {"W": W, "bW": bW, "A": A, "bA": bA}


def _embed_clips(wavs: List[Path], embedder, device: str) -> List[torch.Tensor]:
    embs: List[torch.Tensor] = []
    for wav_path in wavs:
        try:
            wav, sr = _load_audio(wav_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s (load failed: %s)", wav_path, exc)
            continue
        embs.append(embed_wav(wav, sr, embedder=embedder, device=device))
    return embs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", type=Path, help="Root directory of labelled audio.")
    src.add_argument("--manifest", type=Path, help="JSON manifest mapping emotion -> wavs (optionally per speaker).")
    parser.add_argument("--layout", choices=["emotion", "speaker_emotion"], default="speaker_emotion")
    parser.add_argument("--pattern", default="**/*", help="Glob within each emotion dir (default recursive).")
    parser.add_argument("--neutral", default="neutral", help="Label used as the neutral baseline.")
    parser.add_argument("--out", type=Path, default=Path("emotion_directions"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-axes", action="store_true", help="Skip valence/arousal axis synthesis.")
    parser.add_argument(
        "--space", choices=["raw", "lda", "proj"], default="raw",
        help="Build directions in the raw embedding space, the model's post-LDA "
             "space, or the post-speaker-projection hidden space.")
    parser.add_argument(
        "--lda-checkpoint", type=Path, default=None,
        help="Model .pth/.safetensors holding the speaker projection weights "
             "(required for --space lda/proj).")
    args = parser.parse_args()

    lda = None       # for space == "lda" (stores W,b,pinv in the output)
    chain = None     # for space == "proj" (projects to hidden; no matrices stored)
    if args.space in ("lda", "proj"):
        if args.lda_checkpoint is None:
            raise SystemExit(f"--space {args.space} requires --lda-checkpoint pointing at the model weights.")
        if args.space == "lda":
            lda = _load_lda(args.lda_checkpoint)
            logger.info("Loaded LDA projection (in=%d -> out=%d) from %s",
                        lda["weight"].shape[1], lda["weight"].shape[0], args.lda_checkpoint)
        else:
            chain = _load_speaker_chain(args.lda_checkpoint)
            logger.info("Loaded speaker chain LDA+proj (in=%d -> hidden=%d) from %s",
                        chain["W"].shape[1], chain["A"].shape[0], args.lda_checkpoint)

    if args.dir is not None:
        groups = _gather_from_dir(args.dir, args.layout, args.pattern)
    else:
        groups = _gather_from_manifest(args.manifest)

    neutral = _norm_label(args.neutral)
    if not groups:
        raise SystemExit("No audio found for any emotion.")
    logger.info("Found emotions: %s", ", ".join(sorted(groups)))

    from zonos2.models.speaker_cloning import Qwen3SpeakerEmbedding

    logger.info("Loading speaker encoder on %s ...", args.device)
    embedder = Qwen3SpeakerEmbedding(device=args.device)

    # Encode every clip, grouped by emotion and speaker.
    # emb_by[emotion][speaker] -> list[Tensor]
    emb_by: Dict[str, Dict[str, List[torch.Tensor]]] = defaultdict(dict)
    all_embs: List[torch.Tensor] = []
    for emo, by_spk in groups.items():
        for spk, wavs in by_spk.items():
            embs = _embed_clips(wavs, embedder, args.device)
            if lda is not None:
                # Project into the model's post-LDA space: W x + b.
                embs = [(lda["weight"] @ e.cpu() + lda["bias"]) for e in embs]
            elif chain is not None:
                # Project into the post-speaker-projection hidden space:
                # A (W x + bW) + bA.
                embs = [
                    (chain["A"] @ (chain["W"] @ e.cpu() + chain["bW"]) + chain["bA"])
                    for e in embs
                ]
            if embs:
                emb_by[emo][spk] = embs
                all_embs.extend(embs)
            logger.info("  %-12s spk=%-10s clips=%d", emo, spk, len(embs))

    if not all_embs:
        raise SystemExit("No clips could be encoded.")
    dim = all_embs[0].numel()
    ref_base_norm = float(torch.stack([e for e in all_embs]).norm(dim=1).mean())

    def emotion_mean(emo: str, spk: str) -> torch.Tensor | None:
        embs = emb_by.get(emo, {}).get(spk)
        if not embs:
            return None
        return torch.stack(embs).mean(dim=0)

    has_neutral = neutral in emb_by
    if not has_neutral:
        logger.warning(
            "No '%s' baseline found; directions will be relative to the global mean "
            "(less clean, identity not cancelled).", neutral,
        )
    global_mean = torch.stack(all_embs).mean(dim=0)

    # Build per-emotion directions.
    directions: Dict[str, torch.Tensor] = {}
    for emo, by_spk in emb_by.items():
        if emo == neutral:
            continue
        per_speaker_dirs: List[torch.Tensor] = []
        if has_neutral:
            for spk in by_spk:
                e_mean = emotion_mean(emo, spk)
                n_mean = emotion_mean(neutral, spk)
                if e_mean is None or n_mean is None:
                    continue
                per_speaker_dirs.append(e_mean - n_mean)
        if per_speaker_dirs:
            directions[emo] = torch.stack(per_speaker_dirs).mean(dim=0)
        else:
            # Fallback: emotion mean minus neutral global mean (or global mean).
            base = (
                torch.stack([m for spk in emb_by.get(neutral, {})
                             for m in [emotion_mean(neutral, spk)] if m is not None]).mean(dim=0)
                if has_neutral else global_mean
            )
            e_all = torch.stack([m for spk in by_spk
                                 for m in [emotion_mean(emo, spk)] if m is not None]).mean(dim=0)
            directions[emo] = e_all - base

    if not directions:
        raise SystemExit("No emotion directions could be derived (need at least one non-neutral emotion).")

    # Synthesise valence/arousal axes from the named directions.
    axes: Dict[str, torch.Tensor] = {}
    if not args.no_axes:
        for axis_name, signs in (("valence", DEFAULT_VALENCE_SIGNS), ("arousal", DEFAULT_AROUSAL_SIGNS)):
            contributions = [
                signs[emo] * vec
                for emo, vec in directions.items()
                if signs.get(emo, 0.0) != 0.0
            ]
            if contributions:
                axes[axis_name] = torch.stack(contributions).mean(dim=0)
            else:
                logger.warning("No emotions map to the %s axis; skipping it.", axis_name)

    # Write everything out.
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dirs: Dict[str, dict] = {}

    def _save(name: str, vec: torch.Tensor, kind: str) -> None:
        arr = vec.detach().to(torch.float32).cpu().numpy()
        np.save(out_dir / f"{name}.npy", arr)
        manifest_dirs[name] = {
            "file": f"{name}.npy",
            "kind": kind,
            "norm": float(np.linalg.norm(arr)),
        }

    for name, vec in directions.items():
        _save(name, vec, "named")
    for name, vec in axes.items():
        _save(name, vec, "axis")

    manifest = {
        "dim": int(dim),
        "ref_base_norm": ref_base_norm,
        "neutral": neutral,
        "space": args.space,
        "directions": manifest_dirs,
    }
    if lda is not None:
        input_dim = int(lda["weight"].shape[1])
        np.save(out_dir / "lda_weight.npy", lda["weight"].cpu().numpy())
        np.save(out_dir / "lda_bias.npy", lda["bias"].cpu().numpy())
        np.save(out_dir / "lda_pinv.npy", lda["pinv"].cpu().numpy())
        manifest["input_dim"] = input_dim
        manifest["lda"] = {"weight": "lda_weight.npy", "bias": "lda_bias.npy", "pinv": "lda_pinv.npy"}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(
        "Wrote %d directions (%d named, %d axes, space=%s) to %s",
        len(manifest_dirs), len(directions), len(axes), args.space, out_dir,
    )


if __name__ == "__main__":
    main()
