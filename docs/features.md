# Features

## TTS Serving

Zonos2 serves Zonos2 TTS checkpoints from the Hugging Face Hub (e.g. `Zyphra/ZONOS2`) or a local path. The HTTP API exposes:

- `POST /tts/generate` for the native TTS API.
- `POST /v1/audio/speech` for an OpenAI-style speech endpoint.
- `GET /tts/capabilities` for model feature flags.
- `GET/POST /tts/speakers` for speaker reference upload, cache, and selection.

For command-line arguments, run:

```bash
python -m zonos2 --help
```

## Text Prompting

Text is encoded directly as UTF-8 byte IDs. The prompt builder also supports speaking-rate bucket conditioning when the checkpoint config enables it.

## Speaker Conditioning

Speaker-conditioned checkpoints can use saved `.npy`/`.npz` embeddings or uploaded audio. Audio uploads are converted to 2048D embeddings with the retained Qwen3 voice embedding model.

## Emotion Control

Additive emotion-direction vectors nudge a voice toward an emotion (happy/sad/angry/surprised) or along valence/arousal axes while preserving speaker identity. Directions load from `--tts-emotion-directions-dir` (defaults to `./emotion_directions/`, auto-enabling sliders when present); per-request via `emotion_enabled` + `emotion_sliders`/`emotion_valence`/`emotion_arousal`, with `emotion_strength` and `emotion_cfg_scale` controlling intensity. Build sets with `scripts/build_emotion_directions.py` and calibrate strength with `scripts/calibrate_emotion_strength.py`.

## Distributed Serving

Tensor parallelism is available with `--tp-size n`. Rank 0 handles API-facing scheduler messages and broadcasts work to the remaining ranks.

## Attention Backends

Zonos2 supports FlashAttention and FlashInfer backends. Use `--attn` to select a backend, or a comma-separated prefill/decode pair such as `--attn fa,fi`.

## CUDA Graph

Decode CUDA graph capture is enabled by default. Set `--cuda-graph-max-bs 0` to disable it, or use a positive value to control the maximum captured batch size.

## Radix Cache

The KV cache manager defaults to radix caching. Use `--cache naive` to select the naive cache manager.
