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

## Long-form Generation

Text longer than `long_form_chunk_chars` is split into chunks (`long_form_split_mode`: greedy `"word"` packing or `"sentence"` boundaries) and synthesized one chunk per step. Each step regenerates the current chunk as a teacher-forced continuation of the previous `long_form_window_chunks - 1` chunks, so the model stays grounded on already-spoken audio. Only whole chunks are ever fed as context, keeping the acoustic prefix aligned to its transcript text. Set `long_form` to `true`/`false` to force the mode on or off; the default (`null`) auto-engages when the text exceeds the chunk size.

With `long_form_pin_anchor` (default on), the whole first chunk is pinned into every continuation prefix alongside the rolling recent window (the middle is evicted), re-grounding timbre on a stable reference to prevent voice drift over long passages.

`prefix_cfg_scale` applies classifier-free guidance relative to the acoustic prefix, orthogonal to speaker guidance (`cfg_scale`). The unconditional twin drops the prefix and context text but keeps the speaker embedding; `>1` tightens continuity across chunk joins while `<1`/negative loosens it. It has no effect on the first chunk.

## Distributed Serving

Tensor parallelism is available with `--tp-size n`. Rank 0 handles API-facing scheduler messages and broadcasts work to the remaining ranks.

## Attention Backends

Zonos2 supports FlashAttention and FlashInfer backends. Use `--attn` to select a backend, or a comma-separated prefill/decode pair such as `--attn fa,fi`.

## CUDA Graph

Decode CUDA graph capture is enabled by default. Set `--cuda-graph-max-bs 0` to disable it, or use a positive value to control the maximum captured batch size.

## Radix Cache

The KV cache manager defaults to radix caching. Use `--cache naive` to select the naive cache manager.
