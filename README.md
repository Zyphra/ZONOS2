# ZONOS2

<p align="center">
  <img src="./assets/ZONOS2BlogThumbnail.png" alt="ZONOS2 title card" width="750" />
</p>

<div align="center">
  <a href="https://discord.gg/gTW9JwST8q" target="_blank">
    <img src="https://img.shields.io/badge/Join%20Our%20Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
</div>

---


ZONOS2 is our latest text-to-speech model trained on more than 6 million hours of varied multilingual speech, delivering expressiveness and quality on par with—or even surpassing—top TTS providers at low latency with MoE. ZONOS2 excels at high-fidelity and naturalistic voice cloning.


During inference we use nemo TN normalized UTF-8 bytes and an ECAPA-TDNN embedding to generate DAC tokens with our MoE backbone. An inference overview can be seen below.
<p align="center">
  <img src="./assets/zonos2_arlooop_animated.gif" alt="ZONOS2 title card" width="750" />
</p>

For local inference we provide a high-performance TTS inference server built on [Mini-SGLang](https://github.com/sgl-project/mini-sglang).

**For more details and speech samples, check out our [blog](https://www.zyphra.com/our-work/zonos2).**

**We also have a hosted version available at [cloud.zyphra.com/audio-playground](https://cloud.zyphra.com/audio-playground).**

---

## Quick Start

> **Platform Support**: Linux only (x86_64). Requires NVIDIA GPU with CUDA toolkit matching your driver version (`nvidia-smi` to check).

### 1. Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/Zyphra/Zonos2.git
cd Zonos2
uv sync
```

### 2. Launch the TTS Server

```bash
uv run python -m zonos2 --model-path Zyphra/ZONOS2 --tts-default-voices-dir ./default_voices/
```

`uv run` always uses the project environment, so no venv activation is needed.

The server starts on `http://localhost:1919` by default. TTS mode is auto-detected for zonos2 models.
`--tts-default-voices-dir <folder>` pre-populates the web UI with voice-clone
speakers from disk; the folder is scanned recursively for speaker audio
(`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.opus`, `.aac`, `.webm`) and saved
embeddings (`.npy`, `.npz`). The newest voice is selected automatically on
startup.

### 3. Generate Speech

**curl:**

```bash
curl -X POST http://localhost:1919/tts/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "stream": true}' \
  --output output.pcm

# Convert to WAV
ffmpeg -f f32le -ar 44100 -ac 1 -i output.pcm output.wav
```

**Web UI:** Open `http://localhost:1919/` in your browser.

## Python API (offline inference)

You can also run the engine directly in a Python script, without starting a
server, via `TTSLLM`:

```python
from zonos2.message import TTSSamplingParams
from zonos2.tts import TTSLLM

tts = TTSLLM(model_path="Zyphra/ZONOS2")

results = tts.generate(
    ["Hello from the offline Python API.", "Batched prompts work too."],
    TTSSamplingParams(seed=42),
)

for i, result in enumerate(results):
    print(f"frames={len(result['audio_tokens'])}, eos_frame={result['eos_frame']}")
    tts.save_audio(result["audio"], f"output_{i}.wav")
```


## API Reference

### `POST /tts/generate`

Full-featured TTS endpoint with streaming support.

Long text is automatically split into overlapping chunks, each synthesized as a
teacher-forced continuation of prior audio. This avoids boundary discontinuities and
voice drift over long passages; see the `long_form_*` parameters below to tune or
disable it.

**Request body:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to synthesize |
| `language` | string | `en_us` | Text-normalization language: `en_us`, `en_gb`, `fr_fr`, `de`, `es`, `it`, `pt_br`, `ja`, `cmn`, `ko` |
| `text_normalization` | bool | `true` | Verbalize numbers, dates and currency before synthesis (`false` = raw byte tokenization) |
| `temperature` | float | `1.15` | Sampling temperature |
| `topk` | int | `106` | Top-k sampling |
| `top_p` | float | `0.0` | Nucleus (top-p) sampling threshold; `0` disables |
| `min_p` | float | `0.18` | Min-p probability filter; `0` disables |
| `max_tokens` | int \| null | model max | Maximum audio tokens. Omit or set `null` to use the model context limit; long prompts are clamped to remaining context. |
| `fade_out_ms` | float | `0.0` | Cosine fade-out applied to the audio tail; `0` disables |
| `repetition_window` | int | `50` | Recent generated frames to check per codebook; `0` disables |
| `repetition_penalty` | float | `1.2` | Per-codebook repetition penalty strength; `1.0` disables |
| `repetition_codebooks` | int | `8` | Number of codebooks from CB0 upward to penalize; negative means all |
| `seed` | int \| null | `null` | Random seed for reproducibility |
| `speaking_rate_enabled` | bool | `false` | Set `true` to use model speaking-rate conditioning when another speaking-rate field is present |
| `speaking_rate_bucket` | int \| null | `null` | Exact model speaking-rate bucket to prepend before text |
| `speaking_rate` | float \| null | `null` | Target speaking rate in cleaned UTF-8 bytes per second; mapped to a bucket |
| `speed` | float \| null | `null` | OpenAI-style multiplier; `1.0` maps to the model's neutral speaking-rate bucket |
| `quality_enabled` | bool | `true` | Enable quality-bin conditioning on supported models |
| `quality_buckets` | object \| list \| null | `{"trailing_silence_s": 3}` | Per-feature quality bucket indices (keyed by feature name, or a list in feature order) |
| `quality_values` | object \| list \| null | `null` | Raw quality metric values, mapped to buckets server-side (alternative to `quality_buckets`) |
| `clean_speaker_background` | bool | `false` | Mark the reference voice as having a clean background (supported models) |
| `accurate_mode` | bool | `true` | `true` = accurate mode (closer voice match), `false` = expressive mode |
| `cfg_scale` | float | `1.0` | Speaker-embedding classifier-free guidance; `1.0` disables, `>1` pushes generation toward the target speaker |
| `prefix_cfg_scale` | float | `1.0` | Acoustic-prefix classifier-free guidance for long-form continuation chunks; `1.0` disables, `>1` sharpens continuity across chunk joins, `<1`/negative downweights the prefix. No effect on the first chunk |
| `long_form` | bool \| null | `null` | `null` = auto (engage when text exceeds the chunk size), `true`/`false` force long-form on/off |
| `long_form_chunk_chars` | int | `150` | New text generated per step, in characters |
| `long_form_window_chunks` | int | `2` | Total chunks fed per step; `window-1` prior chunks are teacher-forced context. `1` disables teacher forcing |
| `long_form_pin_anchor` | bool | `true` | Pin the whole first chunk into every continuation prefix to prevent timbre drift over long passages |
| `long_form_split_mode` | string | `"word"` | Chunking strategy: `"word"` (greedy word packing) or `"sentence"` (sentence boundaries, falling back to words for over-long sentences) |
| `stream` | bool | `true` | Stream audio chunks |

**Response:** Raw PCM audio (`audio/pcm`, float32, 44.1 kHz, mono). Headers include `X-Audio-Sample-Rate`, `X-Audio-Channels`, `X-Audio-Format`.

### `POST /v1/audio/speech`

OpenAI-compatible endpoint.

**Request body:**

```json
{
  "model": "zonos2",
  "input": "Hello world",
  "voice": "alloy",
  "response_format": "pcm"
}
```

For speaking-rate-enabled checkpoints, set `speaking_rate_enabled` to `true`
and use `speaking_rate_bucket` for exact bucket control, `speaking_rate` for
bytes-per-second control, or `speed` for OpenAI-style multiplier control.



## Citation
If you find this model useful in an academic context please cite as:
```
@misc{zyphra2025zonos,
  title     = {Zonos V2 Technical Report},
  author    = {Gabriel Clark, Sofian Mejjoute, Mohamed Osman, George Close, Beren Millidge},
  year      = {2026},
}
```
