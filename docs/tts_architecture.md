# TTS System Architecture

## Overview

mini-sglang serves a multi-codebook TTS transformer (Zonos2) through a multi-process pipeline with streaming audio output. Text is encoded as UTF-8 byte token frames, fed through an autoregressive transformer that predicts 9 parallel audio codebook streams, and decoded to 44.1 kHz PCM audio by the DAC vocoder - all with incremental streaming to the HTTP client.

## Process Architecture

Four process types communicate over ZMQ:

```
                  ZMQ                ZMQ                ZMQ
  Frontend  ──────────►  Tokenizer  ──────────►  Scheduler  ──────────►  Detokenizer
  (FastAPI)  ◄──────────             ◄──────────  (Engine)   ◄──────────  (Vocoder)
             TTSAudioReply          TTSUserMsg             TTSDetokenizeMsg
```

| Process | Count | Role |
|---------|-------|------|
| Frontend | 1 | FastAPI server. Accepts `POST /tts/generate`, streams PCM chunks back. Logs TTFB, E2E latency, RTF. |
| Tokenizer | N (default 0 = shared with detokenizer) | Runs `TTSPromptBuilder`: text → byte tokens → 2D prompt tensor. Sends `TTSUserMsg` to scheduler. |
| Scheduler | 1 per TP rank | Runs `TTSScheduler` wrapping `Engine`. Autoregressive decode loop: prefill → decode → sample → emit frames. |
| Detokenizer | 1 | Runs `TTSVocoderManager`: audio codes → DAC → PCM bytes. Sends `TTSAudioReply` to frontend. |

Spawned by `launch_server()` in `server/launch.py`. Each scheduler process holds one GPU; tensor parallelism across ranks is coordinated via NCCL.

### Message types

Defined in `message/tts.py`:

| Message | Direction | Key fields |
|---------|-----------|------------|
| `TTSTokenizeMsg` | Frontend → Tokenizer | `uid`, `text`, `sampling_params`, optional speaker/rate conditioning |
| `TTSUserMsg` | Tokenizer → Scheduler | `uid`, `input_ids: Tensor (seq_len, frame_width)`, `sampling_params` |
| `TTSDetokenizeMsg` | Scheduler → Detokenizer | `uid`, `audio_codes: List[int]` (one frame), `finished`, `eos_frame` |
| `TTSAudioReply` | Detokenizer → Frontend | `uid`, `audio_data: bytes` (float32 PCM), `finished`, `sample_rate` |

## Token Format

Each time step is a **frame** of width `n_codebooks + 1`:

```
[cb0, cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, text_token]
 ----------- 9 audio codebooks -----------    byte/text id
```

- **Audio codebook tokens**: integers in `[0, 1023]` (codebook_size = 1024).
- **Special tokens**: `eoa_id = 1024` (end-of-audio), `audio_pad_id = 1025` (padding).
- **Text column**: byte token ID during prompt; `text_vocab` placeholder during generation.
- **Conditioning tokens**: the tail of the text vocabulary holds, in order, speaking-rate buckets, quality buckets (per feature), the clean/noisy speaker-background markers, and the accurate-mode marker. `text_vocab` itself is text padding.

During the prompt phase, audio columns are filled with `audio_pad_id` and the text column carries the byte-token sequence. Pre-computed silence frames (0.2 s, 17 frames) are appended after the text with a shear pattern applied.

### Shear pattern

The model is trained with an inter-codebook delay: codebook `j` is shifted by `j` frames. `shear()` applies this delay to the prompt; `shear_up()` inverts it before vocoder decoding.

## Request Lifecycle

```
1. HTTP POST /tts/generate
   └─ FrontendManager assigns uid, creates TTSTokenizeMsg

2. Tokenizer process
   ├─ TTSPromptBuilder.build(text)         # UTF-8 byte tokenization (tts/prompt.py)
   │   └─ per text token: [audio_pad_id]*9 + [text_token_id]
   ├─ append sheared silence tokens        # 17 frames of pre-computed silence
   └─ emit TTSUserMsg(input_ids: 2D tensor)

3. Scheduler
   ├─ Prefill: engine.forward_batch_tts()  # full prompt in one pass
   │   └─ extract last-token logits per sequence
   └─ Decode loop (frame by frame):
       ├─ engine.forward_batch_tts()       # single frame input
       ├─ sample_tts(logits)               # → [cb0..cb8, text_placeholder]
       ├─ TTSSequence.append_token()       # EOS detection
       ├─ emit TTSDetokenizeMsg per frame
       └─ repeat until is_finished

4. Detokenizer (TTSVocoderManager)
   ├─ accumulate frames per uid
   ├─ shear_up() to remove delay pattern
   ├─ DAC decode → float32 PCM @ 44.1 kHz
   └─ emit TTSAudioReply (streaming chunks)

5. Frontend streams audio_data bytes to HTTP client
```

## Model Architecture (Zonos2)

`Zonos2ForCausalLM` in `models/zonos2.py`.

```
Input (seq_len, frame_width)
  │
  ▼
MultiEmbedding              # sum of per-column embeddings → (seq_len, hidden_size)
  │
  ▼
emb_norm                    # RMSNorm (no learnable params)
  │
  ▼
TransformerBlock × N        # residual stream with fused RMSNorm
  │  ├─ attention_norm (RMSNormFused)
  │  ├─ Attention
  │  │   ├─ wq, wkv, wo projections
  │  │   ├─ QK norm + learnable temperature (per-head)
  │  │   ├─ RoPE (interleaved format)
  │  │   ├─ FlashAttention / FlashInfer backend
  │  │   └─ headwise gating (sigmoid)
  │  ├─ ffn_norm (RMSNormFused)
  │  ├─ FeedForward (dense) or MoEFeedForward
  │
  ▼
out_norm (RMSNormFused)
  │
  ▼
MultiOutputHead             # linear: hidden → (n_codebooks × audio_vocab)
  │                         #   reshape → (*, 9, 1026)
  ▼
softcap(logits, 15.0)      # tanh-based logit capping
```

### MultiEmbedding

Maintains separate `VocabParallelEmbedding` tables for each of the 9 audio codebooks plus the text column. Forward pass sums all column embeddings element-wise.

### Attention

- **QK normalization**: `F.rms_norm` on Q and K, then Q is scaled by `|temp|` (learnable per-head parameter).
- **Gating**: After attention output, applies `sigmoid(gater(x))` either per-head (`headwise`) or per-element (`elementwise`), then multiplies with the attention result.
- **RoPE**: Interleaved format (`is_neox=False`).

### MoE

Activated for middle layers (`moe_start_from_layer` to `num_layers - moe_end_from_layer`).

- **Router**: Projects hidden → `router_dim` via `down_proj`, then a 3-layer MLP with GELUs produces expert logits.
- **Expert Dropout Augmentation (EDA)**: Non-first MoE layers blend the previous layer's router hidden states (`router_states * scale + current`) before routing. This threads router context across consecutive MoE layers.
- **Experts**: `FusedGroupedExperts` using fused gate+up projection (`w1 || w3` → `gate_up_proj`) with SiLU activation, and a down projection (`w2` → `down_proj`). Uses the `FusedMoE` kernel for efficient dispatch.

## Sampling

`sample_tts()` in `tts/sampler.py`. Input: `(B, 9, vocab_size)` logits.

1. **Repetition penalty**: For each request and selected codebook, gather the last `repetition_window` generated tokens from that codebook only and apply `repetition_penalty` to matching logits. `repetition_codebooks=N` applies this to CB0 through CB(N-1), while a negative value means all codebooks. `repetition_window=0` or `repetition_penalty=1.0` disables this path.
2. **Temperature**: Per-sequence scaling, broadcast over codebooks.
3. **Top-k**: Single `torch.topk` with `min(top_ks)` across the batch.
4. **Top-p / Min-p**: Applied to the softmax probabilities using the most restrictive value across the batch; `top_p=0` and `min_p=0` disable them.
5. **Multinomial sampling**: Per-codebook independently. Supports per-request `torch.Generator` for seeded determinism.
6. **Output**: `[cb0, ..., cb8, text_vocab]` — audio tokens plus text placeholder.

Default params (`TTSSamplingParams`): temperature=1.15, topk=106, top_p=0.0, min_p=0.18, max_tokens=1024, repetition_window=50, repetition_penalty=1.2, repetition_codebooks=8.

## EOS Detection

Managed by `TTSReq.check_eos()` in `core.py`.

1. Each frame, check whether any delayed audio codebook sampled `eoa_id` (1024).
2. When detected: align `eos_frame` by subtracting the highest EOA codebook index, then start countdown = `n_codebooks + 1` (10 steps).
3. Sequence marked finished when countdown reaches 0 — this buffer allows frame alignment across codebooks due to the shear pattern.
4. Also terminates if `num_completion_tokens >= max_tokens`.

## Vocoder (DAC)

`TTSVocoderManager` in `tokenizer/vocoder.py`. Uses [DAC](https://github.com/descriptinc/descript-audio-codec) at 44.1 kHz.

### Streaming decode

- Frames accumulate in a per-request buffer.
- With `N` accumulated frames, only `N - (n_codebooks - 1)` can be fully decoded (the rest need future frames for `shear_up`).
- Decodes in chunks of `min_decode_chunk` (default 16) frames for efficiency.
- On stream end, decodes remaining frames with padding fill.

### Decode pipeline

```
raw audio codes (seq_len, 9)
  → shear_up(): remove inter-codebook delay, fill with audio_pad_id
  → clamp to [0, 1023]
  → permute to (batch, 9, seq_len) for DAC
  → dac.quantizer.from_codes() → continuous latent
  → dac.decode() → waveform (float32, 44.1 kHz)
```

## Tensor Parallelism

Weights are sharded across TP ranks:

| Component | Sharding |
|-----------|----------|
| `wq`, `gater` | Column-parallel (split output dim) |
| `wkv` | Split KV dim (dim 1 of 3D weight) |
| `wo` | Row-parallel with all-reduce |
| `w_in` (FFN) | Split intermediate dim (dim 1) |
| `w_out` (FFN) | Row-parallel with all-reduce |
| MoE `gate_up_proj` | Split intermediate (dim 1) |
| MoE `down_proj` | Split intermediate (dim 2) |
| Embeddings | Vocabulary-parallel |
| `multi_output` | Output vocabulary sharded |
| `attention.temp` | Split heads (dim 1) |

Sharding logic in `models/weight.py`. Communication via NCCL all-reduce in output projections.

## Configuration

Key `ServerArgs` fields (in `server/args.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `tts_n_codebooks` | `9` | Number of audio codebooks |
| `tts_audio_pad_id` | `1025` | Audio padding token |
| `tts_text_vocab` | `None` | Text vocabulary size (auto-detected from config) |
| `tts_sample_rate` | `44100` | Output audio sample rate |
| `tts_default_voices_dir` | `None` | Optional folder of audio or `.npy`/`.npz` embedding files pre-populated in the web UI speaker picker |

Model-level config is loaded from checkpoint via `cached_load_checkpoint_config()` and `ModelConfig.from_checkpoint_config()` in `models/config.py`. Key fields: `n_codebooks`, `codebook_size`, `text_vocab`, `eoa_id`, `audio_pad_id`, `loss_softcap`, `moe_n_experts`, `moe_start_from_layer`, `moe_router_dim`.

## Key Files

| Path | Role |
|------|------|
| `server/api_server.py` | HTTP endpoints, `FrontendManager` |
| `server/launch.py` | Process spawning |
| `server/args.py` | `ServerArgs` configuration |
| `tokenizer/server.py` | Tokenizer/detokenizer worker loop |
| `tts/prompt.py` | Byte tokenization, prompt rows, speaking-rate tokens, shear/silence |
| `tokenizer/vocoder.py` | `TTSVocoderManager`, DAC decode |
| `engine/engine.py` | `Engine.forward_batch_tts()` |
| `models/zonos2.py` | `Zonos2ForCausalLM` |
| `models/config.py` | `ModelConfig` |
| `models/weight.py` | Weight loading and TP sharding |
| `tts/sequence.py` | `TTSSequence`, EOS detection |
| `tts/sampler.py` | `sample_tts()` |
| `tts/llm.py` | `TTSLLM` offline generation interface |
| `message/tts.py` | Message dataclasses, `TTSSamplingParams` |
| `attention/` | FlashInfer / FlashAttention backends |
| `layers/` | Linear, norm, RoPE, MoE layers |
