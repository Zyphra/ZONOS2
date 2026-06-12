# Structure

Zonos2 is now a TTS-only serving stack for Zonos2 checkpoints (Hugging Face repo id or local path).

## Processes

- **API server**: FastAPI frontend for `/tts/generate`, `/v1/audio/speech`, capabilities, and speaker cache endpoints.
- **Tokenizer worker**: Builds byte-token TTS prompt frames and runs DAC detokenization for audio frames.
- **Scheduler worker**: Runs one scheduler per tensor-parallel rank, manages request queues, KV cache slots, and decode scheduling.
- **Engine**: Owns the model, attention backend, KV cache, CUDA graphs, and checkpoint weights for one rank.

The processes communicate through ZeroMQ. Tensor-parallel ranks use `torch.distributed` and NCCL/PyNCCL for GPU communication.

## Request Flow

1. The API server receives text plus optional speaker and speaking-rate conditioning.
2. The tokenizer worker turns text into byte prompt frames and inserts the optional speaker slot.
3. Rank 0 scheduler receives the request and broadcasts it to other TP ranks when needed.
4. The engine runs Zonos2 prefill/decode and samples multi-codebook audio frames.
5. The scheduler sends audio code frames to the tokenizer worker.
6. The tokenizer worker decodes frames through DAC and streams PCM bytes back to the API server.

## Code Organization

- `zonos2.core`: TTS request, batch, context, and sampling dataclasses.
- `zonos2.message`: ZeroMQ message dataclasses, with TTS messages in `message/tts.py`.
- `zonos2.tts`: Prompt building, sampling helpers, sequence helpers, and the offline `TTSLLM` wrapper.
- `zonos2.tokenizer`: TTS tokenizer/detokenizer worker and DAC vocoder manager.
- `zonos2.scheduler`: TTS scheduler, decode manager, table manager, and cache wiring.
- `zonos2.engine`: Model execution, checkpoint loading, KV cache setup, and CUDA graph replay.
- `zonos2.models`: Zonos2 model, config conversion, local checkpoint weight loading, and speaker embedding model.
- `zonos2.attention`: FlashAttention and FlashInfer backends.
- `zonos2.kvcache`: KV cache pools and cache managers.
- `zonos2.server`: CLI parsing, process launch, and HTTP API.
- `zonos2.kernel`: Custom CUDA kernels and bindings.
