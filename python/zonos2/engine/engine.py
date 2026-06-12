from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, Tuple

import torch
from zonos2.attention import create_attention_backend
from zonos2.core import (
    Context,
    TTSBatch,
    TTSReq,
    TTSSamplingParams,
    set_global_ctx,
)
from zonos2.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from zonos2.kvcache import create_kvcache
from zonos2.layers import set_rope_device
from zonos2.models import create_model, load_checkpoint_weight
from zonos2.utils import divide_even, init_logger, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB

logger = init_logger(__name__)


def create_page_table(shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.int32, device=device)


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _get_frame_width(model_config) -> int:
    """Get frame width for multi-codebook models.

    Frame width = n_codebooks + (1 if text_vocab).
    """
    width = model_config.n_codebooks
    if model_config.text_vocab is not None:
        width += 1
    return width


class Engine:
    def __init__(self, config: EngineConfig):
        self.model_config = config.model_config
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        # Importing the model eagerly pulls in sgl_kernel, which creates a CUDA
        # context on the current device. The scheduler entry point binds this
        # process to its rank's GPU before that import, so an existing context is
        # expected; only one on the wrong device indicates a real ordering bug.
        if (
            torch.cuda.is_initialized()
            and torch.cuda.current_device() != self.device.index
        ):
            raise RuntimeError(
                f"CUDA was initialized on cuda:{torch.cuda.current_device()} before "
                f"the engine could bind TP rank {config.tp_info.rank} to {self.device}; "
                "set the device before importing model code."
            )
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # load model and determine number of pages
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config)

        # Log expected shapes before loading (for debugging dimension mismatches)
        logger.info_rank0(f"Model config: num_kv_heads={config.model_config.num_kv_heads}, head_dim={config.model_config.head_dim}")
        expected_kv_dim = config.model_config.num_kv_heads * config.model_config.head_dim
        logger.info_rank0(f"Expected kv_dim per TP rank: {expected_kv_dim}")

        state_dict = self._load_weight_state_dict(config)
        self._check_speaker_lda_weights(state_dict)

        # Debug: Check for temp keys in state_dict
        if logger.isEnabledFor(logging.DEBUG):
            temp_keys = [k for k in state_dict.keys() if '.temp' in k]
            logger.debug("Found %d temp keys in state_dict: %s", len(temp_keys), temp_keys[:5])
            if temp_keys:
                first_temp = state_dict[temp_keys[0]]
                logger.debug(
                    "First temp tensor: shape=%s, dtype=%s, values=%s",
                    first_temp.shape, first_temp.dtype, first_temp.flatten()[:4].tolist(),
                )

        # Validate wkv weight shapes match config (catch n_kv_heads mismatch early)
        for key, tensor in state_dict.items():
            if ".wkv." in key and "weight" in key:
                # wkv.weight should be (2, kv_dim, hidden_size) or (2*kv_dim, hidden_size)
                if tensor.dim() == 3:
                    actual_kv_dim = tensor.shape[1]
                else:
                    actual_kv_dim = tensor.shape[0] // 2
                if actual_kv_dim != expected_kv_dim:
                    inferred_kv_heads = actual_kv_dim // config.model_config.head_dim
                    raise ValueError(
                        f"KV dimension mismatch for {key}: checkpoint has kv_dim={actual_kv_dim} "
                        f"(implying n_kv_heads={inferred_kv_heads}), but config expects "
                        f"kv_dim={expected_kv_dim} (n_kv_heads={config.model_config.num_kv_heads}). "
                        f"Please ensure your checkpoint's params.json has the correct n_kv_heads value "
                        f"matching the trained model weights."
                    )
                break  # Only need to check one layer

        self.model.load_state_dict(state_dict)

        # Debug: print attention temp values to verify they loaded correctly
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Model type: %s", type(self.model).__name__)
            logger.debug("Model has layers: %s", hasattr(self.model, 'layers'))
            if hasattr(self.model, 'layers'):
                logger.debug("layers type: %s", type(self.model.layers))
                if hasattr(self.model.layers, 'op_list'):
                    logger.debug("num layers: %d", len(self.model.layers.op_list))
                    for i, layer in enumerate(self.model.layers.op_list[:2]):
                        logger.debug("Layer %d type: %s", i, type(layer).__name__)
                        if hasattr(layer, 'attention'):
                            attn = layer.attention
                            logger.debug("Layer %d attention type: %s", i, type(attn).__name__)
                            logger.debug("Layer %d has_qk_norm: %s", i, getattr(attn, 'has_qk_norm', 'N/A'))
                            if hasattr(attn, 'temp'):
                                temp = attn.temp
                                if temp is not None:
                                    logger.debug("Layer %d attention.temp: shape=%s, values=%s", i, temp.shape, temp.flatten()[:8].tolist())
                                else:
                                    logger.debug("Layer %d attention.temp is None", i)

        self.num_pages = self.dummy_page = self._determine_num_pages(init_free_memory, config)
        self.kv_cache = create_kvcache(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            device=self.device,
            dtype=self.dtype,
        )
        # NOTE: make page table 128 aligned (32 * sizeof(int32) == 128 bytes)
        self.max_seq_len = _align_up_32(min(config.max_seq_len, self.num_pages))
        self.page_table = create_page_table(  # + 1 for dummy request
            (config.max_running_req + 1, self.max_seq_len),
            device=self.device,
        )
        self.attn_backend = create_attention_backend(
            config.attention_backend,
            config.model_config,
            self.kv_cache,
            self.page_table,
        )
        self.ctx = Context(page_size=1, attn_backend=self.attn_backend)
        set_global_ctx(self.ctx)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        frame_width = _get_frame_width(self.model_config)
        self.dummy_req = TTSReq(
            input_ids=torch.zeros((1, frame_width), dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=TTSSamplingParams(),
            cache_handle=None,  # type: ignore
            n_codebooks=self.model_config.n_codebooks,
            eoa_id=self.model_config.eoa_id,
        )
        self.page_table[self.dummy_req.table_idx].fill_(self.dummy_page)

        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=self.max_seq_len,
            vocab_size=self.model_config.codebook_size + 2,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            return {
                k: v.to(self.dtype)
                for k, v in load_checkpoint_weight(config.model_path, self.device).items()
            }

    def _check_speaker_lda_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        lda_keys = [key for key in state_dict if key.startswith("speaker_lda_projection.")]
        if getattr(self.model, "speaker_lda_projection", None) is None:
            for key in lda_keys:
                state_dict.pop(key)
            if lda_keys:
                logger.warning_rank0(
                    "Dropped %d speaker_lda_projection weights because speaker LDA is not enabled.",
                    len(lda_keys),
                )
            return

        missing = {
            "speaker_lda_projection.weight",
            "speaker_lda_projection.bias",
        }.difference(lda_keys)
        if missing:
            raise ValueError(
                "speaker_lda_dim is configured but the checkpoint is missing "
                f"{sorted(missing)}. Use a checkpoint with the LDA projection merged in."
            )

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * self.model_config.head_dim
            * divide_even(self.model_config.num_kv_heads, config.tp_info.size)
            * config.page_size
            * self.dtype.itemsize
            * self.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-tokens"
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_pages} pages for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch_tts(self, batch: TTSBatch) -> torch.Tensor:
        """Forward pass for TTS batch, returning multi-codebook logits.

        Args:
            batch: TTS batch with 2D input tokens

        Returns:
            Logits tensor of shape (batch_size, n_codebooks, vocab_size)
        """
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            # For TTS, the model outputs multi-codebook logits directly
            # The zonos2 model has a MultiParallelLMHead that outputs
            # (batch, seq, n_codebooks, vocab_size)
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        # logits shape: (batch, n_codebooks, vocab) for decode
        # or (total_tokens, n_codebooks, vocab) for prefill
        # We only care about the last token per sequence
        if batch.is_decode:
            return logits[: batch.size]
        else:
            # For prefill, get the LAST token logits for each sequence
            # logits shape: (total_tokens, n_codebooks, vocab)
            # Need to extract logits at the last position of each sequence
            last_indices = []
            cumsum = 0
            for req in batch.reqs:
                cumsum += req.extend_len  # extend_len = device_len - cached_len
                last_indices.append(cumsum - 1)
            last_indices = torch.tensor(last_indices, device=logits.device)
            return logits[last_indices]

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()
