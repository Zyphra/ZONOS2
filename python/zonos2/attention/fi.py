from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Literal

import torch
from zonos2.distributed import get_tp_info
from zonos2.env import ENV
from zonos2.utils import divide_even
from zonos2.utils.logger import init_logger

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData, make_positions

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from zonos2.core import TTSBatch
    from zonos2.kvcache import BaseKVCache
    from zonos2.models import ModelConfig


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


logger = init_logger(__name__)

_fi_debug_counter = 0


@dataclass
class FICaptureData(BaseCaptureData):
    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        return self.page_table


@dataclass
class FIMetadata(BaseAttnMetadata):
    # fmt: off
    cu_seqlens_q_cpu:   torch.Tensor  # on cpu
    cu_seqlens_k_cpu:   torch.Tensor  # on cpu
    cu_seqlens_q_gpu:   torch.Tensor  # on gpu
    indices:            torch.Tensor  # on gpu
    last_page_len_cpu:  torch.Tensor  # on cpu
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          Literal[1] # currently only support page_size=1
    pos_encoding_mode:  str
    seq_lens_cpu:       torch.Tensor  # on cpu
    dtype:              torch.dtype
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized:        bool = False
    sm_scale:           float = 0.0  # softmax scale (1/sqrt(head_dim)), set during prepare_metadata
    # fmt: on

    def __post_init__(self) -> None:
        assert self.page_size == 1, "Currently only page_size=1 is supported."
        assert (
            self.positions.is_cuda
            and self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )

    def get_positions(self) -> torch.Tensor:
        return self.positions

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    def __init__(
        self,
        config: ModelConfig,
        kvcache: BaseKVCache,
        page_table: torch.Tensor,
    ) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.kvcache = kvcache
        self.device = kvcache.device
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is buggy, use fa2 instead
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
        )

        # NOTE: some hack to reuse the int_workspace_buffer
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer

        # initialize some data members
        tp_size = get_tp_info().size
        self.qo_head_local = divide_even(self.config.num_qo_heads, tp_size)
        self.kv_head_local = divide_even(self.config.num_kv_heads, tp_size)

        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)
        # for cuda graph
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}
        self.capture: FICaptureData | None = None
        self.page_table = page_table

    @staticmethod
    def _initialize_metadata_once(metadata: FIMetadata) -> None:
        if metadata.initialized:
            return

        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
        )

        metadata.initialized = True

        is_cuda_graph = isinstance(metadata.wrapper, CUDAGraphBatchDecodeWithPagedKVCacheWrapper)
        is_decode = isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper) or is_cuda_graph

        if is_cuda_graph and logger.isEnabledFor(logging.DEBUG):
            global _fi_debug_counter
            if _fi_debug_counter <= 5:
                logger.debug("plan() for CUDAGraphWrapper: indices shape=%s", metadata.indices.shape)
                logger.debug("  cu_seqlens_k_cpu=%s", metadata.cu_seqlens_k_cpu.tolist())

        if is_decode:
            # Decode wrapper (both regular and CUDA graph variants)
            # sm_scale must be passed to plan() for decode, not run()
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                sm_scale=metadata.sm_scale,
                non_blocking=True,
            )
        else:
            # Prefill wrapper - sm_scale passed to plan()
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                sm_scale=metadata.sm_scale,
                non_blocking=True,
                causal=True,
            )

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: TTSBatch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))

        # Debug: check attention inputs/outputs for first few steps at layer 0
        global _fi_debug_counter
        is_capturing = torch.cuda.is_current_stream_capturing()
        if (logger.isEnabledFor(logging.DEBUG)
                and layer_id == 0 and batch.is_decode and not is_capturing
                and _fi_debug_counter < 10):
            _fi_debug_counter += 1
            step = _fi_debug_counter
            logger.debug("ATTN Step %d Layer 0", step)
            logger.debug("  Q shape: %s, norm: %.4f, mean: %.6f", q.shape, q.norm().item(), q.mean().item())
            logger.debug("  K shape: %s, norm: %.4f, mean: %.6f", k.shape, k.norm().item(), k.mean().item())
            logger.debug("  V shape: %s, norm: %.4f, mean: %.6f", v.shape, v.norm().item(), v.mean().item())
            logger.debug("  out_loc: %s", batch.out_loc.tolist())
            logger.debug("  positions: %s", metadata.positions.tolist())
            logger.debug("  indices shape: %s, first 10: %s", metadata.indices.shape, metadata.indices[:10].tolist())
            logger.debug("  cu_seqlens_k: %s", metadata.cu_seqlens_k_cpu.tolist())
            k_cache, v_cache = kv_cache
            if batch.out_loc.numel() > 0:
                slot = batch.out_loc[0].item()
                logger.debug("  KV cache slot %d: k_norm=%.4f", slot, k_cache.view(-1, k_cache.shape[-1])[slot].norm().item())

        output = metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)

        if (logger.isEnabledFor(logging.DEBUG)
                and layer_id == 0 and batch.is_decode and not is_capturing
                and _fi_debug_counter <= 10):
            logger.debug("  Output shape: %s, norm: %.4f, mean: %.6f", output.shape, output.norm().item(), output.mean().item())

        return output

    def prepare_metadata(self, batch: TTSBatch) -> None:
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        cpu_kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.device
        seq_len_cpu = torch.tensor(seqlens_k, **cpu_kwargs)
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **cpu_kwargs).cumsum_(dim=0)
        if max_seqlen_q == 1:  # decode with all extend_len = 1
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **cpu_kwargs)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **cpu_kwargs).cumsum_(dim=0)
        batch.attn_metadata = FIMetadata(
            positions=make_positions(device, reqs),
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),
            indices=torch.cat([self.page_table[req.table_idx, : req.device_len] for req in reqs]),
            last_page_len_cpu=self._get_ones_cpu(padded_size),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_len_cpu,
            dtype=self.kvcache.dtype,
            wrapper=self.decode_wrappers if batch.is_decode else self.prefill_wrapper,
            sm_scale=1.0 / math.sqrt(self.config.head_dim),
        )

    def init_capture_graph(
        self, max_seq_len: int, bs_list: List[int], frame_width: int = 1
    ) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FICaptureData.create(max_bs, max_seq_len, self.kvcache.device, frame_width)
        capture.page_table = capture.page_table.view(-1)  # use 1D as ragged indices
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    @cached_property
    def use_tensor_cores(self) -> bool:
        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
            logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
            return overriden_value
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        return GQA >= 4

    def prepare_for_capture(self, batch: TTSBatch) -> None:
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        batch.padded_reqs = batch.reqs
        capture = self.capture
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        # Copy metadata tensors to capture buffers and update metadata to use capture buffers
        # This ensures CUDA graph uses fixed memory addresses
        indices_len = metadata.indices.shape[0]
        capture.indices[:indices_len].copy_(metadata.indices)
        capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k_cpu.to(capture.indices.device))
        capture.seq_lens[:bs].copy_(metadata.seq_lens_cpu.to(capture.indices.device))
        # Update metadata to point to capture buffers (critical for CUDA graph)
        metadata.indices = capture.indices[:indices_len]
        metadata.wrapper = self.graph_wrappers[bs]
        metadata.positions = capture.positions[:bs]
        batch.input_ids = capture.input_ids[:bs]
        batch.out_loc = capture.out_loc[:bs]
        self._initialize_metadata_once(metadata)

    def prepare_for_replay(self, batch: TTSBatch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs
        capture = self.capture

        if logger.isEnabledFor(logging.DEBUG) and _fi_debug_counter < 5:
            logger.debug("prepare_for_replay: bs=%d, indices_len=%d", bs, metadata.indices.shape[0])
            logger.debug("  cu_seqlens_k_cpu=%s", metadata.cu_seqlens_k_cpu.tolist())

        # Copy all dynamic tensors to capture buffers (fixed memory addresses for CUDA graph)
        capture.input_ids[:bs].copy_(batch.input_ids)
        capture.out_loc[:bs].copy_(batch.out_loc)
        capture.positions[:bs].copy_(metadata.positions)
        # Copy indices (page table entries) - critical for KV cache lookups
        indices_len = metadata.indices.shape[0]
        capture.indices[:indices_len].copy_(metadata.indices)
        # Copy cu_seqlens_k for the wrapper's indptr
        capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k_cpu.to(capture.indices.device))
        # Copy seq_lens for the wrapper
        capture.seq_lens[:bs].copy_(metadata.seq_lens_cpu.to(capture.indices.device))
        # Update metadata to point to capture buffers (CRITICAL: must match prepare_for_capture)
        # This ensures CUDA graph uses fixed memory addresses for all inputs
        metadata.indices = capture.indices[:indices_len]
        metadata.positions = capture.positions[:bs]
        metadata.wrapper = self.graph_wrappers[bs]
        # Update batch input_ids and out_loc to point to capture buffers
        batch.input_ids = capture.input_ids[:bs]
        batch.out_loc = capture.out_loc[:bs]
        self._initialize_metadata_once(metadata)
