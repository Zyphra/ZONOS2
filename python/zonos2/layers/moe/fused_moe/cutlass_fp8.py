"""Blockwise w8a8 FP8 fused-MoE expert computation (CUTLASS grouped GEMM).

Faithful port of SGLang 0.5.7's ``cutlass_fused_experts_fp8``
(sglang/srt/layers/moe/cutlass_moe.py), which matches the sgl-kernel==0.3.20 op
ABI installed in this environment. Only the SM90 (H100) blockwise FP8 path is kept;
the H200/H20 "expert specialization" variants are dropped.

Weight storage convention (set by the offline converter, models/quantize_fp8.py):
- ``gate_up_proj``: ``[E, 2*I, H]`` float8_e4m3fn (standard [out, in] layout)
- ``down_proj``:    ``[E, H, I]``   float8_e4m3fn
- ``*_scale_inv``:  per-128-block dequant scales of the above (float32)

At call time the weights/scales are passed transposed (``.transpose(1, 2)``) so the
grouped GEMM sees ``w1_q: [E, H, 2I]`` and ``w2_q: [E, I, H]`` (column-major),
exactly as SGLang's reference test does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from sgl_kernel import (
    apply_shuffle_mul_sum,
    fp8_blockwise_scaled_grouped_mm,
    prepare_moe_input,
    shuffle_rows,
    silu_and_mul,
)

try:  # newer sgl-kernel exposes the fused 8-bit quant entry point
    from sgl_kernel import sgl_per_token_group_quant_8bit

    _HAS_8BIT = True
except ImportError:  # pragma: no cover - depends on sgl-kernel build
    from sgl_kernel import sgl_per_token_group_quant_fp8

    _HAS_8BIT = False

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
FP8_MIN = -FP8_MAX
GROUP = 128
# Scratch for the CUTLASS grouped GEMM; matches SGLang's reference allocation.
WORKSPACE_BYTES = 7182 * 1024


@dataclass
class CutlassFp8Buffers:
    """Per-layer constant buffers for the grouped GEMM (depend only on shapes)."""

    a1_strides: torch.Tensor
    c1_strides: torch.Tensor
    a2_strides: torch.Tensor
    c2_strides: torch.Tensor
    workspace: torch.Tensor
    a_ptrs: torch.Tensor
    b_ptrs: torch.Tensor
    out_ptrs: torch.Tensor
    a_scales_ptrs: torch.Tensor
    b_scales_ptrs: torch.Tensor
    expert_offsets: torch.Tensor
    problem_sizes1: torch.Tensor
    problem_sizes2: torch.Tensor
    a_sf_layout: torch.Tensor
    w_sf_layout: torch.Tensor


def make_cutlass_fp8_buffers(
    num_experts: int,
    hidden_size: int,
    gate_up_out: int,
    intermediate_size: int,
    device: torch.device,
) -> CutlassFp8Buffers:
    """Allocate the constant grouped-GEMM buffers for one MoE layer.

    ``gate_up_out`` is the fused gate+up width (``2 * intermediate_size``).
    Strides follow SGLang's reference: gemm1 sees (a=H, c=2I), gemm2 (a=I, c=H).
    """
    i64 = dict(dtype=torch.int64, device=device)
    i32 = dict(dtype=torch.int32, device=device)
    return CutlassFp8Buffers(
        a1_strides=torch.full((num_experts,), hidden_size, **i64),
        c1_strides=torch.full((num_experts,), gate_up_out, **i64),
        a2_strides=torch.full((num_experts,), intermediate_size, **i64),
        c2_strides=torch.full((num_experts,), hidden_size, **i64),
        workspace=torch.empty((WORKSPACE_BYTES,), dtype=torch.uint8, device=device),
        a_ptrs=torch.empty((num_experts,), **i64),
        b_ptrs=torch.empty((num_experts,), **i64),
        out_ptrs=torch.empty((num_experts,), **i64),
        a_scales_ptrs=torch.empty((num_experts,), **i64),
        b_scales_ptrs=torch.empty((num_experts,), **i64),
        expert_offsets=torch.empty((num_experts + 1,), **i32),
        problem_sizes1=torch.empty((num_experts, 3), **i32),
        problem_sizes2=torch.empty((num_experts, 3), **i32),
        a_sf_layout=torch.empty((num_experts, 5), **i32),
        w_sf_layout=torch.empty((num_experts, 5), **i32),
    )


def per_token_group_quant_fp8(x: torch.Tensor, group_size: int = GROUP):
    """Quantize ``x`` ([m, k]) to e4m3 with per-token, per-128-group scales.

    Returns ``(x_q [m, k] e4m3, x_s [m, k // group_size] float32)``. Mirrors the
    non-column-major path of SGLang's ``sglang_per_token_group_quant_fp8``.
    """
    assert x.shape[-1] % group_size == 0
    assert x.is_contiguous()
    x_q = torch.empty(x.shape, device=x.device, dtype=FP8_DTYPE)
    x_s = torch.empty(
        (*x.shape[:-1], x.shape[-1] // group_size),
        device=x.device,
        dtype=torch.float32,
    )
    if x.shape[0] > 0:
        if _HAS_8BIT:
            # enable_v2=False keeps us on the v1 op and avoids the wrapper's
            # optional ``sglang`` import (only used to read the v2 env var).
            sgl_per_token_group_quant_8bit(
                x,
                x_q,
                x_s,
                group_size,
                1e-10,
                FP8_MIN,
                FP8_MAX,
                False,  # scale_ue8m0
                False,  # fuse_silu_and_mul
                None,  # masked_m
                enable_v2=False,
            )
        else:
            sgl_per_token_group_quant_fp8(
                x, x_q, x_s, group_size, 1e-10, FP8_MIN, FP8_MAX, False
            )
    return x_q, x_s


def cutlass_fused_experts_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    bufs: CutlassFp8Buffers,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a blockwise FP8 MoE FFN.

    Args:
        a: input activations ``[m, H]`` (bf16/fp16).
        w1_q: gate_up weights ``[E, H, 2I]`` e4m3 (i.e. ``gate_up_proj.transpose(1, 2)``).
        w2_q: down weights ``[E, I, H]`` e4m3 (i.e. ``down_proj.transpose(1, 2)``).
        w1_scale/w2_scale: per-block dequant scales in ``[E, out/128, in/128]`` layout
            (i.e. the stored ``*_scale_inv`` as-is, NOT transposed — the weights are
            transposed but their scales are not, for this sgl-kernel ABI).
        topk_weights/topk_ids: routing ``[m, topk]`` (ids int32).
        bufs: constant per-layer buffers from ``make_cutlass_fp8_buffers``.
        output: optional ``[m, H]`` output buffer.
    """
    assert w1_q.dtype == FP8_DTYPE and w2_q.dtype == FP8_DTYPE
    assert a.shape[1] == w1_q.shape[1], "hidden size mismatch w1"
    assert w1_q.shape[2] == w2_q.shape[1] * 2, "intermediate size mismatch w2"
    assert a.dtype in (torch.half, torch.bfloat16)

    out_dtype = a.dtype
    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(1)  # hidden
    n = w2_q.size(1)  # intermediate
    topk = topk_ids.size(1)
    device = a.device

    a_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)

    prepare_moe_input(
        topk_ids,
        bufs.expert_offsets,
        bufs.problem_sizes1,
        bufs.problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )

    a_q, a1_scale = per_token_group_quant_fp8(a, GROUP)
    rep_a_q = shuffle_rows(a_q, a_map, (m * topk, k))
    # NB: ``shuffle_rows`` is only valid for the fp8 activation rows; applied to the
    # float32 per-token scales it silently produces zeros (it assumes 1-byte rows).
    # Permute the scales with a plain gather instead.
    rep_a1_scales = a1_scale[a_map.long()]

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=out_dtype)
    c2 = torch.empty((m * topk, k), device=device, dtype=out_dtype)

    fp8_blockwise_scaled_grouped_mm(
        c1,
        bufs.a_ptrs,
        bufs.b_ptrs,
        bufs.out_ptrs,
        bufs.a_scales_ptrs,
        bufs.b_scales_ptrs,
        rep_a_q,
        w1_q,
        rep_a1_scales,
        w1_scale,
        bufs.a1_strides,
        bufs.a1_strides,
        bufs.c1_strides,
        bufs.a_sf_layout,
        bufs.w_sf_layout,
        bufs.problem_sizes1,
        bufs.expert_offsets[:-1],
        bufs.workspace,
    )

    intermediate = torch.empty((m * topk, n), device=device, dtype=out_dtype)
    silu_and_mul(c1, intermediate)

    intermediate_q, a2_scale = per_token_group_quant_fp8(intermediate, GROUP)

    fp8_blockwise_scaled_grouped_mm(
        c2,
        bufs.a_ptrs,
        bufs.b_ptrs,
        bufs.out_ptrs,
        bufs.a_scales_ptrs,
        bufs.b_scales_ptrs,
        intermediate_q,
        w2_q,
        a2_scale,
        w2_scale,
        bufs.a2_strides,
        bufs.a2_strides,
        bufs.c2_strides,
        bufs.a_sf_layout,
        bufs.w_sf_layout,
        bufs.problem_sizes2,
        bufs.expert_offsets[:-1],
        bufs.workspace,
    )

    if output is None:
        output = torch.empty((m, k), device=device, dtype=out_dtype)
    apply_shuffle_mul_sum(c2, output, c_map, topk_weights.to(out_dtype))
    return output
