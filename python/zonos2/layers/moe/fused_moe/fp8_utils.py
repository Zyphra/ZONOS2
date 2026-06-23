"""Shared helpers for blockwise FP8 (e4m3) expert quantization.

Used by both the offline conversion script (``models/quantize_fp8.py``) and the
runtime FP8 MoE method so the two agree on layout and scale conventions.

Convention (matches DeepSeek / SGLang blockwise w8a8):
- Weights are quantized in ``BLOCK x BLOCK`` tiles over the last two dims.
- ``scale_inv[..., i, j] = amax(tile) / FP8_E4M3_MAX`` is the *dequant* multiplier,
  i.e. ``w_bf16 ~= w_fp8.float() * scale_inv``.
"""

from __future__ import annotations

import torch

FP8_E4M3_MAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn
BLOCK = 128


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def scale_inv_shape(num_experts: int, out_dim: int, in_dim: int, block: int = BLOCK):
    """Block-scale tensor shape for an expert weight ``[E, out_dim, in_dim]``."""
    return (num_experts, ceil_div(out_dim, block), ceil_div(in_dim, block))


def per_block_quant_fp8(w: torch.Tensor, block: int = BLOCK):
    """Blockwise-quantize ``w`` (``[..., M, K]``) to e4m3 + per-block dequant scales.

    Returns ``(q, scale_inv)`` where ``q`` is ``float8_e4m3fn`` with the same shape
    as ``w`` and ``scale_inv`` is ``float32`` of shape
    ``[..., ceil(M/block), ceil(K/block)]``.
    """
    assert w.dim() >= 2, f"expected >=2D weight, got {tuple(w.shape)}"
    orig_dtype = w.dtype
    w = w.to(torch.float32)
    *lead, M, K = w.shape
    mb, kb = ceil_div(M, block), ceil_div(K, block)

    q = torch.empty_like(w, dtype=FP8_DTYPE)
    scale_inv = torch.empty((*lead, mb, kb), dtype=torch.float32, device=w.device)

    for i in range(mb):
        r0, r1 = i * block, min((i + 1) * block, M)
        for j in range(kb):
            c0, c1 = j * block, min((j + 1) * block, K)
            tile = w[..., r0:r1, c0:c1]
            amax = tile.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            s = amax / FP8_E4M3_MAX
            q[..., r0:r1, c0:c1] = (tile / s).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_DTYPE)
            scale_inv[..., i, j] = s.squeeze(-1).squeeze(-1)

    del w, orig_dtype
    return q, scale_inv


def sonic_w13_to_gate_up(w13: torch.Tensor) -> torch.Tensor:
    """De-interleave SonicMoE ``w13`` ``[E, 2*I, H]`` (gate=even, up=odd rows) into the
    fused inference layout ``gate_up_proj = cat([gate, up], dim=1)``.

    Mirrors ``FusedGroupedExperts._convert_sonic_w13_to_gate_up`` in models/zonos2.py.
    """
    assert w13.dim() == 3, f"expected rank-3 w13, got {tuple(w13.shape)}"
    assert w13.shape[1] % 2 == 0, f"expected even fused width, got {tuple(w13.shape)}"
    gate = w13[:, 0::2, :]
    up = w13[:, 1::2, :]
    return torch.cat([gate, up], dim=1)
