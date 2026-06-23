"""Numeric check: blockwise FP8 (w8a8) experts vs a reference computation.

Runs the real ``cutlass_fused_experts_fp8`` path (the same call the model makes in
``FusedMoE._forward_fp8``) on random data and compares it to:
  (a) a full-bf16 reference (sanity), and
  (b) a w8a8 reference that dequantizes the fp8 weights and group-quantizes the
      activations to fp8 (isolates kernel math from precision loss).

Run on a Hopper GPU:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python \
        uv run python tests/test_fp8_experts_numeric.py
"""

import torch
import torch.nn.functional as F

from zonos2.layers.moe.fused_moe.cutlass_fp8 import (
    FP8_MAX,
    GROUP,
    cutlass_fused_experts_fp8,
    make_cutlass_fp8_buffers,
)
from zonos2.layers.moe.fused_moe.fp8_utils import per_block_quant_fp8


def cosine(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def calc_diff(x, y):  # DeepGEMM-style relative diff
    x, y = x.double(), y.double()
    return (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item()


def dequant_block(q, s, block=GROUP):
    f = q.float()
    sb = s.repeat_interleave(block, dim=-2).repeat_interleave(block, dim=-1)
    sb = sb[..., : f.shape[-2], : f.shape[-1]]
    return f * sb


def group_quant_act(x, block=GROUP):
    """Round-trip x through per-128-group fp8 (returns the dequantized values)."""
    m, k = x.shape
    xr = x.float().reshape(m, k // block, block)
    amax = xr.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    s = amax / FP8_MAX
    q = (xr / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (q * s).reshape(m, k)  # dequantized fp8 activation


def reference(x, gate_up, down, topk_w, topk_ids, w8a8: bool):
    """Dense reference. gate_up [E,2I,H], down [E,H,I]."""
    E, twoI, H = gate_up.shape
    I = twoI // 2
    m, topk = topk_ids.shape
    out = torch.zeros(m, H, dtype=torch.float32, device=x.device)
    xc = group_quant_act(x) if w8a8 else x.float()
    for i in range(m):
        acc = torch.zeros(H, dtype=torch.float32, device=x.device)
        for j in range(topk):
            e = int(topk_ids[i, j])
            gu = gate_up[e].float()
            gate, up = gu[:I, :], gu[I:, :]
            xi = xc[i]
            h = F.silu(gate @ xi) * (up @ xi)
            if w8a8:
                h = group_quant_act(h.unsqueeze(0)).squeeze(0)
            acc += float(topk_w[i, j]) * (down[e].float() @ h)
        out[i] = acc
    return out


def main():
    assert torch.cuda.is_available(), "needs a CUDA (Hopper) GPU"
    dev = torch.device("cuda")
    torch.manual_seed(0)
    E, H, I, m, topk = 8, 256, 256, 64, 2
    dtype = torch.bfloat16

    x = (torch.randn(m, H, device=dev) * 0.1).to(dtype)
    gate_up = (torch.randn(E, 2 * I, H, device=dev) * 0.05).to(dtype)  # [E,2I,H]
    down = (torch.randn(E, H, I, device=dev) * 0.05).to(dtype)         # [E,H,I]

    # Blockwise-quantize weights to fp8 (as the converter does), on GPU.
    gu_q, gu_s = per_block_quant_fp8(gate_up.float(), GROUP)
    dn_q, dn_s = per_block_quant_fp8(down.float(), GROUP)
    gu_q, gu_s, dn_q, dn_s = gu_q.to(dev), gu_s.to(dev), dn_q.to(dev), dn_s.to(dev)

    # Dequantized weights for the references.
    gu_deq = dequant_block(gu_q, gu_s).to(dtype)
    dn_deq = dequant_block(dn_q, dn_s).to(dtype)

    # Routing (renormalized like the model).
    topk_w = torch.softmax(torch.rand(m, topk, device=dev), dim=-1)
    topk_ids = torch.randint(0, E, (m, topk), dtype=torch.int32, device=dev)

    bufs = make_cutlass_fp8_buffers(E, H, 2 * I, I, dev)
    y = cutlass_fused_experts_fp8(
        a=x,
        w1_q=gu_q.transpose(1, 2),
        w2_q=dn_q.transpose(1, 2),
        w1_scale=gu_s,
        w2_scale=dn_s,
        topk_weights=topk_w.float(),
        topk_ids=topk_ids,
        bufs=bufs,
    )

    ref_bf16 = reference(x, gate_up, down, topk_w, topk_ids, w8a8=False)
    ref_w8a8 = reference(x, gu_deq, dn_deq, topk_w, topk_ids, w8a8=True)

    c_bf16, d_bf16 = cosine(y, ref_bf16), calc_diff(y, ref_bf16)
    c_w8a8, d_w8a8 = cosine(y, ref_w8a8), calc_diff(y, ref_w8a8)
    print(f"cutlass vs full-bf16 : cosine={c_bf16:.5f}  diff={d_bf16:.6f}")
    print(f"cutlass vs w8a8-ref  : cosine={c_w8a8:.5f}  diff={d_w8a8:.6f}")

    assert c_bf16 > 0.97, f"sanity vs bf16 too low: {c_bf16}"
    assert c_w8a8 > 0.99, f"kernel math vs w8a8 ref too low: {c_w8a8}"
    print("FP8 EXPERTS NUMERIC CHECK PASSED")


if __name__ == "__main__":
    main()
