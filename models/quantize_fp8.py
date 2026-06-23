#!/usr/bin/env python
"""Offline FP8 (blockwise w8a8) expert quantization for ZONOS2.

Reads a bf16 ZONOS2 checkpoint and writes a new checkpoint directory where only
the MoE expert weights are stored as ``float8_e4m3fn`` with 128x128 block dequant
scales. Everything else (router, embeddings, attention, dense FFN, norms) is copied
through unchanged, following an "experts-only" quantization strategy.

The output directory is directly loadable by the engine with ``--quantization fp8``.

Usage:
    uv run python models/quantize_fp8.py --in Zyphra/ZONOS2 --out ./models/zonos2-fp8 [--block 128]
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch

from zonos2.layers.moe.fused_moe.fp8_utils import (
    BLOCK,
    per_block_quant_fp8,
    sonic_w13_to_gate_up,
)
from zonos2.distributed import set_tp_info, try_get_tp_info
from zonos2.models.weight import load_checkpoint_weight
from zonos2.utils import resolve_model_path

EXPERTS_MARKER = ".feed_forward.experts."


def _expert_prefix(key: str) -> str | None:
    """Return ``layers.{N}.feed_forward.experts.`` for an expert key, else None."""
    idx = key.find(EXPERTS_MARKER)
    if idx == -1:
        return None
    return key[: idx + len(EXPERTS_MARKER)]


def _fuse_gate_up(group: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    """Build fused gate_up_proj [E, 2*I, H] from a checkpoint expert group."""
    if "gate_up_proj" in group:
        return group["gate_up_proj"]
    if "w13" in group:
        return sonic_w13_to_gate_up(group["w13"])
    if "w1" in group and "w3" in group:
        # w1 = gate, w3 = up
        return torch.cat([group["w1"], group["w3"]], dim=1)
    raise KeyError(f"{prefix}: cannot find gate/up expert weights (have {sorted(group)})")


def _fuse_down(group: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    """Build down_proj [E, H, I] from a checkpoint expert group."""
    for k in ("down_proj", "w2"):
        if k in group:
            return group[k]
    raise KeyError(f"{prefix}: cannot find down expert weight (have {sorted(group)})")


def _check_block_aligned(name: str, t: torch.Tensor, block: int) -> None:
    _, m, k = t.shape
    if m % block or k % block:
        raise SystemExit(
            f"{name}: shape {tuple(t.shape)} is not divisible by block={block} on the "
            f"last two dims; blockwise FP8 requires alignment. Pad the model or use a "
            f"block size that divides {m} and {k}."
        )


def quantize_state_dict(state_dict, block: int):
    """Return a new state_dict with expert weights replaced by FP8 + scale_inv."""
    # Group expert sub-tensors by their layer prefix.
    groups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    expert_keys: set[str] = set()
    for key in state_dict:
        prefix = _expert_prefix(key)
        if prefix is None:
            continue
        sub = key[len(prefix):]
        groups[prefix][sub] = state_dict[key]
        expert_keys.add(key)

    out = {k: v for k, v in state_dict.items() if k not in expert_keys}

    n_layers = 0
    for prefix, group in sorted(groups.items()):
        gate_up = _fuse_gate_up(group, prefix).contiguous()
        down = _fuse_down(group, prefix).contiguous()
        _check_block_aligned(prefix + "gate_up_proj", gate_up, block)
        _check_block_aligned(prefix + "down_proj", down, block)

        gu_q, gu_s = per_block_quant_fp8(gate_up, block)
        dn_q, dn_s = per_block_quant_fp8(down, block)

        out[prefix + "gate_up_proj"] = gu_q
        out[prefix + "gate_up_proj_scale_inv"] = gu_s
        out[prefix + "down_proj"] = dn_q
        out[prefix + "down_proj_scale_inv"] = dn_s
        n_layers += 1
        print(
            f"  {prefix}: gate_up {tuple(gate_up.shape)} -> fp8 + scale {tuple(gu_s.shape)} | "
            f"down {tuple(down.shape)} -> fp8 + scale {tuple(dn_s.shape)}"
        )

    print(f"Quantized experts in {n_layers} MoE layer(s).")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="ZONOS2 FP8 expert quantizer")
    ap.add_argument("--in", dest="src", required=True, help="bf16 checkpoint dir or HF repo id")
    ap.add_argument("--out", dest="dst", required=True, help="output checkpoint directory")
    ap.add_argument("--block", type=int, default=BLOCK, help="FP8 block size (default 128)")
    args = ap.parse_args()

    # Single-rank (no tensor parallelism) for offline conversion.
    if try_get_tp_info() is None:
        set_tp_info(0, 1)

    src_dir = Path(resolve_model_path(args.src))
    dst_dir = Path(args.dst).expanduser()
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {src_dir} ...")
    state_dict = load_checkpoint_weight(args.src, torch.device("cpu"))

    print("Quantizing MoE experts to FP8 (e4m3, blockwise)...")
    out_sd = quantize_state_dict(state_dict, args.block)

    # Copy config/aux files (params.json, tokenizer, etc.) so the output is loadable.
    for f in src_dir.glob("*"):
        if f.suffix in (".json", ".yaml") and f.is_file():
            shutil.copy2(f, dst_dir / f.name)

    out_path = dst_dir / "model.pth"
    print(f"Saving FP8 checkpoint to {out_path} ...")
    torch.save(out_sd, out_path)

    with open(dst_dir / "quant_config.json", "w") as f:
        json.dump(
            {"quantization": "fp8", "method": "blockwise", "block": args.block, "fmt": "e4m3"},
            f,
            indent=2,
        )
    print("Done.")


if __name__ == "__main__":
    main()
