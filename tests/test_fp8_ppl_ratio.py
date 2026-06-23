"""Perplexity-ratio check: FP8 experts vs bf16 baseline, teacher-forced.

Waveform cosine is meaningless for autoregressive TTS (sampling diverges), so we
measure quality the rigorous way: teacher-forced perplexity on the SAME token
sequence under both models.

  1. bf16 generates an audio-token sequence for a fixed prompt+seed.
  2. We build the full input_ids = text rows ++ audio rows (audio frame layout is
     [cb0..cb8, text_vocab], exactly what the sampler emits -- sampler.py:250).
  3. Each model runs ONE prefill over that whole sequence; we capture per-position
     multi-codebook logits by monkeypatching Zonos2ForCausalLM.compute_logits
     (the engine computes logits for ALL prefill positions, then slices to the
     last -- engine.py:328, so the patch sees the full tensor). Targets come from
     the engine's OWN captured input_ids, so alignment is exact even if the
     scheduler prepends a reserved speaker slot.
  4. Teacher-forced NLL over genuine audio targets (cb0 < codebook_size, i.e.
     excluding eoa/pad) -> per-codebook and overall perplexity.
  5. ratio = ppl_fp8 / ppl_bf16 (≈1.0 means FP8 preserves the distribution).

Each model loads in its OWN subprocess: the engine sets a write-once global TP
singleton, so two TTSLLM cannot share a process.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python \
        uv run python tests/test_fp8_ppl_ratio.py \
        --bf16 Zyphra/ZONOS2 --fp8 ./models/zonos2-fp8 --out /tmp/zonos2_fp8_ppl
"""

import argparse
import os
import subprocess
import sys

import numpy as np

# A few sentences -> a few hundred audio frames -> ~thousands of scored tokens.
TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "FP8 mixture-of-experts inference keeps the spine in bfloat16 while halving "
    "the expert weights, and the perplexity should barely move."
)
GEN_SEED = 42


def _score_sequence(tts, full_rows):
    """Teacher-forced NLL of `full_rows` under `tts`. Returns ce [N, n_cb] (nats)."""
    import torch
    import torch.nn.functional as F

    from zonos2.core import get_global_ctx
    from zonos2.message import TTSSamplingParams
    from zonos2.models.zonos2 import Zonos2ForCausalLM

    cap_logits, cap_ids = [], []
    orig = Zonos2ForCausalLM.compute_logits

    def patched(self, hidden_states):
        out = orig(self, hidden_states)
        batch = get_global_ctx()._batch
        if batch is not None and batch.is_prefill:
            cap_logits.append(out.detach().float().cpu())
            cap_ids.append(batch.input_ids.detach().cpu())
        return out

    Zonos2ForCausalLM.compute_logits = patched
    try:
        sp = TTSSamplingParams(seed=0, max_tokens=1, ignore_eos=False)
        tts.generate([full_rows], sp, decode_audio=False)
    finally:
        Zonos2ForCausalLM.compute_logits = orig

    logits = torch.cat(cap_logits, dim=0)              # [T, n_cb, V]
    ids = torch.cat(cap_ids, dim=0)                    # [T, frame_width]
    assert logits.shape[0] == ids.shape[0], (logits.shape, ids.shape)

    n_cb, V = logits.shape[1], logits.shape[2]
    codebook_size = V - 2                              # +2 for eoa, pad
    pred = logits[:-1]                                 # logits[t] predicts frame t+1
    tgt = ids[1:, :n_cb].long()                        # [T-1, n_cb]
    mask = tgt[:, 0] < codebook_size                   # genuine audio (drop text/eoa/pad)
    pred, tgt = pred[mask], tgt[mask]                  # [N, n_cb, V], [N, n_cb]
    logp = F.log_softmax(pred, dim=-1)                 # [N, n_cb, V]
    ce = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [N, n_cb] NLL vs target
    # logp saved as fp16 to stay lean; KLD/Top-1 computed in the orchestrator.
    return ce.numpy(), logp.half().numpy(), tgt.numpy().astype(np.int32)


def _fp8_roundtrip_experts(tts):
    """In place: round-trip every MoE expert weight through the SAME fp8 block-quant
    the converter uses, keeping bf16 compute. Isolates weight-quant error (w8a16).

    Done with copy_ (same memory address) so eager prefill -- and any captured graph
    -- see the new values. The bf16 fused gate_up_proj/down_proj are bit-identical to
    what models/quantize_fp8.py quantizes, so these weights == the w8a8 weights; only
    the compute (bf16 GEMM + bf16 activations) differs.
    """
    import torch

    from zonos2.layers.moe.fused_moe.fp8_utils import BLOCK, per_block_quant_fp8

    def dequant(q, s, block=BLOCK):
        f = q.float()
        sb = s.repeat_interleave(block, -2).repeat_interleave(block, -1)
        sb = sb[..., : f.shape[-2], : f.shape[-1]]
        return f * sb

    def roundtrip(w):
        q, s = per_block_quant_fp8(w.float(), BLOCK)
        return dequant(q, s).to(w.dtype)

    model = tts.engine.model
    seen, stack, n = set(), [model], 0
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        d = getattr(obj, "__dict__", None)
        if not d:
            continue
        gu, dn = d.get("gate_up_proj"), d.get("down_proj")
        if "_fused_moe" in d and isinstance(gu, torch.Tensor) and gu.dim() == 3:
            with torch.no_grad():
                gu.copy_(roundtrip(gu))
                dn.copy_(roundtrip(dn))
            n += 1
        for v in d.values():
            if hasattr(v, "__dict__"):
                stack.append(v)
            elif isinstance(v, (list, tuple)):
                stack.extend(it for it in v if hasattr(it, "__dict__"))
    return n


def run_worker(model_path, quantization, mode, text, ids_file, result_file):
    """Worker subprocess: load one model, (optionally) generate, then score."""
    import torch

    from zonos2.message import TTSSamplingParams
    from zonos2.tts import TTSLLM

    kwargs = {} if quantization == "none" else {"quantization": quantization}
    tts = TTSLLM(model_path=model_path, **kwargs)

    if mode == "gen_and_score":
        # 1. Generate a real audio-token sequence with this (bf16) model.
        res = tts.generate([text], TTSSamplingParams(seed=GEN_SEED), decode_audio=False)
        audio_tokens = res[0]["audio_tokens"]          # List[[cb0..cb8], ...]
        print(f"  [gen] generated {len(audio_tokens)} audio frames")

        # 2. Build full input_ids = text rows ++ audio rows ([cb0..cb8, text_vocab]).
        text_rows = tts._tokenize_one(text).to(torch.int32)          # [n_text, fw]
        audio = torch.tensor(audio_tokens, dtype=torch.int32)        # [F, n_cb]
        text_col = torch.full((audio.shape[0], 1), int(tts.text_vocab), dtype=torch.int32)
        audio_rows = torch.cat([audio, text_col], dim=1)             # [F, fw]
        full_rows = torch.cat([text_rows, audio_rows], dim=0)        # [L, fw]
        np.save(ids_file, full_rows.numpy())
        print(f"  [gen] full scoring sequence: {tuple(full_rows.shape)} "
              f"(text={text_rows.shape[0]} + audio={audio_rows.shape[0]})")
        rows_list = full_rows.tolist()
    else:  # score / w8a16
        full_rows = np.load(ids_file)
        rows_list = full_rows.tolist()

    if mode == "w8a16":
        n = _fp8_roundtrip_experts(tts)
        print(f"  [w8a16] round-tripped {n} MoE expert layers through fp8 block-quant "
              f"(bf16 compute)")

    ce, logp, tgt = _score_sequence(tts, rows_list)    # [N,n_cb], [N,n_cb,V] fp16, [N,n_cb]
    ppl = float(np.exp(ce.mean()))
    print(f"  [{quantization}] scored {ce.shape[0]} frames x {ce.shape[1]} codebooks | "
          f"mean NLL={ce.mean():.4f} nats | ppl={ppl:.4f}")
    np.savez(result_file, ce=ce.astype(np.float32), logp=logp, tgt=tgt)


def spawn(model_path, quantization, mode, text, out_dir, tag, ids_file):
    result_file = os.path.join(out_dir, f"{tag}_ppl.npz")
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--worker", quantization, "--mode", mode,
        "--model", model_path, "--text", text,
        "--ids", ids_file, "--result", result_file,
    ]
    subprocess.run(cmd, check=True, env=os.environ.copy())
    return np.load(result_file)


def _kld_top1(logp_b, logp_f):
    """KL(base||quant) mean and base-vs-quant Top-1 agreement over [N, n_cb, V]."""
    N, n_cb, V = logp_b.shape
    lb = logp_b.reshape(-1, V)        # fp16
    lf = logp_f.reshape(-1, V)
    rows = lb.shape[0]
    top1 = float((lb.argmax(-1) == lf.argmax(-1)).mean())
    kld_sum, chunk = 0.0, 8192
    for s in range(0, rows, chunk):
        a = lb[s:s + chunk].astype(np.float32)
        b = lf[s:s + chunk].astype(np.float32)
        # KL(base||quant) = sum_x p_base(x) * (logp_base - logp_quant)
        kld_sum += float((np.exp(a) * (a - b)).sum())
    return kld_sum / rows, top1


def _dir_size_gb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / 1e9
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16")
    ap.add_argument("--fp8")
    ap.add_argument("--out", default="/tmp/zonos2_fp8_ppl")
    # worker-mode (internal)
    ap.add_argument("--worker", choices=["none", "fp8"], help=argparse.SUPPRESS)
    ap.add_argument("--mode", choices=["gen_and_score", "score", "w8a16"],
                    help=argparse.SUPPRESS)
    ap.add_argument("--model", help=argparse.SUPPRESS)
    ap.add_argument("--text", help=argparse.SUPPRESS)
    ap.add_argument("--ids", help=argparse.SUPPRESS)
    ap.add_argument("--result", help=argparse.SUPPRESS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.worker is not None:
        run_worker(args.model, args.worker, args.mode, args.text, args.ids, args.result)
        return

    assert args.bf16 and args.fp8, "--bf16 and --fp8 required"
    ids_file = os.path.join(args.out, "scoring_ids.npy")

    print("=== bf16: generate reference sequence + score ===")
    d_bf16 = spawn(args.bf16, "none", "gen_and_score", TEXT, args.out, "bf16", ids_file)
    print("=== w8a16: fp8-rounded weights, bf16 compute (isolates WEIGHT error) ===")
    d_w8a16 = spawn(args.bf16, "none", "w8a16", TEXT, args.out, "w8a16", ids_file)
    print("=== w8a8: fp8 weights + fp8 activations (weight + activation error) ===")
    d_fp8 = spawn(args.fp8, "fp8", "score", TEXT, args.out, "fp8", ids_file)

    ce_bf16 = d_bf16["ce"]
    n_frames, n_cb = ce_bf16.shape
    ppl_bf16 = float(np.exp(ce_bf16.mean()))
    size_gb = _dir_size_gb(args.fp8)

    builds = [
        ("fp8-exp (w8a16)", d_w8a16, "weight-only (fp8 weights, bf16 acts/GEMM)"),
        ("fp8-exp (w8a8) ", d_fp8, "weight + activation (fused fp8 GEMM)"),
    ]
    rows = {}
    for name, d, _ in builds:
        ce = d["ce"]
        assert ce.shape == ce_bf16.shape, (name, ce.shape)
        assert np.array_equal(d_bf16["tgt"], d["tgt"]), f"{name} scored different seq!"
        ppl = float(np.exp(ce.mean()))
        kld, top1 = _kld_top1(d_bf16["logp"], d["logp"])
        rows[name.strip()] = dict(ce=ce, ppl=ppl, ratio=ppl / ppl_bf16, kld=kld, top1=top1)

    print("\n=== teacher-forced perplexity (lower = better) ===")
    print(f"  scored {n_frames} audio frames x {n_cb} codebooks "
          f"= {n_frames * n_cb} tokens")
    print(f"  bf16 (baseline)  : mean NLL={ce_bf16.mean():.4f} nats | ppl={ppl_bf16:.4f}")
    for name, d, desc in builds:
        r = rows[name.strip()]
        print(f"  {name} : mean NLL={r['ce'].mean():.4f} nats | ppl={r['ppl']:.4f} "
              f"| ΔNLL={r['ce'].mean() - ce_bf16.mean():+.4f}  [{desc}]")

    w = rows["fp8-exp (w8a16)"]
    a = rows["fp8-exp (w8a8)"]
    dnll_w = w["ce"].mean() - ce_bf16.mean()
    dnll_a = a["ce"].mean() - w["ce"].mean()
    print("\n  === error decomposition (ΔNLL, nats) ===")
    print(f"    weight-quant error  (w8a16 - bf16) = {dnll_w:+.4f}")
    print(f"    activation error    (w8a8 - w8a16) = {dnll_a:+.4f}")
    print(f"    total               (w8a8 - bf16)  = {a['ce'].mean() - ce_bf16.mean():+.4f}")
    if a["ce"].mean() - ce_bf16.mean() > 1e-6:
        frac = dnll_a / (a["ce"].mean() - ce_bf16.mean())
        print(f"    -> activations account for ~{frac * 100:.0f}% of total degradation")

    # Rows in a compact experts-only quantization comparison table.
    print("\n  === comparable rows (experts-only quant table; bf16 spine) ===")
    hdr = "Build               | bpw  |  Size   |  PPL  | ratio | Top-1 | KLD mean"
    print("  " + hdr)
    print("  " + "-" * len(hdr))
    for name in ("fp8-exp (w8a16)", "fp8-exp (w8a8)"):
        r = rows[name]
        print(f"  {name:<19} | 8.00 | {size_gb:4.2f} GB | {r['ppl']:5.2f} | "
              f"{r['ratio']:.3f} | {r['top1'] * 100:4.1f}% | {r['kld']:.3f}")
    # keep legacy single-build names for the trailing NOTE
    ppl_fp8, ratio, top1, kld = a["ppl"], a["ratio"], a["top1"], a["kld"]
    print("\n  NOTE: absolute PPL/ratio are on a self-generated 992-frame sequence, so")
    print("  compare the COLUMNS/methodology, not absolute PPL, against any external")
    print("  table. Top-1 = base-vs-quant argmax agreement; KLD = mean KL(bf16 || fp8)")
    print("  in nats (same definition as llama.cpp --kl-divergence). Also note this")
    print("  build is w8a8 (activations quantized too), vs weight-only K-quant tables.")


if __name__ == "__main__":
    main()
