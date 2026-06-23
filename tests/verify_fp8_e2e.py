"""End-to-end TTS verification: FP8 experts vs bf16 baseline.

Generates the same prompts with the bf16 model and the FP8 model, saves both
waveforms, and reports peak GPU memory for each. Run AFTER converting a checkpoint
with models/quantize_fp8.py.

Each model runs in its OWN subprocess: the engine sets a write-once global TP-info
singleton (zonos2.distributed.set_tp_info), so two TTSLLM instances cannot share a
process. The orchestrator spawns one worker per model and compares their outputs.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python \
        uv run python tests/verify_fp8_e2e.py \
        --bf16 Zyphra/ZONOS2 --fp8 ./models/zonos2-fp8 --out /tmp/zonos2_fp8_check
"""

import argparse
import os
import subprocess
import sys

import numpy as np

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "FP8 mixture-of-experts inference keeps the spine in bfloat16.",
]


def run_one(model_path: str, quantization: str, out_dir: str, tag: str, result_file: str):
    """Worker: load one model, generate, save wavs + a result .npz, print peak mem."""
    import torch

    from zonos2.message import TTSSamplingParams
    from zonos2.tts import TTSLLM

    torch.cuda.reset_peak_memory_stats()
    kwargs = {} if quantization == "none" else {"quantization": quantization}
    tts = TTSLLM(model_path=model_path, **kwargs)
    results = tts.generate(PROMPTS, TTSSamplingParams(seed=42))

    audios = []
    for i, r in enumerate(results):
        path = os.path.join(out_dir, f"{tag}_{i}.wav")
        tts.save_audio(r["audio"], path)
        # r["audio"] is raw float32 PCM bytes (44.1 kHz); decode to a 1-D array.
        audios.append(np.frombuffer(r["audio"], dtype=np.float32).copy())
        print(f"  [{tag}] prompt {i}: frames={len(r['audio_tokens'])} -> {path}")

    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"  [{tag}] peak GPU mem: {peak:.2f} GiB")
    np.savez(
        result_file,
        peak=np.float32(peak),
        **{f"audio_{i}": a for i, a in enumerate(audios)},
    )


def spawn(model_path: str, quantization: str, out_dir: str, tag: str):
    """Launch a worker subprocess and load back its audio arrays + peak mem."""
    result_file = os.path.join(out_dir, f"{tag}_result.npz")
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--worker", quantization,
        "--model", model_path,
        "--out", out_dir,
        "--tag", tag,
        "--result", result_file,
    ]
    subprocess.run(cmd, check=True, env=os.environ.copy())
    data = np.load(result_file)
    audios = [data[k] for k in sorted(data.files) if k.startswith("audio_")]
    return audios, float(data["peak"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", help="bf16 checkpoint (HF id or path)")
    ap.add_argument("--fp8", help="FP8-converted checkpoint dir")
    ap.add_argument("--out", default="/tmp/zonos2_fp8_check")
    # Worker-mode args (internal; one model per subprocess).
    ap.add_argument("--worker", choices=["none", "fp8"], help=argparse.SUPPRESS)
    ap.add_argument("--model", help=argparse.SUPPRESS)
    ap.add_argument("--tag", help=argparse.SUPPRESS)
    ap.add_argument("--result", help=argparse.SUPPRESS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.worker is not None:
        run_one(args.model, args.worker, args.out, args.tag, args.result)
        return

    assert args.bf16 and args.fp8, "--bf16 and --fp8 are required in orchestrator mode"

    print("=== bf16 baseline ===")
    a_bf16, m_bf16 = spawn(args.bf16, "none", args.out, "bf16")
    print("=== fp8 experts ===")
    a_fp8, m_fp8 = spawn(args.fp8, "fp8", args.out, "fp8")

    print("\n=== comparison ===")
    for i, (b, f) in enumerate(zip(a_bf16, a_fp8)):
        n = min(len(b), len(f))
        if n == 0:
            print(f"  prompt {i}: empty audio?")
            continue
        b, f = b[:n], f[:n]
        cos = float(np.dot(b, f) / (np.linalg.norm(b) * np.linalg.norm(f) + 1e-8))
        secs_b, secs_f = len(a_bf16[i]) / 44100, len(a_fp8[i]) / 44100
        print(f"  prompt {i}: dur bf16={secs_b:.2f}s fp8={secs_f:.2f}s | sample-cosine={cos:.4f}")
    print(f"\n  peak mem: bf16={m_bf16:.2f} GiB, fp8={m_fp8:.2f} GiB, saved={m_bf16 - m_fp8:.2f} GiB")
    print("  NOTE: sample-aligned cosine is ~0 by design -- autoregressive sampling diverges")
    print("  after the first rounding-induced token difference, so the two utterances are")
    print("  different (and different length). Validate quality by LISTENING to the .wav files;")
    print("  kernel correctness is covered by tests/test_fp8_experts_numeric.py.")
    print("  Peak GPU mem is ~equal because the scheduler sizes its KV-cache pool to fill the")
    print("  GPU; the expert-weight savings show in the checkpoint footprint (~15.3->8.1 GB).")


if __name__ == "__main__":
    main()
