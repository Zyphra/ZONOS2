"""TTS sampler for multi-codebook audio generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import torch
from zonos2.tts.sampler import sample_tts

if TYPE_CHECKING:
    from zonos2.core import TTSBatch


def make_device_tensor(
    data: List, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Create a tensor on device from a list."""
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(
        device, non_blocking=True
    )


@dataclass
class TTSBatchSamplingArgs:
    """Sampling arguments for a TTS batch."""

    temperatures: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor
    min_ps: torch.Tensor
    text_vocab: int
    repetition_token_ids: torch.Tensor | None = None
    repetition_penalties: torch.Tensor | None = None
    generators: list[torch.Generator | None] = field(default_factory=list)


@dataclass
class TTSSampler:
    """Sampler for multi-codebook TTS output."""

    device: torch.device
    n_codebooks: int
    codebook_size: int
    text_vocab: int

    def prepare(
        self, batch: TTSBatch, token_pool: torch.Tensor | None = None
    ) -> TTSBatchSamplingArgs:
        """Prepare sampling arguments from a TTS batch.

        Args:
            batch: TTS batch with requests
            token_pool: Optional device-side token pool for recent generated history

        Returns:
            TTSBatchSamplingArgs with per-sequence sampling parameters
        """
        params = [r.sampling_params for r in batch.reqs]

        MIN_T = 1e-6
        temps = [max(p.temperature, MIN_T) for p in params]
        top_ks = [p.topk if p.topk >= 1 else self.codebook_size for p in params]
        top_ps = [min(max(p.top_p, 0.0), 1.0) for p in params]
        min_ps = [max(p.min_p, 0.0) for p in params]
        repetition_windows = [max(int(p.repetition_window), 0) for p in params]
        repetition_penalties = [max(float(p.repetition_penalty), 1.0) for p in params]
        repetition_codebooks = [
            self.n_codebooks
            if int(p.repetition_codebooks) < 0
            else min(max(int(p.repetition_codebooks), 0), self.n_codebooks)
            for p in params
        ]
        generators = [r.rng for r in batch.reqs]
        repetition_token_ids = None
        repetition_penalties_tensor = None

        active_windows = [
            min(window, r.total_generated, r.device_len)
            if window > 0 and penalty > 1.0 and codebooks > 0
            else 0
            for r, window, penalty, codebooks in zip(
                batch.reqs,
                repetition_windows,
                repetition_penalties,
                repetition_codebooks,
            )
        ]
        max_window = max(active_windows, default=0)
        if max_window > 0:
            repetition_token_ids = torch.full(
                (len(batch.reqs), self.n_codebooks, max_window),
                -1,
                dtype=torch.int64,
                device=self.device,
            )
            for i, (req, window) in enumerate(zip(batch.reqs, active_windows)):
                if window <= 0:
                    continue

                start = req.device_len - window
                if token_pool is None:
                    history = req.input_ids[start : req.device_len, : self.n_codebooks].to(
                        device=self.device, dtype=torch.int64, non_blocking=True
                    )
                else:
                    history = token_pool[
                        req.table_idx, start : req.device_len, : self.n_codebooks
                    ].to(dtype=torch.int64)

                history = history.transpose(0, 1).contiguous()
                valid = (history >= 0) & (history < self.codebook_size)
                valid[repetition_codebooks[i] :] = False
                repetition_token_ids[i, :, -window:] = torch.where(
                    valid, history, torch.full_like(history, -1)
                )

            repetition_penalties_tensor = make_device_tensor(
                repetition_penalties, torch.float32, self.device
            )

        return TTSBatchSamplingArgs(
            temperatures=make_device_tensor(temps, torch.float32, self.device),
            top_ks=make_device_tensor(top_ks, torch.int32, self.device),
            top_ps=make_device_tensor(top_ps, torch.float32, self.device),
            min_ps=make_device_tensor(min_ps, torch.float32, self.device),
            text_vocab=self.text_vocab,
            repetition_token_ids=repetition_token_ids,
            repetition_penalties=repetition_penalties_tensor,
            generators=generators,
        )

    def sample(
        self, logits: torch.Tensor, args: TTSBatchSamplingArgs
    ) -> List[List[int]]:
        """Sample from multi-codebook logits.

        Args:
            logits: Shape (B, n_codebooks, vocab_size)
            args: Sampling arguments

        Returns:
            List of unpacked tokens [cb0, cb1, ..., cb8, text_placeholder]
        """
        return sample_tts(
            logits=logits,
            temperatures=args.temperatures,
            top_ks=args.top_ks,
            top_ps=args.top_ps,
            min_ps=args.min_ps,
            repetition_token_ids=args.repetition_token_ids,
            repetition_penalties=args.repetition_penalties,
            text_vocab=args.text_vocab,
            generators=args.generators,
        )

    def sample_to_tensor(
        self, logits: torch.Tensor, args: TTSBatchSamplingArgs
    ) -> torch.Tensor:
        """Sample and return as tensor instead of list.

        Args:
            logits: Shape (B, n_codebooks, vocab_size)
            args: Sampling arguments

        Returns:
            Tensor of shape (B, n_codebooks + 1) with sampled tokens
        """
        tokens_list = self.sample(logits, args)
        return torch.tensor(tokens_list, dtype=torch.int32, device=self.device)
