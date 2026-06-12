from __future__ import annotations

from types import SimpleNamespace

import torch
from zonos2.engine.sample import TTSSampler
from zonos2.message.tts import TTSSamplingParams
from zonos2.tts.sampler import apply_repetition_penalty, sample_tts


def test_repetition_penalty_applies_per_codebook_for_greedy_sampling():
    logits = torch.tensor(
        [
            [
                [4.0, 3.0, 0.0],
                [4.0, 3.0, 0.0],
            ]
        ]
    )

    tokens = sample_tts(
        logits=logits,
        temperatures=torch.tensor([0.0]),
        top_ks=torch.tensor([3]),
        top_ps=torch.tensor([0.0]),
        min_ps=torch.tensor([0.0]),
        repetition_token_ids=torch.tensor([[[0], [1]]]),
        repetition_penalties=torch.tensor([2.0]),
        text_vocab=99,
    )

    assert tokens == [[1, 0, 99]]


def test_repetition_penalty_handles_negative_logits_sign_aware():
    logits = torch.tensor([[[-2.0, -3.0]]])

    penalized = apply_repetition_penalty(
        logits,
        repetition_token_ids=torch.tensor([[[0]]]),
        repetition_penalties=torch.tensor([2.0]),
    )

    assert torch.equal(penalized, torch.tensor([[[-4.0, -3.0]]]))


def test_repetition_penalty_ignores_invalid_padding_tokens():
    logits = torch.tensor([[[4.0, 3.0]]])

    penalized = apply_repetition_penalty(
        logits,
        repetition_token_ids=torch.tensor([[[0, -1]]]),
        repetition_penalties=torch.tensor([2.0]),
    )

    assert torch.equal(penalized, torch.tensor([[[2.0, 3.0]]]))


def test_sampler_prepare_limits_repetition_to_first_codebooks():
    token_pool = torch.tensor(
        [
            [
                [99, 99, 99],
                [1, 2, 99],
                [3, 4, 99],
                [5, 6, 99],
            ]
        ],
        dtype=torch.int32,
    )
    req = SimpleNamespace(
        sampling_params=TTSSamplingParams(
            repetition_window=2,
            repetition_penalty=2.0,
            repetition_codebooks=1,
        ),
        total_generated=3,
        device_len=4,
        table_idx=0,
        input_ids=token_pool[0],
        rng=None,
    )
    batch = SimpleNamespace(reqs=[req])
    sampler = TTSSampler(
        device=torch.device("cpu"),
        n_codebooks=2,
        codebook_size=10,
        text_vocab=99,
    )

    args = sampler.prepare(batch, token_pool=token_pool)

    assert args.repetition_token_ids is not None
    assert args.repetition_token_ids.tolist() == [[[3, 5], [-1, -1]]]
