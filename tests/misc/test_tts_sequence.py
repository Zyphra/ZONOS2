from __future__ import annotations

from zonos2.message.tts import TTSSamplingParams
from zonos2.tts.sequence import TTSSequence


def test_tts_sequence_detects_any_delayed_eoa_and_aligns_frame():
    seq = TTSSequence(
        prompt_ids=[[99, 99, 99, 0]],
        sampling_params=TTSSamplingParams(max_tokens=64),
        n_codebooks=3,
        eoa_id=7,
    )

    seq.append_token([1, 2, 3, 0])
    assert seq.eos_frame is None

    seq.append_token([1, 2, 7, 0])
    assert seq.eos_frame == 0
    assert seq.eos_countdown == 3
    assert not seq.is_finished

    seq.append_token([1, 2, 3, 0])
    seq.append_token([1, 2, 3, 0])
    seq.append_token([1, 2, 3, 0])

    assert seq.eos_countdown == 0
    assert seq.is_finished
