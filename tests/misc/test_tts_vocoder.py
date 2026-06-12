from __future__ import annotations

import torch
from zonos2.message import TTSDetokenizeMsg
from zonos2.tokenizer import vocoder


def test_vocoder_does_not_send_eos_frame_or_after_to_dac(monkeypatch):
    captured_codes: list[torch.Tensor] = []

    def fake_decode_dac(codes: torch.Tensor) -> torch.Tensor:
        captured_codes.append(codes.cpu().clone())
        return torch.zeros((codes.shape[0], codes.shape[1]), dtype=torch.float32)

    monkeypatch.setattr(vocoder, "decode_dac", fake_decode_dac)

    manager = vocoder.TTSVocoderManager(
        n_codebooks=3,
        audio_pad_id=1025,
        min_decode_chunk=99,
        overlap_frames=0,
        hop_length=1,
    )
    chunks = manager.decode_frames(
        [
            TTSDetokenizeMsg(uid=1, audio_codes=[10, 20, 30], finished=False),
            TTSDetokenizeMsg(uid=1, audio_codes=[11, 21, 31], finished=False),
            TTSDetokenizeMsg(uid=1, audio_codes=[999, 22, 32], finished=False),
            TTSDetokenizeMsg(uid=1, audio_codes=[13, 999, 33], finished=False),
            TTSDetokenizeMsg(
                uid=1,
                audio_codes=[14, 24, 999],
                finished=True,
                eos_frame=2,
            ),
        ]
    )

    assert chunks[:-1] == [b"", b"", b"", b""]
    assert chunks[-1]
    assert len(captured_codes) == 1
    assert captured_codes[0].tolist() == [
        [
            [10, 21, 32],
            [11, 22, 33],
        ]
    ]
