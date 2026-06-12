from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel

try:
    from zonos.utils import DEFAULT_DEVICE
except Exception:
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Qwen3SpeakerEmbedding(nn.Module):
    """Qwen3 voice embedding extractor used by 2048D speaker-conditioned checkpoints."""

    MODEL_NAME = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"
    TARGET_SAMPLE_RATE = 24_000
    N_FFT = 1024
    HOP_LENGTH = 256
    WIN_LENGTH = 1024
    N_MELS = 128
    F_MIN = 0.0
    F_MAX = 12_000.0

    def __init__(self, device: str = DEFAULT_DEVICE):
        super().__init__()
        self.device = device
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.TARGET_SAMPLE_RATE,
            n_fft=self.N_FFT,
            win_length=self.WIN_LENGTH,
            hop_length=self.HOP_LENGTH,
            f_min=self.F_MIN,
            f_max=self.F_MAX,
            n_mels=self.N_MELS,
            power=1.0,
            center=False,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)

        self.requires_grad_(False).eval()

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @cache
    def _get_resampler(self, orig_sample_rate: int):
        return torchaudio.transforms.Resample(orig_sample_rate, self.TARGET_SAMPLE_RATE).to(self.device)

    def prepare_input(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        assert wav.ndim < 3
        if wav.ndim == 2:
            wav = wav.mean(0, keepdim=True)
        wav = wav.to(self.device, torch.float32)
        if sample_rate != self.TARGET_SAMPLE_RATE:
            wav = self._get_resampler(sample_rate)(wav)
        return wav

    def _make_mel(self, wav: torch.Tensor) -> torch.Tensor:
        # Mirror the standalone demo preprocessing: reflect-pad, magnitude mel, log clamp.
        pad = (self.N_FFT - self.HOP_LENGTH) // 2
        wav = F.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        mel = self.mel_transform(wav)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.transpose(1, 2)

    def forward(self, wav: torch.Tensor, sample_rate: int):
        wav = self.prepare_input(wav, sample_rate)
        mel = self._make_mel(wav)
        return self.model(input_values=mel).last_hidden_state.to(torch.float32)
