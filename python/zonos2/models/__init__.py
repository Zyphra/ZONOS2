from __future__ import annotations

from .config import ModelConfig, RotaryConfig
from .weight import load_checkpoint_weight
from .zonos2 import Zonos2ForCausalLM


def create_model(config: ModelConfig) -> Zonos2ForCausalLM:
    return Zonos2ForCausalLM(config.model_config, config.moe_backend)


__all__ = [
    "Zonos2ForCausalLM",
    "load_checkpoint_weight",
    "create_model",
    "ModelConfig",
    "RotaryConfig",
]
