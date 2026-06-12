from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from zonos2.utils.logger import init_logger
from zonos2.utils.moe_topk import normalize_special_topk_layers

logger = init_logger(__name__)

_HF_REPO_ID_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


@lru_cache()
def resolve_model_path(model_path: str) -> str:
    """Resolve a model path to a local directory.

    Local paths are returned as-is. A Hugging Face repo id (e.g.
    ``Zyphra/ZONOS2``) is downloaded to the local HF cache first.
    """
    if Path(model_path).expanduser().exists():
        return model_path
    if _HF_REPO_ID_RE.match(model_path):
        from huggingface_hub import snapshot_download

        logger.info("Downloading checkpoint from Hugging Face: %s", model_path)
        return snapshot_download(
            model_path,
            allow_patterns=["*.json", "*.pth", "*.pt", "*.yaml"],
        )
    return model_path


@dataclass
class Zonos2Config:
    """Config class for zonos2 models loaded from training checkpoints.

    This mimics the HuggingFace config interface for compatibility with the server.
    """

    model_type: str = "zonos2"
    dtype: str = "bfloat16"

    # Model architecture
    n_layers: int = 8
    dim: int = 512
    head_dim: int = 128
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None
    ffn_dim_multiplier: float = 4.0
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_seqlen: int = 2048

    # TTS-specific
    n_codebooks: int = 9
    codebook_size: int = 1024
    eoa_id: int = 1024
    audio_pad_id: int = 1025
    text_vocab: Optional[int] = None
    speaker_enabled: bool = False
    speaker_embedding_dim: int = 128
    # Optional LDA projection applied to speaker embeddings before the speaker
    # projection. The projection weights live inside the model checkpoint.
    speaker_lda_dim: Optional[int] = None
    # Clean/noisy speaker-background marker tokens (2 ids at the text-vocab tail).
    speaker_background_token_enabled: bool = False
    # Accurate-mode marker token (1 id after the background markers).
    accurate_mode_token_enabled: bool = False
    speaking_rate_num_buckets: int = 0
    speaking_rate_buckets: Optional[list[str]] = None
    quality_num_buckets: int = 0
    quality_features: Optional[list[str]] = None
    quality_buckets: Optional[dict[str, list[str]]] = None
    quality_dropout: Optional[dict[str, float]] = None

    # MoE config
    moe_router_topk: int = 1
    special_topk_layers: dict[int, int] | None = None
    moe_n_experts: int = 1
    moe_router_dim: int = 128
    moe_start_from_layer: int = 0
    moe_end_from_layer: int = 0
    moe_impl: str = "grouped"
    moe_balancing_strategy: str = "legacy"

    # Loss
    loss_softcap: float = 15.0

    def __post_init__(self) -> None:
        self.special_topk_layers = normalize_special_topk_layers(
            self.special_topk_layers
        )
        if self.speaking_rate_buckets is not None:
            self.speaking_rate_buckets = [str(item) for item in self.speaking_rate_buckets]
        if self.quality_features is not None:
            self.quality_features = [str(item) for item in self.quality_features]
        if self.quality_buckets is not None:
            self.quality_buckets = {
                str(feature): [str(item) for item in (buckets or [])]
                for feature, buckets in self.quality_buckets.items()
            }
            if self.quality_features is None:
                self.quality_features = list(self.quality_buckets.keys())
            if int(self.quality_num_buckets or 0) <= 0:
                self.quality_num_buckets = sum(
                    len(self.quality_buckets.get(feature, ()))
                    for feature in (self.quality_features or [])
                )
        if self.quality_dropout is not None:
            self.quality_dropout = {
                str(feature): float(dropout)
                for feature, dropout in self.quality_dropout.items()
            }
        self.speaker_background_token_enabled = bool(self.speaker_background_token_enabled)
        self.accurate_mode_token_enabled = bool(self.accurate_mode_token_enabled)

    def to_dict(self):
        return self.__dict__.copy()


def _load_zonos2_config(model_path: str) -> Zonos2Config:
    """Load config from zonos2 training checkpoint format.

    Supports:
    - config.yaml in parent directory (training run format)
    - params.json in checkpoint directory (release checkpoint format)
    """
    from pathlib import Path

    def _cfg_get(cfg, key: str, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _apply_data_sidecar(model_params: dict, data_cfg) -> None:
        if data_cfg is None:
            return

        rate_buckets = _cfg_get(data_cfg, "speaking_rate_buckets", None) or []
        rate_buckets = [str(item) for item in rate_buckets]
        if rate_buckets:
            model_params["speaking_rate_buckets"] = rate_buckets

        rate_count = int(model_params.get("speaking_rate_num_buckets") or 0)
        rate_enabled = bool(_cfg_get(data_cfg, "speaking_rate_enabled", False))
        if rate_enabled:
            sidecar_rate_count = len(rate_buckets)
            if sidecar_rate_count == 0:
                sidecar_rate_count = int(_cfg_get(data_cfg, "speaking_rate_num_buckets", 0) or 0)
            if sidecar_rate_count > 0:
                rate_count = sidecar_rate_count
                if not int(model_params.get("speaking_rate_num_buckets") or 0):
                    model_params["speaking_rate_num_buckets"] = rate_count

        # Quality conditioning lives in the data section of training configs.
        if bool(_cfg_get(data_cfg, "quality_enabled", False)):
            raw_features = _cfg_get(data_cfg, "quality_features", None)
            if hasattr(raw_features, "items"):
                quality_features = [
                    str(feature) for feature, enabled in raw_features.items() if bool(enabled)
                ]
            else:
                quality_features = [str(item) for item in (raw_features or ())]
            raw_buckets = _cfg_get(data_cfg, "quality_buckets", None) or {}
            quality_buckets = {
                str(feature): [str(item) for item in (raw_buckets.get(feature, None) or ())]
                for feature in (quality_features or raw_buckets.keys())
            }
            if quality_buckets and "quality_buckets" not in model_params:
                model_params["quality_buckets"] = quality_buckets
                model_params["quality_features"] = quality_features or list(
                    quality_buckets.keys()
                )
            raw_dropout = _cfg_get(data_cfg, "quality_dropout", None)
            if raw_dropout is not None and "quality_dropout" not in model_params:
                if hasattr(raw_dropout, "items"):
                    model_params["quality_dropout"] = {
                        str(feature): float(dropout) for feature, dropout in raw_dropout.items()
                    }

        # Training configs name the marker-token flags after their data pipeline.
        background_enabled = _cfg_get(data_cfg, "speaker_embedding_origin_token_enabled", None)
        if background_enabled is not None:
            model_params.setdefault(
                "speaker_background_token_enabled", bool(background_enabled)
            )
        accurate_enabled = _cfg_get(
            data_cfg, "speaker_embedding_cartesia_clone_source_token_enabled", None
        )
        if accurate_enabled is not None:
            model_params.setdefault("accurate_mode_token_enabled", bool(accurate_enabled))

        if rate_count > 0 and model_params.get("text_vocab") is None:
            try:
                from zonos2.tts.prompt import conditioned_text_vocab_size

                quality_count = sum(
                    len(buckets)
                    for buckets in (model_params.get("quality_buckets") or {}).values()
                )
                background_count = (
                    2 if model_params.get("speaker_background_token_enabled") else 0
                )
                accurate_count = (
                    1
                    if model_params.get("accurate_mode_token_enabled") and background_count
                    else 0
                )
                model_params["text_vocab"] = conditioned_text_vocab_size(
                    rate_count, quality_count, background_count, accurate_count
                )
                logger.debug(
                    "Resolved text_vocab=%d from data text vocabulary and conditioning buckets",
                    model_params["text_vocab"],
                )
            except Exception as exc:
                logger.debug("Could not resolve conditioned text_vocab from data config: %s", exc)

    def _validate_model_type(model_params: dict) -> None:
        model_type = model_params.get("model_type")
        if model_type is not None and str(model_type) != "zonos2":
            raise ValueError(
                f"Unsupported model_type={model_type!r}. This release only loads zonos2 checkpoints."
            )

    model_path = Path(model_path)

    # Try params.json in checkpoint dir first
    params_json = model_path / "params.json"
    if params_json.exists():
        with open(params_json, "r") as f:
            params = json.load(f)
        # params.json may have nested structure
        if "model" in params:
            model_params = params["model"]
        else:
            model_params = params

        # Also check config.yaml for tokenizer sidecar fields stored in data.
        for parent in [model_path, model_path.parent, model_path.parent.parent]:
            config_yaml = parent / "config.yaml"
            if config_yaml.exists():
                try:
                    from omegaconf import OmegaConf

                    cfg = OmegaConf.load(config_yaml)
                    _apply_data_sidecar(model_params, getattr(cfg, "data", None))
                except ImportError:
                    import yaml

                    with open(config_yaml, "r") as f:
                        cfg = yaml.safe_load(f)
                    _apply_data_sidecar(
                        model_params, cfg.get("data") if isinstance(cfg, dict) else None
                    )
                break

        _validate_model_type(model_params)
        result = Zonos2Config(
            **{k: v for k, v in model_params.items() if hasattr(Zonos2Config, k)}
        )
        logger.debug("Loaded Zonos2Config from params.json")
        return result

    # Try config.yaml in parent directories
    for parent in [model_path, model_path.parent, model_path.parent.parent]:
        config_yaml = parent / "config.yaml"
        logger.debug("Checking for config at: %s", config_yaml)
        if config_yaml.exists():
            logger.debug("Found config.yaml at: %s", config_yaml)
            try:
                from omegaconf import OmegaConf

                cfg = OmegaConf.load(config_yaml)
                # Navigate to model config
                if hasattr(cfg, "model"):
                    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
                else:
                    model_cfg = OmegaConf.to_container(cfg, resolve=True)
                # Also check data section for tokenizer sidecar fields.
                _apply_data_sidecar(model_cfg, getattr(cfg, "data", None))
                _validate_model_type(model_cfg)
                result = Zonos2Config(
                    **{k: v for k, v in model_cfg.items() if hasattr(Zonos2Config, k)}
                )
                logger.debug("Loaded Zonos2Config from config.yaml")
                return result
            except ImportError:
                # omegaconf not available, try as regular yaml
                import yaml

                with open(config_yaml, "r") as f:
                    cfg = yaml.safe_load(f)
                model_cfg = cfg.get("model", cfg)
                # Also check data section for tokenizer sidecar fields.
                _apply_data_sidecar(model_cfg, cfg.get("data") if isinstance(cfg, dict) else None)
                _validate_model_type(model_cfg)
                result = Zonos2Config(
                    **{k: v for k, v in model_cfg.items() if hasattr(Zonos2Config, k)}
                )
                logger.debug("Loaded Zonos2Config from config.yaml (yaml)")
                return result

    raise ValueError(
        f"Could not find config.yaml or params.json for zonos2 checkpoint at {model_path}"
    )


def _is_zonos2_checkpoint(model_path: str) -> bool:
    """Check if path is a zonos2 checkpoint with a loadable config."""
    model_path = Path(model_path)

    # Check for params.json
    if (model_path / "params.json").exists():
        return True

    # Check for config.yaml in parent dirs (training run format)
    for parent in [model_path, model_path.parent, model_path.parent.parent]:
        if (parent / "config.yaml").exists():
            return True

    return False


@lru_cache()
def _load_config(model_path: str) -> Zonos2Config:
    model_path = resolve_model_path(model_path)
    if not _is_zonos2_checkpoint(model_path):
        raise ValueError(
            f"Unsupported checkpoint at {model_path!r}. This release only loads "
            "Zonos2 TTS checkpoints with config.yaml or params.json."
        )
    return _load_zonos2_config(model_path)


def cached_load_checkpoint_config(model_path: str) -> Zonos2Config:
    config = _load_config(model_path)
    return Zonos2Config(**config.to_dict())
