from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from zonos2.utils.moe_topk import normalize_special_topk_layers, resolve_layer_topk


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


def normalize_moe_balancing_strategy(strategy: str) -> str:
    normalized = strategy.strip().lower().replace("-", "_")
    aliases = {
        "current": "quantile",
        "quantile": "quantile",
        "qbalancing": "quantile",
        "old": "legacy",
        "legacy": "legacy",
        "aux": "legacy",
        "aux_loss": "legacy",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "Unsupported model.moe_balancing_strategy="
            f"{strategy!r}. Expected one of: current, quantile, qbalancing, old, legacy, aux_loss."
        ) from exc


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    special_topk_layers: dict[int, int] | None = None
    # Zonos2-specific fields
    n_codebooks: int = 9
    codebook_size: int = 1024
    text_vocab: int | None = None
    moe_n_experts: int = 1
    moe_start_from_layer: int = 0
    moe_end_from_layer: int = 0
    moe_router_dim: int = 256  # Router hidden dimension
    moe_impl: str = "grouped"
    moe_balancing_strategy: str = "legacy"
    # TTS-specific fields
    eoa_id: int = 1024  # End-of-audio token ID
    audio_pad_id: int = 1025  # Audio padding token ID
    loss_softcap: float = 15.0  # Logit soft capping value
    speaker_enabled: bool = False
    speaker_embedding_dim: int = 128
    speaker_lda_dim: int | None = None
    speaker_background_token_enabled: bool = False
    accurate_mode_token_enabled: bool = False
    speaking_rate_num_buckets: int = 0
    speaking_rate_buckets: tuple[str, ...] = ()
    quality_num_buckets: int = 0
    quality_features: tuple[str, ...] = ()
    quality_buckets: Dict[str, tuple[str, ...]] | None = None
    quality_dropout: Dict[str, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "special_topk_layers",
            normalize_special_topk_layers(self.special_topk_layers),
        )

    def get_num_experts_per_tok(self, layer_idx: int) -> int:
        default_topk = self.num_experts_per_tok if self.num_experts_per_tok > 0 else 1
        return resolve_layer_topk(default_topk, self.special_topk_layers, layer_idx)

    @classmethod
    def from_checkpoint_config(cls, config) -> ModelConfig:
        return cls.from_zonos2_config(config)

    @classmethod
    def from_zonos2_config(cls, config) -> ModelConfig:
        """Create ModelConfig from Zonos2Config (training checkpoint format).

        Zonos2Config uses different field names than HuggingFace:
        - n_layers -> num_layers
        - dim -> hidden_size
        - head_dim -> head_dim
        - n_heads -> num_qo_heads
        - n_kv_heads -> num_kv_heads
        - ffn_dim_multiplier * dim -> intermediate_size
        """
        # Calculate dimensions
        head_dim = config.head_dim
        dim = config.dim
        n_heads = config.n_heads if config.n_heads is not None else dim // head_dim
        n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else n_heads

        # Calculate intermediate size from ffn_dim_multiplier
        ffn_dim = int(config.ffn_dim_multiplier * dim)
        # Round to multiple_of
        multiple_of = getattr(config, "multiple_of", 256)
        intermediate_size = multiple_of * ((ffn_dim + multiple_of - 1) // multiple_of)

        # Calculate vocab size for audio codebooks
        # Total vocab = n_codebooks * (codebook_size + 2) + text_vocab (if present)
        audio_vocab = config.codebook_size + 2  # +2 for eoa_id and audio_pad_id
        vocab_size = config.n_codebooks * audio_vocab
        if config.text_vocab is not None:
            vocab_size += config.text_vocab + 1  # +1 for text padding

        # MoE settings
        moe_n_experts = getattr(config, "moe_n_experts", 1)
        moe_router_topk = getattr(config, "moe_router_topk", 1)
        special_topk_layers = getattr(config, "special_topk_layers", None)
        moe_router_dim = getattr(config, "moe_router_dim", 256)
        moe_impl = getattr(config, "moe_impl", "grouped")
        moe_balancing_strategy = normalize_moe_balancing_strategy(
            getattr(config, "moe_balancing_strategy", "legacy")
        )

        return cls(
            num_layers=config.n_layers,
            num_qo_heads=n_heads,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_size=dim,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            hidden_act="silu",  # Zonos2 uses SiLU
            rms_norm_eps=config.norm_eps,
            tie_word_embeddings=False,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_seqlen,
                base=config.rope_theta,
                scaling=None,
            ),
            num_experts=moe_n_experts if moe_n_experts > 1 else 0,
            num_experts_per_tok=moe_router_topk if moe_n_experts > 1 else 0,
            special_topk_layers=special_topk_layers,
            moe_intermediate_size=intermediate_size if moe_n_experts > 1 else 0,
            norm_topk_prob=False,
            # Zonos2 fields
            n_codebooks=config.n_codebooks,
            codebook_size=config.codebook_size,
            text_vocab=config.text_vocab,
            moe_n_experts=moe_n_experts,
            moe_start_from_layer=getattr(config, "moe_start_from_layer", 0),
            moe_end_from_layer=getattr(config, "moe_end_from_layer", 0),
            moe_router_dim=moe_router_dim,
            moe_impl=moe_impl,
            moe_balancing_strategy=moe_balancing_strategy,
            # TTS fields
            eoa_id=config.eoa_id,
            audio_pad_id=config.audio_pad_id,
            loss_softcap=config.loss_softcap,
            speaker_enabled=getattr(config, "speaker_enabled", False),
            speaker_embedding_dim=getattr(config, "speaker_embedding_dim", 128),
            speaker_lda_dim=getattr(config, "speaker_lda_dim", None),
            speaker_background_token_enabled=bool(
                getattr(config, "speaker_background_token_enabled", False)
            ),
            accurate_mode_token_enabled=bool(
                getattr(config, "accurate_mode_token_enabled", False)
            ),
            speaking_rate_num_buckets=getattr(config, "speaking_rate_num_buckets", 0),
            speaking_rate_buckets=tuple(getattr(config, "speaking_rate_buckets", None) or ()),
            quality_num_buckets=int(getattr(config, "quality_num_buckets", 0) or 0),
            quality_features=tuple(getattr(config, "quality_features", None) or ()),
            quality_buckets={
                str(feature): tuple(str(item) for item in (buckets or ()))
                for feature, buckets in (getattr(config, "quality_buckets", None) or {}).items()
            }
            or None,
            quality_dropout=dict(getattr(config, "quality_dropout", None) or {}) or None,
        )
