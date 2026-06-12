"""Zonos2 model matching checkpoint naming from zonos2 training."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn.functional as F
from zonos2.core import get_global_ctx
from zonos2.distributed import get_tp_info
from zonos2.layers import (
    AttentionLayer,
    BaseOP,
    ChunkedLinear,
    LinearRowParallel,
    OPList,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
)
from zonos2.layers.linear import LinearReplicated
from zonos2.layers.moe.fused_moe.layer import FusedMoE
from zonos2.utils import divide_even, init_logger, nvtx_annotate

from .config import normalize_moe_balancing_strategy
from .speaker_lda import SpeakerLDAProjection

_model_debug_counter = 0
_logger = init_logger(__name__)

if TYPE_CHECKING:
    from .config import ModelConfig


class SimpleLinear(BaseOP):
    """Simple linear layer with .weight attribute for checkpoint compatibility."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty(out_features, in_features)
        self.bias = torch.empty(out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class SimpleLinearOProj(BaseOP):
    """Linear output projection with all-reduce for tensor parallelism.

    Has .weight attribute for checkpoint compatibility (wo.weight).
    """

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        from zonos2.distributed import DistributedCommunicator

        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty(out_features, in_features)
        self.bias = torch.empty(out_features) if has_bias else None
        self._comm = DistributedCommunicator()
        self._tp_size = get_tp_info().size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class MultiOutputHead(BaseOP):
    """Multi-codebook output head with .weight attribute.

    Checkpoint naming: multi_output.weight
    """

    def __init__(self, hidden_size: int, output_size: int):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight = torch.empty(output_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class MultiEmbedding(BaseOP):
    """Multi-codebook embedding for audio + text tokens.

    Checkpoint naming: multi_embedder.embedders.{N}.weight
    All embedders are in a single ModuleList.

    IMPORTANT: Uses padding_idx to zero out embeddings for padding tokens.
    This matches the reference implementation behavior where:
    - Audio codebooks use audio_pad_id (1025) as padding
    - Text embeddings use text_vocab as padding
    """

    def __init__(self, config: ModelConfig):
        self.n_codebooks = config.n_codebooks
        self.text_vocab = config.text_vocab
        self.audio_pad_id = config.audio_pad_id

        # Store padding indices for each embedder column
        # padding_indices[i] is the padding index for column i, or None if no padding
        self.padding_indices: list[int | None] = []

        # All embedders in a single list: audio codebooks first, then text.
        # This matches zonos2 checkpoint naming: multi_embedder.embedders.{0..N}.weight
        embedders_list = []

        # Audio codebook embeddings (padding_idx = audio_pad_id)
        for _ in range(config.n_codebooks):
            embedders_list.append(
                VocabParallelEmbedding(
                    num_embeddings=config.codebook_size + 2,
                    embedding_dim=config.hidden_size,
                )
            )
            self.padding_indices.append(config.audio_pad_id)

        # Optional text embedding (at index n_codebooks, padding_idx = text_vocab)
        if config.text_vocab is not None:
            embedders_list.append(
                VocabParallelEmbedding(
                    num_embeddings=config.text_vocab + 1,
                    embedding_dim=config.hidden_size,
                )
            )
            self.padding_indices.append(config.text_vocab)

        self.embedders = OPList(embedders_list)

    @nvtx_annotate("MultiEmbedding")
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: (batch, seq, n_codebooks + extras) or (total_tokens, n_codebooks + extras)
        # Sum embeddings from all provided codebooks
        # Note: codes[..., i] creates non-contiguous view, must call .contiguous()
        #
        # NOTE: We do NOT mask out padding tokens here. The reference implementation
        # relies on the embedding table having zeros at padding positions, but the
        # checkpoint may have non-zero values there. To match reference behavior,
        # we just sum all embeddings without masking.

        # Get first embedding
        col_codes = codes[..., 0].contiguous()
        result = self.embedders.op_list[0].forward(col_codes)

        # Sum remaining columns
        for i in range(1, codes.size(-1)):
            col_codes = codes[..., i].contiguous()
            emb = self.embedders.op_list[i].forward(col_codes)
            result = result + emb

        return result


class Attention(BaseOP):
    """Attention with QK norm and headwise Qwen-style gating.

    Checkpoint naming:
    - layers.{N}.attention.wq.weight
    - layers.{N}.attention.wkv.weight (3D: [2, kv_dim, hidden])
    - layers.{N}.attention.wo.weight
    - layers.{N}.attention.temp
    - layers.{N}.attention.gater.weight
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        head_dim = config.head_dim
        self.num_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = head_dim

        # Linear projections matching checkpoint naming
        tp_info = get_tp_info()
        local_num_heads = divide_even(self.num_heads, tp_info.size)
        local_num_kv_heads = divide_even(self.num_kv_heads, tp_info.size)

        # wq: [num_heads * head_dim, hidden_size] - uses SimpleLinear for .weight attribute
        self.wq = SimpleLinear(config.hidden_size, local_num_heads * head_dim, has_bias=False)
        # wkv: ChunkedLinear with 3D weight [2, kv_dim, hidden]
        self.wkv = ChunkedLinear(
            in_features=config.hidden_size,
            out_features=local_num_kv_heads * head_dim * 2,
            divisor=2,
            has_bias=False,
        )
        # wo: [hidden_size, num_heads * head_dim] - uses SimpleLinearOProj for all-reduce + .weight attribute
        self.wo = SimpleLinearOProj(local_num_heads * head_dim, config.hidden_size, has_bias=False)

        # QK normalization with learnable temperature.
        # temp: [1, local_num_heads, 1] - sharded across TP ranks.
        self.temp = torch.ones(1, local_num_heads, 1, dtype=torch.bfloat16)

        # Attention backend
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=None,  # We handle norm ourselves for temperature scaling
            k_norm=None,
        )

        # Gater output is sharded across TP ranks (local_num_heads per rank).
        self.gater = SimpleLinear(config.hidden_size, local_num_heads, has_bias=False)

    @nvtx_annotate("Attention")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global _model_debug_counter
        ctx = get_global_ctx()
        is_capturing = torch.cuda.is_current_stream_capturing()
        debug_attn = (
            _logger.isEnabledFor(logging.DEBUG) and ctx.batch.is_decode and not is_capturing and
            self.attn.layer_id == 0 and _model_debug_counter <= 10
        )

        gate = torch.sigmoid(self.gater.forward(x))

        # Q projection
        q = self.wq.forward(x)

        # KV projection (from ChunkedLinear)
        kv = self.wkv.forward(x)

        # Split KV into K and V
        tp_info = get_tp_info()
        local_num_kv_heads = divide_even(self.num_kv_heads, tp_info.size)
        kv_dim = local_num_kv_heads * self.head_dim

        k, v = kv.split([kv_dim, kv_dim], dim=-1)
        # v needs to be contiguous because it's passed directly to the KV cache kernel
        # (k becomes contiguous through flatten/view operations in the QK norm path)
        v = v.contiguous()

        if debug_attn:
            step = _model_debug_counter
            _logger.debug("Attn L0 Step %d Q proj: norm=%.4f", step, q.norm().item())
            _logger.debug("Attn L0 Step %d K proj: norm=%.4f", step, k.norm().item())
            _logger.debug("Attn L0 Step %d V proj: norm=%.4f", step, v.norm().item())

        metadata = ctx.batch.attn_metadata
        local_num_heads = divide_even(self.num_heads, tp_info.size)
        qo_attn_dim = local_num_heads * self.head_dim

        q = q.view(-1, local_num_heads, self.head_dim)
        k = k.view(-1, local_num_kv_heads, self.head_dim)

        q = F.rms_norm(q, (self.head_dim,), eps=1e-6) * self.temp.abs().to(q.dtype).to(q.device)
        k = F.rms_norm(k, (self.head_dim,), eps=1e-6)

        if debug_attn:
            step = _model_debug_counter
            _logger.debug("Attn L0 Step %d Q after norm+temp: norm=%.4f", step, q.norm().item())
            _logger.debug("Attn L0 Step %d K after norm: norm=%.4f", step, k.norm().item())

        # Zonos2 uses interleaved RoPE format (is_neox=False).
        if self.attn.rotary:
            q, k = self.attn.rotary.forward(metadata.positions, q.flatten(-2), k.flatten(-2), is_neox=False)

        if debug_attn:
            step = _model_debug_counter
            _logger.debug("Attn L0 Step %d Q after RoPE: norm=%.4f", step, q.norm().item())
            _logger.debug("Attn L0 Step %d K after RoPE: norm=%.4f", step, k.norm().item())

        # q needs to be 3D for flash attention: (total_tokens, num_heads, head_dim)
        # k and v stay 2D for the KV cache store kernel: (total_tokens, kv_dim)
        q = q.view(-1, local_num_heads, self.head_dim)

        o = ctx.attn_backend.forward(q, k, v, self.attn.layer_id, ctx.batch)
        o = o.view(-1, qo_attn_dim)

        if debug_attn:
            _logger.debug("Attn L0 Step %d Attn output: norm=%.4f", _model_debug_counter, o.norm().item())

        o = o.view(-1, local_num_heads, self.head_dim)
        o = o * gate.unsqueeze(-1)
        o = o.view(-1, local_num_heads * self.head_dim)

        if debug_attn:
            _logger.debug("Attn L0 Step %d After gating: norm=%.4f", _model_debug_counter, o.norm().item())

        # Output projection
        result = self.wo.forward(o)

        if debug_attn:
            _logger.debug("Attn L0 Step %d After wo proj: norm=%.4f", _model_debug_counter, result.norm().item())

        return result


class FeedForward(BaseOP):
    """Dense feedforward layer.

    Checkpoint naming:
    - layers.{N}.feed_forward.w_in.weight (3D: [2, intermediate, hidden])
    - layers.{N}.feed_forward.w_out.weight
    """

    def __init__(self, config: ModelConfig):
        tp_info = get_tp_info()
        local_intermediate = divide_even(config.intermediate_size, tp_info.size)

        # w_in: ChunkedLinear [2, intermediate, hidden] for gate and up
        self.w_in = ChunkedLinear(
            in_features=config.hidden_size,
            out_features=local_intermediate * 2,
            divisor=2,
            has_bias=False,
        )
        # w_out: [hidden, intermediate]
        self.w_out = LinearRowParallel(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            has_bias=False,
        )
        self._inter_size = local_intermediate

    @nvtx_annotate("FeedForward")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # w_in produces [h, gate] where h is "up" and gate gets SiLU
        # Reference does: h * F.silu(gate) = up * SiLU(gate)
        h_gate = self.w_in.forward(x)
        inter_size = self._inter_size
        h = h_gate[..., :inter_size]      # first half = h (up projection)
        gate = h_gate[..., inter_size:]   # second half = gate
        # Match reference: h * F.silu(gate)
        y = h * F.silu(gate)
        return self.w_out.forward(y)


class FusedGroupedExperts(BaseOP):
    """Fused MoE experts with weight fusion during loading.

    Checkpoint naming:
    - layers.{N}.feed_forward.experts.w1.weight (or per-expert: w1)
    - layers.{N}.feed_forward.experts.w2.weight
    - layers.{N}.feed_forward.experts.w3.weight
    - layers.{N}.feed_forward.experts.w13 (SonicMoE, interleaved gate/up)

    Inference format:
    - gate_up_proj: [num_experts, 2 * intermediate, hidden] = concat(w1, w3)
    - down_proj: [num_experts, hidden, intermediate] = w2
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        tp_info = get_tp_info()
        # Use moe_n_experts for zonos2 (not num_experts which is for HF models)
        self.num_experts = config.moe_n_experts
        self.hidden_size = config.hidden_size
        # Use intermediate_size for MoE (moe_intermediate_size may be 0)
        self.intermediate_size = config.moe_intermediate_size if config.moe_intermediate_size > 0 else config.intermediate_size
        self.top_k = config.get_num_experts_per_tok(layer_id)

        local_intermediate = divide_even(self.intermediate_size, tp_info.size)

        # Fused weights: gate_up_proj = [w1, w3], down_proj = w2
        self.gate_up_proj = torch.empty(
            self.num_experts, 2 * local_intermediate, self.hidden_size
        )
        self.down_proj = torch.empty(
            self.num_experts, self.hidden_size, local_intermediate
        )

        self._fused_moe = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            renormalize=config.norm_topk_prob,
            prefix="",
        )
        # Override weights with ours
        self._fused_moe.gate_up_proj = self.gate_up_proj
        self._fused_moe.down_proj = self.down_proj

    def load_state_dict(
        self,
        state_dict,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """Handle loading from both fused and unfused weight formats.

        If checkpoint has w1.weight and w3.weight (unfused), fuse them into gate_up_proj.
        If checkpoint has SonicMoE w13, de-interleave it into gate_up_proj.
        If checkpoint has gate_up_proj, load directly.
        """
        from zonos2.layers.base import _concat_prefix

        # Keys we expect in fused format
        gate_up_key = _concat_prefix(prefix, "gate_up_proj")
        down_proj_key = _concat_prefix(prefix, "down_proj")

        # Keys from zonos2 checkpoint (unfused format)
        w1_key = _concat_prefix(prefix, "w1.weight")
        w2_key = _concat_prefix(prefix, "w2.weight")
        w3_key = _concat_prefix(prefix, "w3.weight")
        sonic_w13_key = _concat_prefix(prefix, "w13")
        sonic_w2_key = _concat_prefix(prefix, "w2")

        def _convert_sonic_w13_to_gate_up(w13: torch.Tensor) -> torch.Tensor:
            assert w13.dim() == 3, f"Expected SonicMoE w13 to be rank-3, got {w13.shape}"
            assert w13.shape[1] % 2 == 0, (
                f"Expected even fused width in SonicMoE w13, got {w13.shape}"
            )
            gate = w13[:, 0::2, :]
            up = w13[:, 1::2, :]
            return torch.cat([gate, up], dim=1)

        # Check if we have unfused weights that need fusion
        if gate_up_key not in state_dict and sonic_w13_key in state_dict:
            state_dict[gate_up_key] = _convert_sonic_w13_to_gate_up(
                state_dict.pop(sonic_w13_key)
            )
        elif gate_up_key not in state_dict and w1_key in state_dict and w3_key in state_dict:
            w1 = state_dict.pop(w1_key)
            w3 = state_dict.pop(w3_key)
            # Fuse: gate_up_proj = [w1, w3] along dim 1
            # w1 = gate projection (SiLU applied), w3 = up projection
            state_dict[gate_up_key] = torch.cat([w1, w3], dim=1)

        # Handle w2 -> down_proj mapping
        if down_proj_key not in state_dict and sonic_w2_key in state_dict:
            state_dict[down_proj_key] = state_dict.pop(sonic_w2_key)
        elif w2_key in state_dict and down_proj_key not in state_dict:
            state_dict[down_proj_key] = state_dict.pop(w2_key)

        # Now load using standard mechanism
        super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)

        # Update _fused_moe's references to point to the newly loaded tensors
        # (setattr in super().load_state_dict replaces self.gate_up_proj/down_proj,
        # but _fused_moe still points to the old meta tensors)
        self._fused_moe.gate_up_proj = self.gate_up_proj
        self._fused_moe.down_proj = self.down_proj

    def forward(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        *,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._fused_moe.forward(
            x,
            router_logits=router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )


class RouterMLP(BaseOP):
    """Router MLP layer matching zonos2's nn.Sequential structure.

    Checkpoint naming:
    - router_mlp.0.weight, router_mlp.0.bias  (Linear)
    - router_mlp.2.weight, router_mlp.2.bias  (Linear, after GELU at index 1)
    - router_mlp.4.weight  (Linear, after GELU at index 3)

    The GELU activations (indices 1, 3) are stateless and don't have weights.
    """

    def __init__(self, router_dim: int, num_experts: int):
        # Match zonos2's nn.Sequential indices exactly
        # 0: Linear(router_dim, router_dim, bias=True)
        # 1: GELU() - no weights
        # 2: Linear(router_dim, router_dim, bias=True)
        # 3: GELU() - no weights
        # 4: Linear(router_dim, num_experts, bias=False)

        # Using SimpleLinear with bias for layers 0, 2 and without bias for layer 4
        self._0 = _RouterLinear(router_dim, router_dim, has_bias=True)
        self._2 = _RouterLinear(router_dim, router_dim, has_bias=True)
        self._4 = _RouterLinear(router_dim, num_experts, has_bias=False)

    def state_dict(self, *, prefix: str = "", result=None):
        """Custom state_dict to match nn.Sequential numeric naming."""
        from zonos2.layers.base import _concat_prefix

        result = result if result is not None else {}
        self._0.state_dict(prefix=_concat_prefix(prefix, "0"), result=result)
        self._2.state_dict(prefix=_concat_prefix(prefix, "2"), result=result)
        self._4.state_dict(prefix=_concat_prefix(prefix, "4"), result=result)
        return result

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False):
        """Custom load_state_dict to match nn.Sequential numeric naming."""
        from zonos2.layers.base import _concat_prefix

        self._0.load_state_dict(state_dict, prefix=_concat_prefix(prefix, "0"), _internal=True)
        self._2.load_state_dict(state_dict, prefix=_concat_prefix(prefix, "2"), _internal=True)
        self._4.load_state_dict(state_dict, prefix=_concat_prefix(prefix, "4"), _internal=True)

        if not _internal and state_dict:
            keys = list(state_dict.keys())
            raise RuntimeError(
                f"Unexpected keys in state_dict: {len(keys)} keys (first 10: {keys[:10]})"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self._0.forward(x))
        x = F.gelu(self._2.forward(x))
        x = self._4.forward(x)
        return x


class _RouterLinear(BaseOP):
    """Simple linear layer for router MLP."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool = True):
        self.weight = torch.empty(out_features, in_features)
        self.bias = torch.empty(out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class Router(BaseOP):
    """MoE Router matching zonos2's EDA Router structure.

    Checkpoint naming:
    - router.down_proj.weight, router.down_proj.bias
    - router.router_mlp.{0,2,4}.weight, router.router_mlp.{0,2}.bias
    - router.rmsnorm_eda.weight
    - router.router_states_scale (optional, for EDA layers)
    - router.balancing_biases
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        self.hidden_size = config.hidden_size
        self.router_dim = config.moe_router_dim
        self.num_experts = config.moe_n_experts
        self.top_k = config.get_num_experts_per_tok(layer_id)
        self.moe_balancing_strategy = normalize_moe_balancing_strategy(
            config.moe_balancing_strategy
        )
        self.use_legacy_balancing = self.moe_balancing_strategy == "legacy"
        self._layer_id = layer_id

        # down_proj: Linear(hidden_size, router_dim, bias=True)
        self.down_proj = _RouterLinear(self.hidden_size, self.router_dim, has_bias=True)

        # router_mlp: Sequential with GELU activations
        self.router_mlp = RouterMLP(self.router_dim, self.num_experts)

        # RMSNorm for EDA
        self.rmsnorm_eda = RMSNorm(size=self.router_dim, eps=config.rms_norm_eps)

        # EDA: use_eda is True for all layers except the first MoE layer
        self.use_eda = layer_id != config.moe_start_from_layer
        if self.use_eda:
            self.router_states_scale = torch.ones(self.router_dim)

        # Balancing biases
        self.balancing_biases = torch.zeros(self.num_experts, dtype=torch.float32)

    def _select_balanced_topk(
        self, expert_prob: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            bias = self.balancing_biases.detach().float()
            routing_scores = (
                expert_prob.float() + bias
                if self.use_legacy_balancing
                else expert_prob.float() - bias
            )
            _, expert_choice = torch.topk(routing_scores, self.top_k, dim=-1)

        route_prob = torch.gather(expert_prob, dim=-1, index=expert_choice)
        return route_prob.reshape(-1, self.top_k), expert_choice.reshape(-1, self.top_k).to(
            dtype=torch.int32
        )

    def load_state_dict(
        self,
        state_dict,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        from zonos2.layers.base import _concat_prefix

        for name in ("router_states_scale", "balancing_biases"):
            key = _concat_prefix(prefix, name)
            if key in state_dict and hasattr(self, name):
                current = getattr(self, name)
                item = state_dict[key]
                if isinstance(current, torch.Tensor) and current.dtype != item.dtype:
                    setattr(self, name, current.to(dtype=item.dtype))

        super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)

    def forward(self, hidden_states: torch.Tensor, router_states: Optional[torch.Tensor] = None):
        """Compute router logits.

        Args:
            hidden_states: [batch, hidden_size]
            router_states: [batch, router_dim] from previous layer (for EDA)

        Returns:
            route_prob_flat: [tokens, top_k] routing weights from pre-bias softmax
            expert_choice_flat: [tokens, top_k] expert indices after bias-aware top-k
            hidden_states_next: [batch, router_dim] for next layer's EDA
        """
        # Down-project to router dimension
        hidden_states = self.down_proj.forward(hidden_states)

        # EDA: blend with previous router states
        if self.use_eda and router_states is not None:
            hidden_states = hidden_states + router_states * self.router_states_scale

        # Save for next layer's EDA
        hidden_states_next = hidden_states.clone()

        # Normalize
        hidden_states = self.rmsnorm_eda.forward(hidden_states)

        expert_prob = torch.softmax(self.router_mlp.forward(hidden_states).float(), dim=-1)
        route_prob_flat, expert_choice_flat = self._select_balanced_topk(expert_prob)

        return route_prob_flat, expert_choice_flat, hidden_states_next


class MoEFeedForward(BaseOP):
    """MoE feedforward layer.

    Checkpoint naming:
    - layers.{N}.feed_forward.router.*
    - layers.{N}.feed_forward.experts.*
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        # Router with EDA support
        self.router = Router(config, layer_id)

        # Experts with weight fusion
        self.experts = FusedGroupedExperts(config, layer_id)

    @nvtx_annotate("MoEFeedForward")
    def forward(self, x: torch.Tensor, router_states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for MoE layer.

        Args:
            x: [num_tokens, hidden_dim]
            router_states: [num_tokens, router_dim] from previous MoE layer (for EDA)

        Returns:
            output: [num_tokens, hidden_dim]
            router_states_next: [num_tokens, router_dim] for next MoE layer
        """
        num_tokens, hidden_dim = x.shape
        x_input = x.view(-1, hidden_dim)

        # Compute router weights/ids and EDA states for next layer.
        topk_weights, topk_ids, router_states_next = self.router.forward(
            x_input, router_states
        )

        # Expert computation
        output = self.experts.forward(
            x_input,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

        return output.view(num_tokens, hidden_dim), router_states_next


class TransformerBlock(BaseOP):
    """Transformer decoder layer.

    Checkpoint naming:
    - layers.{N}.attention.*
    - layers.{N}.attention_norm.weight
    - layers.{N}.ffn_norm.weight
    - layers.{N}.feed_forward.*
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        self.attention = Attention(config, layer_id)

        # Norms with checkpoint naming
        self.attention_norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.ffn_norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Feed forward: MoE or dense based on layer position
        self.is_moe = self._is_moe_layer(config, layer_id)
        if self.is_moe:
            self.feed_forward = MoEFeedForward(config, layer_id)
        else:
            self.feed_forward = FeedForward(config)

        self._layer_id = layer_id

    def _is_moe_layer(self, config: ModelConfig, layer_id: int) -> bool:
        """Check if this layer should be MoE based on config."""
        if config.moe_n_experts <= 1:
            return False
        if layer_id < config.moe_start_from_layer:
            return False
        if (config.num_layers - layer_id) <= config.moe_end_from_layer:
            return False
        return True

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None, router_states: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch = get_global_ctx().batch
        is_capturing = torch.cuda.is_current_stream_capturing()
        debug_this_layer = (
            _logger.isEnabledFor(logging.DEBUG) and batch.is_decode and not is_capturing and
            self._layer_id == 0 and _model_debug_counter <= 10
        )

        if debug_this_layer:
            _logger.debug("Layer 0 Step %d Input x: norm=%.4f, mean=%.6f", _model_debug_counter, x.norm().item(), x.mean().item())

        x, residual = self.attention_norm.forward(x, residual)

        if debug_this_layer:
            _logger.debug("Layer 0 Step %d After attn_norm: x_norm=%.4f, res_norm=%.4f", _model_debug_counter, x.norm().item(), residual.norm().item())

        x = self.attention.forward(x)

        if debug_this_layer:
            _logger.debug("Layer 0 Step %d After attention: x_norm=%.4f, mean=%.6f", _model_debug_counter, x.norm().item(), x.mean().item())
            x_flat = x.view(-1)[:10]
            _logger.debug("Layer 0 Step %d Attn out[:10]: %s", _model_debug_counter, [f'{v:.4f}' for v in x_flat.tolist()])

        x, residual = self.ffn_norm.forward(x, residual)

        if debug_this_layer:
            _logger.debug("Layer 0 Step %d After ffn_norm: x_norm=%.4f, res_norm=%.4f", _model_debug_counter, x.norm().item(), residual.norm().item())

        # MoE layers return router_states for EDA
        if self.is_moe:
            x, router_states = self.feed_forward.forward(x, router_states)
        else:
            x = self.feed_forward.forward(x)
            router_states = None

        return x, residual, router_states


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """Apply soft capping to logits."""
    return cap * torch.tanh(x / cap)


class Zonos2ForCausalLM(BaseOP):
    """Zonos2 causal LM.

    Checkpoint naming (no "model." prefix):
    - multi_embedder.*
    - layers.{N}.*
    - out_norm.weight
    - multi_output.weight
    """

    def __init__(self, config: ModelConfig, moe_backend: str = "fused_moe"):
        self.config = config
        self.n_codebooks = config.n_codebooks
        self.audio_vocab = config.codebook_size + 2

        # All components directly on this class for correct state_dict keys
        self.multi_embedder = MultiEmbedding(config)

        # Training uses elementwise_affine=False (no learnable params).
        self.emb_norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
            elementwise_affine=False,
        )

        # Optional speaker embedding projections (voice cloning capable checkpoints).
        # The LDA projection reduces raw speaker embeddings before the main
        # speaker projection; its weights are merged into the checkpoint.
        self.speaker_lda_projection = (
            SpeakerLDAProjection(
                input_dim=config.speaker_embedding_dim,
                output_dim=int(config.speaker_lda_dim),
            )
            if config.speaker_enabled and config.speaker_lda_dim
            else None
        )
        speaker_projection_input_dim = (
            int(config.speaker_lda_dim)
            if self.speaker_lda_projection is not None
            else config.speaker_embedding_dim
        )
        self.speaker_projection = (
            LinearReplicated(
                input_size=speaker_projection_input_dim,
                output_size=config.hidden_size,
                has_bias=True,
            )
            if config.speaker_enabled
            else None
        )

        self.layers = OPList(
            [TransformerBlock(config, layer_id) for layer_id in range(config.num_layers)]
        )

        self.out_norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Multi-output head: single linear [audio_vocab * n_codebooks, hidden]
        # Uses MultiOutputHead wrapper for checkpoint compatibility (multi_output.weight)
        self.multi_output = MultiOutputHead(config.hidden_size, self.audio_vocab * self.n_codebooks)

        super().__init__()

    def _forward_model(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run model forward pass to get hidden states."""
        global _model_debug_counter
        batch = get_global_ctx().batch
        is_decode = batch.is_decode

        is_capturing = torch.cuda.is_current_stream_capturing()
        should_debug = is_decode and _logger.isEnabledFor(logging.DEBUG) and not is_capturing

        if should_debug:
            _model_debug_counter += 1
            step = _model_debug_counter
            if step <= 10:
                _logger.debug("MODEL Step %d input_ids shape: %s", step, input_ids.shape)
                _logger.debug("  input_ids[0]: %s", input_ids[0].tolist())

        x = self.multi_embedder.forward(input_ids)

        if should_debug and _model_debug_counter <= 10:
            _logger.debug("  After embedding: norm=%.4f, mean=%.6f", x.norm().item(), x.mean().item())

        # Inject projected speaker embeddings at selected token positions.
        if self.speaker_projection is not None:
            speaker_emb_values = getattr(batch, "speaker_emb_values", None)
            speaker_token_positions = getattr(batch, "speaker_token_positions", None)
            if (
                speaker_emb_values is not None
                and speaker_token_positions is not None
                and speaker_emb_values.numel() > 0
                and speaker_token_positions.numel() > 0
            ):
                if self.speaker_lda_projection is not None:
                    speaker_emb_values = self.speaker_lda_projection.forward(
                        speaker_emb_values.to(dtype=self.speaker_lda_projection.weight.dtype)
                    )
                projected_speaker = self.speaker_projection.forward(
                    speaker_emb_values.to(dtype=self.speaker_projection.weight.dtype)
                )
                x = x.index_copy(0, speaker_token_positions, projected_speaker.to(dtype=x.dtype))

        x, _ = self.emb_norm.forward(x, None)

        if should_debug and _model_debug_counter <= 10:
            _logger.debug("  After emb_norm: norm=%.4f, mean=%.6f", x.norm().item(), x.mean().item())

        residual: torch.Tensor | None = None
        router_states: torch.Tensor | None = None
        for layer_id, layer in enumerate(self.layers.op_list):
            x, residual, router_states = layer.forward(x, residual, router_states)

            if should_debug and _model_debug_counter <= 10:
                if layer_id in [0, 1, 5, 10, 15, len(self.layers.op_list) - 1]:
                    res_norm = residual.norm().item() if residual is not None else 0
                    _logger.debug("  After layer %d: x_norm=%.4f, x_mean=%.6f, res_norm=%.4f", layer_id, x.norm().item(), x.mean().item(), res_norm)

        h = self.out_norm.forward(x, residual)[0]

        if should_debug and _model_debug_counter <= 10:
            _logger.debug("  After out_norm: norm=%.4f, mean=%.6f", h.norm().item(), h.mean().item())

        return h

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        hidden = self._forward_model(batch.input_ids)

        is_capturing = torch.cuda.is_current_stream_capturing()
        if (batch.is_decode and _logger.isEnabledFor(logging.DEBUG)
                and not is_capturing and _model_debug_counter <= 10):
            _logger.debug("  Hidden for logits: norm=%.4f, mean=%.6f", hidden.norm().item(), hidden.mean().item())
            h_flat = hidden.view(-1)[:10]
            _logger.debug("  Hidden[:10]: %s", [f'{v:.4f}' for v in h_flat.tolist()])

        return self.compute_logits(hidden)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute multi-codebook logits from hidden states.

        Args:
            hidden_states: [batch, seq, hidden] or [total_tokens, hidden]

        Returns:
            logits: [..., n_codebooks, audio_vocab]
        """
        # Linear projection: [..., audio_vocab * n_codebooks]
        logits = self.multi_output.forward(hidden_states)
        # Reshape to [..., n_codebooks, audio_vocab]
        *batch_dims, _ = logits.shape
        logits = logits.view(*batch_dims, self.n_codebooks, self.audio_vocab)

        # Apply soft capping if configured
        if self.config.loss_softcap > 0:
            logits = softcap(logits, self.config.loss_softcap)

        return logits


__all__ = ["Zonos2ForCausalLM"]
