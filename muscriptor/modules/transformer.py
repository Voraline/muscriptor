import torch
import torch.nn as nn
from torch.nn import functional as F

from muscriptor.modules.streaming import ModelState, State, StatefulModule


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000,
    dtype: torch.dtype = torch.float32,
    inv_freq: torch.Tensor | None = None,
) -> torch.Tensor:
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    if inv_freq is None:
        adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
        max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)
        phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    else:
        phase = positions * inv_freq.to(device=positions.device, dtype=dtype)
    out = torch.empty((*phase.shape[:-1], dim), device=phase.device, dtype=dtype)
    torch.cos(phase, out=out[..., :half_dim])
    torch.sin(phase, out=out[..., half_dim:])
    return out


class StreamingMultiheadAttention(StatefulModule):
    """Causal multi-head self-attention with a preallocated KV cache."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // num_heads

        in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False, **factory_kwargs)
        self.in_proj_weight = in_proj.weight
        self.in_proj_bias = in_proj.bias
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **factory_kwargs)

    def init_state(self, batch_size: int, sequence_length: int) -> State:
        weight = self.in_proj_weight
        # Allocate up to 2048 tokens initially, or full sequence_length if <= 2048.
        # This avoids repeated dynamic reallocations for standard chunks while
        # keeping initial VRAM usage modest.
        initial_capacity = min(2048, sequence_length)
        return {
            # Layout is [2, batch, heads, time, head_dim] so the per-head
            # [time, head_dim] slice SDPA reads is contiguous (no per-layer,
            # per-step transpose of a strided cache view).
            "cache": torch.empty(
                (2, batch_size, self.num_heads, initial_capacity, self.dim_per_head),
                device=weight.device,
                dtype=weight.dtype,
            ),
            # Kept as a plain host int: it is advanced deterministically by the
            # host-side generate loop, and reading it from a device tensor
            # (`.item()`) would force a GPU sync per layer per decode step.
            "offset": 0,
            "max_length": sequence_length,
        }

    def increment_step(self, state: State, increment: int = 1) -> None:
        state["offset"] = state["offset"] + increment

    def _complete_kv(self, k, v, state: State | None):
        """Append k/v ([B, T, H, D]) to the cache; return full k/v as [B, H, T, D]."""
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if state is None:
            return k, v
        cache = state["cache"]
        end = state["offset"]
        T = k.shape[2]

        if end + T > cache.shape[3]:
            # Grow directly to max_length (always >= end + T) to eliminate repeated
            # reallocations, memory copies, and fragmentation across layers.
            max_len = state.get("max_length", end + T)
            new_capacity = max(end + T, max_len)
            new_cache = torch.empty(
                (2, cache.shape[1], self.num_heads, new_capacity, self.dim_per_head),
                device=cache.device,
                dtype=cache.dtype,
            )
            new_cache[:, :, :, :end].copy_(cache[:, :, :, :end])
            state["cache"] = new_cache
            cache = new_cache

        cache[0, :, :, end : end + T] = k
        cache[1, :, :, end : end + T] = v
        return cache[0, :, :, : end + T], cache[1, :, :, : end + T]

    def forward(
        self,
        query: torch.Tensor,
        model_state: ModelState | None = None,
    ):
        B, T, _ = query.shape
        state = self.get_state(model_state)
        projected = nn.functional.linear(query, self.in_proj_weight)
        packed = projected.view(B, T, 3, self.num_heads, self.dim_per_head)
        q, k, v = packed.unbind(dim=2)

        k, v = self._complete_kv(k, v, state)
        dtype = q.dtype

        q_t = q.transpose(1, 2)
        k_t = k  # already [B, H, T, D] from _complete_kv
        v_t = v

        # Causality must be bottom-right aligned so streaming decode steps
        # (T_q=1, T_k=cache_len) attend to all past tokens; PyTorch's
        # is_causal=True is top-left aligned and would mask out all cached
        # tokens except position 0 when T_q < T_k. An explicit attn_mask
        # forces SDPA onto the unfused math fallback, so only build one in
        # the rectangular case that actually needs it — the two shapes this
        # model hits (single-token decode and square prefill) stay mask-free
        # and dispatch to the fused (flash) CPU/CUDA kernels.
        T_q, T_k = q_t.shape[2], k_t.shape[2]
        if T_q == 1:
            # One query row, bottom-right aligned: nothing is masked.
            x = F.scaled_dot_product_attention(q_t, k_t, v_t, dropout_p=0.0)
            # x has shape [B, H, 1, D]. Squeezing dim 2 yields [B, H, D] with strides (H*D, D, 1)
            # which is already contiguous, enabling a zero-copy pointer view without memory allocation.
            x = x.squeeze(2).view(B, 1, self.embed_dim)
        elif T_q == T_k:
            # Square: bottom-right and top-left alignment coincide.
            x = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, dropout_p=0.0
            )
            x = x.transpose(1, 2).to(dtype).reshape(B, T, self.embed_dim)
        else:
            # Unused in practice
            raise NotImplementedError(
                f"Streaming attention with T_q={T_q} and T_k={T_k} is not supported; use T_q=1 or T_q=T_k."
            )
        x = self.out_proj(x)
        return x


class StreamingTransformerLayer(nn.Module):
    """Pre-norm transformer block: self-attention + GELU FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = StreamingMultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, **factory_kwargs
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        model_state: ModelState | None = None,
    ):
        x = x.add_(self.self_attn(self.norm1(x), model_state=model_state))
        x = x.add_(self.linear2(F.gelu(self.linear1(self.norm2(x)))))
        return x


class StreamingTransformer(StatefulModule):
    """Stack of causal streaming transformer layers with sinusoidal positions."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        max_period: float = 10_000,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.max_period = max_period
        half_dim = d_model // 2
        adim = torch.arange(half_dim, dtype=torch.float32)
        inv_freq = 1.0 / (max_period ** (adim / (half_dim - 1)))
        self.register_buffer("inv_freq", inv_freq.view(1, 1, -1), persistent=False)

        # Precompute static sinusoidal position embeddings up to max_period to eliminate
        # dynamic trigonometry kernel launches and tensor allocations during token decoding.
        pos_table = create_sin_embedding(
            torch.arange(int(max_period), dtype=torch.float32).unsqueeze(-1),
            d_model,
            max_period=max_period,
            dtype=torch.float32,
            inv_freq=inv_freq.view(1, 1, -1),
        ).squeeze(1)
        self.register_buffer("pos_table", pos_table, persistent=False)

        self.layers = nn.ModuleList(
            [
                StreamingTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def init_state(self, batch_size: int, sequence_length: int) -> State:
        device = self.layers[0].norm2.weight.device
        return {
            "offsets": torch.zeros(batch_size, dtype=torch.long, device=device),
        }

    def increment_step(self, state: State, increment: int = 1) -> None:
        state["offsets"] = state["offsets"] + increment

    def forward(
        self,
        x: torch.Tensor,
        prepend_length: int = 0,
        model_state: ModelState | None = None,
    ):
        del prepend_length  # unused; positions come from state['offsets']
        B, T, C = x.shape
        state = self.get_state(model_state)
        offsets = (
            state["offsets"]
            if state is not None
            else torch.zeros(B, dtype=torch.long, device=x.device)
        )

        if T == 1:
            if (offsets >= 0).all() and (offsets < self.pos_table.shape[0]).all():
                pos_emb = self.pos_table[offsets].unsqueeze(1)
            else:
                positions = offsets.view(-1, 1, 1)
                pos_emb = create_sin_embedding(
                    positions, C, max_period=self.max_period, dtype=torch.float32, inv_freq=self.inv_freq
                )
            x = x.add_(pos_emb.to(x.dtype))
        else:
            positions = torch.arange(T, device=x.device).view(1, -1, 1) + offsets.view(-1, 1, 1)
            # Always compute the sinusoidal embedding in fp32: fp16 cannot even
            # represent odd integers above 2048, so half-precision positions would
            # collapse neighbouring timesteps to the same embedding.
            pos_emb = create_sin_embedding(
                positions, C, max_period=self.max_period, dtype=torch.float32, inv_freq=self.inv_freq
            )
            x = x + (pos_emb * (positions >= 0).float()).to(x.dtype)

        for layer in self.layers:
            x = layer(x, model_state=model_state)
        return x