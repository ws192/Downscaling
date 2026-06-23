import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================
# 1. Timestep Embeddings
# ======================

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, flip_sin_to_cos: bool = False, downscale_freq_shift: float = 1, scale: float = 1, max_period: int = 10000):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(timesteps, self.num_channels, self.flip_sin_to_cos, self.downscale_freq_shift)
        return t_emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu", cond_proj_dim: Optional[int] = None):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = F.silu if act_fn == "silu" else F.mish if act_fn == "mish" else F.gelu
        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, t_emb, cond=None):
        if cond is not None:
            t_emb = t_emb + self.cond_proj(cond)
        t_emb = self.linear_1(t_emb)
        t_emb = self.act(t_emb)
        t_emb = self.linear_2(t_emb)
        return t_emb


# ======================
# 2. ResNet Blocks
# ======================

class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: Optional[int] = 512,
        groups: int = 32,
        eps: float = 1e-6,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(temb_channels, out_channels)
            elif time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(temb_channels, 2 * out_channels)
            else:
                raise ValueError(f"unknown time_embedding_norm: {time_embedding_norm}")
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = F.silu if non_linearity == "silu" else F.mish if non_linearity == "mish" else F.gelu

        self.use_in_shortcut = self.in_channels != self.out_channels
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input_tensor: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError("`temb` should not be None when `time_embedding_norm` is scale_shift")
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor


class TemporalConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: Optional[int] = None, dropout: float = 0.0, norm_num_groups: int = 32):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv1 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, in_dim),
            nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, hidden_states: torch.Tensor, num_frames: int = 1) -> torch.Tensor:
        hidden_states = hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
        identity = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(hidden_states)
        hidden_states = identity + hidden_states
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape((hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:])
        return hidden_states


# ======================
# 3. Attention
# ======================

class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        norm_num_groups: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        residual_connection: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.scale_qk = scale_qk
        self.scale = dim_head ** -0.5 if self.scale_qk else 1.0
        self.heads = heads
        self.sliceable_head_dim = heads

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, query_dim, bias=out_bias), nn.Dropout(dropout)])

    def _reshape_for_attention(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape tensor from (batch, seq, dim) to (batch, heads, seq, dim_head)"""
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, self.heads, dim // self.heads)
        return tensor.transpose(1, 2)

    def _prepare_mask_for_sdpa(self, attention_mask: Optional[torch.Tensor], batch_size: int, seq_len_q: int, seq_len_kv: int) -> Optional[torch.Tensor]:
        """Convert attention mask to boolean mask for scaled_dot_product_attention.
        True = keep, False = mask out (opposite of original convention)
        """
        if attention_mask is None:
            return None
        # Original mask: 0 = keep, large negative = mask out
        # We need: True = keep, False = mask out
        # The original mask adds to attention scores before softmax
        # For SDPA boolean mask: True means attend, False means ignore
        if attention_mask.dtype == torch.bool:
            return attention_mask
        # Original mask uses additive masking (0 or -inf)
        # Convert to boolean: values >= 0 mean keep
        bool_mask = attention_mask >= 0
        # Ensure shape is (batch, heads, seq_q, seq_kv) or broadcastable
        if bool_mask.dim() == 2:
            # (batch*heads, seq_kv) -> need to reshape
            if bool_mask.shape[0] == batch_size * self.heads:
                bool_mask = bool_mask.reshape(batch_size, self.heads, 1, seq_len_kv)
                bool_mask = bool_mask.expand(batch_size, self.heads, seq_len_q, seq_len_kv)
            elif bool_mask.shape[0] == batch_size:
                bool_mask = bool_mask.unsqueeze(1).unsqueeze(2)
                bool_mask = bool_mask.expand(batch_size, self.heads, seq_len_q, seq_len_kv)
        elif bool_mask.dim() == 3:
            # (batch, 1, seq_kv) or similar
            bool_mask = bool_mask.unsqueeze(1)
            bool_mask = bool_mask.expand(batch_size, self.heads, seq_len_q, seq_len_kv)
        elif bool_mask.dim() == 4:
            # Already has head dim
            pass
        return bool_mask

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, query_seq_len, _ = hidden_states.shape

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # Reshape to (batch, heads, seq, dim_head)
        query = self._reshape_for_attention(query)
        key = self._reshape_for_attention(key)
        value = self._reshape_for_attention(value)

        # Prepare mask for SDPA
        sdpa_mask = self._prepare_mask_for_sdpa(attention_mask, batch_size, query.shape[2], key.shape[2])

        # Use PyTorch 2.0+ scaled_dot_product_attention
        # Automatically selects: FlashAttention > Memory-Efficient > Cudnn > Math
        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=self.scale if self.scale_qk else None,
        )

        # Reshape back: (batch, heads, seq, dim_head) -> (batch, seq, heads * dim_head)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, query_seq_len, self.inner_dim)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, activation_fn: str = "geglu", final_dropout: bool = False, inner_dim: Optional[int] = None, bias: bool = True):
        super().__init__()
        inner_dim = inner_dim or dim * 4
        if activation_fn == "geglu":
            self.net = nn.Sequential(
                GEGLU(dim, inner_dim, bias=bias),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim, bias=bias),
                nn.Dropout(final_dropout) if final_dropout else nn.Identity(),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, inner_dim, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim, bias=bias),
                nn.Dropout(final_dropout) if final_dropout else nn.Identity(),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_layer_norm = norm_type == "layer_norm"

        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_zero":
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        if cross_attention_dim is not None or double_self_attention:
            if norm_type == "ada_norm":
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            else:
                self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.norm2 = None
            self.attn2 = None

        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, encoder_hidden_states: Optional[torch.Tensor] = None, timestep: Optional[torch.Tensor] = None, cross_attention_kwargs: Optional[Dict[str, Any]] = None, class_labels: Optional[torch.Tensor] = None):
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **(cross_attention_kwargs or {}),
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.attn2 is not None:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm2(hidden_states, timestep)
            else:
                norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **(cross_attention_kwargs or {}),
            )
            hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm3(hidden_states)
        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output
        return hidden_states


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim * 2)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim * 2, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = emb.chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class AdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim * 6)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim * 6, embedding_dim * 6)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, class_labels: torch.Tensor, hidden_dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(self.emb(timestep)))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# ======================
# 4. Transformer Models
# ======================

class Transformer2DModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, self.inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, self.inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                self.inner_dim,
                num_attention_heads,
                attention_head_dim,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,
                activation_fn=activation_fn,
                num_embeds_ada_norm=num_embeds_ada_norm,
                attention_bias=attention_bias,
                only_cross_attention=only_cross_attention,
                double_self_attention=double_self_attention,
                upcast_attention=upcast_attention,
                norm_type=norm_type,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ])

        if use_linear_projection:
            self.proj_out = nn.Linear(self.inner_dim, self.out_channels)
        else:
            self.proj_out = nn.Conv2d(self.inner_dim, self.out_channels, kernel_size=1, stride=1, padding=0)

    def _operate_on_continuous_inputs(self, hidden_states):
        batch, _, height, width = hidden_states.shape
        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            hidden_states = self.proj_in(hidden_states)
        return hidden_states, inner_dim

    def _get_output_for_continuous_inputs(self, hidden_states, residual, batch_size, height, width, inner_dim):
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        output = hidden_states + residual
        return output

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None, cross_attention_kwargs: Optional[Dict[str, Any]] = None, return_dict: bool = True):
        batch_size, _, height, width = hidden_states.shape
        residual = hidden_states
        hidden_states, inner_dim = self._operate_on_continuous_inputs(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                timestep=None,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        output = self._get_output_for_continuous_inputs(hidden_states, residual, batch_size, height, width, inner_dim)
        if not return_dict:
            return (output,)
        return {"sample": output}


class TransformerTemporalModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = True,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                double_self_attention=double_self_attention,
                norm_elementwise_affine=norm_elementwise_affine,
                positional_embeddings=positional_embeddings,
                num_positional_embeddings=num_positional_embeddings,
            )
            for _ in range(num_layers)
        ])
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None, num_frames: int = 1, cross_attention_kwargs: Optional[Dict[str, Any]] = None, return_dict: bool = True):
        batch_frames, channel, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames
        residual = hidden_states
        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)
        hidden_states = self.proj_in(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=None,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states[None, None, :].reshape(batch_size, height, width, num_frames, channel).permute(0, 3, 4, 1, 2).contiguous()
        hidden_states = hidden_states.reshape(batch_frames, channel, height, width)
        output = hidden_states + residual
        if not return_dict:
            return (output,)
        return {"sample": output}


# ======================
# 5. Downsampling / Upsampling
# ======================

class Downsample2D(nn.Module):
    def __init__(self, channels: int, use_conv: bool = False, out_channels: Optional[int] = None, padding: int = 1, name: str = "conv", kernel_size=3, norm_type=None, eps=None, elementwise_affine=None, bias=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = None
        else:
            self.norm = None
        if use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)
        if name == "conv":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels
        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Upsample2D(nn.Module):
    def __init__(self, channels: int, use_conv: bool = False, use_conv_transpose: bool = False, out_channels: Optional[int] = None, name: str = "conv", kernel_size: Optional[int] = None, padding=1, norm_type=None, eps=None, elementwise_affine=None, bias=True, interpolate=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate
        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = None
        else:
            self.norm = None
        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose2d(channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias)
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        if name == "conv":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels
        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.use_conv_transpose:
            return self.conv(hidden_states)
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
        if self.interpolate:
            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")
        if self.use_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states


# ======================
# 6. UNet 3D Blocks
# ======================

class UNetMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1280,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = True,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        resnets = [ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm)]
        temp_convs = [TemporalConvLayer(in_channels, in_channels, dropout=0.1, norm_num_groups=resnet_groups)]
        attentions = []
        temp_attentions = []
        for _ in range(num_layers):
            attentions.append(Transformer2DModel(in_channels // num_attention_heads, num_attention_heads, in_channels=in_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups, use_linear_projection=use_linear_projection, upcast_attention=upcast_attention))
            temp_attentions.append(TransformerTemporalModel(in_channels // num_attention_heads, num_attention_heads, in_channels=in_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups))
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            temp_convs.append(TemporalConvLayer(in_channels, in_channels, dropout=0.1, norm_num_groups=resnet_groups))
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, encoder_hidden_states: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, num_frames: int = 1, cross_attention_kwargs: Optional[Dict[str, Any]] = None):
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.temp_convs[0](hidden_states, num_frames=num_frames)
        for attn, temp_attn, resnet, temp_conv in zip(self.attentions, self.temp_attentions, self.resnets[1:], self.temp_convs[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]
            hidden_states = temp_attn(hidden_states, num_frames=num_frames, cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)
        return hidden_states


class CrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        temp_attentions = []
        temp_convs = []
        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            temp_convs.append(TemporalConvLayer(out_channels, out_channels, dropout=0.1, norm_num_groups=resnet_groups))
            attentions.append(Transformer2DModel(out_channels // num_attention_heads, num_attention_heads, in_channels=out_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention, upcast_attention=upcast_attention))
            temp_attentions.append(TransformerTemporalModel(out_channels // num_attention_heads, num_attention_heads, in_channels=out_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups))
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op")])
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, encoder_hidden_states: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, num_frames: int = 1, cross_attention_kwargs: Optional[Dict[str, Any]] = None):
        output_states = ()
        for resnet, temp_conv, attn, temp_attn in zip(self.resnets, self.temp_convs, self.attentions, self.temp_attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]
            hidden_states = temp_attn(hidden_states, num_frames=num_frames, cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            temp_convs.append(TemporalConvLayer(out_channels, out_channels, dropout=0.1, norm_num_groups=resnet_groups))
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op")])
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, num_frames: int = 1):
        output_states = ()
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


class UNetMidBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        for i in range(num_layers):
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            temp_convs.append(TemporalConvLayer(in_channels, in_channels, dropout=0.1, norm_num_groups=resnet_groups))
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, num_frames: int = 1):
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.temp_convs[0](hidden_states, num_frames=num_frames)
        for resnet, temp_conv in zip(self.resnets[1:], self.temp_convs[1:]):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)
        return hidden_states


class CrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        resolution_idx: Optional[int] = None,
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        attentions = []
        temp_attentions = []
        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            temp_convs.append(TemporalConvLayer(out_channels, out_channels, dropout=0.1, norm_num_groups=resnet_groups))
            attentions.append(Transformer2DModel(out_channels // num_attention_heads, num_attention_heads, in_channels=out_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention, upcast_attention=upcast_attention))
            temp_attentions.append(TransformerTemporalModel(out_channels // num_attention_heads, num_attention_heads, in_channels=out_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups))
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(self, hidden_states: torch.Tensor, res_hidden_states_tuple: Tuple[torch.Tensor, ...], temb: Optional[torch.Tensor] = None, encoder_hidden_states: Optional[torch.Tensor] = None, upsample_size: Optional[int] = None, attention_mask: Optional[torch.Tensor] = None, num_frames: int = 1, cross_attention_kwargs: Optional[Dict[str, Any]] = None):
        for resnet, temp_conv, attn, temp_attn in zip(self.resnets, self.temp_convs, self.attentions, self.temp_attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]
            hidden_states = temp_attn(hidden_states, num_frames=num_frames, cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        resolution_idx: Optional[int] = None,
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            temp_convs.append(TemporalConvLayer(out_channels, out_channels, dropout=0.1, norm_num_groups=resnet_groups))
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(self, hidden_states: torch.Tensor, res_hidden_states_tuple: Tuple[torch.Tensor, ...], temb: Optional[torch.Tensor] = None, upsample_size: Optional[int] = None, num_frames: int = 1):
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


def get_down_block(down_block_type, num_layers, in_channels, out_channels, temb_channels, add_downsample, resnet_eps, resnet_act_fn, num_attention_heads, resnet_groups=None, cross_attention_dim=None, downsample_padding=None, dual_cross_attention=False, use_linear_projection=True, only_cross_attention=False, upcast_attention=False, resnet_time_scale_shift="default", dropout=0.0):
    if down_block_type == "DownBlock3D":
        return DownBlock3D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, downsample_padding=downsample_padding, resnet_time_scale_shift=resnet_time_scale_shift, dropout=dropout)
    elif down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock3D")
        return CrossAttnDownBlock3D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, downsample_padding=downsample_padding, cross_attention_dim=cross_attention_dim, num_attention_heads=num_attention_heads, dual_cross_attention=dual_cross_attention, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention, upcast_attention=upcast_attention, resnet_time_scale_shift=resnet_time_scale_shift, dropout=dropout)
    else:
        raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(up_block_type, num_layers, in_channels, out_channels, prev_output_channel, temb_channels, add_upsample, resnet_eps, resnet_act_fn, num_attention_heads, resolution_idx=None, resnet_groups=None, cross_attention_dim=None, dual_cross_attention=False, use_linear_projection=True, only_cross_attention=False, upcast_attention=False, resnet_time_scale_shift="default", dropout=0.0):
    if up_block_type == "UpBlock3D":
        return UpBlock3D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, prev_output_channel=prev_output_channel, temb_channels=temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, resnet_time_scale_shift=resnet_time_scale_shift, resolution_idx=resolution_idx, dropout=dropout)
    elif up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
        return CrossAttnUpBlock3D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, prev_output_channel=prev_output_channel, temb_channels=temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, cross_attention_dim=cross_attention_dim, num_attention_heads=num_attention_heads, dual_cross_attention=dual_cross_attention, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention, upcast_attention=upcast_attention, resnet_time_scale_shift=resnet_time_scale_shift, resolution_idx=resolution_idx, dropout=dropout)
    else:
        raise ValueError(f"{up_block_type} does not exist.")


# ======================
# 7. UNet3DConditionModel
# ======================

class UNet3DConditionModel(nn.Module):
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str, ...] = ("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
        up_block_types: Tuple[str, ...] = ("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1024,
        attention_head_dim: Union[int, Tuple[int]] = 64,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        time_cond_proj_dim: Optional[int] = None,
    ):
        super().__init__()
        self.sample_size = sample_size
        num_attention_heads = num_attention_heads or attention_head_dim
        if len(down_block_types) != len(up_block_types):
            raise ValueError("Must provide the same number of `down_block_types` as `up_block_types`.")
        if len(block_out_channels) != len(down_block_types):
            raise ValueError("Must provide the same number of `block_out_channels` as `down_block_types`.")
        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError("Must provide the same number of `num_attention_heads` as `down_block_types`.")
        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        conv_in_kernel = 3
        conv_out_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding)

        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], True, 0)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn, cond_proj_dim=time_cond_proj_dim)

        self.transformer_in = TransformerTemporalModel(
            num_attention_heads=8,
            attention_head_dim=attention_head_dim,
            in_channels=block_out_channels[0],
            num_layers=1,
            norm_num_groups=norm_num_groups,
        )

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=False,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_groups=norm_num_groups,
        )

        self.num_upsamplers = 0
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False
            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=False,
                resolution_idx=i,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
            self.conv_act = F.silu
        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding)

    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], encoder_hidden_states: torch.Tensor, class_labels: Optional[torch.Tensor] = None, timestep_cond: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, cross_attention_kwargs: Optional[Dict[str, Any]] = None, down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None, mid_block_additional_residual: Optional[torch.Tensor] = None, return_dict: bool = True):
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None
        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        num_frames = sample.shape[2]
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        emb = emb.repeat_interleave(num_frames, dim=0, output_size=emb.shape[0] * num_frames)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0, output_size=encoder_hidden_states.shape[0] * num_frames)

        sample = sample.permute(0, 2, 1, 3, 4).reshape((sample.shape[0] * num_frames, -1) + sample.shape[3:])
        sample = self.conv_in(sample)
        sample = self.transformer_in(sample, num_frames=num_frames, cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, num_frames=num_frames, cross_attention_kwargs=cross_attention_kwargs)
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, num_frames=num_frames)
            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(down_block_res_samples, down_block_additional_residuals):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples

        if self.mid_block is not None:
            sample = self.mid_block(sample, emb, num_frames=num_frames)

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, encoder_hidden_states=encoder_hidden_states, upsample_size=upsample_size, attention_mask=attention_mask, num_frames=num_frames, cross_attention_kwargs=cross_attention_kwargs)
            else:
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size, num_frames=num_frames)

        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        sample = sample[None, :].reshape((-1, num_frames) + sample.shape[1:]).permute(0, 2, 1, 3, 4)

        if not return_dict:
            return (sample,)
        return {"sample": sample}


# ======================
# 8. Test
# ======================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet3DConditionModel(
        sample_size=192,
        in_channels=1,
        out_channels=1,
        down_block_types=("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        cross_attention_dim=768,
        attention_head_dim=40,
        norm_num_groups=32,
    ).to(device).eval()

    batch_size = 1
    num_frames = 10
    height = 192
    width = 192
    in_channels = 1
    encoder_hidden_states_seq_len = 77
    cross_attention_dim = 768

    sample = torch.randn(batch_size, in_channels, num_frames, height, width, device=device)
    timestep = torch.tensor([0], device=device)
    encoder_hidden_states = torch.randn(batch_size, encoder_hidden_states_seq_len, cross_attention_dim, device=device)

    print("Input shapes:")
    print(f"  sample: {sample.shape}")
    print(f"  timestep: {timestep.shape}")
    print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")

    with torch.no_grad():
        output = model(sample, timestep, encoder_hidden_states, return_dict=True)

    print("\nOutput shape:", output["sample"].shape)
    print("Test passed!")
