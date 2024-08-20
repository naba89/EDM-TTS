"""
Modified from:
https://github.com/lucidrains/soundstorm-pytorch/blob/main/soundstorm_pytorch/soundstorm.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, EinMix

from edm_tts.models.conformer.attend import Attend


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return pad, pad - (kernel_size + 1) % 2


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class ChanLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-6 if x.dtype == torch.float32 else 1e-4
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * var.clamp(min=eps).rsqrt() * self.weight


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, flash=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = Attend(flash=flash, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context=None, mask=None, rotary_emb_x=None, rotary_emb_context=None, attn_bias=None, return_attn=False):
        n, device, h, has_context = x.shape[-2], x.device, self.heads, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if exists(rotary_emb_x):
            q = apply_rotary_pos_emb(rotary_emb_x, q)
            if not has_context:
                k = apply_rotary_pos_emb(rotary_emb_x, k)

        if exists(rotary_emb_context) and has_context:
            k = apply_rotary_pos_emb(rotary_emb_context, k)

        out, attn = self.attend(q, k, v, mask=mask, attn_bias=attn_bias, return_attn=return_attn)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), attn


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), Swish(), nn.Dropout(dropout), nn.Linear(dim * mult, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n c -> b c n"),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
            Swish(),
            ChanLayerNorm(inner_dim),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        attn_flash=True,
        ff_dropout=0.0,
        conv_dropout=0.0,
        conv_causal=False,
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=attn_flash)
        self.conv = ConformerConvModule(
            dim=dim,
            causal=conv_causal,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)
        self.gradient_checkpointing = False

    def forward(self, x, mask=None, rotary_emb_x=None, attn_bias=None, return_attn=False):
        if self.gradient_checkpointing and self.training:
            x = self._gradient_checkpointing_func(self.ff1.__call__, x) + x
        else:
            x = self.ff1(x) + x

        attn_out, attn = self.attn(x, mask=mask, rotary_emb_x=rotary_emb_x, attn_bias=attn_bias, return_attn=return_attn)
        x = attn_out + x
        x = self.conv(x) + x

        if self.gradient_checkpointing and self.training:
            x = self._gradient_checkpointing_func(self.ff2.__call__, x) + x
        else:
            x = self.ff2(x) + x

        x = self.post_norm(x)
        return x, attn


class Conformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0,
        conv_causal=False,
        attn_flash=True,
    ):
        super().__init__()

        self.dim = dim
        self.layers = nn.ModuleList([])

        self.rotary_emb = RotaryEmbedding(dim_head)

        for _ in range(depth):
            self.layers.append(
                ConformerBlock(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    ff_mult=ff_mult,
                    conv_expansion_factor=conv_expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    conv_dropout=conv_dropout,
                    conv_causal=conv_causal,
                    attn_flash=attn_flash,
                )
            )

    def forward(self, x, mask=None, output_layer_idx=None, return_attn=False):
        seq_len = x.shape[-2]

        rotary_emb_x = self.rotary_emb(seq_len) if exists(self.rotary_emb) else None
        attns = [] if return_attn else None
        for block_idx, block in enumerate(self.layers):
            x, attn = block(
                x,
                mask=mask,
                rotary_emb_x=rotary_emb_x,
                return_attn=return_attn
            )
            if return_attn:
                attns.append(attn)
            if output_layer_idx is not None and block_idx == output_layer_idx:
                return x, attns

        return x, attns
