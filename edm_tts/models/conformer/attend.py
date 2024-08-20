"""
Modified from:
https://github.com/lucidrains/soundstorm-pytorch/blob/main/soundstorm_pytorch/attend.py
"""

from collections import namedtuple

import torch
import torch.nn.functional as F
from einops import rearrange
from packaging import version
from torch import einsum, nn

# constants

EfficientAttentionConfig = namedtuple(
    "EfficientAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)


def get_sdpa_backend_list(config):
    from torch.nn.attention import SDPBackend
    backend_list = []
    if config.enable_flash:
        backend_list.append(SDPBackend.FLASH_ATTENTION)
    if config.enable_mem_efficient:
        backend_list.append(SDPBackend.EFFICIENT_ATTENTION)
    if config.enable_math:
        backend_list.append(SDPBackend.MATH)
    return backend_list


class Attend(nn.Module):
    def __init__(self, causal=False, dropout=0.0, flash=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.flash = flash
        assert not (
            flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = EfficientAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major >= 8 and self.flash:
            self.cuda_config = EfficientAttentionConfig(True, False, True)
        else:
            self.cuda_config = EfficientAttentionConfig(False, False, True)

    def get_mask(self, i: int, j: int, device: torch.device) -> torch.Tensor:
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    def flash_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # single headed key / values

        if k.ndim == 3:
            k = rearrange(k, "b n d -> b 1 n d")

        if v.ndim == 3:
            v = rearrange(v, "b n d -> b 1 n d")

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if mask is not None and mask.ndim != 4:
            mask = rearrange(mask, "b j -> b 1 1 j")
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        causal = self.causal

        # handle attention bias

        if attn_bias is not None:
            mask_value = -torch.finfo(q.dtype).max // 2
            causal_mask = self.get_mask(q_len, k_len, device)
            attn_bias = attn_bias.masked_fill(causal_mask, mask_value)

            if mask is not None:
                attn_bias = attn_bias.masked_fill(~mask, mask_value)

            mask = attn_bias
            causal = False

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        # depending on pytorch version, the function signature changes
        with torch.nn.attention.sdpa_kernel(get_sdpa_backend_list(config)):
        # with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=causal
            )

        return out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        return_attn: bool = False,
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        if self.flash and not return_attn:
            assert not attn_bias is not None
            return self.flash_attn(q, k, v, mask=mask), None

        # similarity

        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # attention bias

        if attn_bias is not None:
            sim = sim + attn_bias

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(q_len, k_len, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # key padding mask

        if mask is not None:
            if mask.ndim != 4:
                mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        if return_attn:
            return out, attn

        return out, None


# pyright: reportOptionalMemberAccess=false
