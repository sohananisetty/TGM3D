import math
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from functools import wraps, partial

import torch
import torch.nn.functional as F
from core.models.utils import (
    LayerNorm,
    create_causal_mask,
    default,
    dropout_seq,
    exists,
    l2norm,
    print_once,
)
from einops import rearrange, repeat

from packaging import version
from torch import einsum, nn


@dataclass
class AttentionParams:
    dim: int = 768
    heads: int = 8
    causal: bool = False
    qk_norm: bool = False
    qk_norm_scale: int = 8
    dropout: float = 0.0
    cross_attn_tokens_dropout: float = 0.0
    add_null_kv: bool = False


# main class


class Attend(nn.Module):
    def __init__(
        self,
        dropout=0.0,
        causal=False,
        scale=None,
        qk_norm=False,
    ):
        super().__init__()
        self.scale = scale
        self.qk_norm = qk_norm

        self.causal = causal
        self.create_causal_mask = create_causal_mask

        self.attn_fn = (
            partial(F.softmax, dtype=torch.float32) if not qk_norm else F.softmax
        )

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, attn_bias=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, heads, kv_heads, device = q.shape[-2], q.shape[1], k.shape[1], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        # handle grouped multi-query attention

        if kv_heads == 1:
            k, v = map(lambda t: rearrange(t, "b 1 n d -> b n d"), (k, v))
        elif kv_heads < heads:
            k, v = map(
                lambda t: repeat(t, "b kvh n d -> b (r kvh) n d", r=heads // kv_heads),
                (k, v),
            )

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        dots = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        qk_similarities = dots.clone()

        if exists(attn_bias):
            dots = dots + attn_bias

        i, j, dtype = *dots.shape[-2:], dots.dtype

        mask_value = -torch.finfo(dots.dtype).max

        if exists(mask):
            dots = dots.masked_fill(~mask, mask_value)

        if self.causal:
            causal_mask = self.create_causal_mask(i, j, device=device)
            dots = dots.masked_fill(causal_mask, mask_value)

        pre_softmax_attn = dots.clone()

        attn = self.attn_fn(dots, dim=-1)
        attn = attn.type(dtype)

        post_softmax_attn = attn.clone()

        attn = self.attn_dropout(attn)

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out


class Attention(nn.Module):
    def __init__(
        self,
        params: AttentionParams,
    ):
        super().__init__()
        self.dim = params.dim
        self.heads = params.heads
        self.dim_head = self.dim // self.heads
        self.scale = self.dim_head**-0.5

        inner_dim = self.dim_head * self.heads
        self.causal = params.causal
        self.norm = LayerNorm(self.dim)
        self.qk_norm = params.qk_norm
        self.qk_norm_scale = params.qk_norm_scale
        self.add_null_kv = params.add_null_kv

        self.cross_attn_tokens_dropout = params.cross_attn_tokens_dropout

        self.attend = Attend(
            dropout=params.dropout,
            scale=self.qk_norm_scale if self.qk_norm else self.scale,
            causal=self.causal,
            qk_norm=self.qk_norm,
        )

        self.null_kv = nn.Parameter(torch.randn(2, self.heads, 1, self.dim_head))

        self.to_q = nn.Linear(self.dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(self.dim, inner_dim * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(self.dim_head))
        self.k_scale = nn.Parameter(torch.ones(self.dim_head))

        self.to_out = nn.Linear(inner_dim, self.dim, bias=False)

    def forward(self, x, mask=None, context=None, context_mask=None, rel_pos=None):
        n = x.shape[-2]
        h, has_context = self.heads, exists(context)

        if self.training and self.cross_attn_tokens_dropout > 0.0:
            context, context_mask = dropout_seq(
                context, context_mask, self.cross_attn_tokens_dropout
            )

        input_mask = context_mask if has_context else mask

        if exists(input_mask):
            input_mask = rearrange(input_mask, "b j -> b 1 1 j")

        x = self.norm(x)

        kv_input = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if self.add_null_kv:
            nk, nv = self.null_kv
            nk, nv = map(
                lambda t: repeat(t, "h 1 d -> b h 1 d", b=x.shape[0]), (nk, nv)
            )

            k = torch.cat((nk, k), dim=-2)
            v = torch.cat((nv, v), dim=-2)

            input_mask = repeat(input_mask, "b 1 1 j -> b h i j", h=h, i=n)

            input_mask = F.pad(input_mask, (1, 0), value=True)

        if self.qk_norm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        i, j = map(lambda t: t.shape[-2], (q, k))
        attn_bias = None
        if exists(rel_pos):
            attn_bias = rel_pos(i, j)

        out = self.attend(q, k, v, mask=input_mask, attn_bias=attn_bias)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
