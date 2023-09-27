import math
import pathlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from random import random
from typing import Callable, List, Optional, Union

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from core.models.conv_vqvae import ConvVQMotionModel
from core.models.muse.attend import Attend, AttentionParams, Attention
from core.models.muse.positional_embeddings import (
    PositionalEmbeddingParams,
    PositionalEmbeddingType,
)

from core.models.utils import (
    FeedForward,
    LayerNorm,
    default,
    exists,
)
from einops import rearrange, repeat
from torch import einsum, nn
from tqdm.auto import tqdm


class TransformerBlocks(nn.Module):
    def __init__(self, attention_params: AttentionParams, depth: int, ff_mult: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(attention_params),
                        Attention(attention_params),
                        FeedForward(dim=attention_params.dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = LayerNorm(attention_params.dim)

    def forward(self, x, mask=None, context=None, context_mask=None, rel_pos=None):
        for attn, cross_attn, ff in self.layers:
            x = attn(x, mask=mask, rel_pos=rel_pos) + x

            x = cross_attn(x, mask=mask, context=context, context_mask=context_mask) + x

            x = ff(x) + x

        return self.norm(x)


@dataclass
class BERTParams:
    attention_params: AttentionParams = AttentionParams()
    positional_embedding_params: PositionalEmbeddingParams = PositionalEmbeddingParams()
    positional_embedding: PositionalEmbeddingType = PositionalEmbeddingType.SINE
    num_tokens: int = 1024
    dim_out: int = 768
    depth: int = 12
    ff_mult: int = 4
    emb_dropout = 0.1
    post_emb_norm = False


class BERT(nn.Module):
    def __init__(self, transformer_params: BERTParams):
        super().__init__()
        self.dim = transformer_params.attention_params.dim
        self.num_tokens = transformer_params.num_tokens
        self.seq_len = transformer_params.positional_embedding_params.max_seq_len

        self.mask_id = self.num_tokens
        self.bos_index = self.num_tokens + 1
        self.eos_index = self.num_tokens + 2
        self.token_emb = nn.Embedding(
            self.num_tokens + 3, self.dim, padding_idx=self.mask_id
        )

        self.segment_emb = nn.Embedding(3, self.dim, padding_idx=0)

        self.is_abs_pos_emb = transformer_params.positional_embedding.name in [
            "ABS",
            "SINE",
        ]

        self.pos_emb = transformer_params.positional_embedding.value(
            transformer_params.positional_embedding_params
        )

        self.emb_dropout = nn.Dropout(transformer_params.emb_dropout)

        self.transformer_blocks = TransformerBlocks(
            attention_params=transformer_params.attention_params,
            depth=transformer_params.depth,
            ff_mult=transformer_params.ff_mult,
        )
        self.norm = LayerNorm(self.dim)

        self.dim_out = default(transformer_params.dim_out, self.num_tokens)
        self.to_logits = nn.Linear(self.dim, self.dim_out, bias=False)

        self.post_emb_norm = (
            nn.LayerNorm(self.dim)
            if transformer_params.post_emb_norm
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        segment_label: torch.Tensor = None,
        mask: torch.Tensor = None,
        return_embed: bool = False,
        return_logits: bool = False,
        labels=None,
        ignore_index: int = -1,
        pos: torch.Tensor = None,
    ):
        device, b, n = x.device, *x.shape
        assert n <= self.seq_len

        if mask is None:
            mask = x != self.mask_id

        # embed tokens
        if self.is_abs_pos_emb:
            x = (
                self.token_emb(x)
                + self.pos_emb(x, pos=pos)
                + self.segment_emb(segment_label)
            )
            rel_pos = None
        else:
            x = self.token_emb(x) + self.segment_emb(segment_label)
            rel_pos = self.pos_emb

        # post embedding norm, purportedly leads to greater stabilization
        x = self.post_emb_norm(x)

        embed = self.transformer_blocks(
            x,
            mask=mask,
            rel_pos=rel_pos,
        )

        logits = self.to_logits(embed)

        if return_embed:
            return logits, embed

        if not exists(labels):
            return logits

        if self.dim_out == 1:
            loss = F.binary_cross_entropy_with_logits(
                rearrange(logits, "... 1 -> ..."), labels
            )
        else:
            loss = F.cross_entropy(
                rearrange(logits, "b n c -> b c n"), labels, ignore_index=ignore_index
            )

        if not return_logits:
            return loss

        return loss, logits


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.dim)
        self.mask_lm = MaskedLanguageModel(self.bert.dim, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label=segment_label)
        return self.next_sentence(x), self.mask_lm(x)


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
