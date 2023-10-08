import math
import pathlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from random import random
from typing import Callable, List, Optional, Union
import importlib
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from core.models.conv_vqvae import ConvVQMotionModel
from core.models.attend import Attend, AttentionParams, Attention
from core.models.positional_embeddings import (
    PositionalEmbeddingParams,
    PositionalEmbeddingType,
)
from core.models.albef.xbert import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig
from core.datasets.motion_bert_dataset import TokenizerParams

from core.models.utils import (
    FeedForward,
    LayerNorm,
    default,
    exists,
)
from einops import rearrange, repeat
from torch import einsum, nn
from tqdm.auto import tqdm

import os


class MotionTokenizer:
    def __init__(self, config, device):
        path, class_name = config.model_path.rsplit(".", 1)
        self.tokenizer = (
            getattr(
                importlib.import_module(path),
                class_name,
            )
            .eval()
            .to(device)
        )
        self.chk = os.path.join(config.output_dir, "vqvae_motion.pt")

    def load(self):
        self.tokenizer.load(self.chk)
        for param in self.tokenizer.parameters():
            param.requires_grad = False

    def tokenize(self, motion: torch.Tensor) -> torch.LongTensor:
        # motion: b n d -> b n/4
        with torch.no_grad():
            tokens = self.tokenizer.encode(motion, need_transpose=True)

        return tokens.long()

    def embed(self, indices: torch.LongTensor) -> torch.Tensor:
        # b n -> b n/4 d'

        with torch.no_grad():
            quantized, _ = self.tokenizer.decode(indices)

        return quantized


# class TransformerBlocks(nn.Module):
#     def __init__(self, attention_params: AttentionParams, depth: int, ff_mult: int = 4):
#         super().__init__()
#         self.layers = nn.ModuleList([])

#         for _ in range(depth):
#             self.layers.append(
#                 nn.ModuleList(
#                     [
#                         Attention(attention_params),
#                         Attention(attention_params),
#                         FeedForward(dim=attention_params.dim, mult=ff_mult),
#                     ]
#                 )
#             )

#         self.norm = LayerNorm(attention_params.dim)

#     def forward(self, x, mask=None, context=None, context_mask=None, rel_pos=None):
#         for attn, cross_attn, ff in self.layers:
#             x = attn(x, mask=mask, rel_pos=rel_pos) + x

#             x = cross_attn(x, mask=mask, context=context, context_mask=context_mask) + x

#             x = ff(x) + x

#         return self.norm(x)


# @dataclass
# class BERTParams:
#     attention_params: AttentionParams = AttentionParams()
#     positional_embedding_params: PositionalEmbeddingParams = PositionalEmbeddingParams()
#     positional_embedding: PositionalEmbeddingType = PositionalEmbeddingType.SINE
#     codebook_size: int = 1024
#     dim_out: Optional[int] = 768
#     depth: int = 8
#     ff_mult: int = 4
#     emb_dropout = 0.3
#     post_emb_norm = True


# class BERT(nn.Module):
#     def __init__(self, transformer_params: BERTParams):
#         super().__init__()
#         self.dim = transformer_params.attention_params.dim
#         self.dim_out = transformer_params.dim_out
#         self.codebook_size = transformer_params.codebook_size
#         self.seq_len = transformer_params.positional_embedding_params.max_seq_len
#         self.mask_index = self.codebook_size
#         self.pad_index = self.codebook_size + 1
#         self.vocab_size = self.codebook_size + 3
#         # self.token_emb = nn.Embedding(
#         #     self.vocab_size, self.dim, padding_idx=self.pad_index
#         # )
#         # self.spec_token_emb = nn.Embedding(3, self.dim, padding_idx=self.pad_index)
#         # self.motion_emb = torch.nn.Embedding.from_pretrained(codebook)
#         self.token_emb = nn.Embedding(
#             self.vocab_size, self.dim, padding_idx=self.pad_index
#         )

#         # self.segment_emb = nn.Embedding(3, self.dim, padding_idx=0)

#         self.is_abs_pos_emb = True

#         self.pos_emb = transformer_params.positional_embedding.value(
#             transformer_params.positional_embedding_params
#         )

#         self.emb_dropout = nn.Dropout(transformer_params.emb_dropout)

#         self.transformer_blocks = TransformerBlocks(
#             attention_params=transformer_params.attention_params,
#             depth=transformer_params.depth,
#             ff_mult=transformer_params.ff_mult,
#         )
#         self.norm = LayerNorm(self.dim)

#         # self.dim_out = default(transformer_params.dim_out, self.vocab_size)
#         self.to_logits = nn.Linear(self.dim, self.vocab_size, bias=False)
#         self.to_out = nn.Linear(self.dim, self.dim_out, bias=False)

#         # self.softmax = nn.LogSoftmax(dim=-1)

#         self.post_emb_norm = (
#             nn.LayerNorm(self.dim)
#             if transformer_params.post_emb_norm
#             else nn.Identity()
#         )

#     # def next_sentence(self, x):
#     #     return self.softmax(self.nsp_linear(x[:, 0]))

#     # def mask_lm(self, x):
#     #     return self.softmax(self.mlm_linear(x))

#     def embed(self, x):
#         mask = x > self.motion_emb.num_embeddings

#         # You may want to optimize it, you could probably get away without copy, though
#         # I'm not currently sure how
#         pretrained_batch = x.copy()
#         pretrained_batch[mask] = 0

#         embedded_batch = self.motion_emb(pretrained_batch)

#         # Every token without representation has to be brought into appropriate range
#         x -= self.motion_emb.num_embeddings
#         # Zero out the ones which already have pretrained embedding
#         x[~mask] = 0
#         non_pretrained_embedded_batch = self.spec_token_emb(x)

#         # And finally change appropriate tokens from placeholder embedding created by
#         # pretrained into trainable embeddings.
#         embedded_batch[mask] = non_pretrained_embedded_batch[mask]

#         return embedded_batch

#     def forward(
#         self,
#         x: torch.Tensor,
#         mask: torch.Tensor = None,
#         return_embed: bool = False,
#         pos: torch.Tensor = None,
#     ):
#         device, b, n = x.device, *x.shape
#         assert n <= self.seq_len

#         if mask is None:
#             mask = x != self.mask_index

#         # embed tokens

#         x = self.token_emb(x) + self.pos_emb(x, pos=pos)
#         # + self.segment_emb(segment_label)

#         # post embedding norm, purportedly leads to greater stabilization
#         x = self.post_emb_norm(x)

#         embed = self.transformer_blocks(
#             x,
#             mask=mask,
#         )

#         logits = self.to_logits(embed[:, 1:])
#         out = self.to_out(embed[:, 0])
#         # mlm = self.mask_lm(logits)

#         if return_embed:
#             return logits, out, embed

#         return logits, out


class MotionBERT(nn.Module):
    def __init__(self, args, device="cuda") -> None:
        super().__init__()
        self.device = device
        self.bert_args = args.bert_args

        self.bert_config = BertConfig.from_pretrained(self.bert_args.bert_config)
        self.bert_model = BertForMaskedLM(self.bert_config)
        self.mlm_loss_fnc = torch.nn.CrossEntropyLoss()

        self.init_codebook(args.output_dir)

    def init_codebook(self, path):
        codebook = torch.load(
            os.path.join(path, "codebook.pt"),
            map_location=self.device,
        )
        self.bert_model.bert.embeddings.word_embeddings.weight.data[
            : codebook.shape[0]
        ] = codebook

    def prob_mask_like(
        self,
        shape,
        prob,
    ):
        def uniform(shape):
            return torch.zeros(shape, device=self.device).float().uniform_(0, 1)

        if prob == 1:
            return torch.ones(shape, device=self.device, dtype=torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device=self.device, dtype=torch.bool)
        else:
            return uniform(shape, device=self.device) < prob

    def mask_for_mlm(
        self,
        input_ids,
        tokenizer_params: TokenizerParams,
        targets=None,
        masked_indices=None,
        probability_matrix=None,
    ):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == tokenizer_params.pad_index] = False
        masked_indices[input_ids == tokenizer_params.cls_index] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = tokenizer_params.mask_index

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            tokenizer_params.vocab_size, input_ids.shape, dtype=torch.long
        ).to(input_ids.device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        context=None,
        context_mask=None,
        cond_drop_prob=0.0,
    ):
        bs, n = input_ids.shape
        probability_matrix = torch.full(labels.shape, self.bert_args.mlm_probability)
        input_ids, labels = self.mask_for_mlm(
            input_ids, targets=labels, probability_matrix=probability_matrix
        )

        if context_mask is not None and self.training and cond_drop_prob > 0.0:
            mask = self.prob_mask_like((bs, 1), 1.0 - cond_drop_prob)
            context_mask = context_mask & mask

        mask_lm_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_hidden_states=context,
            encoder_attention_mask=context_mask,
            return_dict=True,
        )

        return mask_lm_output
