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
from music_motion.TGM3D.core.models.pretrained_encoders.t5 import (
    DEFAULT_T5_NAME,
    get_encoded_dim,
    t5_encode_text,
)
from core.models.utils import (
    FeedForward,
    LayerNorm,
    default,
    dropout_seq,
    eval_decorator,
    exists,
    get_mask_subset_prob,
    l2norm,
)
from einops import rearrange, repeat
from torch import einsum, nn
from tqdm.auto import tqdm


class AdaIn(nn.Module):
    """
    adaptive instance normalization
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(dim)

    def forward(self, x, style=None):
        if style is None:
            return x
        b, n, d = x.shape
        style_mean = style.mean(1).view(b, 1, d)
        style_std = (style.var(1) + 1e-8).sqrt().view(b, 1, d)
        x = self.norm(x)
        x = x * style_std + style_mean
        return x


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
                        AdaIn(dim=attention_params.dim),
                        FeedForward(dim=attention_params.dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = LayerNorm(attention_params.dim)

    def forward(
        self, x, mask=None, context=None, context_mask=None, style=None, rel_pos=None
    ):
        for attn, cross_attn, ff1, adain, ff2 in self.layers:
            x = attn(x, mask=mask, rel_pos=rel_pos) + x

            x = cross_attn(x, mask=mask, context=context, context_mask=context_mask) + x

            x = ff1(x) + x

            x = adain(x, style=style)

            x = ff2(x) + x

        return self.norm(x)


# transformer - it's all we need


@dataclass
class TransformerParams:
    attention_params: AttentionParams
    positional_embedding_params: PositionalEmbeddingParams
    positional_embedding: PositionalEmbeddingType
    num_tokens: int = 1024
    dim_out: int = None
    depth: int = 12
    ff_mult: int = 4
    self_cond: bool = False
    add_mask_id: bool = True
    emb_dropout = 0.0
    post_emb_norm = False
    context_types = List[str] = ["text"]
    context_dim: List[int] = [768]
    style_dim: int = 768


class Transformer(nn.Module):
    def __init__(self, transformer_params: TransformerParams):
        super().__init__()
        self.dim = transformer_params.attention_params.dim
        self.num_tokens = transformer_params.num_tokens
        self.context_types = transformer_params.context_types
        self.seq_len = transformer_params.positional_embedding_params.max_seq_len

        self.style_dim = transformer_params.style_dim
        self.mask_id = self.num_tokens if transformer_params.add_mask_id else None

        self.token_emb = nn.Embedding(
            self.num_tokens + int(transformer_params.add_mask_id), self.dim
        )

        self.is_abs_pos_emb = transformer_params.positional_embedding.name in [
            "ABS",
            "SINE",
        ]

        self.pos_emb = transformer_params.positional_embedding(
            transformer_params.positional_embedding_params
        )

        self.emb_dropout = nn.Dropout(transformer_params.emb_dropout)

        # self.pos_emb = nn.Embedding(seq_len, dim)

        self.transformer_blocks = TransformerBlocks(
            attention_params=transformer_params.attention_params,
            depth=transformer_params.depth,
            ff_mult=transformer_params.ff_mult,
        )
        self.norm = LayerNorm(self.dim)

        self.dim_out = default(transformer_params.dim_out, self.num_tokens)
        self.to_logits = nn.Linear(self.dim, self.dim_out, bias=False)

        self.context_embed_projs = {}
        for context_dim, context in zip(
            transformer_params.context_dim, transformer_params.context_types
        ):
            self.context_embed_projs[context] = (
                nn.Linear(context_dim, self.dim, bias=False)
                if context_dim != self.dim
                else nn.Identity()
            )

        # context conditioning

        self.style_embed_proj = (
            nn.Linear(self.style_dim, self.dim, bias=False)
            if self.style_dim != self.dim
            else nn.Identity()
        )

        # optional self conditioning

        self.self_cond = transformer_params.self_cond
        self.self_cond_to_init_embed = FeedForward(self.dim)

        self.post_emb_norm = (
            nn.LayerNorm(self.dim)
            if transformer_params.post_emb_norm
            else nn.Identity()
        )

    def prepare_inputs(
        self, x, mask=None, contexts=None, style=None, cond_drop_prob=0.0
    ):
        device, b, n = x.device, *x.shape

        if mask is None:
            mask = (x != self.mask_id).any(dim=-1)

        context = None
        context_mask = None

        for context_type, context_embeds in contexts.items():
            new_context = self.context_embed_projs[context_type](context_embeds)
            new_context_mask = (context_embeds != self.mask_id).any(dim=-1)

            if context is None or context_mask is None:
                context = new_context
                context_mask = new_context_mask
            else:
                context = torch.cat((context, new_context), dim=-2)
                context_mask = torch.cat((context_mask, new_context_mask), dim=-2)

        # classifier free guidance

        if cond_drop_prob > 0.0:
            mask_ = prob_mask_like((b, 1), 1.0 - cond_drop_prob, device)
            context_mask = context_mask & mask_

        style = self.style_embed_proj(style)

        return mask, context, context_mask, style

    def forward_with_cond_scale(
        self, *args, cond_scale=3.0, return_embed=False, **kwargs
    ):
        if cond_scale == 1:
            return self.forward(
                *args, return_embed=return_embed, cond_drop_prob=0.0, **kwargs
            )

        logits, embed = self.forward(
            *args, return_embed=True, cond_drop_prob=0.0, **kwargs
        )

        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward_with_neg_prompt(
        self,
        *args,
        text_embed: torch.Tensor,
        neg_text_embed: torch.Tensor,
        cond_scale=3.0,
        return_embed=False,
        **kwargs,
    ):
        neg_logits = self.forward(
            *args, neg_text_embed=neg_text_embed, cond_drop_prob=0.0, **kwargs
        )
        pos_logits, embed = self.forward(
            *args,
            return_embed=True,
            text_embed=text_embed,
            cond_drop_prob=0.0,
            **kwargs,
        )

        logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        if return_embed:
            return logits, embed

        return logits

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        style: torch.Tensor = None,
        return_embed: bool = False,
        return_logits: bool = False,
        labels=None,
        ignore_index: int = 0,
        self_cond_embed=None,
        cond_drop_prob: float = 0.0,
        context_embeds_dict=None,
        pos: torch.Tensor = None,
        sum_embeds=None,
    ):
        device, b, n = x.device, *x.shape
        assert n <= self.seq_len

        mask, context, context_mask, style = self.prepare_inputs(
            x,
            mask=mask,
            contexts=context_embeds_dict,
            style=style,
            cond_drop_prob=cond_drop_prob,
        )

        # embed tokens
        if self.is_abs_pos_emb:
            x = self.token_emb(x) + self.pos_emb(x, pos=pos)
            rel_pos = None
        else:
            x = self.token_emb(x)
            rel_pos = self.pos_emb

        if exists(sum_embeds):
            x = x + sum_embeds

        # post embedding norm, purportedly leads to greater stabilization
        x = self.post_emb_norm(x)

        if self.self_cond:
            if not exists(self_cond_embed):
                self_cond_embed = torch.zeros_like(x)
            x = x + self.self_cond_to_init_embed(self_cond_embed)

        embed = self.transformer_blocks(
            x,
            mask=mask,
            context=context,
            context_mask=context_mask,
            style=style,
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


# self critic wrapper


class SelfCritic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.to_pred = nn.Linear(net.dim, 1)

    def forward_with_cond_scale(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_cond_scale(
            x, *args, return_embed=True, **kwargs
        )
        return self.to_pred(embeds)

    def forward_with_neg_prompt(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_neg_prompt(
            x, *args, return_embed=True, **kwargs
        )
        return self.to_pred(embeds)

    def forward(self, x, *args, labels=None, **kwargs):
        _, embeds = self.net(x, *args, return_embed=True, **kwargs)
        logits = self.to_pred(embeds)

        if not exists(labels):
            return logits

        logits = rearrange(logits, "... 1 -> ...")
        return F.binary_cross_entropy_with_logits(logits, labels)


# specialized transformers


class MaskGitTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        assert "add_mask_id" not in kwargs
        super().__init__(*args, add_mask_id=True, **kwargs)


class TokenCritic(Transformer):
    def __init__(self, *args, **kwargs):
        assert "dim_out" not in kwargs
        super().__init__(*args, dim_out=1, **kwargs)


# classifier free guidance functions


def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob


# sampling helpers


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs


# noise schedules


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


# main maskgit classes


class MaskGit(nn.Module):
    def __init__(
        self,
        image_size,
        transformer: MaskGitTransformer,
        noise_schedule: Callable = cosine_schedule,
        token_critic: Optional[TokenCritic] = None,
        self_token_critic=False,
        vae: Optional[ConvVQMotionModel] = None,
        cond_drop_prob=0.5,
        self_cond_prob=0.9,
        no_mask_token_prob=0.0,
        critic_loss_weight=1.0,
    ):
        super().__init__()
        self.vae = vae.copy_for_eval() if exists(vae) else None

        self.image_size = image_size

        self.cond_drop_prob = cond_drop_prob

        self.transformer = transformer
        self.self_cond = transformer.self_cond
        assert (
            self.vae.codebook_size == transformer.num_tokens
        ), "transformer num_tokens must be set to be equal to the vae codebook size"

        self.mask_id = transformer.mask_id
        self.noise_schedule = noise_schedule

        assert not (self_token_critic and exists(token_critic))
        self.token_critic = token_critic

        if self_token_critic:
            self.token_critic = SelfCritic(transformer)

        self.critic_loss_weight = critic_loss_weight

        # self conditioning
        self.self_cond_prob = self_cond_prob

        # percentage of tokens to be [mask]ed to remain the same token, so that transformer produces better embeddings across all tokens as done in original BERT paper
        # may be needed for self conditioning
        self.no_mask_token_prob = no_mask_token_prob

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        state_dict = torch.load(str(path))
        self.load_state_dict(state_dict)

    def prepare_mask_and_labels(self, ids, ignore_index=-1):
        batch, seq_len, device = (
            *ids.shape,
            ids.device,
        )

        rand_time = uniform((batch,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((batch, seq_len), device=device).argsort(dim=-1)
        mask = batch_randperm < rearrange(num_token_masked, "b -> b 1")

        labels = torch.where(mask, ids, ignore_index)

        return mask, labels

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        texts: List[str],
        negative_texts: Optional[List[str]] = None,
        cond_images: Optional[torch.Tensor] = None,
        fmap_size=None,
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,
        force_not_use_token_critic=False,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        cond_scale=3,
        critic_noise_scale=1,
    ):
        fmap_size = default(fmap_size, self.vae.get_encoded_fmap_size(self.image_size))

        # begin with all image token ids masked

        device = next(self.parameters()).device

        seq_len = fmap_size**2

        batch_size = len(texts)

        shape = (batch_size, seq_len)

        ids = torch.full(shape, self.mask_id, dtype=torch.long, device=device)
        scores = torch.zeros(shape, dtype=torch.float32, device=device)

        starting_temperature = temperature

        cond_ids = None

        text_embeds = self.transformer.encode_text(texts)

        demask_fn = self.transformer.forward_with_cond_scale

        # whether to use token critic for scores

        use_token_critic = exists(self.token_critic) and not force_not_use_token_critic

        if use_token_critic:
            token_critic_fn = self.token_critic.forward_with_cond_scale

        # negative prompting, as in paper

        neg_text_embeds = None
        if exists(negative_texts):
            assert len(texts) == len(negative_texts)

            neg_text_embeds = self.transformer.encode_text(negative_texts)
            demask_fn = partial(
                self.transformer.forward_with_neg_prompt,
                neg_text_embeds=neg_text_embeds,
            )

            if use_token_critic:
                token_critic_fn = partial(
                    self.token_critic.forward_with_neg_prompt,
                    neg_text_embeds=neg_text_embeds,
                )

        if self.resize_image_for_cond_image:
            assert exists(
                cond_images
            ), "conditioning image must be passed in to generate for super res maskgit"
            with torch.no_grad():
                _, cond_ids, _ = self.cond_vae.encode(cond_images)

        self_cond_embed = None

        for timestep, steps_until_x0 in tqdm(
            zip(
                torch.linspace(0, 1, timesteps, device=device),
                reversed(range(timesteps)),
            ),
            total=timesteps,
        ):
            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim=-1).indices

            ids = ids.scatter(1, masked_indices, self.mask_id)

            logits, embed = demask_fn(
                ids,
                text_embeds=text_embeds,
                self_cond_embed=self_cond_embed,
                conditioning_token_ids=cond_ids,
                cond_scale=cond_scale,
                return_embed=True,
            )

            self_cond_embed = embed if self.self_cond else None

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (
                steps_until_x0 / timesteps
            )  # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            is_mask = ids == self.mask_id

            ids = torch.where(is_mask, pred_ids, ids)

            if use_token_critic:
                scores = token_critic_fn(
                    ids,
                    text_embeds=text_embeds,
                    conditioning_token_ids=cond_ids,
                    cond_scale=cond_scale,
                )

                scores = rearrange(scores, "... 1 -> ...")

                scores = scores + (
                    uniform(scores.shape, device=device) - 0.5
                ) * critic_noise_scale * (steps_until_x0 / timesteps)

            else:
                probs_without_temperature = logits.softmax(dim=-1)

                scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
                scores = rearrange(scores, "... 1 -> ...")

                if not can_remask_prev_masked:
                    scores = scores.masked_fill(~is_mask, -1e5)
                else:
                    assert (
                        self.no_mask_token_prob > 0.0
                    ), "without training with some of the non-masked tokens forced to predict, not sure if the logits will be meaningful for these token"

        # get ids

        ids = rearrange(ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)

        if not exists(self.vae):
            return ids

        images = self.vae.decode_from_ids(ids)
        return images

    def forward(
        self,
        ids: torch.Tensor,
        ignore_index: int = -1,
        context_texts: Optional[List[str]] = None,
        context_text_embeds: Optional[torch.Tensor] = None,
        context_music: Optional[torch.Tensor] = None,
        context_music_embeds: Optional[torch.Tensor] = None,
        style_texts: Optional[List[str]] = None,
        style_text_embeds: Optional[torch.Tensor] = None,
        cond_drop_prob: Optional[float] = None,
        train_only_generator: bool = False,
        sample_temperature: Optional[float] = None,
    ):
        # get some basic variables

        ids = rearrange(ids, "b ... -> b (...)")

        device, cond_drop_prob = (
            ids.device,
            default(cond_drop_prob, self.cond_drop_prob),
        )

        mask, labels = self.prepare_mask_and_labels(ids, ignore_index)

        if self.training and self.no_mask_token_prob > 0.0:
            no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
            mask &= ~no_mask_mask

        x = torch.where(mask, self.mask_id, ids)

        if exists(context_texts):
            pass

        if exists(context_music):
            pass

        if exists(style_texts):
            pass

        context_embeds_dict = {
            "text": context_text_embeds,
            "music": context_music_embeds,
        }

        # self conditioning

        self_cond_embed = None

        if self.transformer.self_cond and random() < self.self_cond_prob:
            with torch.no_grad():
                _, self_cond_embed = self.transformer(
                    x,
                    mask=mask,
                    style=style_text_embeds,
                    context_embeds_dict=context_embeds_dict,
                    cond_drop_prob=0.0,
                    return_embed=True,
                )

                self_cond_embed.detach_()

        # get loss

        ce_loss, logits = self.transformer(
            x,
            mask=mask,
            style=style_text_embeds,
            context_embeds_dict=context_embeds_dict,
            self_cond_embed=self_cond_embed,
            labels=labels,
            cond_drop_prob=cond_drop_prob,
            ignore_index=ignore_index,
            return_logits=True,
        )

        if not exists(self.token_critic) or train_only_generator:
            return ce_loss

        # token critic loss

        sampled_ids = gumbel_sample(
            logits, temperature=default(sample_temperature, random())
        )

        critic_input = torch.where(mask, sampled_ids, x)
        critic_labels = (ids != critic_input).float()

        bce_loss = self.token_critic(
            critic_input,
            mask=mask,
            style=style_text_embeds,
            context_embeds_dict=context_embeds_dict,
            labels=critic_labels,
            cond_drop_prob=cond_drop_prob,
        )

        return ce_loss + self.critic_loss_weight * bce_loss
