from functools import wraps

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
import importlib
from core.datasets.motion_bert_dataset import TokenizerParams


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


def create_causal_mask(i, j, device):
    return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)


def l2norm(t):
    return F.normalize(t, dim=-1)


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


# tensor helpers
def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device=device)

    if exists(mask):
        mask_value = max_neg_value(logits)
        logits = logits.masked_fill(~mask, mask_value)

    keep_prob = 1.0 - dropout
    num_keep = max(1, int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim=1).indices

    batch_indices = torch.arange(b, device=device)
    batch_indices = rearrange(batch_indices, "b -> b 1")

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim=-1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device=device) < rearrange(
            seq_keep_counts, "b -> b 1"
        )

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask


def get_mask_subset_prob(mask, prob, min_mask=0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim=-1, keepdim=True) * prob).clamp(min=min_mask)
    logits = torch.rand((batch, seq), device=device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim=-1).float()

    num_padding = (~mask).sum(dim=-1, keepdim=True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask


def prob_mask_like(shape, prob, device=None):
    def uniform(shape, min=0, max=1, device=None):
        return torch.zeros(shape, device=device).float().uniform_(0, 1)

    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    """https://arxiv.org/abs/2002.05202"""

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return gate * F.gelu(x)


def FeedForward(dim, mult=4):
    """https://arxiv.org/abs/2110.09456"""

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias=False),
    )


# def AdaIN(x, style=None):
#     assert x.shape == style.shape
#     b, n, d = x.shape

#     torch.nn.Ada
#     if style:
#         x_mean = x.mean(1).view(b, 1, d)
#         x_std = (x.var(1) + 1e-8).sqrt().view(b, 1, d)
#         style_mean = style.mean(1).view(b, 1, d)
#         style_std = (style.var(1) + 1e-8).sqrt().view(b, 1, d)
#         return ((x - x_mean) / x_std) * style_std + style_mean
#     else:
#         return x


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


def get_obj_from_str(string, reload=False):
    """
    Get object from string
    """

    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    Instantiate object from config
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(config)


def mask_for_mlm(
    input_ids,
    tokenizer_params=TokenizerParams,
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
