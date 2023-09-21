import logging
from typing import List, Union

import torch
import transformers
from beartype import beartype
from transformers import T5Config, T5EncoderModel, T5Tokenizer

transformers.logging.set_verbosity_error()


def exists(val):
    return val is not None


# config

MAX_LENGTH = 256

DEFAULT_T5_NAME = "google/t5-v1_1-base"

T5_CONFIGS = {}

# singleton globals


def get_tokenizer(name):
    tokenizer = T5Tokenizer.from_pretrained(name)
    return tokenizer


def get_model(name):
    model = T5EncoderModel.from_pretrained(name)
    return model


def get_model_and_tokenizer(name):
    global T5_CONFIGS

    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()
    if "model" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["model"] = get_model(name)
    if "tokenizer" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    return T5_CONFIGS[name]["model"], T5_CONFIGS[name]["tokenizer"]


def get_encoded_dim(name):
    if name not in T5_CONFIGS:
        # avoids loading the model if we only want to get the dim
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config=config)
    elif "config" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["config"]
    elif "model" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["model"].config
    else:
        assert False
    return config.d_model


# encoding text


@beartype
def t5_encode_text(
    texts: Union[str, List[str]],
    name=DEFAULT_T5_NAME,
    output_device=None,
    mask_id: float = 0.0,
):
    if isinstance(texts, str):
        texts = [texts]

    t5, tokenizer = get_model_and_tokenizer(name)

    if torch.cuda.is_available():
        t5 = t5.cuda()

    device = next(t5.parameters()).device

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors="pt",
        padding="longest",
        max_length=MAX_LENGTH,
        truncation=True,
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    t5.eval()

    with torch.no_grad():
        output = t5(input_ids=input_ids, attention_mask=attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask.bool()
    encoded_text = encoded_text.masked_fill(attn_mask[..., None], mask_id)

    if not exists(output_device):
        return encoded_text

    encoded_text.to(output_device)
    return encoded_text
