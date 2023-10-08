from typing import List, Union
import torch
from transformers import T5Config, T5EncoderModel, T5Tokenizer
import clip

DEFAULT_T5_NAME = "google/t5-v1_1-base"
# "google/t5-v1_1-base"
# "google/flan-t5-xl"

from dataclasses import dataclass


@dataclass
class TextEncoderParams:
    padding: str = "longest"
    target: str = "google/t5-v1_1-base"
    max_length: int = 128


class T5:
    def __init__(
        self, params: TextEncoderParams = TextEncoderParams, device=torch.device("cuda")
    ) -> None:
        self.device = device
        self.max_length = params.max_length
        self.padding = params.padding
        self.config = T5Config.from_pretrained(params.name)
        self.dim = self.config.d_model
        self.tokenizer = T5Tokenizer.from_pretrained(params.name)
        self.encoder = T5EncoderModel.from_pretrained(params.name).to(self.device)

    def load(self, path):
        pkg = torch.load(str(path), map_location="cuda")
        self.encoder.load_state_dict(pkg["model"])

    def tokenize(self, texts: Union[str, List[str]]):
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = encoded.input_ids.to(self.device)
        attn_mask = encoded.attention_mask.to(self.device)

        return input_ids, attn_mask

    def get_text_embedding(
        self,
        texts: Union[str, List[str]],
        mask_id: float = 0.0,
    ):
        if isinstance(texts, str):
            texts = [texts]

        input_ids, attn_mask = self.tokenize(texts)

        with torch.no_grad():
            encoded_text = self.encoder(
                input_ids=input_ids, attention_mask=attn_mask
            ).last_hidden_state.detach()

        attn_mask = attn_mask.bool()
        encoded_text = encoded_text.masked_fill(~attn_mask[..., None], mask_id)

        return encoded_text.to("cpu")


class Clip:
    def __init__(
        self, params: TextEncoderParams = None, device=torch.device("cuda")
    ) -> None:
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device)
        clip.model.convert_weights(self.clip_model)
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.clip_model = self.clip_model.eval()

    def get_text_embedding(self, texts, mask_id=None):
        if isinstance(texts, str):
            texts = [texts]

        return self.clip_model.encode_text(clip.tokenize(texts).to(self.device)).float()
