from typing import List, Union
import torch
from transformers import T5Config, T5EncoderModel, T5Tokenizer
import clip

DEFAULT_T5_NAME = "google/t5-v1_1-base"


class T5:
    def __init__(
        self, max_length, name=DEFAULT_T5_NAME, device=torch.device("cuda")
    ) -> None:
        self.device = device
        self.max_length = max_length
        self.config = T5Config.from_pretrained(name)
        self.dim = self.config.d_model
        self.tokenizer = T5Tokenizer.from_pretrained(name)
        self.encoder = T5EncoderModel.from_pretrained(name).eval().to(self.device)

    def tokenize(self, texts: Union[str, List[str]]):
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = encoded.input_ids.to(self.device)
        attn_mask = encoded.attention_mask.to(self.device)

        return input_ids, attn_mask

    def t5_encode_text(
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
        encoded_text = encoded_text.masked_fill(attn_mask[..., None], mask_id)

        return encoded_text.to(self.device)


class Clip:
    def __init__(self, device) -> None:
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device).eval()
        clip.model.convert_weights(self.clip_model)
        for p in self.clip_model.parameters():
            p.requires_grad = False

    def get_text_embedding(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        return self.clip_model.encode_text(clip.tokenize(texts).to(self.device)).float()
