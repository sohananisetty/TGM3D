import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


from glob import glob
import codecs as cs
from scipy.spatial.transform import Rotation as R
from core.models.text_encoders import T5
from dataclasses import dataclass
from functools import partial


@dataclass
class TokenizerParams:
    padding: str = "longest"
    cls_index: int = 1024
    pad_index: int = 1025
    mask_index: int = 1026
    vocab_size: int = 1027
    model_max_length: int = 512


class MotionCollator:
    def __init__(self, text_encoder, tokenizer_prams=TokenizerParams):
        self.tokenizer_prams = tokenizer_prams
        self.text_encoder = text_encoder

    def __call__(self, samples: List) -> Dict[str, torch.Tensor]:
        pad_batch_inputs = []
        pad_batch_mask = []
        motion_lengths = []
        contexts = []
        no_attend = []
        max_len = max([len(sample) for sample, _ in samples])

        for indx, (inp, context) in enumerate(samples):
            n = len(inp)
            diff = max_len - n
            mask = torch.BoolTensor([1] * n + [0] * diff)
            padded = torch.concatenate(
                (
                    torch.LongTensor(inp),
                    torch.ones(diff, dtype=torch.long) * self.tokenizer_prams.pad_index,
                )
            )
            pad_batch_inputs.append(padded)
            pad_batch_mask.append(mask)
            motion_lengths.append(n)
            contexts.append(context)
            if context == "":
                no_attend.append(indx)

        context_embed = self.text_encoder.get_text_embedding(
            contexts, mask_id=self.tokenizer_prams.pad_index
        )
        context_mask = (context_embed != self.tokenizer_prams.pad_index).any(dim=-1)
        context_mask[no_attend] = 0.0

        batch = {
            "input_ids": torch.stack(pad_batch_inputs, 0),
            "input_lengths": torch.Tensor(motion_lengths),
            "attention_mask": torch.stack(pad_batch_mask, 0),
            "context": context_embed,
            "context_mask": context_mask,
            "texts": np.array(contexts),
        }

        return batch


class BERTMotionDataset(data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        window_size=20,
        max_motion_length=512,
        codebook_size=1024,
        mask_prob=0.15,
        split: str = "train",
    ):
        self.window_size = window_size // 4
        self.max_motion_length = max_motion_length
        self.dataset_name = dataset_name
        self.split = split
        self.codebook_size = codebook_size
        self.mask_prob = mask_prob
        self.fps = 20
        self.mask_index = self.codebook_size
        self.pad_index = self.codebook_size + 1
        self.cls_index = self.codebook_size + 2
        self.vocab_size = self.codebook_size + 3

        if dataset_name == "t2m":
            self.data_root = os.path.join(data_root, "HumanML3D")
            self.motion_dir = os.path.join(self.data_root, "joint_indices")
            self.text_dir = os.path.join(
                "/srv/hays-lab/scratch/sanisetty3/music_motion/HumanMotion/HumanML3D",
                "texts",
            )
            split = split + "_og"

        if dataset_name == "aist":
            self.data_root = os.path.join(data_root, "AIST/")
            self.motion_dir = os.path.join(self.data_root, "joint_indices")
            self.music_dir = os.path.join(self.data_root, "music")

        if dataset_name == "cm":
            self.data_root = os.path.join(data_root, "Choreomaster/")
            self.motion_dir = os.path.join(self.data_root, "joint_indices")
            self.music_dir = os.path.join(self.data_root, "music")

        split_file = os.path.join(self.data_root, f"{split}.txt")

        self.data = []
        self.context = {}
        self.id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())

        for name in tqdm(self.id_list):
            motion = np.load(os.path.join(self.motion_dir, name + ".npy"))[0]

            if motion.shape[0] <= self.window_size:
                continue
            self.data.append(motion)

            if dataset_name == "t2m":
                captions = self.get_caption(os.path.join(self.text_dir, name + ".txt"))
                self.context[name] = captions

        print(f"Total number of motions {len(self.data)}")

    def __len__(self) -> int:
        return len(self.data)

    def get_caption(self, path):
        text_data = []
        captions = []
        ranges = []
        flag = False
        with cs.open(path) as f:
            for line in f.readlines():
                text_dict = {}
                line_split = line.strip().split("#")
                caption = line_split[0]
                captions.append(caption)
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag
                ranges.append((int(f_tag * self.fps), int(to_tag * self.fps)))

        text_dict["captions"] = captions
        text_dict["ranges"] = ranges
        return text_dict

    def random_mask(self, tokens):
        # tokens = sentence.split()

        tokens = list(tokens)

        output_label = []
        mask = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                mask.append(False)

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(self.vocab_size)

                # 10% randomly change token to current token
                else:
                    tokens[i] = token

                output_label.append(token)

            else:
                tokens[i] = token
                output_label.append(self.pad_index)
                mask.append(True)

        return tokens, output_label, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        m1 = self.data[item]
        m1_len = len(m1)
        name = self.id_list[item]
        context = ""

        if self.dataset_name == "t2m":
            text_dict = self.context[name]
            idx = np.random.choice(np.arange(len(text_dict["captions"])))
            context = text_dict["captions"][idx]
            f_tag, to_tag = text_dict["ranges"][idx]
            if f_tag != 0.0 and to_tag != 0.0:
                m1 = m1[f_tag:to_tag]

        m1_random, m1_label, mask = self.random_mask(m1)

        m1 = [self.cls_index] + m1_random
        mask = [True] + mask
        m1_label = [self.pad_index] + m1_label

        bert_input = (m1)[: self.max_motion_length]
        bert_label = (m1_label)[: self.max_motion_length]
        mask = (mask)[: self.max_motion_length]

        padding = [self.pad_index] * (self.max_motion_length - len(bert_input))
        mask_pad = [False] * (self.max_motion_length - len(bert_input))
        bert_input.extend(padding), bert_label.extend(padding), mask.extend(mask_pad)

        output = {
            "bert_input": (np.array(bert_input)),
            "bert_label": (np.array(bert_label)),
            "context": context,
            "mask": (np.array(mask)),
        }

        return output


class BERTPretrainMotionDataset(data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        window_size=5,
        tokenization_params=TokenizerParams,
        split: str = "train",
    ):
        self.window_size = window_size
        self.max_motion_length = tokenization_params.model_max_length
        self.dataset_name = dataset_name
        self.split = split
        self.fps = 20
        self.mask_index = tokenization_params.mask_index
        self.pad_index = tokenization_params.pad_index
        self.cls_index = tokenization_params.cls_index
        self.vocab_size = tokenization_params.vocab_size

        if dataset_name == "t2m":
            self.data_root = os.path.join(data_root, "HumanML3D")
            self.motion_dir = os.path.join(self.data_root, "joint_indices")
            self.text_dir = os.path.join(
                "/srv/hays-lab/scratch/sanisetty3/music_motion/HumanMotion/HumanML3D/texts"
            )

        if dataset_name == "aist":
            self.data_root = os.path.join(data_root, "AIST/")
            self.motion_dir = os.path.join(self.data_root, "joint_indices")

        if dataset_name == "cm":
            self.data_root = os.path.join(data_root, "Choreomaster/")
            self.motion_dir = os.path.join(self.data_root, "joint_indices")

        if dataset_name == "moyo":
            self.data_root = os.path.join(data_root, "MOYO/")
            self.motion_dir = os.path.join(self.data_root, "joint_indices")

        split_file = os.path.join(self.data_root, f"{split}.txt")

        self.data = []
        self.context = {}
        self.id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())

        for name in tqdm(self.id_list):
            motion = np.load(os.path.join(self.motion_dir, name + ".npy"))[0]

            if motion.shape[0] <= self.window_size:
                continue
            self.data.append(motion)
            if dataset_name == "t2m":
                captions = self.get_caption(os.path.join(self.text_dir, name + ".txt"))
                self.context[name] = captions

        print(f"Total number of motions {len(self.data)}")

    def __len__(self) -> int:
        return len(self.data)

    def get_caption(self, path):
        captions = []
        ranges = []
        with open(path) as f:
            for line in f.readlines():
                text_dict = {}
                line_split = line.strip().split("#")
                caption = line_split[0]
                captions.append(caption)
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag
                ranges.append((int(f_tag * self.fps), int(to_tag * self.fps)))

        text_dict["captions"] = captions
        text_dict["ranges"] = ranges
        return text_dict

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        name = self.id_list[item]
        m1 = list(self.data[item])
        context = ""

        if len(m1) > self.max_motion_length:
            idx = np.random.randint(0, len(m1) - self.max_motion_length)
            m1 = m1[idx : idx + self.max_motion_length]

        m1 = [self.cls_index] + m1
        bert_input = m1[: self.max_motion_length]

        if self.dataset_name == "t2m":
            text_dict = self.context[name]
            idx = np.random.choice(np.arange(len(text_dict["captions"])))
            context = text_dict["captions"][idx]

        return bert_input, context


def simple_collate(
    samples: List[Tuple[torch.Tensor, str]], t5
) -> Dict[str, torch.Tensor]:
    new_batch = []
    bert_label = []
    mask = []
    context_embed = []
    contexts = []

    for batch in samples:
        new_batch.append(torch.LongTensor(batch["bert_input"]))
        bert_label.append(torch.LongTensor(batch["bert_label"]))
        mask.append(torch.BoolTensor(batch["mask"]))
        contexts.append((batch["context"]))

    context_embed = t5.t5_encode_text(contexts)

    batch = {
        "bert_input": torch.stack(new_batch),
        "bert_label": torch.stack(bert_label),
        "mask": torch.stack(mask),
        "context_embed": (context_embed),
        "context": np.array(contexts),
    }

    return batch


def DATALoader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    text_encoder,
    num_workers: int = 0,
    shuffle: bool = False,
    sampler: torch.utils.data.Sampler = None,
    collate_fn: Optional[
        Callable[[List[Tuple[torch.Tensor, str]]], Dict[str, torch.Tensor]]
    ] = None,
) -> torch.utils.data.DataLoader:
    if collate_fn is None:
        collate_fn = MotionCollator(text_encoder)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return train_loader
