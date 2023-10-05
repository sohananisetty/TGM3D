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


def simple_collate(samples: List[Tuple[torch.Tensor, str]]) -> Dict[str, torch.Tensor]:
    new_batch = []
    names = []
    lens = []

    for inp, name, length in samples:
        new_batch.append(torch.Tensor(inp))
        names.append(name)
        lens.append(length)

    batch = {
        "motion": torch.stack(new_batch, 0),
        "names": np.array(names),
        "motion_lengths": np.array(lens),
    }

    return batch


class MotionCollator:
    def __init__(self, mask_id: Optional[int] = None):
        self.mask_id = torch.LongTensor(([mask_id]))

    def __call__(
        self, samples: List[Tuple[torch.Tensor, str]]
    ) -> Dict[str, torch.Tensor]:
        pad_batch_inputs = []
        pad_batch_mask = []
        motion_lengths = []
        names = []
        max_len = max([lens for sample, name, lens in samples])

        for inp, name, lens in samples:
            n, d = inp.shape
            diff = max_len - n
            mask = torch.LongTensor([1] * n + [0] * diff)
            padded = torch.concatenate((torch.tensor(inp), torch.zeros((diff, d))))
            pad_batch_inputs.append(padded)
            pad_batch_mask.append(mask)
            motion_lengths.append(n)
            names.append(name)

        batch = {
            "motion": torch.stack(pad_batch_inputs, 0),
            "motion_lengths": torch.Tensor(motion_lengths),
            "motion_mask": torch.stack(pad_batch_mask, 0),
            "names": np.array(names),
        }

        return batch


class MUSEMotionDataset(data.Dataset):
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

        self.t5 = T5(128)

        if dataset_name == "t2m":
            self.data_root = os.path.join(data_root, "HumanML3D")
            self.motion_dir = os.path.join(self.data_root, "joint_indices")
            self.text_dir = os.path.join(self.data_root, "texts")
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
        self.style = []
        self.context = {}
        self.id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())

        for name in tqdm(self.id_list):
            motion = np.load(os.path.join(self.motion_dir, name + ".npy"))[0]
            # if name[0] == "M":
            #     name = name[1:]
            if motion.shape[0] <= self.window_size:
                continue
            self.data.append(motion)

            if dataset_name == "t2m":
                captions = self.get_caption(os.path.join(self.text_dir, name + ".txt"))
                self.context[name] = captions
            else:
                music = np.load(self.music_dir, name + ".npy")
                self.context[name] = music

        print(
            f"Total number of motions {len(self.data)} total text: {len(self.texts)} , total music {len(self.music)}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def get_caption(self, path):
        text_data = []
        captions = []
        flag = False
        with cs.open(path) as f:
            for line in f.readlines():
                text_dict = {}
                line_split = line.strip().split("#")
                caption = line_split[0]
                captions.append(caption)
                tokens = line_split[1].split(" ")
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                text_dict["caption"] = caption
                text_dict["tokens"] = tokens
                text_dict["f"] = int(f_tag * self.fps)
                text_dict["t"] = int(to_tag * self.fps)
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

        if self.dataset_name == "aist" or self.dataset_name == "cm":
            if m1_len > self.max_motion_length:
                idx = random.randint(0, m1_len - self.max_motion_length)
                m1 = m1[idx : idx + self.max_motion_length]
                context = self.context[item][idx : idx + self.max_motion_length]
            else:
                m1 = m1[: self.max_motion_length]
                context = self.context[item][: self.max_motion_length]

        if self.dataset_name == "t2m":
            text_dict = self.context[name]
            context = text_dict["caption"]
            # context_embed = self.t5.t5_encode_text(context)
            if text_dict["f"] != 0.0 and text_dict["t"] != 0.0:
                m1 = m1[text_dict["f"] : text_dict["t"]]

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
            "mask": np.array(mask),
        }

        return {key: torch.tensor(value) for key, value in output.items()}


# def simple_collate(samples: List[Tuple[torch.Tensor, str]]) -> Dict[str, torch.Tensor]:
#     new_batch = []
#     bert_label = []

#     for batch in samples:
#         new_batch.append(torch.Tensor(batch["bert_input"]))
#         bert_label.append(torch.Tensor(batch["bert_label"]))

#     batch = {
#         "bert_input": torch.stack(new_batch),
#         "bert_label": torch.stack(bert_label),
#     }

#     return batch


def DATALoader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
    sampler: torch.utils.data.Sampler = None,
    collate_fn: Optional[
        Callable[[List[Tuple[torch.Tensor, str]]], Dict[str, torch.Tensor]]
    ] = None,
) -> torch.utils.data.DataLoader:
    if collate_fn is None:
        collate_fn = simple_collate

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=None,
        drop_last=True,
    )

    return train_loader
