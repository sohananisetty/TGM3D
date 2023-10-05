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

from functools import partial


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
    t5,
    num_workers: int = 0,
    shuffle: bool = False,
    sampler: torch.utils.data.Sampler = None,
) -> torch.utils.data.DataLoader:
    collate_fn = partial(simple_collate, t5=t5)

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
