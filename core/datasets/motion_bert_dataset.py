import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


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


class BERTMotionDataset(data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        window_size=20,
        max_motion_length=512,
        codebook_size=1024,
        enable_var_len=False,
        fps: int = 20,
        split: str = "train",
    ):
        self.fps = fps

        self.window_size = window_size
        self.max_motion_length = max_motion_length
        self.dataset_name = dataset_name
        self.split = split
        self.enable_var_len = enable_var_len
        self.vocab_size = codebook_size
        self.mask_index = self.vocab_size
        self.bos_index = self.vocab_size + 1
        self.eos_index = self.vocab_size + 2

        if dataset_name == "t2m":
            self.data_root = os.path.join(data_root, "HumanML3D_SMPL")
            self.motion_dir = os.path.join(self.data_root, "joint_indices")
            self.text_dir = os.path.join(self.data_root, "texts")

        if dataset_name == "aist":
            self.data_root = os.path.join(data_root, "AIST_SMPL/")
            self.motion_dir = os.path.join(self.data_root, "joint_indices")
            self.music_dir = os.path.join(self.data_root, "music")

        if dataset_name == "cm":
            self.data_root = os.path.join(data_root, "Choreomaster_SMPL/")
            self.motion_dir = os.path.join(self.data_root, "joint_indices")
            self.music_dir = os.path.join(self.data_root, "music")

        split_file = os.path.join(self.data_root, f"{split}.txt")

        self.data = []
        self.id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())

        for name in tqdm(self.id_list):
            motion = np.load(os.path.join(self.motion_dir, name + ".npy"))[0]
            self.data.append(motion)

        print("Total number of motions {}".format(len(self.data)))

    def __len__(self) -> int:
        return len(self.data)

    def random_motion_nmp(self, motion_tokens):
        motion_len = motion_tokens.shape[0]

        if motion_len > self.max_motion_length:
            st = random.choice(np.arange(motion_len - self.max_motion_length))
            m = st + random.choice(
                np.arange(self.window_size // 2, self.max_motion_length // 2)
            )
            e = st + self.max_motion_length

        else:
            st = 0
            m = motion_len // 2
            e = motion_len

        m1 = motion_tokens[st:m]
        m2 = motion_tokens[m:e]

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return m1, m2, 1
        else:
            dl = list(np.arange(len(self.data)))
            del dl[6]
            return m1, self.data[np.random.choice(dl)], 0

    def random_mask(self, tokens):
        # tokens = sentence.split()

        tokens = list(tokens)

        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

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
                output_label.append(-1)

        return tokens, output_label

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        tokens = self.data[item]
        m1, m2, is_next_label = self.random_motion_nmp(tokens)
        m1_len = len(m1)
        m2_len = len(m2)
        m1_random, m1_label = self.random_mask(m1)
        m2_random, m2_label = self.random_mask(m2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        m1 = [self.bos_index] + m1_random + [self.eos_index]
        m2 = m2_random + [self.eos_index]

        m1_label = [self.mask_index] + m1_label + [self.mask_index]
        m2_label = m2_label + [self.mask_index]

        segment_label = ([1 for _ in range(len(m1))] + [2 for _ in range(len(m2))])[
            : self.max_motion_length
        ]
        bert_input = (m1 + m2)[: self.max_motion_length]
        bert_label = (m1_label + m2_label)[: self.max_motion_length]

        padding = [
            self.mask_index for _ in range(self.max_motion_length - len(bert_input))
        ]
        padding2 = [0 for _ in range(self.max_motion_length - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(
            padding2
        )

        output = {
            "bert_input": (np.array(bert_input)),
            "bert_label": (np.array(bert_label)),
            "segment_label": (np.array(segment_label)),
            "motion_lengths": np.array([m1_len, m2_len]),
            "is_next_label": (np.array([is_next_label])),
            # "name": self.id_list[item],
        }

        return {key: torch.tensor(value) for key, value in output.items()}


def simple_collate(samples: List[Tuple[torch.Tensor, str]]) -> Dict[str, torch.Tensor]:
    new_batch = []
    bert_label = []
    segment_label = []
    is_next_label = []

    for batch in samples:
        print(batch["is_next_label"])
        new_batch.append(torch.Tensor(batch["bert_input"]))
        bert_label.append(torch.Tensor(batch["bert_label"]))
        segment_label.append(torch.Tensor(batch["segment_label"]))
        is_next_label.append(torch.Tensor(batch["is_next_label"]))

    batch = {
        "bert_input": torch.stack(new_batch),
        "bert_label": torch.stack(bert_label),
        "segment_label": torch.stack(segment_label),
        "is_next_label": torch.stack(is_next_label),
    }

    return batch


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
        collate_fn=collate_fn,
        drop_last=True,
    )

    return train_loader
