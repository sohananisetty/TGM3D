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


class VQMotionDataset(data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        window_size: int = 80,
        max_motion_seconds=60,
        enable_var_len=False,
        fps: int = 20,
        split: str = "train",
    ):
        self.fps = fps

        self.window_size = window_size
        self.max_motion_length = max_motion_seconds * fps
        self.dataset_name = dataset_name
        self.split = split
        self.joints_num = 22
        self.enable_var_len = enable_var_len

        if dataset_name == "t2m":
            self.data_root = os.path.join(data_root, "HumanML3D_SMPL")
            self.motion_dir = os.path.join(self.data_root, "new_joint_vecs")
            self.text_dir = os.path.join(self.data_root, "texts")

        if dataset_name == "aist":
            self.data_root = os.path.join(data_root, "AIST_SMPL/")
            self.motion_dir = os.path.join(self.data_root, "new_joint_vecs")
            self.music_dir = os.path.join(self.data_root, "music")

        if dataset_name == "cm":
            self.data_root = os.path.join(data_root, "Choreomaster_SMPL/")
            self.motion_dir = os.path.join(self.data_root, "new_joint_vecs")
            self.music_dir = os.path.join(self.data_root, "music")

        joints_num = self.joints_num

        self.mean = np.load(os.path.join(data_root, "Mean.npy"))
        self.std = np.load(os.path.join(data_root, "Std.npy"))

        split_file = os.path.join(self.data_root, f"{split}.txt")

        self.data = []
        self.lengths = []
        self.id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())

        for name in tqdm(self.id_list):
            motion = np.load(os.path.join(self.motion_dir, name + ".npy"))
            window_size = (
                self.window_size if self.window_size != -1 else motion.shape[0]
            )
            if motion.shape[0] <= self.window_size:
                continue
            self.lengths.append(motion.shape[0] - self.window_size)
            self.data.append(motion)

        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data: torch.Tensor) -> torch.Tensor:
        return data * (torch.Tensor(self.std).to(data.device) - 1e-8) + torch.Tensor(
            self.mean
        ).to(data.device)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        motion = self.data[item]
        prob = random.random()

        if self.enable_var_len and prob < 0.3:
            if self.max_motion_length < 0:
                window_size = motion.shape[0]
            else:
                window_size = np.random.randint(
                    self.window_size, min(motion.shape[0], self.max_motion_length)
                )

        else:
            if self.window_size == -1:
                window_size = (motion).shape[0]
            else:
                window_size = min(self.window_size, (motion).shape[0])

        idx = random.randint(0, (motion).shape[0] - window_size)

        motion = motion[idx : idx + window_size]
        "Z Normalization"
        motion = (motion - self.mean) / (self.std + 1e-8)
        return motion, self.id_list[item], window_size


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
