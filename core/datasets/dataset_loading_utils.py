import itertools
from typing import Dict, List, Optional

import torch

from .vq_dataset import VQMotionDataset


def load_dataset(
    dataset_names: List[str],
    args: Dict,
    split: str = "train",
    weight_scale: List[int] = [1, 1, 1],
):
    assert len(dataset_names) == len(weight_scale), "mismatch in size"
    dataset_list = []
    weights = []
    for dataset_name in dataset_names:
        dataset_list.append(
            VQMotionDataset(
                dataset_name,
                data_root=args.dataset.dataset_root,
                window_size=args.vqvae.window_size,
                split=split,
            )
        )

    # if len(dataset_names) == 1:
    #     return dataset_list[0], None, None
    concat_dataset = torch.utils.data.ConcatDataset(dataset_list)

    if split != "train" or len(dataset_names) == 1:
        return concat_dataset, None, None

    for i, ds in enumerate(dataset_list):
        weights.append(
            [weight_scale[i] * concat_dataset.__len__() / (ds.__len__())] * ds.__len__()
        )

    weights = list(itertools.chain.from_iterable(weights))

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(weights)
    )

    return concat_dataset, sampler, weights
