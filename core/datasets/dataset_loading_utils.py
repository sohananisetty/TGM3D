import itertools
from typing import Dict, List, Optional
from enum import Enum
import torch

from .vq_dataset import VQMotionDataset


class AIST_GENRE(Enum):
    GBR = "Break"
    GPO = "Pop"
    GLO = "Lock"
    GMH = "Middle Hip-hop"
    GLH = "LA style Hip-hop"
    GHO = "House"
    GWA = "Waack"
    GKR = "Krump"
    GJS = "Street Jazz"
    GJB = "Ballet Jazz"


def load_dataset(
    dataset_names: List[str],
    args: Dict,
    split: str = "train",
    weight_scale: Optional[List[int]] = None,
):
    if weight_scale is None:
        weight_scale = [1] * len(dataset_names)
    assert len(dataset_names) == len(weight_scale), "mismatch in size"
    dataset_list = []
    weights = []
    for dataset_name in dataset_names:
        dataset_list.append(
            VQMotionDataset(
                dataset_name,
                data_root=args.dataset.dataset_root,
                window_size=args.vqvae.window_size,
                max_motion_seconds=args.vqvae.max_length_seconds,
                enable_var_len=args.dataset.var_len,
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
