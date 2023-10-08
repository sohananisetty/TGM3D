from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.conformer import ConformerBlock
from core.quantization.vector_quantize import VectorQuantize
from einops import rearrange
from einops.layers.torch import Rearrange


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim=256,
        output_dim=768,
        down_sampling_ratio=4,
        n_heads=8,
        dim=768,
        depth=3,
        conv_expansion_factor=2,
        dropout=0.2,
        conv_kernel_size=5,
    ):
        super().__init__()

        self.blocks = []
        self.layers = nn.ModuleList([])
        self.blocks.append(nn.Conv1d(input_dim, dim, 3, 1, 1))
        self.blocks.append(nn.SiLU())

        for i in range(int(np.log2(down_sampling_ratio))):
            self.blocks.append(
                Rearrange("b c n -> b n c"),
            )
            for _ in range(depth):
                block = ConformerBlock(
                    dim=dim,
                    dim_head=dim // n_heads,
                    heads=n_heads,
                    conv_expansion_factor=conv_expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    attn_dropout=dropout,
                    ff_dropout=dropout,
                    conv_dropout=dropout,
                )
            self.blocks.append(block)
            self.blocks.append(
                Rearrange("b n c -> b c n"),
            )
            self.blocks.append(
                nn.Conv1d(dim, dim, 3, 2, 1),
            )
        self.blocks.append(nn.Conv1d(dim, output_dim, 3, 1, 1))
        self.model = nn.Sequential(*self.blocks)

    def forward(self, x, mask=None, need_transpose=False):
        if need_transpose:
            x = rearrange(x, "b n d -> b d n")

        if mask is not None:
            for l in self.blocks:
                if isinstance(l, ConformerBlock):
                    x = l(x, mask.bool()).float()
                    mask = torch.nn.functional.max_pool1d(
                        mask.float(), 3, stride=2, padding=1
                    )
                else:
                    x = l(x.float())

            out = x

        else:
            out = self.model(x)

        if need_transpose:
            out = rearrange(out, "b d n -> b n d")

        return out, mask


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim=256,
        output_dim=768,
        up_sampling_ratio=4,
        n_heads=8,
        dim=768,
        depth=3,
        conv_expansion_factor=2,
        dropout=0.2,
        conv_kernel_size=5,
    ):
        super().__init__()

        self.blocks = []
        self.blocks.append(nn.Conv1d(output_dim, dim, 3, 1, 1))
        self.blocks.append(nn.SiLU())

        for i in range(int(np.log2(up_sampling_ratio))):
            self.blocks.append(
                Rearrange("b c n -> b n c"),
            )
            for _ in range(depth):
                block = ConformerBlock(
                    dim=dim,
                    dim_head=dim // n_heads,
                    heads=n_heads,
                    conv_expansion_factor=conv_expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    attn_dropout=dropout,
                    ff_dropout=dropout,
                    conv_dropout=dropout,
                )
            self.blocks.append(block)
            self.blocks.append(
                Rearrange("b n c -> b c n"),
            )
            self.blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.blocks.append(
                nn.Conv1d(dim, dim, 3, 1, 1),
            )
        self.blocks.append(nn.Conv1d(dim, input_dim, 3, 1, 1))
        self.model = nn.Sequential(*self.blocks)

    def forward(self, x, mask=None, need_transpose=False):
        if need_transpose:
            x = rearrange(x, "b n d -> b d n")
        if mask is not None:
            i = 0
            for l in self.blocks:
                if isinstance(l, ConformerBlock):
                    mask_ = torch.nn.functional.max_pool1d(
                        mask.float(), 3, stride=2 ** (2 - i), padding=1
                    )
                    x = l(x, mask_.bool())

                    i += 1
                else:
                    x = l(x.float())

            out = x
            mask = mask.bool()

        else:
            out = self.model(x)

        if need_transpose:
            out = rearrange(out, "b d n -> b n d")

        return out


class ConformerVQMotionModel(nn.Module):
    """Audio Motion VQGAN model."""

    def __init__(self, args):
        """Initializer for VQGANModel.

        Args:
        config: `VQGANModel` instance.
        is_training: bool. true for training model, false for eval model. Controls
                whether dropout will be applied.
        """
        super(ConformerVQMotionModel, self).__init__()

        self.dim = args.enc_dec_dim

        self.motionEncoder = Encoder(
            input_dim=args.motion_dim,
            output_dim=args.enc_dec_dim,
            dim=args.enc_dec_dim,
            down_sampling_ratio=args.down_sampling_ratio,
            depth=args.depth,
            n_heads=args.heads,
        )

        self.motionDecoder = Decoder(
            input_dim=args.motion_dim,
            output_dim=args.enc_dec_dim,
            dim=args.enc_dec_dim,
            up_sampling_ratio=args.down_sampling_ratio,
            depth=args.depth,
            n_heads=args.heads,
        )

        self.vq = VectorQuantize(
            dim=args.enc_dec_dim,
            codebook_dim=args.codebook_dim,
            codebook_size=args.codebook_size,  # codebook size
            kmeans_init=True,  # set to True
            kmeans_iters=100,
            threshold_ema_dead_code=10,
            stochastic_sample_codes=True,
            sample_codebook_temp=0.2,
            affine_param=True,
            sync_update_v=0.2,
            sync_codebook=False,
        )

    def load(self, path):
        pkg = torch.load(str(path), map_location="cuda")
        self.vq._codebook.batch_mean = pkg["model"]["vq._codebook.batch_mean"]
        self.vq._codebook.batch_variance = pkg["model"]["vq._codebook.batch_variance"]
        self.load_state_dict(pkg["model"])

    def forward(
        self, motion: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict sequences from inputs.

        This is a single forward pass that been used during training.

        Args:
                inputs: Input dict of tensors. The dict should contains
                `motion_input` ([batch_size, motion_seq_length, motion_feature_dimension])

        Returns:
                Final output after the cross modal transformer. A tensor with shape
                [batch_size, motion_seq_length, motion_feature_dimension]
        """
        # Computes motion features.
        motion_input = motion  # b n d

        embed_motion_features, mask_downsampled = self.motionEncoder(
            motion_input, mask, True
        )  # b n d

        ##codebook
        quantized_enc_motion, indices, commit_loss = self.vq(
            embed_motion_features, mask_downsampled
        )

        # b n d , b n/4 , q

        ## decoder
        decoded_motion_features = self.motionDecoder(
            quantized_enc_motion, mask, True
        )  # b n d

        # print(commit_loss.shape)
        # commit_loss = torch.Tensor([1]).to("cuda")

        return decoded_motion_features, indices, commit_loss.sum()

    def encode(self, motion_input, mask=None):
        with torch.no_grad():
            embed_motion_features, mask_downsampled = self.motionEncoder(
                motion_input, mask, True
            )  # b n d

            ##codebook
            quantized_enc_motion, indices, commit_loss = self.vq(
                embed_motion_features, mask_downsampled
            )
            return indices

    def decode(self, indices, mask=None):
        with torch.no_grad():
            quantized = self.vq.get_codes_from_indices(indices).reshape(
                indices.shape[0], -1, self.dim
            )
            decoded_motion_features = self.motionDecoder(quantized, mask, True)  # b n d
            return quantized, decoded_motion_features
