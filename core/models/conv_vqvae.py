import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..quantization.vector_quantize import VectorQuantize
from .resnet import Resnet1D


class Encoder(nn.Module):
    def __init__(
        self,
        input_emb_width=256,
        output_emb_width=512,
        down_sampling_ratio=4,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        blocks = []
        k, p = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(int(np.log2(down_sampling_ratio))):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, k, stride_t, p),
                Resnet1D(
                    width, depth, dilation_growth_rate, activation=activation, norm=norm
                ),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x, need_transpose=False):
        if need_transpose:
            x = rearrange(x, "b n d -> b d n")
        out = self.model(x)

        if need_transpose:
            out = rearrange(out, "b d n -> b n d")

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        input_emb_width=256,
        output_emb_width=512,
        down_sampling_ratio=4,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = []

        k, p = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(int(np.log2(down_sampling_ratio))):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    reverse_dilation=True,
                    activation=activation,
                    norm=norm,
                ),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(width, out_dim, 3, 1, 1),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x, need_transpose=False):
        if need_transpose:
            x = rearrange(x, "b n d -> b d n")
        out = self.model(x)

        if need_transpose:
            out = rearrange(out, "b d n -> b n d")

        return out


class ConvVQMotionModel(nn.Module):
    """Audio Motion VQGAN model."""

    def __init__(self, args, device="cuda"):
        """Initializer for VQGANModel.

        Args:
        config: `VQGANModel` instance.
        is_training: bool. true for training model, false for eval model. Controls
                whether dropout will be applied.
        """
        super(ConvVQMotionModel, self).__init__()

        self.device = device
        self.dim = args.motion_dim

        self.motionEncoder = Encoder(
            input_emb_width=args.motion_dim,
            output_emb_width=args.enc_dec_dim,
            width=args.width,
            down_sampling_ratio=args.down_sampling_ratio,
        )

        self.motionDecoder = Decoder(
            input_emb_width=args.motion_dim,
            output_emb_width=args.enc_dec_dim,
            width=args.width,
            down_sampling_ratio=args.down_sampling_ratio,
        )

        self.vq = VectorQuantize(
            dim=args.enc_dec_dim,
            codebook_size=args.codebook_size,  # codebook size
            kmeans_init=True,  # set to True
            kmeans_iters=100,
            threshold_ema_dead_code=10,
            stochastic_sample_codes=True,
            sample_codebook_temp=0.1,
            affine_param=True,
            sync_update_v=0.2,
        )

    def forward(self, motion: torch.Tensor):
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

        embed_motion_features = self.motionEncoder(motion_input, True)  # b n d

        ##codebook
        quantized_enc_motion, indices, commit_loss = self.vq(embed_motion_features)
        # b n d , b n q , q

        ## decoder
        decoded_motion_features = self.motionDecoder(
            quantized_enc_motion, True
        )  # b n d

        return decoded_motion_features, indices, commit_loss.sum()

    def encode(self, motion_input):
        with torch.no_grad():
            embed_motion_features = self.motionEncoder(motion_input)
            quantized_enc_motion, indices, commit_loss = self.vq(embed_motion_features)
            return indices

    def decode(self, indices):
        with torch.no_grad():
            quantized = self.vq.get_codes_from_indices(indices).reshape(
                quantized.shape[0], -1, self.dim
            )
            out_motion = self.motionDecoder(quantized)
            return quantized, out_motion
