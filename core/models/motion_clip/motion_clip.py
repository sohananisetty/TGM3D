from typing import List, Union
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import clip


@dataclass
class MotionClipEncoderParams:
    modeltype = "cvae"
    njoints = None
    nfeats = None
    num_frames = None
    num_classes = None
    translation = None
    pose_rep = None
    glob = None
    glob_rot = None
    latent_dim = 256
    ff_size = 1024
    num_layers = 4
    num_heads = 4
    dropout = 0.1
    ablation = None
    activation = "gelu"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class MotionClipEncoder(nn.Module):
    def __init__(
        self,
        # modeltype="cvae",
        njoints=25,
        nfeats=6,
        num_frames=60,
        # num_classes=0,
        # translation=True,
        # pose_rep="rot6d",
        # glob=True,
        # glob_rot=[3.141592653589793, 0, 0],
        latent_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        # ablation=None,
        activation="gelu",
        # **kargs
    ):
        super().__init__()

        # self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        # self.num_classes = num_classes

        # self.pose_rep = pose_rep
        # self.glob = glob
        # self.glob_rot = glob_rot
        # self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # self.ablation = ablation
        self.activation = activation

        self.input_feats = self.njoints * self.nfeats

        self.muQuery = nn.Parameter(torch.randn(1, self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(1, self.latent_dim))
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=self.num_layers
        )

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(
            len(lengths), max_len
        )
        mask = index < lengths.unsqueeze(1)
        return mask

    def load(self, path):
        state_dict = torch.load(path, map_location="cpu")
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

    def forward(self, batch):
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        # embedding of the skeleton
        x = self.skelEmbedding(x)

        # Blank Y to 0's , no classes in our model, only learned token
        # y = y - y
        xseq = torch.cat(
            (
                torch.repeat_interleave(self.muQuery, bs, 0)[None],
                torch.repeat_interleave(self.sigmaQuery, bs, 0)[None],
                x,
            ),
            axis=0,
        )
        # xseq = torch.cat((self.muQuery[y][None], self.sigmaQuery[y][None], x), axis=0)

        # add positional encoding
        xseq = self.sequence_pos_encoder(xseq)

        # create a bigger mask, to allow attend to mu and sigma
        muandsigmaMask = torch.ones((bs, 2), dtype=bool, device=x.device)

        maskseq = torch.cat((muandsigmaMask, mask), axis=1)

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
        mu = final[0]
        logvar = final[1]

        return {"mu": mu}

    def encode_motions(self, motions):
        return self.forward(
            {
                "x": motions,
                "y": torch.zeros(motions.shape[0], dtype=int, device=motions.device),
                "mask": self.lengths_to_mask(
                    torch.ones(motions.shape[0], dtype=int, device=motions.device)
                    * motions.shape[-1]
                ),
            }
        )["mu"]


def get_clip(device):
    clip_model, clip_preprocess = clip.load(
        "ViT-B/32", device=device, jit=False
    ).eval()  # Must set jit=False for training
    clip.model.convert_weights(
        clip_model
    )  # Actually this line is unnecessary since clip by default already on float16
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model
