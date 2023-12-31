import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.smpl.body_model import BodyModel
import utils.rotation_conversions as geomtry


class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints, motion_dim):
        super(ReConsLoss, self).__init__()

        if recons_loss == "l1":
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == "l2":
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == "l1_smooth":
            self.Loss = torch.nn.SmoothL1Loss()

        self.bm = BodyModel(
            "/srv/hays-lab/scratch/sanisetty3/music_motion/motion-diffusion-model/body_models/smplh/neutral/model.npz"
        )

        # 3 global motion associated to root
        # 12 motion (6 rot6d, 3 positions, 3 vel xyz, )
        # 4 foot contact
        self.nb_joints = nb_joints
        self.motion_dim = motion_dim

    def forward(self, motion_pred, motion_gt, mask=None):
        ## pred: b n d, gt: b n d, mask: b n
        if mask is None:
            loss = self.Loss(
                motion_pred[..., : self.motion_dim], motion_gt[..., : self.motion_dim]
            )
        else:
            # F.mse_loss(batch["motion"] * batch["motion_mask"][...,None] , pred_motion*batch["motion"] * batch["motion_mask"][...,None], reduction = "sum")
            norm = motion_pred.numel() / (mask.sum() * motion_pred.shape[-1])
            loss = (
                self.Loss(
                    motion_pred[..., : self.motion_dim] * mask[..., None],
                    motion_gt[..., : self.motion_dim] * mask[..., None],
                )
                * norm
            )

        return loss

    def forward_new(self, motion_pred, motion_gt, mask=None):
        bs, n, d = motion_gt.shape  ## d = 135
        motion_pred_aa = geomtry.matrix_to_axis_angle(
            geomtry.rotation_6d_to_matrix(motion_pred[:, :, 3:135].reshape(-1, 22, 6))
        ).reshape(-1, 66)
        body_pred = self.bm.forward(
            root_orient=motion_pred_aa[:, :3],
            pose_body=motion_pred_aa[:, 3:],
            trans=motion_pred[:, :, :3].reshape(-1, 3),
            return_dict=True,
        )

        motion_gt_aa = geomtry.matrix_to_axis_angle(
            geomtry.rotation_6d_to_matrix(motion_gt[:, :, 3:135].reshape(-1, 22, 6))
        ).reshape(-1, 66)
        body_gt = self.bm.forward(
            root_orient=motion_gt_aa[:, :3],
            pose_body=motion_gt_aa[:, 3:],
            trans=motion_gt[:, :, :3].reshape(-1, 3),
            return_dict=True,
        )

        j_pred = body_pred["Jtr"].reshape(bs, n, 52, 3)[:, : self.nb_joints, :]
        j_gt = body_gt["Jtr"].reshape(bs, n, 52, 3)[:, : self.nb_joints, :]

        loss_pos = self.Loss(
            j_pred,
            j_gt,
        )
        loss_vel = self.Loss(
            (j_pred[:, 1:] - j_pred[:, :-1]), (j_gt[:, 1:] - j_gt[:, :-1])
        )

        return loss_pos, loss_vel

    def forward_vel(self, motion_pred, motion_gt, mask=None):
        if motion_pred.shape[-1] == 135 and motion_gt.shape[-1] == 135:
            loss = self.Loss(
                (motion_pred[:, 1:, 3:] - motion_pred[:, :-1, 3:]),
                (motion_gt[:, 1:, 3:] - motion_gt[:, :-1, 3:]),
            )

            return loss
        if mask is None:
            loss = self.Loss(
                motion_pred[..., (3 + self.nb_joints * 9) : (3 + self.nb_joints * 12)],
                motion_gt[..., (3 + self.nb_joints * 9) : (3 + self.nb_joints * 12)],
            )

        else:
            norm = motion_pred[
                ..., (3 + self.nb_joints * 9) : (3 + self.nb_joints * 12)
            ].numel() / (mask.sum() * motion_pred.shape[-1])
            loss = (
                self.Loss(
                    motion_pred[
                        ..., (3 + self.nb_joints * 6) : (3 + self.nb_joints * 9)
                    ]
                    * mask[..., None],
                    motion_gt[..., (3 + self.nb_joints * 6) : (3 + self.nb_joints * 9)]
                    * mask[..., None],
                )
                * norm
            )

        return loss


class InfoNceLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, z_i, z_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        batch_size = z_i.shape[0]
        # z_i = F.normalize(emb_i, dim=1)
        # z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        negatives_mask = (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool, device=z_i.device)
        ).float()

        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss


class CLIPLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))

    def cross_entropy(self, preds, targets, reduction="none"):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def forward(self, z_i, z_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        # z_i = F.normalize(emb_i, dim=1)
        # z_j = F.normalize(emb_j, dim=1)

        logits = (z_i @ z_j.T) / self.temperature
        images_similarity = z_j @ z_j.T
        texts_similarity = z_i @ z_i.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = self.cross_entropy(logits, targets, reduction="none")
        images_loss = self.cross_entropy(logits.T, targets.T, reduction="none")
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()
