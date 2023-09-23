import itertools
import os
from collections import Counter
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs

# from core.datasets import dataset_TM_eval
from core.datasets.dataset_loading_utils import load_dataset
from core.datasets.vq_dataset import DATALoader, MotionCollator
from core.models.conformer_vqvae import ConformerVQMotionModel
from core.models.conv_vqvae import ConvVQMotionModel

# from core.models.evaluator_wrapper import EvaluatorModelWrapper
from core.models.loss import ReConsLoss
from core.optimizer import get_optimizer
from torch import nn
from tqdm import tqdm
from transformers import AdamW, get_scheduler

# from utils.eval_trans import evaluation_vqvae, evaluation_vqvae_loss
from utils.vis_utils.render_final import Renderer
from yacs.config import CfgNode

# from utils.word_vectorizer import WordVectorizer


def exists(val):
    return val is not None


def noop(*args, **kwargs):
    pass


def cycle(dl):
    while True:
        for data in dl:
            yield data


def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)


def yes_or_no(question):
    answer = input(f"{question} (y/n) ")
    return answer.lower() in ("yes", "y")


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


# auto data to module keyword argument routing functions


def has_duplicates(tup):
    counts = dict()
    for el in tup:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
    return any(filter(lambda count: count > 1, counts.values()))


# main trainer class


class VQVAEMotionTrainer(nn.Module):
    def __init__(
        self,
        args: CfgNode,
        accelerate_kwargs: dict = dict(),
    ):
        super().__init__()
        self.model_name = args.vqvae_model_name

        transformers.set_seed(42)

        self.args = args
        self.vqvae_args = args.vqvae
        self.training_args = args.train
        self.dataset_args = args.dataset
        self.eval_args = args.eval_model
        self.dataset_name = args.dataset.dataset_name
        self.enable_var_len = self.dataset_args.var_len
        self.num_train_steps = self.training_args.num_train_iters
        self.num_stages = self.training_args.num_stages
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.register_buffer("steps", torch.Tensor([0]))

        self.vqvae_model = ConformerVQMotionModel(
            self.vqvae_args,
        ).to(self.device)
        total = sum(p.numel() for p in self.vqvae_model.parameters() if p.requires_grad)
        print("Total training params: %.2fM" % (total / 1e6))

        if self.vqvae_args.freeze_model:
            print("freezing encoder and decoder")
            for name, param in self.vqvae_model.motionEncoder.named_parameters():
                param.requires_grad = False
            for name, param in self.vqvae_model.motionDecoder.named_parameters():
                param.requires_grad = False

            total = sum(
                p.numel() for p in self.vqvae_model.parameters() if p.requires_grad
            )
            print("Total training params: %.2fM" % (total / 1e6))

        self.grad_accum_every = self.training_args.gradient_accumulation_steps

        self.loss_fnc = ReConsLoss(
            self.vqvae_args.recons_loss, self.vqvae_args.nb_joints
        )

        self.optim = get_optimizer(
            self.vqvae_model.parameters(),
            lr=self.training_args.learning_rate,
            wd=self.training_args.weight_decay,
        )

        self.lr_scheduler = get_scheduler(
            name=self.training_args.lr_scheduler_type,
            optimizer=self.optim,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=self.num_train_steps,
        )

        self.max_grad_norm = self.training_args.max_grad_norm

        if self.dataset_args.dataset_name == "mix":
            train_ds, sampler_train, weights_train = load_dataset(
                dataset_names=["t2m", "aist", "cm"],
                args=self.args,
                split="train",
                weight_scale=[1, 1, 1],
            )
            test_ds, _, _ = load_dataset(
                dataset_names=["t2m", "aist", "cm"], args=self.args, split="test"
            )
            self.render_ds, _, _ = load_dataset(
                dataset_names=["t2m", "aist", "cm"], args=self.args, split="render"
            )

            # if self.is_main:
            self.print(
                f"training with training {len(train_ds)} and test dataset of  and  {len(test_ds)} samples and reder of  {len(self.render_ds)}"
            )

        else:
            train_ds, sampler_train, weights_train = load_dataset(
                [self.dataset_args.dataset_name], self.args, "train"
            )
            test_ds, _, _ = load_dataset(
                [self.dataset_args.dataset_name], self.args, "test"
            )
            self.render_ds, _, _ = load_dataset(
                [self.dataset_args.dataset_name], self.args, "render"
            )

            # if self.is_main:
            self.print(
                f"training with training {len(train_ds)} and test dataset of  and  {len(test_ds)} samples and render of  {len(self.render_ds)}"
            )

        # dataloader
        collate_fn = MotionCollator() if self.enable_var_len else None

        self.dl = DATALoader(
            train_ds,
            batch_size=self.training_args.train_bs,
            sampler=sampler_train,
            shuffle=False if sampler_train else True,
            collate_fn=collate_fn,
        )
        self.valid_dl = DATALoader(
            test_ds,
            batch_size=self.training_args.eval_bs,
            shuffle=False,
            collate_fn=collate_fn,
        )
        self.render_dl = DATALoader(
            self.render_ds, batch_size=1, shuffle=False, collate_fn=collate_fn
        )

        self.dl_iter = cycle(self.dl)
        # self.valid_dl_iter = cycle(self.valid_dl)

        self.renderer = Renderer(self.device)

        self.save_model_every = self.training_args.save_steps
        self.log_losses_every = self.training_args.logging_steps
        self.evaluate_every = self.training_args.evaluate_every
        self.calc_metrics_every = self.training_args.evaluate_every
        self.wandb_every = self.training_args.wandb_every

        self.best_fid = float("inf")
        self.best_div = float("-inf")
        self.best_top1 = float("-inf")
        self.best_top2 = float("-inf")
        self.best_top3 = float("-inf")
        self.best_matching = float("inf")

        # if self.is_main:
        wandb.login()
        wandb.init(project=self.model_name)

    def print(self, msg):
        # self.accelerator.print(msg)
        print(msg)

    @property
    def device(self):
        return torch.device("cuda")

    # self.accelerator.device

    @property
    def is_distributed(self):
        return not (
            self.accelerator.distributed_type == DistributedType.NO
            and self.accelerator.num_processes == 1
        )

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def save(self, path, loss=None):
        pkg = dict(
            model=self.vqvae_model.state_dict(),
            optim=self.optim.state_dict(),
            steps=self.steps,
            total_loss=self.best_loss if loss is None else loss,
        )
        # self.accelerator.wait_for_everyone()
        # self.accelerator.save_model(pkg, path)
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location="cuda")
        self.vqvae_model.vq._codebook.batch_mean = pkg["model"][
            "vq._codebook.batch_mean"
        ]
        self.vqvae_model.vq._codebook.batch_variance = pkg["model"][
            "vq._codebook.batch_variance"
        ]

        self.vqvae_model.load_state_dict(pkg["model"])
        # self.vqvae_model = self.vqvae_model.to(self.device)

        self.optim.load_state_dict(pkg["optim"])
        self.steps = pkg["steps"]
        self.best_loss = pkg["total_loss"]

    def train_step(self):
        steps = int(self.steps.item())

        log_losses = self.log_losses_every > 0 and not (steps % self.log_losses_every)

        self.vqvae_model = self.vqvae_model.train()

        # logs

        logs = {}

        # for idx in range(self.grad_accum_every):
        #     batch = next(self.dl_iter)
        #     gt_motion = batch["motion"]
        #     print(gt_motion.device, self.vqvae_model.device)

        #     if idx != (self.grad_accum_every - 1):
        #         with self.accelerator.no_sync(self.vqvae_model):
        #             pred_motion, indices, commit_loss = self.vqvae_model(gt_motion)
        #             loss_motion = self.loss_fnc(pred_motion, gt_motion)
        #             loss_vel = self.loss_fnc.forward_vel(pred_motion, gt_motion)
        #             loss = (
        #                 loss_motion
        #                 + self.vqvae_args.commit * commit_loss
        #                 + self.vqvae_args.loss_vel * loss_vel
        #             )

        #             self.accelerator.backward(loss)

        #     else:
        #         pred_motion, indices, commit_loss = self.vqvae_model(gt_motion)
        #         loss_motion = self.loss_fnc(pred_motion, gt_motion)
        #         loss_vel = self.loss_fnc.forward_vel(pred_motion, gt_motion)
        #         loss = (
        #             loss_motion
        #             + self.vqvae_args.commit * commit_loss
        #             + self.vqvae_args.loss_vel * loss_vel
        #         )

        #         self.accelerator.backward(loss)

        #         self.optim.step()
        #         self.lr_scheduler.step()
        #         self.optim.zero_grad()

        # used_indices = indices.flatten().tolist()
        # for ui in used_indices:
        #     self.codebook_indices_usage[f"{ui}"] += 1
        # usage = (
        #     len([k for k, v in self.codebook_indices_usage.items() if v != 0])
        #     / self.vqvae_args.codebook_size
        # )

        # # accum_log(
        # logs = (
        #     dict(
        #         loss=loss.detach().cpu(),
        #         loss_motion=loss_motion.detach().cpu(),
        #         loss_vel=loss_vel.detach().cpu(),
        #         commit_loss=commit_loss.detach().cpu(),
        #         usage=usage,
        #     ),
        # )
        # # )

        for _ in range(self.grad_accum_every):
            batch = next(self.dl_iter)

            gt_motion = batch["motion"].to(self.device)
            # print(self.vqvae_model.device, self.vqvae_model.device)

            pred_motion, indices, commit_loss = self.vqvae_model(gt_motion)
            loss_motion = self.loss_fnc(pred_motion, gt_motion)
            loss_vel = self.loss_fnc.forward_vel(pred_motion, gt_motion)
            # print(loss_motion.shape, loss_vel.shape, commit_loss.shape)

            loss = (
                self.vqvae_args.loss_motion * loss_motion
                + self.vqvae_args.commit * commit_loss
                + self.vqvae_args.loss_vel * loss_vel
            ) / self.grad_accum_every

            used_indices = indices.flatten().tolist()
            # for ui in used_indices:
            #     self.codebook_indices_usage[f"{ui}"] += 1
            usage = len(set(used_indices)) / self.vqvae_args.codebook_size

            # print(loss,loss.shape)

            # self.accelerator.backward(loss / self.grad_accum_every)
            loss.backward()

            accum_log(
                logs,
                dict(
                    loss=loss.detach().cpu(),
                    loss_motion=loss_motion.detach().cpu() / self.grad_accum_every,
                    loss_vel=loss_vel.detach().cpu() / self.grad_accum_every,
                    commit_loss=commit_loss.detach().cpu() / self.grad_accum_every,
                    usage=usage / self.grad_accum_every,
                ),
            )

        # if exists(self.max_grad_norm):
        #     self.accelerator.clip_grad_norm_(
        #         self.vqvae_model.parameters(), self.max_grad_norm
        #     )

        self.optim.step()
        self.lr_scheduler.step()
        self.optim.zero_grad()

        # build pretty printed losses

        losses_str = f"{steps}: vqvae model total loss: {logs['loss'].float():.3} reconstruction loss: {logs['loss_motion'].float():.3} loss_vel: {logs['loss_vel'].float():.3} commitment loss: {logs['commit_loss'].float():.3} codebook usage: {logs['usage']}"

        # if log_losses:
        #     self.accelerator.log(
        #         {
        #             "total_loss": logs["loss"],
        #             "loss_motion": logs["loss_motion"],
        #             "loss_vel": logs["loss_vel"],
        #             "commit_loss": logs["commit_loss"],
        #             "average_max_length": logs["avg_max_length"],
        #             "codebook_usage": logs["usage"],
        #         },
        #         step=steps,
        #     )

        # log
        if steps % self.wandb_every == 0:
            for key, value in logs.items():
                wandb.log({f"train_loss/{key}": value})

            self.print(losses_str)

        if steps % self.evaluate_every == 0:
            self.validation_step()
            self.sample_render(os.path.join(self.output_dir, "samples"))

        # if self.is_main and steps % self.evaluate_every == 0:
        #     self.sample_render(os.path.join(self.output_dir, "samples"))

        # save model

        # if self.is_main and not (steps % self.save_model_every) and steps > 0:
        if not (steps % self.save_model_every):
            os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
            model_path = os.path.join(
                self.output_dir, "checkpoints", f"vqvae_motion.{steps}.pt"
            )
            self.save(model_path)

            if float(logs["loss"]) <= self.best_loss:
                model_path = os.path.join(self.output_dir, f"vqvae_motion.pt")
                self.save(model_path)
                self.best_loss = logs["loss"]

            self.print(
                f'{steps}: saving model to {str(os.path.join(self.output_dir , "checkpoints") )}'
            )

        self.steps += 1
        return logs

    def validation_step(self):
        self.vqvae_model.eval()
        val_loss_ae = {}
        all_loss = 0.0

        self.print(f"validation start")

        with torch.no_grad():
            for batch in tqdm(
                (self.valid_dl),
                position=0,
                leave=True,
                # disable=not self.accelerator.is_main_process,
            ):
                gt_motion = batch["motion"].to(self.device)

                pred_motion, indices, commit_loss = self.vqvae_model(gt_motion)
                # (
                #     all_pred_motion,
                #     all_commit_loss,
                #     all_gt_motion,
                # ) = self.accelerator.gather_for_metrics(
                #     (pred_motion, commit_loss, gt_motion)
                # )
                # loss_motion = self.loss_fnc(all_pred_motion, all_gt_motion)
                # loss_vel = self.loss_fnc.forward_vel(all_pred_motion, all_gt_motion)
                loss_motion = self.loss_fnc(pred_motion, gt_motion)
                loss_vel = self.loss_fnc.forward_vel(pred_motion, gt_motion)
                loss = (
                    loss_motion
                    + self.vqvae_args.commit * commit_loss
                    + self.vqvae_args.loss_vel * loss_vel
                )

                loss_dict = {
                    "total_loss": loss.detach().cpu(),
                    "loss_motion": loss_motion.detach().cpu(),
                    "loss_vel": loss_vel.detach().cpu(),
                    "commit_loss": commit_loss.detach().cpu(),
                }

                val_loss_ae.update(loss_dict)

                sums_ae = dict(Counter(val_loss_ae) + Counter(loss_dict))
                means_ae = {
                    k: sums_ae[k] / float((k in val_loss_ae) + (k in loss_dict))
                    for k in sums_ae
                }
                val_loss_ae.update(means_ae)

        for key, value in val_loss_ae.items():
            wandb.log({f"val_loss_vqgan/{key}": value})

        print(
            "val/rec_loss",
            val_loss_ae["loss_motion"],
        )
        print(
            f"val/total_loss ",
            val_loss_ae["total_loss"],
        )

        self.vqvae_model.train()

    def sample_render(self, save_path):
        save_file = os.path.join(save_path, f"{int(self.steps.item())}")
        os.makedirs(save_file, exist_ok=True)

        # assert self.render_dl.batch_size == 1 , "Batch size for rendering should be 1!"
        dataset_lens = self.render_ds.cumulative_sizes
        self.vqvae_model.eval()
        print(f"render start")
        with torch.no_grad():
            for idx, batch in tqdm(
                enumerate(self.render_dl),
            ):
                gt_motion = batch["motion"].to(self.device)
                name = str(batch["names"][0])

                curr_dataset_idx = np.searchsorted(dataset_lens, idx + 1)

                motion_len = int(batch.get("motion_lengths", [gt_motion.shape[1]])[0])

                gt_motion = gt_motion[:, :motion_len, :]

                pred_motion, _, _ = self.vqvae_model(gt_motion)

                gt_motion = (
                    self.render_ds.datasets[curr_dataset_idx]
                    .inv_transform(gt_motion.cpu())
                    .squeeze()
                    .float()
                )
                pred_motion = (
                    self.render_ds.datasets[curr_dataset_idx]
                    .inv_transform(pred_motion.cpu())
                    .squeeze()
                    .float()
                )

                self.renderer.render(
                    motion_vec=gt_motion,
                    outdir=save_path,
                    step=int(self.steps.item()),
                    name=f"{name}_gt",
                )
                self.renderer.render(
                    motion_vec=pred_motion,
                    outdir=save_path,
                    step=int(self.steps.item()),
                    name=f"{name}_pred",
                )

        self.vqvae_model.train()

    def train(self, resume=False, log_fn=noop):
        self.best_loss = float("inf")
        print(self.output_dir)

        if resume:
            save_path = "/srv/hays-lab/scratch/sanisetty3/music_motion/TGM3D/checkpoints/conformer_512_1024_affine/checkpoints/vqvae_motion.150000.pt"
            # chk = sorted(os.listdir(save_path), key=lambda x: int(x.split(".")[1]))[-1]
            print("resuming from ", save_path)
            self.load(save_path)

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print("training complete")
