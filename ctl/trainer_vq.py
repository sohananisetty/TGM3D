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
from core.datasets import dataset_TM_eval
from utils.eval_trans import evaluation_vqvae_loss, evaluation_vqvae

from core.models.conformer_vqvae import ConformerVQMotionModel
from core.models.conv_vqvae import ConvVQMotionModel
from utils.motion_processing.hml_process import recover_from_ric
from core.models.evaluator_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer

# from core.models.evaluator_wrapper import EvaluatorModelWrapper
from core.models.loss import ReConsLoss
from core.optimizer import get_optimizer
from torch import nn
from tqdm import tqdm
from transformers import AdamW, get_scheduler

# from utils.eval_trans import evaluation_vqvae, evaluation_vqvae_loss
from utils.vis_utils.render_final import Renderer
import utils.vis_utils.plot_3d_global as plot_3d
from core.models.utils import instantiate_from_config
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
        print(self.vqvae_args)

        self.vqvae_model = instantiate_from_config(self.vqvae_args).to(self.device)

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
            self.vqvae_args.recons_loss,
            self.vqvae_args.nb_joints,
            self.vqvae_args.motion_dim,
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
                dataset_names=["t2m", "aist", "cm", "moyo"],
                args=self.args,
                split="train",
                weight_scale=[2.0, 1, 1, 0.2],
            )
            test_ds, _, _ = load_dataset(
                dataset_names=["t2m", "aist", "cm", "moyo"],
                args=self.args,
                split="test",
            )
            self.render_ds, _, _ = load_dataset(
                dataset_names=["t2m", "aist", "cm", "moyo"],
                args=self.args,
                split="render",
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
            self.render_ds, batch_size=1, shuffle=False, collate_fn=None
        )

        self.dl_iter = cycle(self.dl)
        # self.valid_dl_iter = cycle(self.valid_dl)

        if self.vqvae_args.motion_dim == 263:
            self.w_vectorizer = WordVectorizer(
                "/srv/hays-lab/scratch/sanisetty3/music_motion/T2M-GPT/glove", "our_vab"
            )
            self.eval_wrapper = EvaluatorModelWrapper(self.eval_args)
            self.tm_eval = dataset_TM_eval.DATALoader(
                32,
                self.w_vectorizer,
                unit_length=4,
            )

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

        for _ in range(self.grad_accum_every):
            batch = next(self.dl_iter)

            gt_motion = batch["motion"][:, :, : self.vqvae_args.motion_dim].to(
                self.device
            )
            mask = batch.get("motion_mask", None)

            pred_motion, indices, commit_loss = self.vqvae_model(gt_motion, mask)
            loss_motion = self.loss_fnc(pred_motion, gt_motion, mask)
            loss_vel = self.loss_fnc.forward_vel(pred_motion, gt_motion, mask)

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

        self.optim.step()
        self.lr_scheduler.step()
        self.optim.zero_grad()

        # build pretty printed losses

        losses_str = f"{steps}: vqvae model total loss: {logs['loss'].float():.3} reconstruction loss: {logs['loss_motion'].float():.3} loss_vel: {logs['loss_vel'].float():.3}, commitment loss: {logs['commit_loss'].float():.3} codebook usage: {logs['usage']}"

        # log
        if steps % self.wandb_every == 0:
            for key, value in logs.items():
                wandb.log({f"train_loss/{key}": value})

            self.print(losses_str)

        if steps % self.evaluate_every == 0:
            self.validation_step()
            if self.vqvae_args.motion_dim == 263:
                self.calculate_metrics(steps, loss)
                self.sample_render_hmlvec(os.path.join(self.output_dir, "samples"))

            else:
                self.sample_render(os.path.join(self.output_dir, "samples"))

        # save model

        # if self.is_main and not (steps % self.save_model_every) and steps > 0:
        if not (steps % self.save_model_every):
            os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
            model_path = os.path.join(
                self.output_dir, "checkpoints", f"vqvae_motion.{steps}.pt"
            )
            self.save(model_path)
            print(float(logs["loss"]), self.best_loss)

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
                gt_motion = batch["motion"][:, :, : self.vqvae_args.motion_dim].to(
                    self.device
                )
                mask = batch.get("motion_mask", None)

                pred_motion, indices, commit_loss = self.vqvae_model(gt_motion, mask)

                loss_motion = self.loss_fnc(pred_motion, gt_motion, mask)
                loss_vel = self.loss_fnc.forward_vel(pred_motion, gt_motion, mask)

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
                gt_motion = batch["motion"][:, :, : self.vqvae_args.motion_dim].to(
                    self.device
                )
                mask = batch.get("motion_mask", None)
                name = str(batch["names"][0])

                curr_dataset_idx = np.searchsorted(dataset_lens, idx + 1)

                motion_len = int(batch.get("motion_lengths", [gt_motion.shape[1]])[0])

                gt_motion = gt_motion[:, :motion_len, :]

                pred_motion, _, _ = self.vqvae_model(gt_motion, mask)

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

    def sample_render_hmlvec(self, save_path):
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

                gt_motion = gt_motion

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

                gt_motion_xyz = recover_from_ric(gt_motion, 22)
                pred_motion_xyz = recover_from_ric(pred_motion, 22)

                gt_pose_vis = plot_3d.draw_to_batch(
                    gt_motion_xyz.numpy().squeeze()[None],
                    None,
                    [os.path.join(save_file, name + "_gt.gif")],
                )
                pred_pose_vis = plot_3d.draw_to_batch(
                    pred_motion_xyz.numpy().squeeze()[None],
                    None,
                    [os.path.join(save_file, name + "_pred.gif")],
                )

                # render(pred_motion_xyz, outdir=save_path, step=self.steps, name=f"{name}", pred=True)
                # render(gt_motion_xyz, outdir=save_path, step=self.steps, name=f"{name}", pred=False)

        self.vqvae_model.train()

    def calculate_metrics(self, steps, loss):
        (
            best_fid,
            best_iter,
            best_div,
            best_top1,
            best_top2,
            best_top3,
            best_matching,
        ) = evaluation_vqvae(
            best_fid=self.best_fid,
            best_div=self.best_div,
            best_top1=self.best_top1,
            best_top2=self.best_top2,
            best_top3=self.best_top3,
            best_matching=self.best_matching,
            val_loader=self.tm_eval,
            net=self.vqvae_model,
            nb_iter=steps,
            eval_wrapper=self.eval_wrapper,
            save=False,
        )
        if best_fid < self.best_fid:
            model_path = os.path.join(self.output_dir, f"vqvae_motion_best_fid.pt")
            self.save(model_path, loss=loss)

        wandb.log({f"best_fid": best_fid})
        wandb.log({f"best_div": best_div})
        wandb.log({f"best_top1": best_top1})
        wandb.log({f"best_top2": best_top2})
        wandb.log({f"best_top3": best_top3})
        wandb.log({f"best_matching": best_matching})

        (
            self.best_fid,
            self.best_iter,
            self.best_div,
            self.best_top1,
            self.best_top2,
            self.best_top3,
            self.best_matching,
        ) = (
            best_fid,
            best_iter,
            best_div,
            best_top1,
            best_top2,
            best_top3,
            best_matching,
        )

    def train(self, resume=False, log_fn=noop):
        self.best_loss = float("inf")
        print(self.output_dir)

        if resume:
            save_dir = self.args.output_dir
            # "/srv/hays-lab/scratch/sanisetty3/music_motion/TGM3D/checkpoints/conformer_512_1024_affine/checkpoints/vqvae_motion.150000.pt"
            save_path = os.path.join(save_dir, "vqvae_motion.pt")
            print("resuming from ", save_path)
            self.load(save_path)

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print("training complete")
