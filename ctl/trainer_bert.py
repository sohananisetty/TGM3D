from pathlib import Path

import os
import numpy as np
import torch
from torch import nn

from transformers import AdamW, get_scheduler
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import DistributedType
import wandb
import transformers

from core.optimizer import get_optimizer

from tqdm import tqdm
from collections import Counter
from core.datasets.dataset_loading_utils import load_dataset_bert
from core.datasets.motion_bert_dataset import DATALoader
from music_motion.TGM3D.core.models.motion_bert import MotionBERT
from core.models.albef.xbert import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig
from core.datasets.motion_bert_dataset import TokenizerParams, MotionCollator
from core.models.utils import mask_for_mlm
from core.models.text_encoders import Clip, T5, TextEncoderParams


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


class BERTTrainer(nn.Module):
    def __init__(
        self,
        args,
        accelerate_kwargs: dict = dict(),
    ):
        super().__init__()

        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[kwargs], **accelerate_kwargs)

        transformers.set_seed(42)

        self.args = args
        self.bert_args = args.bert
        self.training_args = args.train
        self.dataset_args = args.dataset
        self.eval_args = args.eval_model
        self.dataset_name = args.dataset.dataset_name
        self.model_name = args.bert_model_name
        self.num_train_steps = self.training_args.num_train_iters
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.register_buffer("steps", torch.Tensor([0]))

        self.bert_model = MotionBERT(self.args)
        self.text_encoder = self.bert_args.text_encoder(device=self.device)

        self.tokenizer_prams = TokenizerParams(
            pad_index=self.bert_model.bert_config.pad_token_id,
            vocab_size=self.bert_model.bert_config.vocab_size,
            model_max_length=self.bert_model.bert_config.max_position_embeddings,
        )

        self.pad_index = self.tokenizer_prams.pad_index

        total = sum(p.numel() for p in self.bert_model.parameters() if p.requires_grad)
        print("Total training params: %.2fM" % (total / 1e6))

        self.grad_accum_every = self.training_args.gradient_accumulation_steps

        self.mlm_loss_fnc = torch.nn.CrossEntropyLoss()

        self.optim = get_optimizer(
            self.bert_model.parameters(),
            lr=self.training_args.learning_rate,
            wd=self.training_args.weight_decay,
        )

        self.lr_scheduler = get_scheduler(
            name=self.training_args.lr_scheduler_type,
            optimizer=self.optim,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=self.num_train_steps,
        )

        # self.max_grad_norm = max_grad_norm

        if self.dataset_args.dataset_name == "mix":
            train_ds, sampler_train, weights_train = load_dataset_bert(
                dataset_names=["t2m", "aist", "cm"],
                args=self.args,
                split="train",
                weight_scale=[2, 1, 1],
            )
            test_ds, _, _ = load_dataset_bert(
                dataset_names=["t2m", "aist", "cm"], args=self.args, split="test"
            )

            # if self.is_main:
            self.print(
                f"training with training {len(train_ds)} and test dataset of  and  {len(test_ds)} samples"
            )

        else:
            train_ds, sampler_train, weights_train = load_dataset_bert(
                [self.dataset_args.dataset_name], self.args, "train"
            )
            test_ds, _, _ = load_dataset_bert(
                [self.dataset_args.dataset_name], self.args, "test"
            )

            self.print(
                f"training with training {len(train_ds)} and test dataset of  and  {len(test_ds)} samples"
            )

        collate_fn = MotionCollator(self.text_encoder)

        self.dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.training_args.train_bs,
            sampler=sampler_train,
            shuffle=False if sampler_train else True,
            collate_fn=collate_fn,
        )
        self.valid_dl = torch.utils.data.DataLoader(
            test_ds,
            batch_size=self.training_args.eval_bs,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # prepare with accelerator

        (
            self.bert_model,
            self.text_encoder,
            self.optim,
            self.dl,
            self.valid_dl,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.bert_model,
            self.text_encoder,
            self.optim,
            self.dl,
            self.valid_dl,
            self.lr_scheduler,
        )

        self.dl_iter = cycle(self.dl)
        # self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = self.training_args.save_steps
        self.log_losses_every = self.training_args.logging_steps
        self.evaluate_every = self.training_args.evaluate_every
        self.calc_metrics_every = self.training_args.evaluate_every
        self.wandb_every = self.training_args.wandb_every

        self.best_fid = float("inf")

        hps = {
            "learning_rate": self.training_args.learning_rate,
        }
        self.accelerator.init_trackers(f"{self.model_name}", config=hps)

        if self.is_main:
            wandb.login()
            wandb.init(project=self.model_name)

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

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
            model=self.accelerator.get_state_dict(self.bert_model),
            optim=self.optim.state_dict(),
            steps=self.steps,
            total_loss=self.best_loss if loss is None else loss,
        )
        torch.save(pkg, path)

    @property
    def unwrapped_vqvae_model(self):
        return self.accelerator.unwrap_model(self.bert_model)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location="cpu")

        self.unwrapped_vqvae_model.load_state_dict(pkg["model"])

        self.optim.load_state_dict(pkg["optim"])
        self.steps = pkg["steps"]
        self.best_loss = pkg["total_loss"]

    def train_step(self):
        steps = int(self.steps.item())

        log_losses = self.log_losses_every > 0 and not (steps % self.log_losses_every)

        self.bert_model.train()

        logs = {}

        for _ in range(self.grad_accum_every - 1):
            data = next(self.dl_iter)

            with self.accelerator.no_sync(self.bert_model):
                input_ids = data["input_ids"]
                labels = input_ids.clone()

                mask_lm_output = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=data["attention_mask"],
                    labels=labels,
                    context=data["context"],
                    context_mask=data["context_mask"],
                    cond_drop_prob=1,
                )

                mask_loss = mask_lm_output.loss

                loss = self.bert_args.loss_mlm * mask_loss
                correct = (
                    mask_lm_output.logits.argmax(-1)[labels != -100]
                    == labels[labels != -100]
                )

                correct = correct.sum() / len(correct)

                self.accelerator.backward(loss)

                accum_log(
                    logs,
                    dict(
                        mask_loss=mask_loss.detach().cpu() / self.grad_accum_every,
                        correct=correct.cpu() / self.grad_accum_every,
                    ),
                )

        data = next(self.dl_iter)

        input_ids = data["input_ids"]
        labels = input_ids.clone()

        mask_lm_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=data["attention_mask"],
            labels=labels,
            context=data["context"],
            context_mask=data["context_mask"],
            cond_drop_prob=1,
        )

        mask_loss = mask_lm_output.loss

        loss = self.bert_args.loss_mlm * mask_loss
        correct = (
            mask_lm_output.logits.argmax(-1)[labels != -100] == labels[labels != -100]
        )

        correct = correct.sum() / len(correct)

        self.accelerator.backward(loss)

        accum_log(
            logs,
            dict(
                mask_loss=mask_loss.detach().cpu() / self.grad_accum_every,
                correct=correct.cpu() / self.grad_accum_every,
            ),
        )

        self.optim.step()
        self.lr_scheduler.step()
        self.optim.zero_grad()

        if log_losses:
            losses_str = f"{steps}: bert model loss:  mlm loss: {logs['mask_loss'].float():.3} correct: {correct.float():.3} "
            self.print(losses_str)
            self.accelerator.log(
                {
                    "mask_loss": logs["mask_loss"],
                },
                step=steps,
            )

        # log
        if self.is_main and (steps % self.wandb_every == 0):
            for key, value in logs.items():
                wandb.log({f"train_loss/{key}": value})

        if self.is_main and (steps % self.evaluate_every == 0):
            self.validation_step()

        # save model

        if self.is_main and not (steps % self.save_model_every) and steps > 0:
            os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
            model_path = os.path.join(
                self.output_dir, "checkpoints", f"bert_motion.{steps}.pt"
            )
            self.save(model_path)

            if float(logs["loss"]) <= self.best_loss:
                model_path = os.path.join(self.output_dir, f"bert_motion.pt")
                self.save(model_path)
                self.best_loss = logs["loss"]

            self.print(
                f'{steps}: saving model to {str(os.path.join(self.output_dir , "checkpoints") )}'
            )

        self.steps += 1
        return logs

    def validation_step(self):
        self.bert_model.eval()
        val_loss_ae = {}

        print(f"validation start")

        with torch.no_grad():
            for data in tqdm((self.valid_dl), position=0, leave=True):
                input_ids = data["input_ids"]
                labels = input_ids.clone()

                mask_lm_output = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=data["attention_mask"],
                    labels=labels,
                    context=data["context"],
                    context_mask=data["context_mask"],
                    cond_drop_prob=1,
                )
                mask_loss = mask_lm_output.loss

                correct = (
                    mask_lm_output.logits.argmax(-1)[labels != -100]
                    == labels[labels != -100]
                )

                correct = correct.sum() / len(correct)

                loss_dict = {
                    "mask_loss": mask_loss.detach().cpu(),
                    "correct": correct.cpu(),
                }

                val_loss_ae.update(loss_dict)

                sums_ae = dict(Counter(val_loss_ae) + Counter(loss_dict))
                means_ae = {
                    k: sums_ae[k] / float((k in val_loss_ae) + (k in loss_dict))
                    for k in sums_ae
                }
                val_loss_ae.update(means_ae)

        for key, value in val_loss_ae.items():
            wandb.log({f"val_loss_bert/{key}": value})

        print(
            f"val/total_loss ",
            val_loss_ae["total_loss"],
        )

        self.bert_model.train()

    def train(self, resume=False, log_fn=noop):
        self.best_loss = float("inf")
        print(self.output_dir)

        if resume:
            save_path = os.path.join(self.output_dir, "checkpoints")
            chk = sorted(os.listdir(save_path), key=lambda x: int(x.split(".")[1]))[-1]
            print("resuming from ", os.path.join(save_path, f"{chk}"))
            self.load(os.path.join(save_path, f"{chk}"))

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print("training complete")
