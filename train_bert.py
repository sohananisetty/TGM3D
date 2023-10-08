from configs.config import cfg, get_cfg_defaults
from ctl.trainer_bert import BERTTrainer


def main(cfg):
    trainer = BERTTrainer(
        args=cfg,
    ).cuda()

    trainer.train(cfg.train.resume)


if __name__ == "__main__":
    path = "/srv/hays-lab/scratch/sanisetty3/music_motion/TGM3D/checkpoints/bert_12_768/bert_12_768.yaml"
    cfg = get_cfg_defaults()
    print("loading config from:", path)
    cfg.merge_from_file(path)
    cfg.freeze()
    print("output_dir: ", cfg.output_dir)

    main(cfg)


# accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train.py
# accelerate launch --mixed_precision=fp16 --num_processes=1 train_vqvae.py


# accelerate configuration saved at /nethome/sanisetty3/.cache/huggingface/accelerate/default_config.yaml
