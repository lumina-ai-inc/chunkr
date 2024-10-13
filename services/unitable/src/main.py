from typing import Any
import hydra
import logging
import os
import wandb
import torch
import tokenizers as tk
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, instantiate
from pathlib import Path
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from src.utils import printer, count_total_parameters

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    ddp_setup()
    device = int(os.environ["LOCAL_RANK"])
    cwd = Path(get_original_cwd())
    exp_dir = Path(os.getcwd())  # experiment directory

    if cfg.trainer.mode == "train":
        (exp_dir / "snapshot").mkdir(parents=True, exist_ok=True)
        (exp_dir / "model").mkdir(parents=True, exist_ok=True)
        if device == 0:
            wandb.init(project=cfg.wandb.project, name=cfg.name, resume=True)

    # vocab is used in finetuning, not in self-supervised pretraining
    vocab = None
    if cfg.vocab.need_vocab:
        log.info(
            printer(
                device,
                f"Loading {cfg.vocab.type} vocab from {(cwd / cfg.vocab.dir).resolve()}",
            )
        )
        vocab = tk.Tokenizer.from_file(str(cwd / cfg.vocab.dir))

    # dataset
    if cfg.trainer.mode == "train":
        log.info(printer(device, "Loading training dataset"))
        train_dataset = instantiate(cfg.dataset.train_dataset)

        log.info(printer(device, "Loading validation dataset"))
        valid_dataset = instantiate(cfg.dataset.valid_dataset)

        train_kwargs = {
            "dataset": train_dataset,
            "sampler": DistributedSampler(train_dataset),
            "vocab": vocab,
            "max_seq_len": cfg.trainer.max_seq_len,
        }

        valid_kwargs = {
            "dataset": valid_dataset,
            "sampler": DistributedSampler(valid_dataset),
            "vocab": vocab,
            "max_seq_len": cfg.trainer.max_seq_len,
        }

        train_dataloader = instantiate(cfg.trainer.train.dataloader, **train_kwargs)
        valid_dataloader = instantiate(cfg.trainer.valid.dataloader, **valid_kwargs)
    elif cfg.trainer.mode == "test":
        # load testing dataset, same as valid for ssl
        log.info(printer(device, "Loading testing dataset"))
        test_dataset = instantiate(cfg.dataset.test_dataset)

        test_kwargs = {
            "dataset": test_dataset,
            "sampler": DistributedSampler(test_dataset),
            "vocab": vocab,
            "max_seq_len": cfg.trainer.max_seq_len,
        }

        test_dataloader = instantiate(cfg.trainer.test.dataloader, **test_kwargs)

    # model
    log.info(printer(device, "Loading model ..."))
    model_name = str(cfg.model.model._target_).split(".")[-1]
    if model_name == "DiscreteVAE":
        model = instantiate(cfg.model.model)
    elif model_name == "BeitEncoder":
        max_seq_len = (
            cfg.trainer.trans_size[0] // cfg.model.backbone_downsampling_factor
        ) * (cfg.trainer.trans_size[1] // cfg.model.backbone_downsampling_factor)
        model = instantiate(
            cfg.model.model,
            max_seq_len=max_seq_len,
        )
        # load pretrained vqvae
        model_vqvae = instantiate(cfg.model.model_vqvae)

        log.info(printer(device, "Loading pretrained VQVAE model ..."))
        assert Path(
            cfg.trainer.vqvae_weights
        ).is_file(), f"VQVAE weights doesn't exist: {cfg.trainer.vqvae_weights}"
        model_vqvae.load_state_dict(
            torch.load(cfg.trainer.vqvae_weights, map_location="cpu")
        )
    elif model_name == "EncoderDecoder":
        max_seq_len = max(
            (cfg.trainer.img_size[0] // cfg.model.backbone_downsampling_factor)
            * (cfg.trainer.img_size[1] // cfg.model.backbone_downsampling_factor),
            cfg.trainer.max_seq_len,
        )  # for positional embedding
        model = instantiate(
            cfg.model.model,
            max_seq_len=max_seq_len,
            vocab_size=vocab.get_vocab_size(),
            padding_idx=vocab.token_to_id("<pad>"),
        )

    log.info(
        printer(device, f"Total parameters: {count_total_parameters(model) / 1e6:.2f}M")
    )

    # trainer
    log.info(printer(device, "Loading trainer ..."))
    trainer_name = str(cfg.trainer.trainer._target_).split(".")[-1]
    trainer_kwargs = {
        "device": device,
        "model": model,
        "log": log,
        "exp_dir": exp_dir,
        "snapshot": (
            exp_dir / "snapshot" / cfg.trainer.trainer.snapshot
            if cfg.trainer.trainer.snapshot
            else None
        ),
    }

    if trainer_name == "VqvaeTrainer":
        trainer = instantiate(cfg.trainer.trainer, **trainer_kwargs)
    elif trainer_name == "BeitTrainer":
        trainer_kwargs["model_vqvae"] = model_vqvae
        trainer = instantiate(cfg.trainer.trainer, **trainer_kwargs)
    elif trainer_name == "TableTrainer":
        trainer_kwargs["vocab"] = vocab
        trainer = instantiate(cfg.trainer.trainer, **trainer_kwargs)
    else:
        raise ValueError(f"The provided trainer type {trainer_name} is not supported.")

    if cfg.trainer.mode == "train":
        log.info(printer(device, "Training starts ..."))
        trainer.train(
            train_dataloader, valid_dataloader, cfg.trainer.train, cfg.trainer.valid
        )
    elif cfg.trainer.mode == "test":
        log.info(printer(device, "Evaluation starts ..."))
        save_to = exp_dir / cfg.name
        save_to.mkdir(parents=True, exist_ok=True)
        trainer.test(test_dataloader, cfg.trainer.test, save_to=save_to)
    else:
        raise NotImplementedError

    destroy_process_group()


def ddp_setup():
    init_process_group(backend="nccl")


if __name__ == "__main__":
    main()
