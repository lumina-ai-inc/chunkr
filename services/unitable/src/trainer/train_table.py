from typing import Tuple, List, Union, Dict, Optional
import torch
import wandb
import json
import os
from torch import nn, Tensor, autograd
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from hydra.utils import instantiate
import logging
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
import tokenizers as tk
import torch.nn.functional as F

from src.trainer.utils import (
    Batch,
    configure_optimizer_weight_decay,
    turn_off_beit_grad,
    VALID_HTML_TOKEN,
    INVALID_CELL_TOKEN,
    VALID_BBOX_TOKEN,
)
from src.utils import (
    printer,
    compute_grad_norm,
    count_total_parameters,
    batch_autoregressive_decode,
    combine_filename_pred_gt,
)

SNAPSHOT_KEYS = set(["EPOCH", "STEP", "OPTIMIZER", "LR_SCHEDULER", "MODEL", "LOSS"])


class TableTrainer:
    """A trainer for table recognition. The supported tasks are:
    1) table structure extraction
    2) table cell bbox detection
    3) table cell content recognition

    Args:
    ----
        device: gpu id
        vocab: a vocab shared among all tasks
        model: model architecture
        log: logger
        exp_dir: the experiment directory that saves logs, wandb files, model weights, and checkpoints (snapshots)
        snapshot: specify which snapshot to use, only used in training
        model_weights: specify which model weight to use, only used in testing
        beit_pretrained_weights: load SSL pretrained visual encoder
        freeze_beit_epoch: freeze beit weights for the first {freeze_beit_epoch} epochs
    """

    def __init__(
        self,
        device: int,
        vocab: tk.Tokenizer,
        model: nn.Module,
        log: logging.Logger,
        exp_dir: Path,
        snapshot: Path = None,
        model_weights: str = None,
        beit_pretrained_weights: str = None,
        freeze_beit_epoch: int = None,
    ) -> None:
        self.device = device
        self.log = log
        self.exp_dir = exp_dir
        self.vocab = vocab
        self.padding_idx = vocab.token_to_id("<pad>")
        self.freeze_beit_epoch = freeze_beit_epoch

        # loss for training html, cell
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.padding_idx)

        self.model = model

        if (
            beit_pretrained_weights is not None
            and Path(beit_pretrained_weights).is_file()
        ):
            self.load_pretrained_beit(Path(beit_pretrained_weights))

        assert (
            snapshot is None or model_weights is None
        ), "Cannot set snapshot and model_weights at the same time!"

        if snapshot is not None and snapshot.is_file():
            self.snapshot = self.load_snapshot(snapshot)
            self.model.load_state_dict(self.snapshot["MODEL"])
            self.start_epoch = self.snapshot["EPOCH"]
            self.global_step = self.snapshot["STEP"]
        elif model_weights is not None and Path(model_weights).is_file():
            self.load_model(Path(model_weights))
        else:
            self.snapshot = None
            self.start_epoch = 0
            self.global_step = 0

        if freeze_beit_epoch and freeze_beit_epoch > 0:
            self._freeze_beit()

        self.model = self.model.to(device)
        self.model = DDP(self.model, device_ids=[device])

        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        torch.cuda.set_device(device)  # master gpu takes up extra memory
        torch.cuda.empty_cache()

    def _freeze_beit(self):
        if self.start_epoch < self.freeze_beit_epoch:
            turn_off_beit_grad(self.model)
            self.log.info(
                printer(
                    self.device,
                    f"Lock SSL params for {self.freeze_beit_epoch} epochs (params: {count_total_parameters(self.model) / 1e6:.2f}M) - Current epoch {self.start_epoch + 1}",
                )
            )
        else:
            self.log.info(
                printer(
                    self.device,
                    f"Unlock all weights (params: {count_total_parameters(self.model) / 1e6:.2f}M) - Current epoch {self.start_epoch + 1}",
                )
            )

    def train_epoch(
        self,
        epoch: int,
        target: str,
        loss_weights: List[float],
        grad_clip: float = None,
    ):
        avg_loss = 0.0

        # load data from dataloader
        for i, obj in enumerate(self.train_dataloader):
            batch = Batch(device=self.device, target=target, vocab=self.vocab, obj=obj)

            with autograd.detect_anomaly():
                loss, _ = batch.inference(
                    self.model,
                    criterion=self.criterion,
                    criterion_bbox=self.criterion_bbox,
                    loss_weights=loss_weights,
                )

                total_loss = loss["total"]

                self.optimizer.zero_grad()
                total_loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=grad_clip
                    )
                self.optimizer.step()

            total_loss = total_loss.detach().cpu().data
            avg_loss += total_loss
            self.lr_scheduler.step()
            self.global_step += 1

            if i % 10 == 0:
                grad_norm = compute_grad_norm(self.model)
                lr = self.optimizer.param_groups[0]["lr"]
                # elapsed = time.time() - start

                loss_info = f"Loss {total_loss:.3f} ({avg_loss / (i + 1):.3f})"
                if not isinstance(loss["html"], int):
                    loss_info += f" Html {loss['html'].detach().cpu().data:.3f}"
                if not isinstance(loss["cell"], int):
                    loss_info += f" Cell {loss['cell'].detach().cpu().data:.3f}"
                if not isinstance(loss["bbox"], int):
                    loss_info += f" Bbox {loss['bbox'].detach().cpu().data:.3f}"
                self.log.info(
                    printer(
                        self.device,
                        f"Epoch {epoch} Step {i + 1}/{len(self.train_dataloader)} | {loss_info} | Grad norm {grad_norm:.3f} | lr {lr:5.1e}",
                    )
                )

                if i % 100 == 0 and self.device == 0:
                    log_info = {
                        "epoch": epoch,
                        "train_total_loss": total_loss,
                        "learning rate": lr,
                        "grad_norm": grad_norm,
                    }

                    wandb.log(
                        log_info,
                        step=self.global_step,
                    )

    def train(
        self,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        train_cfg: DictConfig,
        valid_cfg: DictConfig,
    ):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        # ensure correct weight decay: https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L215
        optim_params = configure_optimizer_weight_decay(
            self.model.module, weight_decay=train_cfg.optimizer.weight_decay
        )

        self.optimizer = instantiate(train_cfg.optimizer, optim_params)

        self.lr_scheduler = instantiate(
            train_cfg.lr_scheduler, optimizer=self.optimizer
        )

        if self.snapshot is not None:
            self.optimizer.load_state_dict(self.snapshot["OPTIMIZER"])
            self.lr_scheduler.load_state_dict(self.snapshot["LR_SCHEDULER"])

        self.criterion_bbox = None
        if "bbox" in train_cfg.target:
            tmp = [
                self.vocab.token_to_id(i)
                for i in VALID_BBOX_TOKEN[
                    : train_cfg.img_size[0] + 2
                ]  # +1 for <eos> +1 for bbox == img_size
            ]
            tmp = [1.0 if i in tmp else 0.0 for i in range(self.vocab.get_vocab_size())]
            self.criterion_bbox = nn.CrossEntropyLoss(
                weight=torch.tensor(tmp, device=self.device),
                ignore_index=self.padding_idx,
            )

        best_loss = float("inf")
        self.model.train()

        if self.freeze_beit_epoch and self.start_epoch < self.freeze_beit_epoch:
            max_epoch = self.freeze_beit_epoch
        else:
            max_epoch = train_cfg.epochs
        for epoch in range(self.start_epoch, max_epoch):
            train_dataloader.sampler.set_epoch(epoch)

            self.train_epoch(
                epoch,
                grad_clip=train_cfg.grad_clip,
                target=train_cfg.target,
                loss_weights=train_cfg.loss_weights,
            )

            torch.cuda.empty_cache()

            valid_loss = self.valid(valid_cfg)

            if self.device == 0:
                wandb.log(
                    {"valid loss (epoch)": valid_loss},
                    step=self.global_step,
                )

                if epoch % train_cfg.save_every == 0:
                    self.save_snapshot(epoch, best_loss)
                if valid_loss < best_loss:
                    self.save_model(epoch)
                    best_loss = valid_loss

    def valid(self, cfg: DictConfig):
        total_loss = 0.0
        avg_loss = 0.0
        total_samples = 0

        self.model.eval()
        for i, obj in enumerate(self.valid_dataloader):
            batch = Batch(
                device=self.device, target=cfg.target, vocab=self.vocab, obj=obj
            )
            with torch.no_grad():
                loss, _ = batch.inference(
                    self.model,
                    criterion=self.criterion,
                    criterion_bbox=self.criterion_bbox,
                    loss_weights=cfg.loss_weights,
                )

            total_loss = loss["total"]
            total_loss = total_loss.detach().cpu().data
            avg_loss += total_loss * batch.image.shape[0]
            total_samples += batch.image.shape[0]

            if i % 10 == 0:
                loss_info = f"Loss {total_loss:.3f} ({avg_loss / total_samples:.3f})"
                if not isinstance(loss["html"], int):
                    loss_info += f" Html {loss['html'].detach().cpu().data:.3f}"
                if not isinstance(loss["cell"], int):
                    loss_info += f" Cell {loss['cell'].detach().cpu().data:.3f}"
                if not isinstance(loss["bbox"], int):
                    loss_info += f" Bbox {loss['bbox'].detach().cpu().data:.3f}"
                self.log.info(
                    printer(
                        self.device,
                        f"Valid: Step {i + 1}/{len(self.valid_dataloader)} | {loss_info}",
                    )
                )

        return avg_loss / total_samples

    def test(self, test_dataloader: DataLoader, cfg: DictConfig, save_to: str):
        total_result = dict()
        for i, obj in enumerate(test_dataloader):
            batch = Batch(
                device=self.device, target=cfg.target, vocab=self.vocab, obj=obj
            )

            if cfg.target == "html":
                prefix = [self.vocab.token_to_id("[html]")]
                valid_token_whitelist = [
                    self.vocab.token_to_id(i) for i in VALID_HTML_TOKEN
                ]
                valid_token_blacklist = None
            elif cfg.target == "cell":
                prefix = [self.vocab.token_to_id("[cell]")]
                valid_token_whitelist = None
                valid_token_blacklist = [
                    self.vocab.token_to_id(i) for i in INVALID_CELL_TOKEN
                ]
            elif cfg.target == "bbox":
                prefix = [self.vocab.token_to_id("[bbox]")]
                valid_token_whitelist = [
                    self.vocab.token_to_id(i)
                    for i in VALID_BBOX_TOKEN[: cfg.img_size[0]]
                ]
                valid_token_blacklist = None
            else:
                raise NotImplementedError

            pred_id = batch_autoregressive_decode(
                device=self.device,
                model=self.model,
                batch_data=batch,
                prefix=prefix,
                max_decode_len=cfg.max_seq_len,
                eos_id=self.vocab.token_to_id("<eos>"),
                valid_token_whitelist=valid_token_whitelist,
                valid_token_blacklist=valid_token_blacklist,
                sampling=cfg.sampling,
            )

            if cfg.target == "html":
                result = combine_filename_pred_gt(
                    filename=batch.name,
                    pred_id=pred_id,
                    gt_id=batch.html_tgt,
                    vocab=self.vocab,
                    type="html",
                )
            elif cfg.target == "cell":
                result = combine_filename_pred_gt(
                    filename=batch.name,
                    pred_id=pred_id,
                    gt_id=batch.cell_tgt,
                    vocab=self.vocab,
                    type="cell",
                )
            elif cfg.target == "bbox":
                result = combine_filename_pred_gt(
                    filename=batch.name,
                    pred_id=pred_id,
                    gt_id=batch.bbox_tgt,
                    vocab=self.vocab,
                    type="bbox",
                )
            else:
                raise NotImplementedError

            total_result.update(result)

            if i % 10 == 0:
                self.log.info(
                    printer(
                        self.device,
                        f"Test: Step {i + 1}/{len(test_dataloader)}",
                    )
                )

        self.log.info(
            printer(
                self.device,
                f"Converting {len(total_result)} samples to html tables ...",
            )
        )

        with open(
            os.path.join(save_to, cfg.save_to_prefix + f"_{self.device}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(total_result, f, indent=4)

        return total_result

    def save_model(self, epoch: int):
        filename = Path(self.exp_dir) / "model" / f"epoch{epoch}_model.pt"
        torch.save(self.model.module.state_dict(), filename)
        self.log.info(printer(self.device, f"Saving model to {filename}"))
        filename = Path(self.exp_dir) / "model" / "best.pt"
        torch.save(self.model.module.state_dict(), filename)

    def load_model(self, path: Union[str, Path]):
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.log.info(printer(self.device, f"Loading model from {path}"))

    def save_snapshot(self, epoch: int, best_loss: float):
        state_info = {
            "EPOCH": epoch + 1,
            "STEP": self.global_step,
            "OPTIMIZER": self.optimizer.state_dict(),
            "LR_SCHEDULER": self.lr_scheduler.state_dict(),
            "MODEL": self.model.module.state_dict(),
            "LOSS": best_loss,
        }

        snapshot_path = Path(self.exp_dir) / "snapshot" / f"epoch{epoch}_snapshot.pt"
        torch.save(state_info, snapshot_path)

        self.log.info(printer(self.device, f"Saving snapshot to {snapshot_path}"))

    def load_snapshot(self, path: Path):
        self.log.info(printer(self.device, f"Loading snapshot from {path}"))
        snapshot = torch.load(path, map_location="cpu")
        assert SNAPSHOT_KEYS.issubset(snapshot.keys())
        return snapshot

    def load_pretrained_beit(self, path: Path):
        self.log.info(printer(self.device, f"Loading pretrained BEiT from {path}"))
        beit = torch.load(path, map_location="cpu")
        redundant_keys_in_beit = [
            "cls_token",
            "mask_token",
            "generator.weight",
            "generator.bias",
        ]
        for key in redundant_keys_in_beit:
            if key in beit:
                del beit[key]

        # max_seq_len in finetuning may go beyond the length in pretraining
        if (
            self.model.pos_embed.embedding.weight.shape[0]
            != beit["pos_embed.embedding.weight"].shape[0]
        ):
            emb_shape = self.model.pos_embed.embedding.weight.shape
            ckpt_emb = beit["pos_embed.embedding.weight"].clone()
            assert emb_shape[1] == ckpt_emb.shape[1]

            ckpt_emb = ckpt_emb.unsqueeze(0).permute(0, 2, 1)
            ckpt_emb = F.interpolate(ckpt_emb, emb_shape[0], mode="nearest")
            beit["pos_embed.embedding.weight"] = ckpt_emb.permute(0, 2, 1).squeeze()

        out = self.model.load_state_dict(beit, strict=False)

        # ensure missing keys are just token_embed, decoder, and generator
        missing_keys_prefix = ("token_embed", "decoder", "generator")
        for key in out[0]:
            assert key.startswith(
                missing_keys_prefix
            ), f"Key {key} should be loaded from BEiT, but missing in current state dict."
        assert len(out[1]) == 0, f"Unexpected keys from BEiT: {out[1]}"
