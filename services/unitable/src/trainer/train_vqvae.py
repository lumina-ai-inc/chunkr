import math
import wandb
from pathlib import Path
from typing import Tuple, List, Union, Dict
from omegaconf import DictConfig
from hydra.utils import instantiate
import logging
import torch
import time
from functools import partial
from torch import nn, Tensor, autograd
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision.utils import make_grid

from src.utils import printer, compute_grad_norm

SNAPSHOT_KEYS = set(["EPOCH", "STEP", "OPTIMIZER", "LR_SCHEDULER", "MODEL", "LOSS"])


class VqvaeTrainer:
    def __init__(
        self,
        device: int,
        model: nn.Module,
        log: logging.Logger,
        exp_dir: Path,
        snapshot: Path = None,
        model_weights: Path = None,  # only for testing
    ) -> None:
        self.device = device
        self.log = log
        self.exp_dir = exp_dir
        assert (
            snapshot is None or model_weights is None
        ), "Snapshot and model weights cannot be set at the same time."

        self.model = model
        if snapshot is not None and snapshot.is_file():
            self.snapshot = self.load_snapshot(snapshot)
            self.model.load_state_dict(self.snapshot["MODEL"])
            self.start_epoch = self.snapshot["EPOCH"]
            self.global_step = self.snapshot["STEP"]
        elif model_weights is not None and model_weights.is_file():
            self.load_model(model_weights)
        else:
            self.snapshot = None
            self.start_epoch = 0

        self.model = self.model.to(device)
        self.model = DDP(self.model, device_ids=[device])

        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        torch.cuda.set_device(device)  # master gpu takes up extra memory
        torch.cuda.empty_cache()

    def train_epoch(
        self,
        epoch: int,
        starting_temp: float,
        anneal_rate: float,
        temp_min: float,
        grad_clip: float = None,
    ):
        start = time.time()
        total_loss = 0.0
        total_samples = 0

        # load data from dataloader
        for i, obj in enumerate(self.train_dataloader):
            if isinstance(obj, Tensor):
                img = obj.to(self.device)
            elif isinstance(obj, (list, tuple)):
                img = obj[0].to(self.device)
            else:
                raise ValueError(f"Unrecognized object type {type(obj)}")

            # temperature annealing
            self.temp = max(
                starting_temp * math.exp(-anneal_rate * self.global_step), temp_min
            )

            with autograd.detect_anomaly():
                loss, soft_recons = self.model(
                    img, return_loss=True, return_recons=True, temp=self.temp
                )

                self.optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=grad_clip
                    )
                self.optimizer.step()

            loss = loss.detach().cpu().data
            total_loss += loss * img.shape[0]
            total_samples += img.shape[0]

            self.lr_scheduler.step()
            self.global_step += 1

            if i % 10 == 0:
                grad_norm = compute_grad_norm(self.model)
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                self.log.info(
                    printer(
                        self.device,
                        f"Epoch {epoch} Step {i + 1}/{len(self.train_dataloader)} | Loss {loss:.4f} ({total_loss / total_samples:.4f}) | Grad norm {grad_norm:.3f} | {total_samples / elapsed:4.1f} images/s | lr {lr:5.1e} | Temp {self.temp:.2e}",
                    )
                )

            # visualize reconstruction images
            if i % 100 == 0 and self.device == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                k = 4  # num of images saved for visualization
                codes = self.model.module.get_codebook_indices(img[:k])
                hard_recons = self.model.module.decode(codes)

                img = img[:k].detach().cpu()
                soft_recons = soft_recons[:k].detach().cpu()
                codes = codes.flatten(start_dim=1).detach().cpu()
                hard_recons = hard_recons.detach().cpu()

                make_vis = partial(make_grid, nrow=int(math.sqrt(k)), normalize=True)
                img, soft_recons, hard_recons = map(
                    make_vis, (img, soft_recons, hard_recons)
                )

                log_info = {
                    "epoch": epoch,
                    "train_loss": loss,
                    "temperature": self.temp,
                    "learning rate": lr,
                    "original images": wandb.Image(
                        img, caption=f"step: {self.global_step}"
                    ),
                    "soft reconstruction": wandb.Image(
                        soft_recons, caption=f"step: {self.global_step}"
                    ),
                    "hard reconstruction": wandb.Image(
                        hard_recons, caption=f"step: {self.global_step}"
                    ),
                    "codebook_indices": wandb.Histogram(codes),
                }

                wandb.log(
                    log_info,
                    step=self.global_step,
                )

        return total_loss, total_samples

    def train(
        self,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        train_cfg: DictConfig,
        valid_cfg: DictConfig,
    ):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = instantiate(
            train_cfg.optimizer, params=self.model.parameters()
        )

        self.lr_scheduler = instantiate(
            train_cfg.lr_scheduler, optimizer=self.optimizer
        )

        if self.snapshot is not None:
            self.optimizer.load_state_dict(self.snapshot["OPTIMIZER"])
            self.lr_scheduler.load_state_dict(self.snapshot["LR_SCHEDULER"])

        best_loss = float("inf")
        self.model.train()
        self.global_step = 0
        # self.temp = train_cfg.starting_temp
        for epoch in range(self.start_epoch, train_cfg.epochs):
            train_dataloader.sampler.set_epoch(epoch)
            epoch_loss, epoch_samples = self.train_epoch(
                epoch,
                starting_temp=train_cfg.starting_temp,
                anneal_rate=train_cfg.temp_anneal_rate,
                temp_min=train_cfg.temp_min,
                grad_clip=train_cfg.grad_clip,
            )

            torch.cuda.empty_cache()

            valid_loss, valid_samples = self.valid(valid_cfg)

            # reduce loss to gpu 0
            training_info = torch.tensor(
                [epoch_loss, epoch_samples, valid_loss, valid_samples],
                device=self.device,
            )

            dist.reduce(
                training_info,
                dst=0,
                op=dist.ReduceOp.SUM,
            )

            if self.device == 0:
                grad_norm = compute_grad_norm(self.model)
                epoch_loss, epoch_samples, valid_loss, valid_samples = training_info
                epoch_loss, valid_loss = (
                    float(epoch_loss) / epoch_samples,
                    float(valid_loss) / valid_samples,
                )

                log_info = {
                    "train loss (epoch)": epoch_loss,
                    "valid loss (epoch)": valid_loss,
                    "train_samples": epoch_samples,
                    "valid_samples": valid_samples,
                    "grad_norm": grad_norm,
                }

                wandb.log(
                    log_info,
                    step=self.global_step,
                )

                if epoch % train_cfg.save_every == 0:
                    self.save_snapshot(epoch, best_loss)
                if valid_loss < best_loss:
                    self.save_model(epoch)
                    best_loss = valid_loss

    def valid(self, cfg: DictConfig):
        total_samples = 0
        total_loss = 0.0

        self.model.eval()
        for i, obj in enumerate(self.valid_dataloader):
            if isinstance(obj, Tensor):
                img = obj.to(self.device)
            elif isinstance(obj, (list, tuple)):
                img = obj[0].to(self.device)
            else:
                raise ValueError(f"Unrecognized object type {type(obj)}")

            with torch.no_grad():
                loss = self.model(
                    img, return_loss=True, return_recons=False, temp=self.temp
                )

            loss = loss.detach().cpu().data
            total_loss += loss * img.shape[0]
            total_samples += img.shape[0]

            if i % 10 == 0:
                self.log.info(
                    printer(
                        self.device,
                        f"Valid: Step {i + 1}/{len(self.valid_dataloader)} | Loss {loss:.4f} ({total_loss / total_samples:.4f})",
                    )
                )

        return total_loss, total_samples

    def save_model(self, epoch: int):
        filename = Path(self.exp_dir) / "model" / f"epoch{epoch}_model.pt"
        torch.save(self.model.module.state_dict(), filename)
        self.log.info(printer(self.device, f"Saving model to {filename}"))
        filename = Path(self.exp_dir) / "model" / f"best.pt"
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
