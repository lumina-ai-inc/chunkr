from typing import List, Tuple, Dict
import torch
from torch import Tensor, nn
from torchtext.vocab import Vocab
import tokenizers as tk

from src.utils import pred_token_within_range, subsequent_mask
from src.vocab.constant import (
    HTML_TOKENS,
    TASK_TOKENS,
    RESERVED_TOKENS,
    BBOX_TOKENS,
)


VALID_HTML_TOKEN = ["<eos>"] + HTML_TOKENS
INVALID_CELL_TOKEN = (
    ["<sos>", "<pad>", "<empty>", "<sep>"] + TASK_TOKENS + RESERVED_TOKENS
)
VALID_BBOX_TOKEN = [
    "<eos>"
] + BBOX_TOKENS  # image size will be addressed after instantiation


class Batch:
    """Wrap up a batch of training samples with different training targets.
    The input is not torch tensor
    Shape of the image (src): B, S, E
    Shape of the text (tgt): B, N, S, E (M includes 1 table detection, 1 structure, 1 cell, and multiple bbox)
    Reshape text to (B * N, S, E) and inflate the image to match the shape of the text

    Args:
    ----
        device: gpu id
    """

    def __init__(
        self,
        device: torch.device,
        target: str,
        vocab: Vocab,
        obj: List,
    ) -> None:
        self.device = device
        self.image = obj[0].to(device)
        self.name = obj[1]["filename"]
        self.target = target
        self.vocab = vocab
        self.image_size = self.image.shape[-1]

        if "table" in target:
            raise NotImplementedError

        if "html" in target:
            self.valid_html_token = [vocab.token_to_id(i) for i in VALID_HTML_TOKEN]
            (
                self.html_src,
                self.html_tgt,
                self.html_casual_mask,
                self.html_padding_mask,
            ) = self._prepare_transformer_input(obj[1]["html"])

        if "cell" in target:
            self.invalid_cell_token = [vocab.token_to_id(i) for i in INVALID_CELL_TOKEN]
            (
                self.cell_src,
                self.cell_tgt,
                self.cell_casual_mask,
                self.cell_padding_mask,
            ) = self._prepare_transformer_input(obj[1]["cell"])

        if "bbox" in target:
            (
                self.bbox_src,
                self.bbox_tgt,
                self.bbox_casual_mask,
                self.bbox_padding_mask,
            ) = self._prepare_transformer_input(obj[1]["bbox"])

    def _prepare_transformer_input(
        self, seq: List[tk.Encoding]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        tmp = [i.ids for i in seq]
        tmp = torch.tensor(tmp, dtype=torch.int32)
        src = tmp[:, :-1].to(self.device)
        tgt = tmp[:, 1:].type(torch.LongTensor).to(self.device)
        casual_mask = subsequent_mask(src.shape[-1]).to(self.device)
        tmp = [i.attention_mask[:-1] for i in seq]  # padding mask
        tmp = torch.tensor(tmp, dtype=torch.bool)
        padding_mask = (~tmp).to(self.device)

        return src, tgt, casual_mask, padding_mask

    def _inference_one_task(
        self, model, memory, src, casual_mask, padding_mask, use_ddp
    ):
        if use_ddp:
            out = model.module.decode(memory, src, casual_mask, padding_mask)
            out = model.module.generator(out)
        else:
            out = model.decode(memory, src, casual_mask, padding_mask)
            out = model.generator(out)

        return out

    def inference(
        self,
        model: nn.Module,
        criterion: nn.Module,
        criterion_bbox: nn.Module = None,
        loss_weights: dict = None,
        use_ddp: bool = True,
    ) -> Tuple[Dict, Dict]:
        pred = dict()
        loss = dict(table=0, html=0, cell=0, bbox=0)

        if use_ddp:
            memory = model.module.encode(self.image)
        else:
            memory = model.encode(self.image)

        # inference + suppress invalid logits + compute loss
        if "html" in self.target:
            out_html = self._inference_one_task(
                model,
                memory,
                self.html_src,
                self.html_casual_mask,
                self.html_padding_mask,
                use_ddp,
            )

            pred["html"] = pred_token_within_range(
                out_html, white_list=self.valid_html_token
            ).permute(0, 2, 1)
            loss["html"] = criterion(pred["html"], self.html_tgt)

        if "cell" in self.target:
            out_cell = self._inference_one_task(
                model,
                memory,
                self.cell_src,
                self.cell_casual_mask,
                self.cell_padding_mask,
                use_ddp,
            )

            pred["cell"] = pred_token_within_range(
                out_cell, black_list=self.invalid_cell_token
            ).permute(0, 2, 1)
            loss["cell"] = criterion(pred["cell"], self.cell_tgt)

        if "bbox" in self.target:
            assert criterion_bbox is not None

            out_bbox = self._inference_one_task(
                model,
                memory,
                self.bbox_src,
                self.bbox_casual_mask,
                self.bbox_padding_mask,
                use_ddp,
            )
            pred["bbox"] = out_bbox.permute(0, 2, 1)
            loss["bbox"] = criterion_bbox(pred["bbox"], self.bbox_tgt)

        total = 0.0
        for k, v in loss_weights.items():
            total += loss[k] * v
        loss["total"] = total

        return loss, pred


def configure_optimizer_weight_decay(
    model: nn.Module, weight_decay: float
) -> List[Dict]:
    weight_decay_blacklist = (nn.LayerNorm, nn.BatchNorm2d, nn.Embedding)

    if hasattr(model, "no_weight_decay"):
        skip_list = model.no_weight_decay()
    decay = set()
    no_decay = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, weight_decay_blacklist):
                no_decay.add(fpn)
            elif pn in skip_list:
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    decay = param_dict.keys() - no_decay

    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]

    return optim_groups


def turn_off_beit_grad(model: nn.Module):
    "Freeze BEiT pretrained weights."
    for param in model.encoder.parameters():
        param.requires_grad = False

    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.pos_embed.parameters():
        param.requires_grad = False


def turn_on_beit_grad(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True
