import math
import jsonlines
from pathlib import Path
from typing import Dict, Tuple, List, Union
from torch import Tensor, nn

__all__ = [
    "cosine_schedule_with_warmup",
    "load_json_annotations",
    "bbox_augmentation_resize",
    "count_total_parameters",
    "compute_grad_norm",
    "printer",
    "html_table_template",
]

printer = lambda device, output: f"[GPU {device}] " + output

html_table_template = (
    lambda table: f"""<html>
        <head> <meta charset="UTF-8">
        <style>
        table, th, td {{
            border: 1px solid black;
            font-size: 10px;
        }}
        </style> </head>
        <body>
        <table frame="hsides" rules="groups" width="100%%">
            {table}
        </table> </body> </html>"""
)


# adpated from https://github.com/huggingface/transformers/blob/v4.33.0/src/transformers/optimization.py
def cosine_schedule_with_warmup(
    step: int,
    *,
    warmup: int,
    min_ratio: float,
    total_step: int,
    cycle: float = 0.5,
):
    if step < warmup:
        if step == 0:
            step = 1
        return float(step) / float(max(1, warmup))

    if step >= total_step:
        step = total_step
    progress = float(step - warmup) / float(max(1, total_step - warmup))
    return max(
        min_ratio, 0.5 * (1.0 + math.cos(math.pi * float(cycle) * 2.0 * progress))
    )


def load_json_annotations(json_file_dir: Path, split: str):
    """Preprocess jsonl in dataset."""
    image_label_pair = list()
    with jsonlines.open(json_file_dir) as f:
        for obj in f:
            if obj["split"] == split:
                image_label_pair.append((obj["filename"], obj["html"]))

    return image_label_pair


def bbox_augmentation_resize(
    bbox: List[int], image_size: List[int], target_size: int
) -> List[int]:
    """Modify the bbox coordinates according to the image resizing."""
    # Assuming the bbox is [xmin, ymin, xmax, ymax]
    assert len(image_size) == 2
    ratio = [target_size / i for i in image_size]
    ratio = ratio * 2
    bbox = [int(round(i * j)) for i, j in zip(bbox, ratio)]
    return bbox


def count_total_parameters(model: nn.Module) -> int:
    """Count total parameters that need training."""
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_parameters


def compute_grad_norm(model: nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm
