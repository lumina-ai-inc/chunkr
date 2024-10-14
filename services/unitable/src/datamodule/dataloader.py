from typing import Any
from torch.utils.data import DataLoader, Dataset, Sampler
from functools import partial
import tokenizers as tk
import torch
from torch.utils.data import default_collate
from src.utils.mask_generator import MaskGenerator
from src.utils import (
    prepare_html_seq,
    prepare_cell_seq,
    prepare_bbox_seq,
)


class Collator:
    def __init__(
        self,
        vocab: tk.Tokenizer,
        max_seq_len: int,
        label_type: str,
    ) -> None:
        self.vocab = vocab
        self.vocab.enable_truncation(max_seq_len)
        self.label_type = label_type

    def __call__(self, batch) -> Any:
        return self._collate_batch(batch, self.vocab, self.label_type)

    def _collate_batch(
        self,
        batch: list[dict],
        vocab: tk.Tokenizer,
        label_type: str,
    ):
        if "cell" in label_type:
            image_list = [j for i in batch for j in i[0]]
        else:
            image_list = [i["image"] for i in batch]
        image_list = default_collate(image_list)

        if "cell" in label_type:
            filename = [(j["filename"], j["bbox_id"]) for i in batch for j in i[1]]
        else:
            filename = [i["filename"] for i in batch]
        label = dict(filename=filename)

        if "html" in label_type:
            html_list = ["".join(prepare_html_seq(i["html"])) for i in batch]
            label["html"] = vocab.encode_batch(html_list)

        if "cell" in label_type:
            cell_list = [
                " ".join(prepare_cell_seq(j["cell"])) for i in batch for j in i[1]
            ]
            label["cell"] = vocab.encode_batch(cell_list)

        if "bbox" in label_type:
            bbox_list = [" ".join(prepare_bbox_seq(i["bbox"])) for i in batch]
            label["bbox"] = vocab.encode_batch(bbox_list)

        return image_list, label


def generate_mask_for_batch_samples(
    batch, grid_size: int, num_mask_patches: int, min_num_patches: int
):
    N = len(batch)
    mg = MaskGenerator(
        input_size=grid_size,
        num_mask_patches=num_mask_patches,
        min_num_patches=min_num_patches,
    )
    mask_list = [mg() for _ in range(N)]
    return default_collate(batch), default_collate(mask_list)


def dataloader_vae(
    dataset: Dataset, batch_size: int, sampler: Sampler = None, **kwargs
) -> DataLoader:
    dataloader = DataLoader(
        dataset, batch_size, sampler=sampler, num_workers=8, pin_memory=True
    )

    return dataloader


def dataloader_beit(
    dataset: Dataset,
    grid_size: int,
    num_mask_patches: int,
    min_num_patches: int,
    batch_size: int,
    sampler: Sampler = None,
    **kwargs
):
    dataloader = DataLoader(
        dataset,
        batch_size,
        sampler=sampler,
        collate_fn=partial(
            generate_mask_for_batch_samples,
            grid_size=grid_size,
            num_mask_patches=num_mask_patches,
            min_num_patches=min_num_patches,
        ),
        num_workers=8,
        pin_memory=True,
    )

    return dataloader


def dataloader_html(
    dataset: Dataset,
    batch_size: int,
    vocab: tk.Tokenizer,
    max_seq_len: int,
    label_type: str,
    sampler=None,
) -> DataLoader:
    collate_fn = Collator(vocab, max_seq_len, label_type)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=sampler,
    )

    return dataloader
