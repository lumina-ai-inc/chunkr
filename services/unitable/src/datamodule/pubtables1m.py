from typing import Any, Literal, Union
from pathlib import Path
import jsonlines
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import json

from src.utils import bbox_augmentation_resize


class PubTables(Dataset):
    """PubTables-1M-Structure"""

    def __init__(
        self,
        root_dir: Union[Path, str],
        label_type: Literal["image", "cell", "bbox"],
        split: Literal["train", "val", "test"],
        transform: transforms = None,
        cell_limit: int = 100,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.split = split
        self.label_type = label_type
        self.transform = transform
        self.cell_limit = cell_limit

        tmp = os.listdir(self.root_dir / self.split)

        self.image_list = [i.split(".xml")[0] for i in tmp]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index: int) -> Any:
        name = self.image_list[index]
        img = Image.open(os.path.join(self.root_dir, "images", name + ".jpg"))

        if self.label_type == "image":
            if self.transform:
                img = self.transform(img)
            return img
        elif "bbox" in self.label_type:
            img_size = img.size
            if self.transform:
                img = self.transform(img)
            tgt_size = img.shape[-1]
            with open(
                os.path.join(self.root_dir, "words", name + "_words.json"), "r"
            ) as f:
                obj = json.load(f)

            obj[:] = [
                v
                for i in obj
                if "bbox" in i.keys()
                and all([i["bbox"][w + 2] > i["bbox"][w] for w in range(2)])
                for v in bbox_augmentation_resize(
                    [
                        min(max(i["bbox"][0], 0), img_size[0]),
                        min(max(i["bbox"][1], 0), img_size[1]),
                        min(max(i["bbox"][2], 0), img_size[0]),
                        min(max(i["bbox"][3], 0), img_size[1]),
                    ],
                    img_size,
                    tgt_size,
                )
            ]

            sample = {"filename": name, "image": img, "bbox": obj}
            return sample

        elif "cell" in self.label_type:
            img_size = img.size
            with open(
                os.path.join(self.root_dir, "words", name + "_words.json"), "r"
            ) as f:
                obj = json.load(f)

            bboxes_texts = [
                (i["bbox"], i["text"])
                for idx, i in enumerate(obj)
                if "bbox" in i
                and i["bbox"][0] < i["bbox"][2]
                and i["bbox"][1] < i["bbox"][3]
                and i["bbox"][0] >= 0
                and i["bbox"][1] >= 0
                and i["bbox"][2] < img_size[0]
                and i["bbox"][3] < img_size[1]
                and idx < self.cell_limit
            ]

            img_bboxes = [self.transform(img.crop(bbox[0])) for bbox in bboxes_texts]

            text_bboxes = [
                {"filename": name, "bbox_id": i, "cell": j[1]}
                for i, j in enumerate(bboxes_texts)
            ]
            return img_bboxes, text_bboxes
