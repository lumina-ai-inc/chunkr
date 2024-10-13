from typing import Any, Literal, Union
from pathlib import Path
import jsonlines
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os

from src.utils import load_json_annotations, bbox_augmentation_resize

# invalid data pairs: image_000000_1634629424.098128.png has 4 channels
INVALID_DATA = [
    {
        "dataset": "fintabnet",
        "split": "train",
        "image": "image_009379_1634631303.201671.png",
    },
    {
        "dataset": "marketing",
        "split": "train",
        "image": "image_000000_1634629424.098128.png",
    },
]


class Synthtabnet(Dataset):
    def __init__(
        self,
        root_dir: Union[Path, str],
        label_type: Literal["image", "html", "all"],
        split: Literal["train", "val", "test"],
        transform: transforms = None,
        json_html: Union[Path, str] = None,
        cell_limit: int = 100,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir) / "images"
        self.split = split
        self.label_type = label_type
        self.transform = transform
        self.cell_limit = cell_limit

        # SSP only needs image
        self.img_list = os.listdir(self.root_dir / self.split)
        if label_type != "image":
            self.image_label_pair = load_json_annotations(
                json_file_dir=Path(root_dir) / json_html, split=split
            )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index: int) -> Any:
        if self.label_type == "image":
            img = Image.open(self.root_dir / self.split / self.img_list[index])
            if self.transform:
                sample = self.transform(img)
            return sample
        else:
            obj = self.image_label_pair[index]
            img = Image.open(self.root_dir / self.split / obj[0])

            if self.label_type == "html":
                if self.transform:
                    img = self.transform(img)
                sample = dict(
                    filename=obj[0], image=img, html=obj[1]["structure"]["tokens"]
                )
                return sample
            elif self.label_type == "cell":
                bboxes_texts = [
                    (i["bbox"], "".join(i["tokens"]))
                    for idx, i in enumerate(obj[1]["cells"])
                    if "bbox" in i
                    and i["bbox"][0] < i["bbox"][2]
                    and i["bbox"][1] < i["bbox"][3]
                    and idx < self.cell_limit
                ]

                img_bboxes = [
                    self.transform(img.crop(bbox[0])) for bbox in bboxes_texts
                ]  # you can limit the total cropped cells to lower gpu memory

                text_bboxes = [
                    {"filename": obj[0], "bbox_id": i, "cell": j[1]}
                    for i, j in enumerate(bboxes_texts)
                ]
                return img_bboxes, text_bboxes
            else:
                img_size = img.size
                if self.transform:
                    img = self.transform(img)
                tgt_size = img.shape[-1]
                sample = dict(filename=obj[0], image=img)

                bboxes = [
                    entry["bbox"]
                    for entry in obj[1]["cells"]
                    if "bbox" in entry
                    and entry["bbox"][0] < entry["bbox"][2]
                    and entry["bbox"][1] < entry["bbox"][3]
                ]

                bboxes[:] = [
                    i
                    for entry in bboxes
                    for i in bbox_augmentation_resize(entry, img_size, tgt_size)
                ]

                sample["bbox"] = bboxes

                return sample
