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


class TableBank(Dataset):
    """tablebank recognition"""

    def __init__(
        self,
        root_dir: Union[Path, str],
        label_type: Literal["image"],
        split: Literal["train", "val", "test"],
        transform: transforms = None,
    ) -> None:
        super().__init__()

        assert label_type == "image", "No annotations"

        self.root_dir = Path(root_dir)
        self.label_type = label_type
        self.transform = transform
        self.image_list = os.listdir(self.root_dir / "images")

        if split == "val" or split == "test":
            self.image_list = self.image_list[:1000]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index: int) -> Any:
        name = self.image_list[index]
        img = Image.open(os.path.join(self.root_dir, "images", name))
        if self.transform:
            img = self.transform(img)

        if self.label_type == "image":
            return img
        else:
            raise ValueError("TableBank doesn't have HTML annotations.")
