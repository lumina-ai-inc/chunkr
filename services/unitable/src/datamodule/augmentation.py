from typing import Tuple, Any, Optional, Union
from torch import Tensor
import random
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import datasets, transforms

from torchvision.transforms.transforms import _setup_size


_PIL_INTERPOLATION = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "hamming": Image.HAMMING,
}

get_interpolation = lambda method: _PIL_INTERPOLATION.get(method, Image.BILINEAR)


class RandomResizedCropAndInterpolationWithTwoPic(transforms.RandomResizedCrop):
    """Ensure both crops of vqvae and visual encoder have the same scale and size."""

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],  # transformer
        second_size: Union[int, Tuple[int, int]],  # vqvae
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: str = "bilinear",
        second_interpolation: str = "lanczos",
    ):
        self.second_size = _setup_size(
            second_size,
            error_msg="Please provide only two dimensions (h, w) for second size.",
        )

        if interpolation == "random":
            interpolation = random.choice(
                [get_interpolation("bilinear"), get_interpolation("bicubic")]
            )
        else:
            interpolation = get_interpolation(interpolation)
        self.second_interpolation = get_interpolation(second_interpolation)

        super().__init__(
            size=size, scale=scale, ratio=ratio, interpolation=interpolation
        )

    def forward(self, img: Image):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        out = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        out_second = F.resized_crop(
            img, i, j, h, w, self.second_size, self.second_interpolation
        )

        return out, out_second


class AugmentationForMIM(object):
    def __init__(
        self,
        mean: float,
        std: float,
        trans_size: Union[int, Tuple[int, int]],
        vqvae_size: Union[int, Tuple[int, int]],
        trans_interpolation: str,
        vqvae_interpolation: str,
    ) -> None:
        self.common_transform = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(p=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=trans_size,
                    second_size=vqvae_size,
                    interpolation=trans_interpolation,
                    second_interpolation=vqvae_interpolation,
                ),
            ]
        )

        self.trans_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

        self.vqvae_transform = transforms.ToTensor()

    def __call__(self, img: Image) -> Tuple[Tensor, Tensor]:
        trans_img, vqvae_img = self.common_transform(img)
        trans_img = self.trans_transform(trans_img)
        vqvae_img = self.vqvae_transform(vqvae_img)

        return trans_img, vqvae_img


if __name__ == "__main__":
    mean = [240.380, 240.390, 240.486]
    std = [45.735, 45.785, 45.756]

    T = RandomResizedCropAndInterpolationWithTwoPic(
        size=(256, 256),
        second_size=(256, 256),
        interpolation="bicubic",
        second_interpolation="lanczos",
    )

    print(T)
