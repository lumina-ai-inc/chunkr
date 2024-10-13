import random
import math
from typing import Any
import numpy as np

"""
Code adapted from beit mask generator: https://github.com/microsoft/unilm/blob/ecff36188001e9b12a90b01bbbaf9058d2b8bda6/beit/masking_generator.py .
"""

__all__ = ["MaskGenerator"]


class MaskGenerator:
    def __init__(
        self,
        input_size: int,
        num_mask_patches: int,
        min_num_patches: int = 4,
        max_num_patches: int = None,
        min_aspect: float = 0.3,
        max_aspect: float = None,
    ) -> None:
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width

        self.num_mask_patches = num_mask_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = (
            num_mask_patches if max_num_patches is None else max_num_patches
        )

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_mask_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask: np.array, max_mask_patches: int) -> int:
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self) -> Any:
        mask = np.zeros((self.height, self.width), dtype=np.int32)
        mask_count = 0
        while mask_count < self.num_mask_patches:
            max_mask_patches = self.num_mask_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


if __name__ == "__main__":
    mg = MaskGenerator(input_size=14, num_mask_patches=75)
    mask = mg()
    print(mask)
    print(mg, mask.sum())
