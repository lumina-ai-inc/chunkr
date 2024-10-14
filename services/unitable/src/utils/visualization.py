from torchvision import transforms
import numpy as np


def normalize_image_for_visualization(mean: float, std: float):
    invNormalization = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0] * 3, std=1.0 / np.array(std)),
            transforms.Normalize(mean=-1.0 * np.array(mean), std=[1.0] * 3),
        ]
    )

    return invNormalization
