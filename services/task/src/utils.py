import cv2
import matplotlib.pyplot as plt
import mimetypes
import os
from paddleocr import draw_ocr
from pathlib import Path
import re
import subprocess

from src.models.segment_model import Segment, BoundingBox, BaseSegment


def check_imagemagick_installed():
    try:
        # Check ImageMagick version
        version_output = subprocess.run(
            ['magick', '-version'], check=True, capture_output=True, text=True)
        print(f"ImageMagick is installed: {version_output.stdout.strip()}")

        # Check for OpenCL support
        config_output = subprocess.run(
            ['magick', 'identify', '-list', 'configure'], check=True, capture_output=True, text=True)
        opencl_support = 'OpenCL' in config_output.stdout
        print(f"OpenCL support: {'Yes' if opencl_support else 'No'}")

        # Check if GPU is being used
        benchmark_output = subprocess.run(
            ['magick', 'benchmark', 'rose:', '-resize', '1000x1000', 'null:'], check=True, capture_output=True, text=True)
        gpu_match = re.search(r'(\d+\.\d+) fps \(GPU\)',
                              benchmark_output.stdout)

        if gpu_match:
            print(f"GPU acceleration is being used: {gpu_match.group(0)}")
        else:
            print("GPU acceleration is not being used")

    except subprocess.CalledProcessError as e:
        print(f"Error running ImageMagick command: {e}")
        raise RuntimeError("ImageMagick is not functioning correctly")
    except FileNotFoundError:
        raise RuntimeError(
            "ImageMagick is not installed or not in the system PATH")


def needs_conversion(file: Path) -> bool:
    mime_type, _ = mimetypes.guess_type(file)
    return mime_type in [
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-powerpoint',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation'
    ]


def save_ocr(img_path, out_path, result, font):
    os.makedirs(out_path, exist_ok=True)
    save_path = os.path.join(out_path, img_path.split('/')[-1] + 'output')

    image = cv2.imread(img_path)

    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]

    im_show = draw_ocr(image, boxes, txts, scores, font_path=font)

    cv2.imwrite(save_path, im_show)

    img = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
    plt.imshow(img)


def convert_base_segment_to_segment(base_segment: BaseSegment) -> Segment:
    """
    Convert a BaseSegment instance to a Segment instance.

    This function creates a new Segment object using the data from a BaseSegment,
    constructing a BoundingBox from the individual position attributes.
    """
    bbox = BoundingBox(
        top_left=[base_segment.left, base_segment.top],
        top_right=[base_segment.left + base_segment.width, base_segment.top],
        bottom_right=[base_segment.left + base_segment.width,
                      base_segment.top + base_segment.height],
        bottom_left=[base_segment.left, base_segment.top + base_segment.height]
    )

    return Segment(
        segment_id=base_segment.segment_id,
        bbox=bbox,
        page_number=base_segment.page_number,
        page_width=base_segment.page_width,
        page_height=base_segment.page_height,
        text=base_segment.text,
        type=base_segment.segment_type
    )
