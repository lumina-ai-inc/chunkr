import subprocess
import os
import tempfile
import base64
from pathlib import Path
from typing import Dict
from PIL import Image
import shutil
from io import BytesIO

from src.utils import needs_conversion


def convert_to_img(file: Path, density: int, extension: str = "png") -> Dict[int, str]:
    temp_dir = tempfile.mkdtemp()
    result = {}
    try:
        if needs_conversion(file):
            subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', temp_dir, str(file)],
                           check=True, capture_output=True, text=True)
            pdf_file = next(Path(temp_dir).glob('*.pdf'))
        else:
            pdf_file = file

        output_pattern = os.path.join(temp_dir, f'output-%d.{extension}')
        subprocess.run(['magick', str(pdf_file), '-density', str(density),
                        '-background', 'white', '-alpha', 'remove', '-alpha', 'off',
                        output_pattern],
                       check=True, capture_output=True, text=True)

        for img_file in sorted(os.listdir(temp_dir)):
            if img_file.startswith('output-') and img_file.endswith(f".{extension}"):
                page_num = int(img_file.split('-')[1].split('.')[0])
                with open(os.path.join(temp_dir, img_file), 'rb') as img:
                    img_base64 = base64.b64encode(img.read()).decode('utf-8')
                    result[page_num] = img_base64

        return result
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert image: {e.stderr}")
    finally:
        shutil.rmtree(temp_dir)


def crop_image(input_path: str, left: int, top: int, right: int, bottom: int, extension: str = "png") -> str:
    """
    Crop an image given the input path and crop coordinates, and return as base64.

    :param input_path: Path to the input image file
    :param left: Left coordinate of the crop box
    :param top: Top coordinate of the crop box
    :param right: Right coordinate of the crop box
    :param bottom: Bottom coordinate of the crop box
    :param extension: Output file extension (default: "png")
    :return: Base64 encoded string of the cropped image
    """
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with Image.open(input_path) as img:
            if left < 0 or top < 0 or right > img.width or bottom > img.height:
                raise ValueError("Invalid crop coordinates")

            cropped_img = img.crop((left, top, right, bottom))

            format_map = {
                "png": "PNG",
                "jpg": "JPEG",
                "jpeg": "JPEG"
            }

            img_format = format_map.get(extension.lower(), "PNG")

            buffer = BytesIO()
            cropped_img.save(buffer, format=img_format)
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return img_str
    except (FileNotFoundError, ValueError) as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"Failed to crop image: {str(e)}")
