import subprocess
import os
import tempfile
import base64
from pathlib import Path
from typing import Dict
from PIL import Image, UnidentifiedImageError
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
        subprocess.run(['magick', '-density', str(density), str(pdf_file),
                        '-background', 'white', '-alpha', 'remove', '-alpha', 'off',
                        output_pattern],
                       check=True, capture_output=True, text=True)

        for img_file in sorted(os.listdir(temp_dir)):
            if img_file.startswith('output-') and img_file.endswith(f".{extension}"):
                page_num = int(img_file.split('-')[1].split('.')[0]) + 1
                with open(os.path.join(temp_dir, img_file), 'rb') as img:
                    img_base64 = base64.b64encode(img.read()).decode('utf-8')
                    result[page_num] = img_base64

        return result
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert image: {e.stderr}")
    finally:
        shutil.rmtree(temp_dir)


def crop_image(input_path: Path, left: int, top: int, right: int, bottom: int, density: int = 300, extension: str = "png", quality: int = 100, resize: str = None) -> str:
    """
    Crop an image using ImageMagick, given the input path and crop coordinates, and return as base64.

    :param input_path: Path to the input image file
    :param left: Left coordinate of the crop box
    :param top: Top coordinate of the crop box
    :param right: Right coordinate of the crop box
    :param bottom: Bottom coordinate of the crop box
    :param density: Input image density (DPI)
    :param extension: Output file extension (default: "png")
    :param quality: Output image quality (0-100, default: 95)
    :param resize: Optional resize dimensions (e.g., "800x600")
    :return: Base64 encoded string of the cropped image
    """
    try:
        input_path = str(input_path)
        print(f"Cropping image: {input_path}")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Check file size and type
        file_size = os.path.getsize(input_path)
        file_type = subprocess.run(
            ['file', '-b', '--mime-type', input_path], capture_output=True, text=True).stdout.strip()
        print(f"File size: {file_size} bytes, File type: {file_type}")

        # Create a temporary file for the cropped image
        with tempfile.NamedTemporaryFile(suffix=f".{extension}", delete=False) as temp_file:
            temp_output_path = temp_file.name

        # Construct the ImageMagick command
        crop_geometry = f"{right - left}x{bottom - top}+{left}+{top}"
        command = [
            'magick', 'convert',
            input_path,
            '-density', str(density),
            '-crop', crop_geometry,
        ]

        # Add resize option if specified
        if resize:
            command.extend(['-resize', resize])

        # Add quality option
        if extension.lower() in ['jpg', 'jpeg']:
            command.extend(['-quality', str(quality)])
        elif extension.lower() == 'png':
            command.extend(['-quality', str(quality), '-define',
                           f'png:compression-level={9 - quality // 10}'])

        command.append(temp_output_path)

        # Run the ImageMagick command
        result = subprocess.run(
            command, capture_output=True, text=True, check=True)
        print(f"ImageMagick output: {result.stdout}")

        # Read the cropped image and convert to base64
        with open(temp_output_path, 'rb') as img_file:
            img_str = base64.b64encode(img_file.read()).decode('utf-8')

        # Clean up the temporary file
        os.unlink(temp_output_path)

        return img_str
    except FileNotFoundError as e:
        raise e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ImageMagick error: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Failed to crop image: {str(e)}")
