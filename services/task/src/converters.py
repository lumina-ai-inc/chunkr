import subprocess
import os
import tempfile
import base64
from pathlib import Path
from typing import Dict
import shutil

from src.utils import needs_conversion
from src.models.ocr_model import BoundingBox


import time

def convert_to_img(file: Path, density: int, extension: str = "png") -> Dict[int, str]:
    start_time = time.time()
    temp_dir = tempfile.mkdtemp()
    result = {}
    try:
        conversion_start = time.time()
        if needs_conversion(file):
            subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', temp_dir, str(file)],
                           check=True, capture_output=True, text=True)
            pdf_file = next(Path(temp_dir).glob('*.pdf'))
        else:
            pdf_file = file
        conversion_end = time.time()
        
        output_pattern = os.path.join(temp_dir, f'output-%d.{extension}')
        magick_start = time.time()
        subprocess.run(['convert', 
                        '-density', str(density),
                        '-limit', 'thread', '8',  
                        '-define', 'opencl:device=gpu',
                        '-define', 'opencl:gpu-acceleration=on',
                        '-opencl', 'enable',
                        str(pdf_file),
                        '-background', 'white',
                        '-alpha', 'remove',
                        '-alpha', 'off',
                        output_pattern],
                    check=True, capture_output=True, text=True)
        magick_end = time.time()

        processing_start = time.time()
        for img_file in sorted(os.listdir(temp_dir)):
            if img_file.startswith('output-') and img_file.endswith(f".{extension}"):
                page_num = int(img_file.split('-')[1].split('.')[0]) + 1
                with open(os.path.join(temp_dir, img_file), 'rb') as img:
                    img_base64 = base64.b64encode(img.read()).decode('utf-8')
                    result[page_num] = img_base64
        processing_end = time.time()

        print(f"Conversion time: {conversion_end - conversion_start:.2f}s")
        print(f"ImageMagick time: {magick_end - magick_start:.2f}s")
        print(f"Processing time: {processing_end - processing_start:.2f}s")
        print(f"Total time: {time.time() - start_time:.2f}s")

        return result
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert image: {e.stderr}")
    finally:
        shutil.rmtree(temp_dir)


def crop_image(input_path: Path, bounding_box: BoundingBox, density: int = 300, extension: str = "png", quality: int = 100, resize: str = None) -> str:
    """
    Crop an image using ImageMagick, given the input path and a BoundingBox, and return as base64.
    This function creates a copy of the input image to ensure thread-safety.
    """
    try:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = Path(temp_dir) / f"input_copy{input_path.suffix}"
            shutil.copy2(input_path, temp_input)

            temp_output = Path(temp_dir) / f"output.{extension}"

            left = min(bounding_box.top_left[0], bounding_box.bottom_left[0])
            top = min(bounding_box.top_left[1], bounding_box.top_right[1])
            right = max(bounding_box.top_right[0],
                        bounding_box.bottom_right[0])
            bottom = max(
                bounding_box.bottom_left[1], bounding_box.bottom_right[1])

            width = right - left
            height = bottom - top

            crop_geometry = f"{width}x{height}+{left}+{top}"
            command = [
                'magick', 'convert',
                str(temp_input),
                '-density', str(density),
                '-crop', crop_geometry,
            ]

            if resize:
                command.extend(['-resize', resize])

            if extension.lower() in ['jpg', 'jpeg']:
                command.extend(['-quality', str(quality)])
            elif extension.lower() == 'png':
                command.extend(['-quality', str(quality), '-define',
                               f'png:compression-level={9 - quality // 10}'])

            command.append(str(temp_output))

            subprocess.run(command, capture_output=True, text=True, check=True)

            with open(temp_output, 'rb') as img_file:
                img_str = base64.b64encode(img_file.read()).decode('utf-8')

        return img_str
    except FileNotFoundError as e:
        raise e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ImageMagick error: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Failed to crop image: {str(e)}")
