import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import io
from pathlib import Path
from pdf2image import convert_from_path
import tempfile
import time
from typing import Dict, Tuple
import shutil
import subprocess

from src.utils import needs_conversion
from src.models.ocr_model import BoundingBox


def to_base64(image: str) -> str:
    with open(image, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def convert_to_pdf(file: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp())
    if needs_conversion(file):
        subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', str(temp_dir), str(file)],
                       check=True, capture_output=True, text=True)
        return next(temp_dir.glob('*.pdf'))
    else:
        return file


def process_image(args):
    i, img, pil_format = args
    with io.BytesIO() as output:
        img.save(output, format=pil_format)
        img_base64 = base64.b64encode(output.getvalue()).decode('utf-8')
    return i, img_base64


async def convert_to_img(file: Path, density: int, extension: str = "png") -> Dict[int, str]:
    start_time = time.time()
    temp_dir = tempfile.mkdtemp()
    result = {}
    try:
        pdf_to_img_start = time.time()

        extension = extension.lower()
        if not extension.startswith('.'):
            extension = f'.{extension}'

        format_mapping = {'.jpg': 'JPEG', '.jpeg': 'JPEG',
                          '.png': 'PNG', '.tiff': 'TIFF'}
        pil_format = format_mapping.get(extension, extension[1:].upper())

        pdf_images = convert_from_path(
            str(file), dpi=density, fmt=pil_format)
        pdf_to_img_end = time.time()

        processing_start = time.time()
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, (i, img, pil_format))
                       for i, img in enumerate(pdf_images, start=1)]
            for future in as_completed(futures):
                i, img_base64 = future.result()
                result[i] = img_base64
        processing_end = time.time()

        print(f"PDF to Image time: {pdf_to_img_end - pdf_to_img_start:.2f}s")
        print(f"Processing time: {processing_end - processing_start:.2f}s")
        print(f"Total time: {time.time() - start_time:.2f}s")

        return result
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert to PDF: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Failed to convert image: {str(e)}")
    finally:
        shutil.rmtree(temp_dir)


def crop_image(input_path: Path, bounding_box: BoundingBox, extension: str = "png", quality: int = 100, resize: str = None) -> str:
    """
    Crop an image using OpenCV with GPU acceleration (if available), given the input path and a BoundingBox, and return as base64.
    This function creates a copy of the input image to ensure thread-safety.
    """
    try:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = Path(temp_dir) / f"input_copy{input_path.suffix}"
            shutil.copy2(input_path, temp_input)

            img = cv2.imread(str(temp_input))

            if use_gpu:
                img_gpu = cv2.cuda_GpuMat()
                img_gpu.upload(img)

            # Calculate cropping coordinates using the new BoundingBox
            left = round(bounding_box.left)
            top = round(bounding_box.top)
            right = round(bounding_box.left + bounding_box.width)
            bottom = round(bounding_box.top + bounding_box.height)

            if use_gpu:
                cropped_gpu = img_gpu[top:bottom, left:right]
                cropped = cropped_gpu.download()
            else:
                cropped = img[top:bottom, left:right]

            if resize:
                new_width, new_height = map(int, resize.split('x'))
                if use_gpu:
                    resized_gpu = cv2.cuda.resize(
                        cropped_gpu, (new_width, new_height))
                    cropped = resized_gpu.download()
                else:
                    cropped = cv2.resize(cropped, (new_width, new_height))

            encode_params = []
            if extension.lower() in ['jpg', 'jpeg']:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif extension.lower() == 'png':
                encode_params = [
                    cv2.IMWRITE_PNG_COMPRESSION, 9 - quality // 10]

            _, buffer = cv2.imencode(f'.{extension}', cropped, encode_params)
            img_str = base64.b64encode(buffer).decode('utf-8')

        return img_str
    except FileNotFoundError as e:
        raise e
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to crop image: {str(e)}")


def resize_image(image_path: Path, size: Tuple[int, int], extension: str = "png", quality: int = 100) -> str:
    """
    Resize an image to the specified width and height while maintaining the aspect ratio, and return as base64.
    Uses GPU acceleration if available.
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input file not found: {image_path}")

        use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0

        # Read the image
        img = cv2.imread(str(image_path))

        if use_gpu:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)

        original_height, original_width = img.shape[:2]

        target_width, target_height = size
        aspect_ratio = original_width / original_height

        if target_width / target_height > aspect_ratio:
            target_width = int(target_height * aspect_ratio)
        else:
            target_height = int(target_width / aspect_ratio)

        if use_gpu:
            resized_gpu = cv2.cuda.resize(
                gpu_img, (target_width, target_height))
            resized_img = resized_gpu.download()
        else:
            resized_img = cv2.resize(img, (target_width, target_height))

        encode_params = []
        if extension.lower() in ['jpg', 'jpeg']:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif extension.lower() == 'png':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - quality // 10]

        _, buffer = cv2.imencode(f'.{extension}', resized_img, encode_params)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return img_str
    except FileNotFoundError as e:
        raise e
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to resize image: {str(e)}")
