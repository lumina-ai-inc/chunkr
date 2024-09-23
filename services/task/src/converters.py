import base64
import cv2
import io
from pathlib import Path
from pdf2image import convert_from_path
import tempfile
import time
from typing import Dict
import shutil
import subprocess

from src.utils import needs_conversion
from src.models.ocr_model import BoundingBox



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

        pdf_to_img_start = time.time()
        pdf_images = convert_from_path(str(pdf_file), dpi=density, fmt=extension)
        pdf_to_img_end = time.time()

        processing_start = time.time()
        for i, img in enumerate(pdf_images, start=1):
            with io.BytesIO() as output:
                img.save(output, format=extension.upper())
                img_base64 = base64.b64encode(output.getvalue()).decode('utf-8')
                result[i] = img_base64
        processing_end = time.time()

        print(f"Conversion time: {conversion_end - conversion_start:.2f}s")
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


def crop_image(input_path: Path, bounding_box: BoundingBox, density: int = 300, extension: str = "png", quality: int = 100, resize: str = None) -> str:
    """
    Crop an image using OpenCV with GPU acceleration (if available), given the input path and a BoundingBox, and return as base64.
    This function creates a copy of the input image to ensure thread-safety.
    """
    try:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Check if GPU is available
        use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = Path(temp_dir) / f"input_copy{input_path.suffix}"
            shutil.copy2(input_path, temp_input)

            # Read the image
            img = cv2.imread(str(temp_input))

            if use_gpu:
                img_gpu = cv2.cuda_GpuMat()
                img_gpu.upload(img)

            # Calculate crop coordinates
            left = min(bounding_box.top_left[0], bounding_box.bottom_left[0])
            top = min(bounding_box.top_left[1], bounding_box.top_right[1])
            right = max(bounding_box.top_right[0],
                        bounding_box.bottom_right[0])
            bottom = max(
                bounding_box.bottom_left[1], bounding_box.bottom_right[1])

            # Crop the image
            if use_gpu:
                cropped_gpu = img_gpu[top:bottom, left:right]
                cropped = cropped_gpu.download()
            else:
                cropped = img[top:bottom, left:right]

            # Resize if specified
            if resize:
                new_width, new_height = map(int, resize.split('x'))
                if use_gpu:
                    resized_gpu = cv2.cuda.resize(
                        cropped_gpu, (new_width, new_height))
                    cropped = resized_gpu.download()
                else:
                    cropped = cv2.resize(cropped, (new_width, new_height))

            # Apply quality settings
            encode_params = []
            if extension.lower() in ['jpg', 'jpeg']:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif extension.lower() == 'png':
                encode_params = [
                    cv2.IMWRITE_PNG_COMPRESSION, 9 - quality // 10]

            # Encode the image to bytes
            _, buffer = cv2.imencode(f'.{extension}', cropped, encode_params)
            img_str = base64.b64encode(buffer).decode('utf-8')

        return img_str
    except FileNotFoundError as e:
        raise e
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to crop image: {str(e)}")
