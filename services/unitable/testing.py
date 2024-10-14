import concurrent.futures
import cv2
import dotenv
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import requests
import shutil
dotenv.load_dotenv(override=True)

rapidocr_url = os.getenv("RAPIDOCR_URL")
unitable_url = os.getenv("UNITABLE_URL")


def call_unitable_bbox(image_path):
    response = requests.post(f"{unitable_url}/bbox",
                             files={"image": open(image_path, "rb")})
    response.raise_for_status()
    return response.json()


def call_rapidocr(image_path):
    response = requests.post(f"{rapidocr_url}/ocr",
                             files={"file": open(image_path, "rb")})
    response.raise_for_status()
    return response.json()["result"]


def preprocess_image(image_path):
    image = Image.open(image_path)

    gray_image = image.convert('L')

    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced_image = enhancer.enhance(2.0)

    denoised_image = enhanced_image.filter(ImageFilter.MedianFilter(size=3))

    np_image = np.array(denoised_image)

    binary_image = cv2.adaptiveThreshold(
        np_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    preprocessed_image = Image.fromarray(binary_image)

    return preprocessed_image


def process_image(image_path, output_path, preprocess=True):
    input_paths = [image_path]
    output_paths = [output_path]

    ocr_image = image_path

    filename = os.path.basename(image_path)
    parent_dir = os.path.dirname(output_path)
    temp_output_path = os.path.join(parent_dir, f"temp_{filename}")
    temp_input_path = os.path.join(parent_dir, f"temp_input_{filename}")

    if preprocess:
        preprocessed_image = preprocess_image(image_path)
        preprocessed_image.save(temp_input_path)
        input_paths.append(temp_input_path)
        output_paths.append(temp_output_path)
        ocr_image = temp_input_path

    ocr_results = call_rapidocr(ocr_image)
    bboxes = call_unitable_bbox(ocr_image)

    for input_file, output_file in zip(input_paths, output_paths):
        print(f"path: {input_file}")
        image = Image.open(input_file)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 24)
        for i, bbox in enumerate(bboxes):
            left, top, right, bottom = bbox
            width = right - left
            height = bottom - top
            draw.rectangle([left, top, left + width, top + height],
                           outline="red", width=2)
            draw.text((left - 5, top), str(i), fill="red", font=font)

        for i, result in enumerate(ocr_results):
            bbox, text, confidence = result
            top_left, top_right, bottom_right, bottom_left = bbox
            left = top_left[0]
            top = top_left[1]
            width = bottom_right[0] - left
            height = bottom_right[1] - top
            draw.rectangle([left, top, left + width, top + height],
                           outline="blue", width=2)
            draw.text((left + width, top), str(i), fill="blue", font=font)
        image.save(output_file)
        image.close()


def main(input_dir, output_dir, preprocess=True):

    def process_image_wrapper(args):
        try:
            input_path, output_path = args
            process_image(input_path, output_path, preprocess)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

    image_paths = [os.path.join(input_dir, f) for f in os.listdir(
        input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    output_paths = [os.path.join(output_dir, f) for f in os.listdir(
        input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_image_wrapper, zip(image_paths, output_paths))


if __name__ == "__main__":
    input_dir = "./input"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    main(input_dir, output_dir, preprocess=True)
