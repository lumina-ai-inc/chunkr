import concurrent.futures
import cv2
import dotenv
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import requests
from typing import List

dotenv.load_dotenv(override=True)

rapidocr_url = os.getenv("RAPIDOCR_URL")
unitable_url = os.getenv("UNITABLE_URL")


def call_unitable_structure(image_path):
    response = requests.post(f"{unitable_url}/structure",
                             files={"image": open(image_path, "rb")})
    response.raise_for_status()
    return response.json()


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


def html_table_template(table: str) -> str:
    return f"""<html>
        <head> <meta charset="UTF-8">
        <style>
        table, th, td {{
            border: 1px solid black;
            font-size: 10px;
        }}
        </style> </head>
        <body>
        <table frame="hsides" rules="groups" width="100%">
            {table}
        </table> </body> </html>"""


def build_table_from_html_and_cell(
    structure: List[str], content: List[str] = None
) -> List[str]:
    """Build table from html and cell token list"""
    assert structure is not None
    html_code = list()

    # deal with empty table
    if content is None:
        content = ["placeholder"] * len(structure)

    for tag in structure:
        if tag in ("<td>[]</td>", ">[]</td>"):
            if len(content) == 0:
                continue
            cell = content.pop(0)
            html_code.append(tag.replace("[]", cell))
        else:
            html_code.append(tag)

    return html_code

def map_paddle_text_onto_unitable(paddle_bbox, unitable_bbox):
    """Map paddle text onto unitable bbox"""
    def get_center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def point_inside_bbox(point, bbox):
        px, py = point
        x1, y1, x2, y2 = bbox
        return x1 <= px <= x2 and y1 <= py <= y2

    paddle_centers = [get_center(bbox) for bbox, text, _ in paddle_bbox]
    
    mapped_text = []

    for unitable_box in unitable_bbox:
        matching_texts = []
        for (bbox, text, _), center in zip(paddle_bbox, paddle_centers):
            if point_inside_bbox(center, unitable_box):
                matching_texts.append(text)
        
        combined_text = " ".join(matching_texts)
        mapped_text.append(combined_text)

    return mapped_text

def process_image(image_path, output_path, preprocess=True):
    input_paths = [image_path]
    output_paths = [output_path]

    ocr_image = image_path

    filename = os.path.basename(image_path)
    parent_dir = os.path.dirname(output_path)
    temp_output_path = os.path.join(parent_dir, f"temp_{filename}")
    temp_input_path = os.path.join(parent_dir, f"temp_input_{filename}")
    html_output_path = os.path.join(parent_dir, f"{filename}.html")

    if preprocess:
        preprocessed_image = preprocess_image(image_path)
        preprocessed_image.save(temp_input_path)
        input_paths.append(temp_input_path)
        output_paths.append(temp_output_path)
        ocr_image = temp_input_path

    ocr_results = call_rapidocr(ocr_image)
    bboxes = call_unitable_bbox(ocr_image)
    structure = call_unitable_structure(ocr_image)
    mapped_text = map_paddle_text_onto_unitable(ocr_results, bboxes)
    table = build_table_from_html_and_cell(structure, mapped_text)
    table = html_table_template(table)
    with open(html_output_path, "w") as f:
        f.write(table)
        
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
