import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import io
import time
import requests
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import concurrent.futures
from pdf2image import convert_from_path
import numpy as np
import json
from tabulate import tabulate
from sklearn.cluster import KMeans
try:
    import pytesseract
except ImportError:
    pytesseract = None
import pdfplumber

from tokenization_bros import BrosTokenizer

ANNOTATED_IMAGES_DIR = Path("annotated_images")
os.makedirs(ANNOTATED_IMAGES_DIR, exist_ok=True)


import pickle
import shutil
import numpy as np
class Rectangle:
    def __init__(self, left: int, top: int, right: int, bottom: int):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.width = self.right - self.left
        self.height = self.bottom - self.top

    def to_bbox(self):
        return [self.left, self.top, self.width, self.height]

    def get_intersection_percentage(self, other):
        intersection_area = max(0, min(self.right, other.right) - max(self.left, other.left)) * \
                              max(0, min(self.bottom, other.bottom) - max(self.top, other.top))
        return intersection_area / (self.width * self.height)

    def merge_rectangles(rectangles):
        left = min(rect["left"] for rect in rectangles)
        top = min(rect["top"] for rect in rectangles)
        right = max(rect["left"] + rect["width"] for rect in rectangles)
        bottom = max(rect["top"] + rect["height"] for rect in rectangles)
        return Rectangle(left, top, right, bottom)

tokenizer = BrosTokenizer.from_pretrained("naver-clova-ocr/bros-base-uncased")

def rectangle_to_bbox(rectangle: Rectangle):
    return rectangle.to_bbox()

def get_words_positions(text: str, rectangle: Rectangle):
    text = text.strip()
    text_len = len(text)

    width_per_letter = rectangle.width / text_len

    words_bboxes = [Rectangle(rectangle.left, rectangle.top, rectangle.left + 5, rectangle.bottom)]
    words_bboxes[-1].width = 0
    words_bboxes[-1].right = words_bboxes[-1].left

    for letter in text:
        if letter == " ":
            left = words_bboxes[-1].right + width_per_letter
            words_bboxes.append(Rectangle(left, words_bboxes[-1].top, left + 5, words_bboxes[-1].bottom))
            words_bboxes[-1].width = 0
            words_bboxes[-1].right = words_bboxes[-1].left
        else:
            words_bboxes[-1].right = words_bboxes[-1].right + width_per_letter
            words_bboxes[-1].width = words_bboxes[-1].width + width_per_letter

    words = text.split()
    return words, words_bboxes

def get_subwords_positions(word: str, rectangle: Rectangle):
    width_per_letter = rectangle.width / len(word)
    word_tokens = [x.replace("#", "") for x in tokenizer.tokenize(word)]

    if not word_tokens:
        return [], []

    ids = [x[-2] for x in tokenizer(word_tokens)["input_ids"]]

    right = rectangle.left + len(word_tokens[0]) * width_per_letter
    bboxes = [Rectangle(rectangle.left, rectangle.top, right, rectangle.bottom)]

    for subword in word_tokens[1:]:
        right = bboxes[-1].right + len(subword) * width_per_letter
        bboxes.append(Rectangle(bboxes[-1].right, rectangle.top, right, rectangle.bottom))

    return ids, bboxes

def get_grid_words_dict(text_data, bounding_boxes):
    texts, bbox_texts_list, inputs_ids, bbox_subword_list = [], [], [], []
    for text, bounding_box in zip(text_data, bounding_boxes):
        words, words_bboxes = get_words_positions(text, bounding_box)
        texts += words
        bbox_texts_list += [rectangle_to_bbox(r) for r in words_bboxes]
        for word, word_box in zip(words, words_bboxes):
            ids, subwords_bboxes = get_subwords_positions(word, word_box)
            inputs_ids += ids
            bbox_subword_list += [rectangle_to_bbox(r) for r in subwords_bboxes]

    return {
        "input_ids": np.array(inputs_ids),
        "bbox_subword_list": np.array(bbox_subword_list),
        "texts": texts,
        "bbox_texts_list": np.array(bbox_texts_list),
    }

def get_tesseract_ocr_data(pil_image):
    if not pytesseract:
        print("Tesseract OCR is not installed")
        return json.dumps({"data": []})
    print("Tesseract OCR is installed")
    start_time = time.time()
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    elapsed_time = time.time() - start_time
    print(f"Tesseract OCR processing time: {elapsed_time:.2f} seconds")

    ocr_words = []
    bounding_boxes = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        left = float(data["left"][i])
        top = float(data["top"][i])
        width = float(data["width"][i])
        height = float(data["height"][i])
        confidence = float(data["conf"][i]) if isinstance(data["conf"][i], str) and data["conf"][i].isdigit() else 0.0

        ocr_words.append({
            "bbox": {
                "left": left,
                "top": top,
                "width": width,
                "height": height
            },
            "text": text,
            "confidence": confidence
        })
        bounding_boxes.append(Rectangle(left, top, left + width, top + height))

    grid_dict = get_grid_words_dict([word["text"] for word in ocr_words], bounding_boxes)
    
    # Convert NumPy arrays to lists and structure data properly
    return json.dumps({
        "data": ocr_words  # Send the OCR words directly instead of grid_dict
    })

def get_pdfplumber_ocr_data(pdf_path):
    pdf = pdfplumber.open(pdf_path)
    ocr_words = []
    bounding_boxes = []
    for page in pdf.pages:
        words = page.extract_words()
        for word in words:
            ocr_words.append({
                "bbox": {
                    "left": word["x0"],
                    "top": word["top"],
                    "width": word["x1"] - word["x0"],
                    "height": word["bottom"] - word["top"]
                },
                "text": word["text"],
                "confidence": 1.0  # pdfplumber does not provide confidence
            })
            bounding_boxes.append(Rectangle(word["x0"], word["top"], word["x1"], word["bottom"]))
    return ocr_words, bounding_boxes

def  visualize_predictions(images, predictions, subfolder_path):
    class_labels = [
        "Caption","Footnote","Formula","ListItem","PageFooter","PageHeader",
        "Picture","SectionHeader","Table","Text","Title"
    ]
    for i, (image, pred_dict) in enumerate(zip(images, predictions)):
        pred_inst = pred_dict.get("instances", {})
        image_resized = image.resize((image.width, image.height))
        draw = ImageDraw.Draw(image_resized)
        boxes = pred_inst.get("boxes", [])
        scores = pred_inst.get("scores", [])
        classes = pred_inst.get("classes", [])
        print(f"visualiting image {i}")
        for order, (box, score, cls_idx) in enumerate(zip(boxes, scores, classes), 1):
            if score <= 0:
                continue
            scaled_box = [box["left"], box["top"], box["left"]+box["width"], box["top"]+box["height"]]
            draw.rectangle(scaled_box, outline="red", width=3)
            clabel = class_labels[cls_idx] if cls_idx < len(class_labels) else "Unknown"
            t = f"{order}: {score:.2f} ({clabel})"
            pos = (scaled_box[0], max(0, scaled_box[1]-20))
            w = len(t)*6
            h = 15
            lb = [pos[0], pos[1], pos[0]+w, pos[1]+h]
            draw.rectangle(lb, fill="red")
            try:
                font = ImageFont.truetype("DejaVuSans", 50)
            except OSError:
                font = ImageFont.load_default()
            draw.text((pos[0]+2, pos[1]+2), t, fill="black", font=font)
        annotated_name = subfolder_path / f"annotated_page_{i}.jpg"
        image_resized.save(annotated_name)


def post_image_to_async(server_url, img_bytes, ocr_data_json):
    start_time = time.time()
    response = requests.post(
        server_url,
        files=[("file", ("image.jpg", img_bytes, "image/jpeg"))],
        data={"ocr_data": ocr_data_json}
    )
    elapsed = time.time() - start_time

    return response, elapsed


from concurrent.futures import ThreadPoolExecutor


if __name__ == "__main__":
    pdf_path = "figures/test_batch3.pdf"
    server_url = "http://localhost:8001/batch"

    # Create output directories in parallel
    with ThreadPoolExecutor() as executor:
        dirs = [ANNOTATED_IMAGES_DIR / "with_ocr", ANNOTATED_IMAGES_DIR / "without_ocr"]
        executor.map(lambda d: d.mkdir(parents=True, exist_ok=True), dirs)

    for use_tesseract_ocr in [True]:
        for use_reading_order in [False]:
            ocr_mode = "with_ocr" if use_tesseract_ocr else "without_ocr"
            subfolder_path = ANNOTATED_IMAGES_DIR / ocr_mode
            
            start_time = time.time()
            pdf_images = convert_from_path(str(pdf_path), dpi=300, fmt="jpg")
            end_time = time.time()

            # Prepare batch data
            files_data = []
            ocr_data_list = []
            
            if use_tesseract_ocr:
                # Process OCR in parallel
                with ThreadPoolExecutor() as executor:
                    ocr_data_list = list(executor.map(
                        lambda img: json.loads(get_tesseract_ocr_data(img))["data"], 
                        pdf_images
                    ))
            else:
                ocr_words, bounding_boxes = get_pdfplumber_ocr_data(pdf_path)
                ocr_data_list = [ocr_words] * len(pdf_images)

            # Prepare image data
            for pil_img in pdf_images:
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format='JPEG')
                files_data.append(('files', ('image.jpg', img_byte_arr.getvalue(), 'image/jpeg')))

            # Send batch request
            start_time = time.time()
            response = requests.post(
                server_url,
                files=files_data,
                data={"ocr_data": json.dumps({"data": ocr_data_list})}
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                all_predictions = response.json()
            else:
                all_predictions = [{"instances": {}}] * len(pdf_images)

            table_data = [
                ["OCR Mode", "Pages", "Total(s)", "PPS"],
                [ocr_mode, len(pdf_images), f"{elapsed:.2f}", f"{len(pdf_images)/elapsed:.2f}"]
            ]
            print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

            if all_predictions:
                visualize_predictions(pdf_images, all_predictions, subfolder_path)
