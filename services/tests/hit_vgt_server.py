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

try:
    import pytesseract
except ImportError:
    pytesseract = None

ANNOTATED_IMAGES_DIR = Path("annotated_images")
os.makedirs(ANNOTATED_IMAGES_DIR, exist_ok=True)


def get_tesseract_ocr_data(pil_image):
    if not pytesseract:
        return "[]"

    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)

    ocr_words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        left = float(data["left"][i])
        top = float(data["top"][i])
        width = float(data["width"][i])
        height = float(data["height"][i])
        confidence = float(data["conf"][i]) if data["conf"][i].isdigit() else 0.0

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

    return json.dumps(ocr_words)


def visualize_predictions(images, predictions, subfolder_path):
    class_labels = [
        "Caption", "Footnote", "Formula", "ListItem", "PageFooter",
        "PageHeader", "Picture", "SectionHeader", "Table", "Text", "Title"
    ]

    for i, (image, pred_dict) in enumerate(zip(images, predictions)):
        pred_dict = pred_dict.get("instances", {})

        image_resized = image.resize((image.width * 2, image.height * 2))
        draw = ImageDraw.Draw(image_resized)

        boxes = pred_dict.get("boxes", [])
        scores = pred_dict.get("scores", [])
        classes = pred_dict.get("classes", [])

        try:
            for order, (box, score, cls_idx) in enumerate(zip(boxes, scores, classes), start=1):
                if score <= 0:
                    continue
                scaled_box = [
                    box["left"] * 2,
                    box["top"] * 2,
                    (box["left"] + box["width"]) * 2,
                    (box["top"] + box["height"]) * 2
                ]
                draw.rectangle(scaled_box, outline="red", width=3)

                class_label = class_labels[cls_idx] if cls_idx < len(class_labels) else "Unknown"
                label_text = f"{order}: {score:.2f} ({class_label})"

                text_position = (scaled_box[0], max(0, scaled_box[1] - 35))
                text_width = len(label_text) * 10
                text_height = 30
                label_bbox = [
                    text_position[0],
                    text_position[1],
                    text_position[0] + text_width,
                    text_position[1] + text_height
                ]
                draw.rectangle(label_bbox, fill="red")

                try:
                    font = ImageFont.truetype("DejaVuSans", 100)
                except OSError:
                    font = ImageFont.load_default()

                draw.text(
                    (text_position[0] + 2, text_position[1] + 2),
                    label_text,
                    fill="black",
                    font=font
                )
        except Exception as ex:
            print(f"Error drawing predictions on image {i}: {ex}")

        annotated_name = subfolder_path / f"annotated_page_{i}.jpg"
        image_resized.save(annotated_name)


def post_image_to_async(server_url, img_bytes, ocr_data_json):
    start_time = time.time()
    response = requests.post(
        server_url,
        files=[("file", ("image.jpg", img_bytes, "image/jpeg"))],
        data={"ocr_data": json.dumps({"data": json.loads(ocr_data_json)})}
    )
    elapsed = time.time() - start_time
    return response, elapsed


if __name__ == "__main__":

    pdf_path = "figures/test_batch5.pdf"
    server_url = "http://localhost:8001/batch_async"

    for use_tesseract_ocr in [True, False]:
        ocr_mode = "with_ocr" if use_tesseract_ocr else "without_ocr"
        subfolder_path = ANNOTATED_IMAGES_DIR / ocr_mode
        subfolder_path.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        print(f"Converting PDF to images for OCR mode: {ocr_mode}...")
        pdf_images = convert_from_path(str(pdf_path), dpi=300, fmt="jpg")
        end_time = time.time()
        print(f"Conversion completed in {end_time - start_time:.2f} seconds")

        request_data_list = []
        for image_idx, pil_img in enumerate(pdf_images):
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()

            if use_tesseract_ocr:
                ocr_data_json = get_tesseract_ocr_data(pil_img)
            else:
                ocr_data_json = "[]"

            request_data_list.append((img_bytes, ocr_data_json))

        all_predictions = []
        request_times = []
        print(f"Sending {len(request_data_list)} requests to: {server_url}")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(post_image_to_async, server_url, data[0], data[1])
                for data in request_data_list
            ]

        for i, fut in enumerate(futures):
            try:
                response, req_time = fut.result()
                request_times.append(req_time)

                if response.status_code == 200:
                    predictions = response.json()
                    all_predictions.append(predictions)
                else:
                    print(f"Error processing image {i}: HTTP {response.status_code}")
                    print(f"Response: {response.text}")
                    all_predictions.append({"instances": {}})
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                all_predictions.append({"instances": {}})

        total_duration = max(request_times)
        avg_request_time = total_duration / len(request_times)
        pages_per_sec = len(request_times) / total_duration

        print(f"Average request time: {avg_request_time:.2f} seconds")
        print(f"Min request time: {min(request_times):.2f} seconds")
        print(f"Max request time: {max(request_times):.2f} seconds")
        print(f"Total time for all requests: {total_duration:.2f} seconds")
        print(f"Pages per second: {pages_per_sec:.2f}")

        table_data = [
            ["OCR Mode", "Total Requests", "Total Time (s)", "Avg Time per Request (s)", "Pages per Second"],
            [ocr_mode, len(request_times), f"{total_duration:.2f}", f"{avg_request_time:.2f}", f"{pages_per_sec:.2f}"]
        ]

        print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

        if all_predictions:
            print(f"Visualizing predictions for all successful responses in {ocr_mode} mode...")
            visualize_predictions(pdf_images, all_predictions, subfolder_path)
