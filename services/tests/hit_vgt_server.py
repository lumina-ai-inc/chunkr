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
    
    start_time = time.time()
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    elapsed_time = time.time() - start_time
    print(f"Tesseract OCR processing time: {elapsed_time:.2f} seconds")

    ocr_words = []
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
        
    return json.dumps(ocr_words)


def apply_reading_order(instances):
    bxs = instances.get("boxes", [])
    scs = instances.get("scores", [])
    cls = instances.get("classes", [])
    if not bxs:
        return instances

    def is_wide_element(box, page_width=1000, threshold=0.7):
        """Check if element spans multiple columns"""
        return box["width"] / page_width > threshold

    def get_column_assignment(box, col_boundaries):
        """Determine which column a box belongs to"""
        center_x = box["left"] + box["width"]/2
        for i, (left, right) in enumerate(col_boundaries):
            if left <= center_x <= right:
                return i
        return 0

    # Create segments with additional metadata
    segments = []
    page_width = max(box["left"] + box["width"] for box in bxs) if bxs else 1000
    
    for i, (box, score, class_id) in enumerate(zip(bxs, scs, cls)):
        if score > 0.2:
            is_wide = is_wide_element(box, page_width)
            segments.append({
                'idx': i,
                'box': box,
                'score': score,
                'class': class_id,
                'is_wide': is_wide,
                'center_x': box["left"] + box["width"]/2,
                'center_y': box["top"] + box["height"]/2
            })

    # Separate headers and footers
    headers = [s for s in segments if s['class'] == 5]
    footers = [s for s in segments if s['class'] in [1, 4]]
    body = [s for s in segments if s['class'] not in [1, 4, 5]]

    # Define column boundaries (assuming 2-column layout as default)
    col_width = page_width / 2
    col_boundaries = [(0, col_width), (col_width, page_width)]

    def process_body_segments(segments):
        # Sort by vertical position first
        segments.sort(key=lambda s: s['box']['top'])
        
        # Initialize columns
        col1, col2 = [], []
        current_y = float('-inf')
        temp_segments = []
        
        for segment in segments:
            # Check if we're starting a new row
            if abs(segment['box']['top'] - current_y) > 20:
                # Process accumulated segments
                if temp_segments:
                    # If any segment in the row is wide, add all as wide
                    if any(s['is_wide'] for s in temp_segments):
                        for s in temp_segments:
                            s['is_wide'] = True
                    
                    # Distribute segments to columns
                    for s in temp_segments:
                        if s['is_wide']:
                            # Add any accumulated column content first
                            if col1 or col2:
                                yield from col1
                                yield from col2
                                col1, col2 = [], []
                            yield s
                        else:
                            col = get_column_assignment(s['box'], col_boundaries)
                            if col == 0:
                                col1.append(s)
                            else:
                                col2.append(s)
                
                temp_segments = [segment]
                current_y = segment['box']['top']
            else:
                temp_segments.append(segment)
        
        # Process remaining segments
        if temp_segments:
            if any(s['is_wide'] for s in temp_segments):
                if col1 or col2:
                    yield from col1
                    yield from col2
                yield from temp_segments
            else:
                for s in temp_segments:
                    col = get_column_assignment(s['box'], col_boundaries)
                    if col == 0:
                        col1.append(s)
                    else:
                        col2.append(s)
        
        # Yield any remaining column content
        if col1 or col2:
            yield from col1
            yield from col2

    # Process each section
    ordered_segments = []
    ordered_segments.extend(sorted(headers, key=lambda s: s['box']['top']))
    ordered_segments.extend(process_body_segments(body))
    ordered_segments.extend(sorted(footers, key=lambda s: s['box']['top']))

    # Reorder the original lists
    if ordered_segments:
        final_indices = [s['idx'] for s in ordered_segments]
        instances["boxes"] = [bxs[i] for i in final_indices]
        instances["scores"] = [scs[i] for i in final_indices]
        instances["classes"] = [cls[i] for i in final_indices]

    return instances


def visualize_predictions(images, predictions, subfolder_path, apply_ro=False):
    class_labels = [
        "Caption","Footnote","Formula","ListItem","PageFooter","PageHeader",
        "Picture","SectionHeader","Table","Text","Title"
    ]
    for i, (image, pred_dict) in enumerate(zip(images, predictions)):
        pred_inst = pred_dict.get("instances", {})
        if apply_ro:
            pred_inst = apply_reading_order(pred_inst)
        image_resized = image.resize((image.width, image.height))
        draw = ImageDraw.Draw(image_resized)
        boxes = pred_inst.get("boxes", [])
        scores = pred_inst.get("scores", [])
        classes = pred_inst.get("classes", [])
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
        data={"ocr_data": json.dumps({"data": json.loads(ocr_data_json)})}
    )
    elapsed = time.time() - start_time
    return response, elapsed


if __name__ == "__main__":
    pdf_path = "figures/test_batch5.pdf"
    server_url = "http://localhost:8001/batch_async"
    for use_tesseract_ocr in [False]:
        for use_reading_order in [False, True]:
            mode = "with_ocr" if use_tesseract_ocr else "without_ocr"
            ro_mode = "with_reading_order" if use_reading_order else "without_reading_order"
            subfolder_path = ANNOTATED_IMAGES_DIR / mode / ro_mode
            subfolder_path.mkdir(parents=True, exist_ok=True)
            start_time = time.time()
            pdf_images = convert_from_path(str(pdf_path), dpi=150, fmt="jpg")
            end_time = time.time()
            request_data_list = []
            for pil_img in pdf_images:
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                ocr_data_json = get_tesseract_ocr_data(pil_img) if use_tesseract_ocr else "[]"
                request_data_list.append((img_bytes, ocr_data_json))
            all_predictions = []
            request_times = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(post_image_to_async, server_url, d[0], d[1])
                    for d in request_data_list
                ]
            for i, fut in enumerate(futures):
                try:
                    response, req_time = fut.result()
                    request_times.append(req_time)
                    if response.status_code == 200:
                        all_predictions.append(response.json())
                    else:
                        all_predictions.append({"instances": {}})
                except:
                    all_predictions.append({"instances": {}})
            total_duration = max(request_times) if request_times else 0
            avg_request_time = total_duration / len(request_times) if request_times else 0
            pages_per_sec = len(request_times) / total_duration if total_duration else 0
            table_data = [
                ["OCR Mode","Reading Order","Reqs","Total(s)","Avg(s)","PPS"],
                [mode, ro_mode, len(request_times),
                 f"{total_duration:.2f}", f"{avg_request_time:.2f}", f"{pages_per_sec:.2f}"]
            ]
            print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
            if all_predictions:
                visualize_predictions(pdf_images, all_predictions, subfolder_path, apply_ro=use_reading_order)
