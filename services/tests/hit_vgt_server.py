import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Or suppress specific warnings
# warnings.filterwarnings("ignore", category=UserWarning, module='torch')

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
ANNOTATED_IMAGES_DIR = Path("annotated_images")
os.makedirs(ANNOTATED_IMAGES_DIR, exist_ok=True)

def send_word_grids_to_server(grid_data_list: list[dict],
                              images: list[Image.Image],
                              server_url: str) -> None:
    """
    Demonstrates sending multiple images + associated grid data to a server in one batch.

    grid_data_list: list of dictionaries (from create_word_grid) for each page
    images: list of PIL images that correspond to each item in grid_data_list by index
    server_url: the endpoint accepting a POST with 'files' and 'grid_dicts'
    """

    batch_files = []
    batch_grid_data = []

    for i, (img, grid_data) in enumerate(zip(images, grid_data_list)):
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        serializable_data = {
            "input_ids": grid_data["input_ids"].tolist(),
            "bbox_subword_list": grid_data["bbox_subword_list"].tolist(),
            "texts": grid_data["texts"],
            "bbox_texts_list": grid_data["bbox_texts_list"].tolist()
        }

        batch_files.append(("files", (f"image_page_{i}.jpg", img_bytes, "image/jpeg")))
        batch_grid_data.append(serializable_data)

    grid_data_json = json.dumps(batch_grid_data)

    print(f"Sending {len(batch_files)} images to {server_url}")
    try:
        start_time = time.time()
        response = requests.post(
            server_url,
            files=batch_files,
            data={"grid_dicts": grid_data_json}
        )
        end_time = time.time()
        print("Response received.")
        print(f"Batch request completed in {end_time - start_time:.2f} seconds")

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return

        response_data = response.json()  
        print("Server response format:", response_data)

        visualize_predictions(images, response_data)

    except Exception as e:
        print(f"Error sending batch request: {str(e)}")


def visualize_predictions(images: list[Image.Image], predictions: list[dict]) -> None:
    """
    Example visualization: draws boxes, labels, etc. on images (resized x2).
    Adjust the predicted structure to match your server's actual response format.
    """

    class_labels = [
        "Caption", "Footnote", "Formula", "List-item", "Page-footer",
        "Page-header", "Picture", "Section-header", "Table", "Text", "Title"
    ]

    for i, (image, pred_dict) in enumerate(zip(images, predictions)):

        image_resized = image.resize((image.width * 2, image.height * 2))
        draw = ImageDraw.Draw(image_resized)

        instances = pred_dict.get("instances", {})
        boxes = instances.get("boxes", [])
        scores = instances.get("scores", [])
        classes = instances.get("classes", [])

        try:
            for order, (box, score, cls_idx) in enumerate(zip(boxes, scores, classes), start=1):
                if score <= 0:
                    continue  
                
                scaled_box = [
                    box["x1"] * 2,
                    box["y1"] * 2,
                    box["x2"] * 2,
                    box["y2"] * 2
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
            continue

        annotated_name = ANNOTATED_IMAGES_DIR / f"annotated_page_{i}.jpg"
        image_resized.save(annotated_name)
        # print(f"Annotated image saved as: {annotated_name}")


def post_image(server_url, img_bytes, grid_data_json):
    start_time = time.time()
    response = requests.post(
        server_url,
        files=[("file", ("image.jpg", img_bytes, "image/jpeg"))],
        data={"grid_dict": grid_data_json}
    )
    elapsed = time.time() - start_time
    return response, elapsed


def prepare_serializable_grid_data(grid_data):
    def ensure_pylist(value):
        return value.tolist() if hasattr(value, "tolist") else value

    return {
        "input_ids": ensure_pylist(grid_data["input_ids"]),
        "bbox_subword_list": ensure_pylist(grid_data["bbox_subword_list"]),
        "texts": grid_data["texts"],
        "bbox_texts_list": ensure_pylist(grid_data["bbox_texts_list"]),
    }


if __name__ == "__main__":

    use_create_grid = True
    pdf_path = "figures/test_batch3.pdf"

    start_time = time.time()
    print("Converting PDF to images...")
    pdf_images = convert_from_path(str(pdf_path), dpi=300, fmt="jpg")
    end_time = time.time()
    print(f"Conversion completed in {end_time - start_time:.2f} seconds")

   

    try:
        server_url = "http://localhost:8000/batch_async"
        request_futures = []
        request_times = []
        request_data_list = []

        total_preparing_data_start_time = time.time()
        if use_create_grid:
            print("Creating grid...")
            from create_grid import return_word_grid, create_grid_dict, select_tokenizer
            word_grid = return_word_grid(pdf_path)
            print(f"Got word grid with {len(word_grid)} pages")
            print(f"word_grid: ")
            tokenizer = select_tokenizer("google-bert/bert-base-uncased")
            grid_data_list = []
            for page_idx in range(len(word_grid)):
                print(f"Processing page {page_idx}")
                grid = create_grid_dict(tokenizer, word_grid[page_idx])
                grid_data_list.append(grid)
            print(f"Processed {len(grid_data_list)} grids")

            request_data_list = []
            for image, grid_data in zip(pdf_images, grid_data_list):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                grid_data_serializable = prepare_serializable_grid_data(grid_data)
                request_data_list.append((img_bytes, json.dumps(grid_data_serializable)))
            print(f"Prepared {len(request_data_list)} requests")
        else:
            for image in pdf_images:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                grid_data = {
                    "input_ids": np.array([]),
                    "bbox_subword_list": np.array([]),
                    "texts": [],
                    "bbox_texts_list": np.array([])
                }
                grid_data_serializable = prepare_serializable_grid_data(grid_data)
                grid_data_json = json.dumps(grid_data_serializable)
                request_data_list.append((img_bytes, grid_data_json))

        total_preparing_data_end_time = time.time()
        total_preparing_data_duration = total_preparing_data_end_time - total_preparing_data_start_time
        print(f"Total time for preparing data: {total_preparing_data_duration:.2f} seconds")

        total_start_time = time.time()
        print("hitting server")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(post_image, server_url, data[0], data[1])
                for data in request_data_list
            ]
            request_futures.extend(futures)

        try:
            import time
            all_predictions = []
            for i, future in enumerate(request_futures):
                try:
                    response, request_time = future.result()
                    request_times.append(request_time)
                    if response.status_code == 200:
                        all_predictions.append(response.json())
                    else:
                        print(f"Error processing image {i}: {response.status_code}")
                        print(f"Response: {response.text}")
                except Exception as e:
                    print(f"Error processing image {i}: {str(e)}")
            total_end_time = time.time()
            total_duration = total_end_time - total_start_time
            print(f"Average request time: {sum(request_times)/len(request_times):.2f} seconds")
            print(f"Min request time: {min(request_times):.2f} seconds")
            print(f"Max request time: {max(request_times):.2f} seconds")
            print(f"Total time to get all predictions: {total_duration:.2f} seconds")
            if all_predictions:
                print("Visualizing predictions for all successful responses")
                visualize_predictions(pdf_images, all_predictions)
        except Exception as e:
            print(f"Error sending data to server: {str(e)}")
    except Exception as e:
        print(f"Error sending data to server: {str(e)}")

