import io
import json
import time
import requests
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import concurrent.futures

# Create directories for saving annotated images
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

    # Prepare data for the server
    batch_files = []
    batch_grid_data = []

    for i, (img, grid_data) in enumerate(zip(images, grid_data_list)):
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        # Convert numpy arrays in grid_data to lists (for JSON serialization)
        serializable_data = {
            "input_ids": grid_data["input_ids"].tolist(),
            "bbox_subword_list": grid_data["bbox_subword_list"].tolist(),
            "texts": grid_data["texts"],
            "bbox_texts_list": grid_data["bbox_texts_list"].tolist()
        }

        # Add each file as part of a "files" tuple
        batch_files.append(("files", (f"image_page_{i}.jpg", img_bytes, "image/jpeg")))
        batch_grid_data.append(serializable_data)

    # Convert entire grid data to JSON
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

        # Process the JSON response
        response_data = response.json()  # expected format: list of predictions
        print("Server response format:", response_data)

        # Optional: If we want to visualize bounding boxes returned by the server
        # We'll assume server returns predictions[i]['instances'] ... with boxes, scores, classes
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
        # We assume pred_dict has the structure: {
        #   "instances": {
        #       "boxes": [{"x1": float, "y1": float, "x2": float, "y2": float}, ...],
        #       "scores": [...],
        #       "classes": [...]
        #   }
        # }
        image_resized = image.resize((image.width * 2, image.height * 2))
        draw = ImageDraw.Draw(image_resized)

        instances = pred_dict.get("instances", {})
        boxes = instances.get("boxes", [])
        scores = instances.get("scores", [])
        classes = instances.get("classes", [])

        try:
            for order, (box, score, cls_idx) in enumerate(zip(boxes, scores, classes), start=1):
                if score <= 0:
                    continue  # skip invalid or dummy predictions
                
                # Handle box coordinates from BoundingBox object
                scaled_box = [
                    box["x1"] * 2,
                    box["y1"] * 2,
                    box["x2"] * 2,
                    box["y2"] * 2
                ]
                draw.rectangle(scaled_box, outline="red", width=3)

                # Prepare label text
                class_label = class_labels[cls_idx] if cls_idx < len(class_labels) else "Unknown"
                label_text = f"{order}: {score:.2f} ({class_label})"

                # Draw background for label
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

                # Draw label text
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

        # Save annotated image to directory
        annotated_name = ANNOTATED_IMAGES_DIR / f"annotated_page_{i}.jpg"
        image_resized.save(annotated_name)
        print(f"Annotated image saved as: {annotated_name}")


def post_image(server_url, img_bytes, grid_data_json):
    start_time = time.time()
    response = requests.post(
        server_url,
        files=[("file", ("image.jpg", img_bytes, "image/jpeg"))],
        data={"grid_dict": grid_data_json}
    )
    elapsed = time.time() - start_time
    return response, elapsed


if __name__ == "__main__":
    import os
    from pathlib import Path
    from PIL import Image
    from pdf2image import convert_from_path
    import numpy as np
    import json
    from tokenization import BrosTokenizer

    # Initialize tokenizer
    tokenizer = BrosTokenizer.from_pretrained("naver-clova-ocr/bros-base-uncased")

    # Convert PDF to images and create word grid
    pdf_path = "figures/0a8c1d78-1e43-4893-ac26-1078c63c4952.pdf"
    batch = False
    print("Converting PDF to images...")
    pdf_images = convert_from_path(str(pdf_path), dpi=300, fmt="jpg")
    if batch==True:
        batch_files = []
        batch_grid_data = []

        # Convert PDF pages to images


        # Create empty grid data for each page
        grid_list = []
        for i, image in enumerate(pdf_images):
            # Create a minimal grid structure similar to test_server.py
            # but without relying on PdfFeatures/PdfTokens
            grid_data = {
                "input_ids": np.array([]),  # Empty array for input IDs
                "bbox_subword_list": np.array([]),  # Empty array for subword bboxes
                "texts": [],  # Empty list for texts
                "bbox_texts_list": np.array([])  # Empty array for text bboxes
            }
            grid_list.append(grid_data)

            # Convert image to bytes for sending
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Prepare serializable grid data
            grid_data_serializable = {
                "input_ids": grid_data["input_ids"].tolist(),
                "bbox_subword_list": grid_data["bbox_subword_list"].tolist(),
                "texts": grid_data["texts"],
                "bbox_texts_list": grid_data["bbox_texts_list"].tolist()
            }
            
            batch_files.append(("files", ("image.jpg", img_byte_arr, "image/jpeg")))
            batch_grid_data.append(grid_data_serializable)

        # Send to server
        server_url = "http://localhost:8000/batch/"
        print(f"Sending batch request with {len(batch_files)} images")
        
        try:
            grid_data_json = json.dumps(batch_grid_data)
            start_time = time.time()
            response = requests.post(
                server_url,
                files=batch_files,
                data={"grid_dicts": grid_data_json}
            )
            end_time = time.time()
            print(f"Request completed in {end_time - start_time:.2f} seconds")
            if response.status_code == 200:
                print("Successfully sent data to server")
                visualize_predictions(pdf_images, response.json())
                
            else:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"Error sending data to server: {str(e)}")
            
    else:
        # Process a single image
        server_url = "http://localhost:8000/batch_async/"
        
        # Process all images in parallel
        request_futures = []
        request_times = []
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for image in pdf_images:
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                
                # Create grid data for image
                grid_data = {
                    "input_ids": np.array([]),  # Empty array for input IDs
                    "bbox_subword_list": np.array([]),  # Empty array for subword bboxes 
                    "texts": [],  # Empty list for texts
                    "bbox_texts_list": np.array([])  # Empty array for text bboxes
                }
                
                # Prepare serializable grid data
                grid_data_serializable = {
                    "input_ids": grid_data["input_ids"].tolist(),
                    "bbox_subword_list": grid_data["bbox_subword_list"].tolist(), 
                    "texts": grid_data["texts"],
                    "bbox_texts_list": grid_data["bbox_texts_list"].tolist()
                }
                
                grid_data_json = json.dumps(grid_data_serializable)
                
                # Now call our helper
                future = executor.submit(post_image, server_url, img_bytes, grid_data_json)
                request_futures.append(future)
        
        try:
            # Process responses
            all_predictions = []
            for i, future in enumerate(request_futures):
                try:
                    response, request_time = future.result()
                    request_times.append(request_time)
                    
                    if response.status_code == 200:
                        print(f"Successfully processed image {i} in {request_time:.2f} seconds")
                        all_predictions.append(response.json())
                    else:
                        print(f"Error processing image {i}: {response.status_code}")
                        print(f"Response: {response.text}")
                
                except Exception as e:
                    print(f"Error processing image {i}: {str(e)}")
            
            print(f"Average request time: {sum(request_times)/len(request_times):.2f} seconds")
            print(f"Min request time: {min(request_times):.2f} seconds")
            print(f"Max request time: {max(request_times):.2f} seconds")
            
            if all_predictions:
                print("Visualizing predictions for all successful responses")
                visualize_predictions(pdf_images, all_predictions)
                
        except Exception as e:
            print(f"Error sending data to server: {str(e)}")
