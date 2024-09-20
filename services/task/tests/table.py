import requests
import os
import multiprocessing
import time
import json
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def draw_boxes_on_image(image_path: str, results: List[dict], output_path: str):
    """
    Draw bounding boxes on the image and tag them with an index.

    :param image_path: Path to the input image
    :param bounding_boxes: List of bounding box dictionaries
    :param output_path: Path to save the output image
    """
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        for idx, result in enumerate(results):
            box = result['bbox']
            coords = box['top_left'] + box['top_right'] + box['bottom_right'] + box['bottom_left']
            draw.polygon(coords, outline="red")
            draw.text((coords[0], coords[1]), str(idx), font=font, fill="red")

        img.save(output_path)

def send_image_to_ocr(args: Tuple[str, str, str]) -> dict:
    """
    Send an image file to the OCR service and return the results.

    :param args: Tuple containing (image_path, service_url, output_dir)
    :return: Dictionary containing OCR results and time taken
    """
    image_path, service_url, output_dir = args
    files = {'file': open(image_path, 'rb')}

    start_time = time.time()
    response = requests.post(f"{service_url}/paddle_table", files=files)
    time_taken = time.time() - start_time

    print(
        f"Time taken for {os.path.basename(image_path)}: {time_taken:.2f} seconds")

    if response.status_code == 200:
        results = response.json()
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Save JSON result
        json_path = os.path.join(output_dir, f"{base_name}_result.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save HTML result
        html_path = os.path.join(output_dir, f"{base_name}_result.html")
        html_head = """
        <head>
            <style>
                table {
                    background-color: white;
                    color: black;
                    border-collapse: separate;
                    border-spacing: 0;
                    width: 100%;
                    max-width: 800px;
                    margin: 20px auto;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    overflow: hidden;
                }

                th,
                td {
                    border: 1px solid #e0e0e0;
                    padding: 12px 15px;
                    text-align: left;
                }

                thead {
                    background-color: #f8f9fa;
                    font-weight: 600;
                }

                tr:hover {
                    background-color: #f5f5f5;
                    transition: background-color 0.3s ease;
                }
            </style>
        </head>
        """
        with open(html_path, "w") as f:
            f.write(html_head + results['html'])

        output_image_path = os.path.join(output_dir, f"{base_name}_boxes.png")
        draw_boxes_on_image(image_path, results['results'], output_image_path)

        return results
    else:
        raise Exception(
            f"Error for {os.path.basename(image_path)}: {response.status_code} - {response.text}")


def process_images(image_dir: str, service_url: str, output_dir: str) -> List[dict]:
    """
    Process all images in the given directory using multiprocessing.

    :param image_dir: Directory containing input images
    :param service_url: URL of the OCR service
    :param output_dir: Directory to save output files
    :return: List of results for each image
    """
    os.makedirs(output_dir, exist_ok=True)

    image_paths = [os.path.join(image_dir, f) for f in os.listdir(
        image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    args = [(path, service_url, output_dir) for path in image_paths]

    with multiprocessing.Pool() as pool:
        results = pool.map(send_image_to_ocr, args)

    return results


if __name__ == "__main__":
    image_dir = "/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/table_ocr/jpg"
    service_url = os.getenv('SERVICE_URL')
    output_dir = "/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/output/table_ocr"

    start_time = time.time()
    results = process_images(image_dir, service_url, output_dir)
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time / len(results):.2f} seconds")
