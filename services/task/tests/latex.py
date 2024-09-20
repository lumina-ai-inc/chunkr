import requests
import os
import multiprocessing
import time
import json
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def send_image_to_ocr(args: Tuple[str, str, str]) -> dict:
    """
    Send an image file to the OCR service and return the results.

    :param args: Tuple containing (image_path, service_url, output_dir)
    :return: Dictionary containing OCR results and time taken
    """
    image_path, service_url, output_dir = args
    files = {'file': open(image_path, 'rb')}

    start_time = time.time()
    response = requests.post(f"{service_url}/latex_ocr", files=files)
    time_taken = time.time() - start_time

    print(
        f"Time taken for {os.path.basename(image_path)}: {time_taken:.2f} seconds")

    if response.status_code == 200:
        results = response.text
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Save LaTeX result
        latex_path = os.path.join(output_dir, f"{base_name}_result.tex")
        with open(latex_path, "w") as f:
            f.write(results)

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
    image_dir = "/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/latex_ocr/jpg"
    service_url = os.getenv('SERVICE_URL')
    output_dir = "/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/output/latex_ocr"

    start_time = time.time()
    results = process_images(image_dir, service_url, output_dir)
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time / len(results):.2f} seconds")
