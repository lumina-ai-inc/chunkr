import requests
import os
from PIL import Image
import io
import time
from dotenv import load_dotenv
from typing import List, Dict, Any
import json

load_dotenv(override=True)

QWEN_URL = os.getenv('QWEN_URL')


def process_images(prompt: str, image_files: List[str]):
    """
    Prepare the data for sending to the QWEN model, aligning messages with images.
    """
    files = []

    # Attach images as separate files
    for image_path in image_files:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        files.append(("images", (os.path.basename(image_path), image_data, "image/png")))

    messages_batch = []
    for _ in image_files:
        messages_batch.append([
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "text": "Attached Image"}
                ],
            },
        ])

    data = {
        "messages": json.dumps(messages_batch)
    }

    print(f"QWEN URL: {QWEN_URL}")
    print(f"Sending batch request to QWEN URL: {QWEN_URL}")
    print(f"Number of images to send: {len(image_files)}")
    print(f"Number of image references in messages: {len(messages_batch)}")

    try:
        response = requests.post(QWEN_URL, files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while sending batch request to QWEN URL: {QWEN_URL}")
        return {"error": f"Error processing batch request: {e}"}


def test_batch_request():
    script_dir = os.path.dirname(__file__)
    test_dir = os.path.join(script_dir, "test_images")

    print(f"Script directory: {script_dir}")
    print(f"Test directory: {test_dir}")

    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    image_files = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    print(f"Processing {len(image_files)} images")
    print(f"Image files: {image_files}")

    if not image_files:
        print(f"No image files found in {test_dir}")
        return

    prompt = "What is the text in the image?"

    start_time = time.time()
    print("Starting batch request...")
    result = process_images(prompt, image_files)
    end_time = time.time()

    total_time = end_time - start_time
    print("Processing result:")

    if "responses" in result:
        for i, response in enumerate(result["responses"]):
            print(f"Response {i + 1}:")
            print(json.dumps(response, indent=2))
            print("-" * 50)
    else:
        print("No 'responses' in result:")
        print(result)

    print(f"Total execution time: {total_time:.2f} seconds")


def test_single_request():
    script_dir = os.path.dirname(__file__)
    test_dir = os.path.join(script_dir, "test_images")

    print(f"Script directory: {script_dir}")
    print(f"Test directory: {test_dir}")

    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    image_file = os.path.join(test_dir, "list.png")

    if not os.path.isfile(image_file):
        print(f"Error: Image file not found at {image_file}")
        return

    prompt = "What is the text in the image?"

    messages_batch = [[{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "text": "Attached Image"}
        ],
    }]]

    print(f"Messages batch: {json.dumps(messages_batch, indent=2)}")

    start_time = time.time()
    print("Starting single request...")
    result = process_images(messages_batch, [image_file])
    end_time = time.time()

    total_time = end_time - start_time
    print("Processing result:")

    if "responses" in result:
        for i, response in enumerate(result["responses"]):
            print(f"Response {i + 1}:")
            print(json.dumps(response, indent=2))
            print("-" * 50)
    else:
        print("No 'responses' in result:")
        print(result)

    print(f"Total execution time: {total_time:.2f} seconds")



if __name__ == "__main__":
    print("Testing Qwen Processing:")
    print("Single Request Test:")
    # test_single_request()
    print("\nBatch Request Test:")
    test_batch_request()