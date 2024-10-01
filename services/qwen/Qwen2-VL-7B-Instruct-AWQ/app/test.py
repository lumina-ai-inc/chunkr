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


def process_images(messages_batch: List[List[Dict[str, Any]]], image_files: List[str]):
    """
    Prepare the data for sending to the QWEN model, aligning messages with images.
    """
    files = []

    # Attach images as separate files
    for image_path in image_files:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        files.append(("images", (os.path.basename(image_path), image_data, "image/png")))

    # **Modified**: Retain image references in messages to allow the server to associate images.
    # For each message, ensure there is exactly one image reference.
    for batch in messages_batch:
        for message in batch:
            if message.get("role") == "user":
                # Clear existing image content to prevent duplicates
                message["content"] = [
                    content for content in message.get("content", []) if content.get("type") != "image"
                ]
                # Append a single image content item
                message["content"].append({"type": "image", "text": "Attached Image"})

    data = {
        "messages": json.dumps(messages_batch)
    }

    print(f"QWEN URL: {QWEN_URL}")
    print(f"Sending batch request to QWEN URL: {QWEN_URL}")
    print(f"Number of images to send: {len(image_files)}")
    print(f"Number of image references in messages: {sum(1 for batch in messages_batch for msg in batch if msg.get("role") == "user" and any(c.get("type") == "image" for c in msg.get("content", [])))}")

    try:
        response = requests.post(QWEN_URL, files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while sending batch request to QWEN URL: {QWEN_URL}")
        return {"error": f"Error processing batch request: {e}"}


def test_qwen():
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

    prompt = """Your main goal is to convert tables into JSON.
Return the provided complex table in JSON format that preserves information and hierarchy from the table at 100 percent accuracy. 
Preserve all text, and structure it in a logical way. 

YOU MUST:
- HAVE ALL THE TEXT IN THE TABLE ACCOUNTED FOR, DO NOT MISS ANY KEY FACTS OR FIGURES.
- YOU MUST OUTPUT VALID JSON

Put your plan in <plan></plan> tag for how you will preserve the tables full information and text and hierarchy in json, and then make <json></json> tags. 
For each table, put the output in its own <json></json> tag. Your final answer will be your <plan>, and then the <json>. Start planning how you will preserve the table:"""

    print(f"Prompt: {prompt}")

    messages_batch = []
    for image_file in image_files:
        messages_batch.append([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this table and convert it to JSON."},
                {"type": "image", "text": "Attached Image"}  # **Added** image reference
            ],
        }])

    print(f"Messages batch: {json.dumps(messages_batch, indent=2)}")

    start_time = time.time()
    print("Starting batch image processing...")
    result = process_images(messages_batch, image_files)
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
    test_qwen()