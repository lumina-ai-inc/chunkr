import requests
import os
from PIL import Image
import io
import time
from dotenv import load_dotenv
import base64
from typing import List, Dict, Any

load_dotenv()

QWEN_URL = os.getenv('QWEN_URL')

def process_images(messages: List[Dict[str, Any]], prompt: str):
    files = []
    for message in messages:
        if message["role"] == "user":
            for content in message["content"]:
                if content["type"] == "image":
                    image_path = content["image"]
                    with open(image_path, "rb") as image_file:
                        image_data = image_file.read()
                    files.append(("images", (os.path.basename(image_path), image_data, "image/png")))

    data = {
        "messages": json.dumps(messages),
        "prompt": prompt
    }

    try:
        response = requests.post(QWEN_URL, files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error processing request: {e}"}

def test_qwen():
    script_dir = os.path.dirname(__file__)
    test_dir = os.path.join(script_dir, "test_images")

    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Processing {len(image_files)} images")

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

    messages = []
    for image_file in image_files:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_file,
                },
                {"type": "text", "text": "Analyze this table and convert it to JSON."},
            ],
        })

    start_time = time.time()
    result = process_images(messages, prompt)
    end_time = time.time()

    total_time = end_time - start_time
    print("Processing result:")
    
    if "responses" in result:
        for response in result["responses"]:
            print(json.dumps(response, indent=2))
            print("-" * 50)
    else:
        print(result)
    
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    print("Testing Qwen Processing:")
    test_qwen()