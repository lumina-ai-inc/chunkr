import requests
import os
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def process_images(image_paths, url):
    files = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        files.append(("images", (os.path.basename(image_path), image_data, "image/png")))

    data = {
        "prompt": "Each image is a segment. First detect what type of segment it is. It can also be a page (a whole page of a document), if it is, then extract the text as OCR for the whole page put it in <page></page> tags. It can also be <type>Table</type> or <type>Graph</type> or <type>Infograph</type> or <type>Text</type> or <type>Image</type> or <type>Formula</type>. If it is a table, extract the table in markdown format that i can then render later. put markdown in <markdown></markdown> tags, along with a description in <description></description> tags. If it is a picture or graph , describe the picture in detail. If it is Text, then do OCR and return the text exactly as written (regardless if its handwrittten or typed)."
    }

    try:
        start_time = time.time()
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        end_time = time.time()
        processing_time = end_time - start_time
        return f"Batch response (processed in {processing_time:.2f} seconds):\n" + response.text
    except requests.exceptions.RequestException as e:
        return f"Error processing batch request: {e}"

def test_qwen_generate():
    script_dir = os.path.dirname(__file__)
    test_dir = os.path.join(script_dir, "test_images")

    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No image files found in {test_dir}")
        return

    url = "http://35.197.120.88:8000/generate"  # Assuming the server is running on localhost:8000

    batch_size = 4
    start_time = time.time()
    
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]
        result = process_images(batch, url)
        print(result)
        print("-" * 50)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    test_qwen_generate()