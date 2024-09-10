import requests
import os
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_image(image_path, url):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    # Prepare the request
    files = {
        "images": (os.path.basename(image_path), image_data, "image/png")
    }
    data = {
        "prompt": "Each image is a segment. First detect what type of segment it is. It can either by <type>Table</type> or <type>Graph</type> or <type>Infograph</type> or <type>Text</type> or <type>Image</type> or <type>Formula</type>. If it is a table, extract the table in markdown format that i can then render later. put markdown in <markdown></markdown> tabs, along with a description in <description></description> tabs. If it is a picture or graph , describe the picture in detail. If it is Text, then do OCR and return the text exactly as written (regardless if its handwrittten or typed)."
    }

    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        return f"Response for {os.path.basename(image_path)}:\n" + response.text
    except requests.exceptions.RequestException as e:
        return f"Error processing {os.path.basename(image_path)}: {e}"

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

    url = "http://localhost:8000/generate"  # Assuming the server is running on localhost:8000

    prompt = "Each image is a segment. First detect what type of segment it is. It can either by <type>Table</type> or <type>Graph</type> or <type>Infograph</type> or <type>Text</type> or <type>Image</type> or <type>Formula</type>. If it is a table, extract the table in markdown format that i can then render later. put markdown in <markdown></markdown> tags, along with a description in <description></description> tags. If it is a picture or graph , describe the picture in detail. If it is Text, then do OCR and return the text exactly as written (regardless if its handwrittten or typed)."

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_image, img_path, url) for img_path in image_files]
        
        for future in as_completed(futures):
            result = future.result()
            print(result)
            print("-" * 50)

if __name__ == "__main__":
    test_qwen_generate()