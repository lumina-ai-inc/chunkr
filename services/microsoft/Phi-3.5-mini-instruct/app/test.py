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
        "prompt": "First detect what type of image it is. It can either by <type>Table</type> or <type>Graph</type> <type>Infograph</type>  If it is a table, extract the table in markdown format that i can then render later. put markdown in <markdown></markdown> tabs, along with a description in <description></description> tabs. If it is a picture or graph , describe the picture in detail."
    }

    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        return f"Response for {os.path.basename(image_path)}:\n" + response.text
    except requests.exceptions.RequestException as e:
        return f"Error processing {os.path.basename(image_path)}: {e}"

def test_phi_generate():
    url = "http://localhost:8040/generate"
    script_dir = os.path.dirname(__file__)
    test_phi_dir = os.path.join(script_dir, "test_phi")

    if not os.path.exists(test_phi_dir):
        print(f"Error: Test directory not found at {test_phi_dir}")
        return

    image_files = [os.path.join(test_phi_dir, f) for f in os.listdir(test_phi_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No image files found in {test_phi_dir}")
        return

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_image = {executor.submit(process_image, image, url): image for image in image_files}
        for future in as_completed(future_to_image):
            image = future_to_image[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f'{image} generated an exception: {exc}')

if __name__ == "__main__":
    test_phi_generate()