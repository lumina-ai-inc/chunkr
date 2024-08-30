import requests
import os
from PIL import Image
import io

def test_phi_generate():
    # URL of the /generate endpoint
    url = "http://localhost:8040/generate"

    # Path to the test PNG file
    script_dir = os.path.dirname(__file__)
    test_png_path = os.path.join(script_dir, "test_phi/test.png")

    # Check if the file exists
    if not os.path.exists(test_png_path):
        print(f"Error: Test image not found at {test_png_path}")
        return

    # Open and prepare the image
    with open(test_png_path, "rb") as image_file:
        image_data = image_file.read()

    # Prepare the request
    files = {
        "images": ("example.png", image_data, "image/png")
    }
    data = {
        "prompt": "Describe this image in detail. Extract values and put it in markdown table. Extrapolate expected values from any graph and estimate."
    }

    try:
        # Send the POST request
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()

        # Print the streaming response
        print("Response from Phi-3.5 Vision API:")
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                print(chunk.decode(), end='', flush=True)

    except requests.exceptions.RequestException as e:
        print(f"Error occurred while making the request: {e}")

if __name__ == "__main__":
    test_phi_generate()