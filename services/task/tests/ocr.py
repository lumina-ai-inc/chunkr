import requests
from pathlib import Path
import json

def send_image_to_ocr(image_path: str, service_url: str) -> dict:
    """
    Send an image file to the OCR service and return the results.

    :param image_path: Path to the image file
    :param service_url: URL of the OCR service
    :return: Dictionary containing OCR results
    """
    # Prepare the file for sending
    files = {'file': open(image_path, 'rb')}

    # Send POST request to the OCR service
    response = requests.post(f"{service_url}/paddle", files=files)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

# Usage example
if __name__ == "__main__":
    image_path = "/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/05-receipt1.jpg"
    service_url = "http://35.184.192.150:3000"  
    try:
        results = send_image_to_ocr(image_path, service_url)
        print("OCR Results:")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"An error occurred: {str(e)}")