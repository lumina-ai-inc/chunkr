import requests
from pathlib import Path
import json
import cv2
import os
import matplotlib.pyplot as plt
from paddleocr import draw_ocr

def save_ocr(img_path, out_path, result, font):
    os.makedirs(out_path, exist_ok=True)
    # Ensure the output file has a proper image extension
    save_path = os.path.join(out_path, os.path.splitext(os.path.basename(img_path))[0] + '_output.png')
    
    image = cv2.imread(img_path)
    
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]

    im_show = draw_ocr(image, boxes, txts, scores, font_path=font)
    
    cv2.imwrite(save_path, im_show)
 
    img = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()  # Add this line to display the image

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
        results = response.json()
        with open("output/results.json", "w") as f:
            json.dump(results, f)
        save_ocr(image_path, "output", results[0], "font/simfang.ttf")
        return results
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

# Usage example
if __name__ == "__main__":
    image_path = "/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/page_70.png"
    service_url = "http://34.169.115.7:3000"  
    try:
        results = send_image_to_ocr(image_path, service_url)
        print("OCR Results:")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"An error occurred: {str(e)}")