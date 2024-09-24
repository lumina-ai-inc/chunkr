import requests
import json
import cv2
import os
import matplotlib.pyplot as plt
from paddleocr import draw_ocr
import multiprocessing
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def save_ocr(img_path, out_path, result, font):
    os.makedirs(out_path, exist_ok=True)
    save_path = os.path.join(out_path, os.path.splitext(os.path.basename(img_path))[0] + '_output.png')
    
    image = cv2.imread(img_path)
    
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]

    im_show = draw_ocr(image, boxes, txts, scores, font_path=font)
    
    cv2.imwrite(save_path, im_show)
 
    img = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def send_image_to_ocr(image_path: str, service_url: str) -> dict:
    """
    Send an image file to the OCR service and return the results.

    :param image_path: Path to the image file
    :param service_url: URL of the OCR service
    :return: Dictionary containing OCR results and time taken
    """
    # Prepare the file for sending
    files = {'file': open(image_path, 'rb')}

    # Record start time
    start_time = time.time()

    # Send POST request to the OCR service
    response = requests.post(f"{service_url}/paddle_ocr", files=files)

    time_taken = time.time() - start_time

    print(f"Time taken: {time_taken} seconds")

    # Check if the request was successful
    if response.status_code == 200:
        results = response.json()
        with open("output/results_2.json", "w") as f:
            json.dump(results, f)
        # save_ocr(image_path, "output", results[0], "font/simfang.ttf")
        return results
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")


def process_image(args):
    image_path, service_url = args
    try:
        return send_image_to_ocr(image_path, service_url)
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Usage example
if __name__ == "__main__":
    image_path = "/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/Picture.jpg"
    service_url = os.getenv('SERVICE_URL')

    # process_image((image_path, service_url))

    process_image((image_path, service_url))
    n = 10 # Number of times to send the image

    start_time = time.time()
    # Create a pool of worker processes
    with multiprocessing.Pool() as pool:
        # Create a list of arguments for each process
        args = [(image_path, service_url) for _ in range(n)]
        
        # Map the process_image function to the arguments
        results = pool.map(process_image, args)

    end_time = time.time() 
    print(f"Total time taken: {end_time - start_time} seconds")
    print(f"Average time taken: {(end_time - start_time) / n} seconds")