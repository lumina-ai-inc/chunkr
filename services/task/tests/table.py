import requests
import os
from paddleocr import draw_structure_result, save_structure_res
import multiprocessing
import time
from PIL import Image
import json

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
    response = requests.post(f"{service_url}/paddle_table", files=files)

    time_taken = time.time() - start_time

    print(f"Time taken: {time_taken} seconds")

    # Check if the request was successful
    if response.status_code == 200:
        results = response.json()
        with open(f"./output/{os.path.basename(image_path).split('.')[0]}_result.json", "w") as f:
            json.dump(results, f, indent=2)
        save_structure_res(results, "./output",
                           os.path.basename(image_path).split('.')[0])
        font_path = 'fonts/simfang.ttf'
        image = Image.open(image_path).convert('RGB')
        im_show = draw_structure_result(image, results, font_path=font_path)
        im_show = Image.fromarray(im_show)
        im_show.save(
            f"./output/{os.path.basename(image_path).split('.')[0]}_result.jpg")
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
    image_path = "/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/table.jpg"
    service_url = "http://35.236.179.125:3000"

    # process_image((image_path, service_url))

    process_image((image_path, service_url))
    # n = 1  # Number of times to send the image

    # start_time = time.time()
    # # Create a pool of worker processes
    # with multiprocessing.Pool() as pool:
    #     # Create a list of arguments for each process
    #     args = [(image_path, service_url) for _ in range(n)]

    #     # Map the process_image function to the arguments
    #     results = pool.map(process_image, args)

    # end_time = time.time()
    # print(f"Total time taken: {end_time - start_time} seconds")
    # print(f"Average time taken: {(end_time - start_time) / n} seconds")
