import concurrent.futures
import dotenv
import os
from PIL import Image, ImageDraw
import requests

dotenv.load_dotenv(override=True)

rapidocr_url = os.getenv("RAPIDOCR_URL")
unitable_url = os.getenv("UNITABLE_URL")


def call_unitable_bbox(image_path):
    response = requests.post(f"{unitable_url}/bbox",
                             files={"image": open(image_path, "rb")})
    response.raise_for_status()
    return response.json()


def call_rapidocr(image_path):
    response = requests.post(f"{rapidocr_url}/ocr",
                             files={"file": open(image_path, "rb")})
    response.raise_for_status()
    return response.json()["result"]


def process_image(image_path, output_path):
    try:
        ocr_results = call_rapidocr(image_path)
    except Exception as e:
        print(f"Error: {e}")
        return
    bboxes = call_unitable_bbox(image_path)

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for i, bbox in enumerate(bboxes):
        left, top, right, bottom = bbox
        width = right - left
        height = bottom - top
        draw.rectangle([left, top, left + width, top + height],
                       outline="red", width=2)

    for i, result in enumerate(ocr_results):
        bbox, text, confidence = result
        top_left, top_right, bottom_right, bottom_left = bbox
        left = top_left[0]
        top = top_left[1]
        width = bottom_right[0] - left
        height = bottom_right[1] - top
        draw.rectangle([left, top, left + width, top + height],
                       outline="blue", width=2)

    image.save(output_path)


def main(input_dir, output_dir):

    def process_image_wrapper(args):
        try:
            input_path, output_path = args
            process_image(input_path, output_path)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

    image_paths = [os.path.join(input_dir, f) for f in os.listdir(
        input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    output_paths = [os.path.join(output_dir, f) for f in os.listdir(
        input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_image_wrapper, zip(image_paths, output_paths))


if __name__ == "__main__":
    input_dir = "./input"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    main(input_dir, output_dir)
