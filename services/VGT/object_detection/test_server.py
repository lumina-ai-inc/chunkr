import pickle
import shutil

import numpy as np
from os import makedirs
from os.path import join, exists
from test_modules.pdf_features.PdfToken import PdfToken
from test_modules.pdf_features.Rectangle import Rectangle
from test_modules.pdf_features.PdfFeatures import PdfFeatures
import json
from tokenization import BrosTokenizer
from pathlib import Path
ROOT_PATH = Path(__file__).parent.parent.absolute()
WORD_GRIDS_PATH = Path(join(ROOT_PATH, "word_grids"))
tokenizer = BrosTokenizer.from_pretrained("naver-clova-ocr/bros-base-uncased")


def rectangle_to_bbox(rectangle: Rectangle):
    return [rectangle.left, rectangle.top, rectangle.width, rectangle.height]


def get_words_positions(text: str, rectangle: Rectangle):
    text = text.strip()
    text_len = len(text)

    width_per_letter = rectangle.width / text_len

    words_bboxes = [Rectangle(rectangle.left, rectangle.top, rectangle.left + 5, rectangle.bottom)]
    words_bboxes[-1].width = 0
    words_bboxes[-1].right = words_bboxes[-1].left

    for letter in text:
        if letter == " ":
            left = words_bboxes[-1].right + width_per_letter
            words_bboxes.append(Rectangle(left, words_bboxes[-1].top, left + 5, words_bboxes[-1].bottom))
            words_bboxes[-1].width = 0
            words_bboxes[-1].right = words_bboxes[-1].left
        else:
            words_bboxes[-1].right = words_bboxes[-1].right + width_per_letter
            words_bboxes[-1].width = words_bboxes[-1].width + width_per_letter

    words = text.split()
    return words, words_bboxes


def get_subwords_positions(word: str, rectangle: Rectangle):
    width_per_letter = rectangle.width / len(word)
    word_tokens = [x.replace("#", "") for x in tokenizer.tokenize(word)]

    if not word_tokens:
        return [], []

    ids = [x[-2] for x in tokenizer(word_tokens)["input_ids"]]

    right = rectangle.left + len(word_tokens[0]) * width_per_letter
    bboxes = [Rectangle(rectangle.left, rectangle.top, right, rectangle.bottom)]

    for subword in word_tokens[1:]:
        right = bboxes[-1].right + len(subword) * width_per_letter
        bboxes.append(Rectangle(bboxes[-1].right, rectangle.top, right, rectangle.bottom))

    return ids, bboxes


def get_grid_words_dict(tokens: list[PdfToken]):
    texts, bbox_texts_list, inputs_ids, bbox_subword_list = [], [], [], []
    for token in tokens:
        words, words_bboxes = get_words_positions(token.content, token.bounding_box)
        texts += words
        bbox_texts_list += [rectangle_to_bbox(r) for r in words_bboxes]
        for word, word_box in zip(words, words_bboxes):
            ids, subwords_bboxes = get_subwords_positions(word, word_box)
            inputs_ids += ids
            bbox_subword_list += [rectangle_to_bbox(r) for r in subwords_bboxes]


    return {
        "input_ids": np.array(inputs_ids),
        "bbox_subword_list": np.array(bbox_subword_list),
        "texts": texts,
        "bbox_texts_list": np.array(bbox_texts_list),
    }


def create_word_grid(pdf_features_list: list[PdfFeatures]):
    grid_list = []
    makedirs(WORD_GRIDS_PATH, exist_ok=True)
    for pdf_features in pdf_features_list:
        for page in pdf_features.pages:
            image_id = f"{pdf_features.file_name}_{page.page_number - 1}"
            print(f"Image ID: {image_id}")
            # if exists(join(WORD_GRIDS_PATH, image_id + ".pkl")):
            #     continue
            grid_words_dict = get_grid_words_dict(page.tokens)
            grid_list.append(grid_words_dict)

    return grid_list

# def remove_word_grids():
#     shutil.rmtree(WORD_GRIDS_PATH, ignore_errors=True)

if __name__ == "__main__":
    # For APACHE 2.0:
# The following code differs from the source version in these key aspects:
# 1. It includes additional parameters 'density' and 'extension' in the from_pdf_path method.
# 2. The convert_from_path function now uses the 'density' parameter and explicitly sets the output format to 'jpeg'.
# 3. The pdf_name assignment logic has been updated to use the new parameters.
# These changes allow for more flexibility in image conversion settings and file naming.


    import os
    import shutil

    import cv2
    import numpy as np
    from os import makedirs
    from os.path import join
    from pathlib import Path
    from PIL import Image
    from pdf2image import convert_from_path
    from test_modules.pdf_features.PdfFeatures import PdfFeatures

    IMAGES_ROOT_PATH = "tests/images"
    XMLS_PATH = "tests/xmls"
    class PdfImages:
        def __init__(self, pdf_features: PdfFeatures, pdf_images: list[Image]):
            self.pdf_features: PdfFeatures = pdf_features
            self.pdf_images: list[Image] = pdf_images
            self.save_images()

        def show_images(self, next_image_delay: int = 2):
            for image_index, image in enumerate(self.pdf_images):
                image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                cv2.imshow(f"Page: {image_index + 1}", image_np)
                cv2.waitKey(next_image_delay * 1000)
                cv2.destroyAllWindows()

        def save_images(self):
            makedirs(IMAGES_ROOT_PATH, exist_ok=True)
            for image_index, image in enumerate(self.pdf_images):
                image_name = f"{self.pdf_features.file_name}_{image_index}.jpg"
                image.save(join(IMAGES_ROOT_PATH, image_name))

        @staticmethod
        def remove_images():
            shutil.rmtree(IMAGES_ROOT_PATH)

        @staticmethod
        def from_pdf_path(pdf_path: str | Path, pdf_name: str = "", xml_file_name: str = "", density: int = 72, extension: str = "jpeg"):
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            xml_path = Path(join(XMLS_PATH, pdf_path.stem + ".xml"))
            
            if xml_path and not xml_path.parent.exists():
                os.makedirs(xml_path.parent, exist_ok=True)
            print(f"Using XML path: {xml_path}")
            pdf_features = PdfFeatures.from_pdf_path(pdf_path, str(xml_path) if xml_path else None)
            if pdf_features is None:
                raise ValueError(f"Failed to extract features from PDF: {pdf_path}")

            if pdf_name:
                pdf_features.file_name = pdf_name
            else:
                pdf_name = pdf_path.parent.name if pdf_path.name == "document.pdf" else pdf_path.stem
                pdf_features.file_name = pdf_name
                
            pdf_images = convert_from_path(str(pdf_path), dpi=density, fmt=extension)
            return PdfImages(pdf_features, pdf_images)

    # Convert PDF to images and create word grid
    pdf_path = "figures/test_batch.pdf"
    batch_files = []
    batch_grid_data = [] 
    pdf_images_list = [PdfImages.from_pdf_path(pdf_path, "", "test", density=300, extension="jpg")]
    print(len(pdf_images_list))
    
    print("boutta start creating word grid")
    # Create word grid pkl files
    grid_list = create_word_grid([pdf_images.pdf_features for pdf_images in pdf_images_list])
    # Send requests to server for each page
    import requests
    import time
    import pickle
    from pathlib import Path
    import io
    from PIL import ImageDraw
    server_url = "http://localhost:8000/batch/"
    for pdf_images, page_grid_list in zip(pdf_images_list, [grid_list]):
        for i, image in enumerate(pdf_images.pdf_images):
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Get corresponding grid data
            grid_data = page_grid_list[i]
            grid_data_serializable = {
                "input_ids": grid_data["input_ids"].tolist(),
                "bbox_subword_list": grid_data["bbox_subword_list"].tolist(),
                "texts": [""],
                "bbox_texts_list": grid_data["bbox_texts_list"].tolist()
            }
            
            batch_files.append(("files", ("image.jpg", img_byte_arr, "image/jpeg")))
            batch_grid_data.append(grid_data_serializable)
    
    print(f"Sending batch request with {len(batch_files)} images")
    
    # Send batch request
    try:
        start_time = time.time()
        
        # Create the proper GridList structure
        grid_data_json = json.dumps(batch_grid_data)

        response = requests.post(
            server_url,
            files=batch_files,
            data={"grid_dicts": grid_data_json}  # Wrap in GridList structure
        )
        end_time = time.time()
        print("Response received")
            
        print(f"Batch request completed in {end_time - start_time:.2f} seconds")
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
        else:
            # Process the batch response
            predictions = response.json()
            print(f"Received predictions for {len(predictions)} images")
            
            # Create visualizations for each prediction
            for i, (pred, image) in enumerate(zip(predictions, batch_files)):
                # Convert image bytes back to PIL Image for visualization
                image_bytes = image[1][1]  # Get image bytes from tuple
                annotated_image = Image.open(io.BytesIO(image_bytes))
                draw = ImageDraw.Draw(annotated_image)
                
                # Draw boxes from predictions
                for box in pred["boxes"]:
                    draw.rectangle(box, outline="red", width=3)
                
                # Save the annotated image
                annotated_image_path = Path(IMAGES_ROOT_PATH) / f"annotated_img_{i}.jpg"
                annotated_image.save(annotated_image_path)
                print(f"Annotated image saved at: {annotated_image_path}")
                
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")