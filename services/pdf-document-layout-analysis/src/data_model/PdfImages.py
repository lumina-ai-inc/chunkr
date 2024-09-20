import os
import shutil
import requests
import base64
import io

import cv2
import numpy as np
from os import makedirs
from os.path import join
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.convert_to_img import convert_file_to_images

from src.configuration import IMAGES_ROOT_PATH, XMLS_PATH

# Load environment variables
load_dotenv(override=True)

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
    def from_pdf_path(pdf_path: str | Path, pdf_name: str = "", xml_file_name: str = ""):
        xml_path = Path(join(XMLS_PATH, xml_file_name)) if xml_file_name else None

        if xml_path and not xml_path.parent.exists():
            os.makedirs(xml_path.parent, exist_ok=True)
        print(f"Creating PDF images from path: {pdf_path}")
        pdf_features: PdfFeatures = PdfFeatures.from_pdf_path(pdf_path, str(xml_path) if xml_path else None)

        if pdf_name:
            pdf_features.file_name = pdf_name
        else:
            pdf_name = Path(pdf_path).parent.name if Path(pdf_path).name == "document.pdf" else Path(pdf_path).stem
            pdf_features.file_name = pdf_name

        # Use the convert_file_to_images function
        output_dir = Path(IMAGES_ROOT_PATH) / pdf_name
        os.makedirs(output_dir, exist_ok=True)

        convert_file_to_images(pdf_path, output_dir, density=300, extension="png")

        pdf_images = []
        for image_file in sorted(output_dir.glob("*.png")):
            with Image.open(image_file) as img:
                pdf_images.append(img.copy())
            os.remove(image_file)  # Remove the temporary image file

        return PdfImages(pdf_features, pdf_images)

