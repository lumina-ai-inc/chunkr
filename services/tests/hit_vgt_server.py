import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import io
import time
import requests
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import concurrent.futures
from pdf2image import convert_from_path
import numpy as np
import json
from tabulate import tabulate

try:
    import pytesseract
except ImportError:
    pytesseract = None

ANNOTATED_IMAGES_DIR = Path("annotated_images")
os.makedirs(ANNOTATED_IMAGES_DIR, exist_ok=True)


def get_tesseract_ocr_data(pil_image):
    if not pytesseract:
        return "[]"
    
    start_time = time.time()
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    elapsed_time = time.time() - start_time
    print(f"Tesseract OCR processing time: {elapsed_time:.2f} seconds")

    ocr_words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        left = float(data["left"][i])
        top = float(data["top"][i])
        width = float(data["width"][i])
        height = float(data["height"][i])
        confidence = float(data["conf"][i]) if isinstance(data["conf"][i], str) and data["conf"][i].isdigit() else 0.0

        ocr_words.append({
            "bbox": {
                "left": left,
                "top": top,
                "width": width,
                "height": height
            },
            "text": text,
            "confidence": confidence
        })

    # Double the OCR words if a certain condition is met
#     stuff=True
#     if stuff:  # Replace 'some_condition' with the actual condition
#         ocr_words= []
#         print("fuddi")
#         ocr_words = [
#   {
#     "bbox": {
#       "left": 141.25781,
#       "top": 96.67969,
#       "width": 71.15625,
#       "height": 10.828125
#     },
#     "text": "UNIVERSITY",
#     "confidence": 0.99796045
#   },
#   {
#     "bbox": {
#       "left": 213.1875,
#       "top": 95.90625,
#       "width": 18.5625,
#       "height": 11.6015625
#     },
#     "text": "OF",
#     "confidence": 0.9996611
#   },
#   {
#     "bbox": {
#       "left": 232.52345,
#       "top": 96.67969,
#       "width": 47.95311,
#       "height": 10.828125
#     },
#     "text": "OREGON",
#     "confidence": 0.99554425
#   },
#   {
#     "bbox": {
#       "left": 71.64844,
#       "top": 137.67188,
#       "width": 27.070312,
#       "height": 14.6953125
#     },
#     "text": "PDF",
#     "confidence": 0.8920853
#   },
#   {
#     "bbox": {
#       "left": 101.03906,
#       "top": 139.21875,
#       "width": 78.11719,
#       "height": 13.921875
#     },
#     "text": "Accessibility",
#     "confidence": 0.58746594
#   },
#   {
#     "bbox": {
#       "left": 180.70312,
#       "top": 140.76562,
#       "width": 12.375015,
#       "height": 10.0546875
#     },
#     "text": "-",
#     "confidence": 0.9780402
#   },
#   {
#     "bbox": {
#       "left": 193.07814,
#       "top": 137.67188,
#       "width": 42.539047,
#       "height": 13.921875
#     },
#     "text": "Tables",
#     "confidence": 0.98220414
#   },
#   {
#     "bbox": {
#       "left": 70.875,
#       "top": 157.78125,
#       "width": 21.65625,
#       "height": 10.828125
#     },
#     "text": "This",
#     "confidence": 0.78676957
#   },
#   {
#     "bbox": {
#       "left": 91.75782,
#       "top": 158.55469,
#       "width": 9.281242,
#       "height": 10.828125
#     },
#     "text": "is",
#     "confidence": 0.50338566
#   },
#   {
#     "bbox": {
#       "left": 100.265625,
#       "top": 158.55469,
#       "width": 9.28125,
#       "height": 10.828125
#     },
#     "text": "a",
#     "confidence": 0.64166534
#   },
#   {
#     "bbox": {
#       "left": 108.77344,
#       "top": 158.55469,
#       "width": 34.03125,
#       "height": 11.6015625
#     },
#     "text": "sample",
#     "confidence": 0.9984598
#   },
#   {
#     "bbox": {
#       "left": 143.57812,
#       "top": 157.78125,
#       "width": 47.953125,
#       "height": 10.828125
#     },
#     "text": "document",
#     "confidence": 0.98707706
#   },
#   {
#     "bbox": {
#       "left": 191.53125,
#       "top": 158.55469,
#       "width": 11.6015625,
#       "height": 10.0546875
#     },
#     "text": "to",
#     "confidence": 0.999806
#   },
#   {
#     "bbox": {
#       "left": 203.13281,
#       "top": 157.78125,
#       "width": 60.328125,
#       "height": 10.828125
#     },
#     "text": "demonstrate",
#     "confidence": 0.9875628
#   },
#   {
#     "bbox": {
#       "left": 263.46094,
#       "top": 157.78125,
#       "width": 22.429688,
#       "height": 10.828125
#     },
#     "text": "how",
#     "confidence": 0.6343609
#   },
#   {
#     "bbox": {
#       "left": 285.1172,
#       "top": 159.32812,
#       "width": 12.375,
#       "height": 9.28125
#     },
#     "text": "to",
#     "confidence": 0.9997217
#   },
#   {
#     "bbox": {
#       "left": 297.4922,
#       "top": 157.78125,
#       "width": 27.070312,
#       "height": 10.828125
#     },
#     "text": "make",
#     "confidence": 0.9437105
#   },
#   {
#     "bbox": {
#       "left": 324.5625,
#       "top": 157.78125,
#       "width": 47.179688,
#       "height": 10.828125
#     },
#     "text": "accessible",
#     "confidence": 0.96961606
#   },
#   {
#     "bbox": {
#       "left": 372.51562,
#       "top": 157.78125,
#       "width": 31.710938,
#       "height": 10.828125
#     },
#     "text": "tables.",
#     "confidence": 0.9898823
#   },
#   {
#     "bbox": {
#       "left": 71.64844,
#       "top": 180.21094,
#       "width": 36.351562,
#       "height": 14.6953125
#     },
#     "text": "Simple",
#     "confidence": 0.9586192
#   },
#   {
#     "bbox": {
#       "left": 108.77344,
#       "top": 180.98438,
#       "width": 30.9375,
#       "height": 11.6015625
#     },
#     "text": "Table",
#     "confidence": 0.8390789
#   },
#   {
#     "bbox": {
#       "left": 70.875,
#       "top": 197.22656,
#       "width": 25.523438,
#       "height": 10.828125
#     },
#     "text": "Sales",
#     "confidence": 0.9996563
#   },
#   {
#     "bbox": {
#       "left": 96.39844,
#       "top": 197.22656,
#       "width": 32.484375,
#       "height": 11.6015625
#     },
#     "text": "figures",
#     "confidence": 0.9479058
#   },
#   {
#     "bbox": {
#       "left": 128.88281,
#       "top": 196.45312,
#       "width": 14.6953125,
#       "height": 13.1484375
#     },
#     "text": "by",
#     "confidence": 0.775718
#   },
#   {
#     "bbox": {
#       "left": 142.80469,
#       "top": 198.0,
#       "width": 54.140625,
#       "height": 10.828125
#     },
#     "text": "salesperson",
#     "confidence": 0.95091385
#   },
#   {
#     "bbox": {
#       "left": 196.94531,
#       "top": 196.45312,
#       "width": 20.109375,
#       "height": 11.6015625
#     },
#     "text": "and",
#     "confidence": 0.9953701
#   },
#   {
#     "bbox": {
#       "left": 216.28125,
#       "top": 198.0,
#       "width": 23.203125,
#       "height": 11.6015625
#     },
#     "text": "year",
#     "confidence": 0.9990633
#   },
#   {
#     "bbox": {
#       "left": 238.71094,
#       "top": 196.45312,
#       "width": 14.6953125,
#       "height": 13.1484375
#     },
#     "text": "(in",
#     "confidence": 0.99220204
#   },
#   {
#     "bbox": {
#       "left": 252.63281,
#       "top": 196.45312,
#       "width": 52.59375,
#       "height": 13.1484375
#     },
#     "text": "thousands)",
#     "confidence": 0.99768996
#   },
#   {
#     "bbox": {
#       "left": 75.515625,
#       "top": 219.65625,
#       "width": 24.75,
#       "height": 11.6015625
#     },
#     "text": "Year",
#     "confidence": 0.99556845
#   },
#   {
#     "bbox": {
#       "left": 154.40625,
#       "top": 220.42969,
#       "width": 30.164062,
#       "height": 10.828125
#     },
#     "text": "Susan",
#     "confidence": 0.9968917
#   },
#   {
#     "bbox": {
#       "left": 232.52345,
#       "top": 219.65625,
#       "width": 33.257797,
#       "height": 11.6015625
#     },
#     "text": "Gerald",
#     "confidence": 0.99821365
#   },
#   {
#     "bbox": {
#       "left": 310.64062,
#       "top": 219.65625,
#       "width": 34.804688,
#       "height": 11.6015625
#     },
#     "text": "Bobbie",
#     "confidence": 0.9169473
#   },
#   {
#     "bbox": {
#       "left": 387.98438,
#       "top": 220.42969,
#       "width": 33.257812,
#       "height": 10.828125
#     },
#     "text": "Keisha",
#     "confidence": 0.9997429
#   },
#   {
#     "bbox": {
#       "left": 465.32812,
#       "top": 219.65625,
#       "width": 18.5625,
#       "height": 11.6015625
#     },
#     "text": "Art",
#     "confidence": 0.99980325
#   },
#   {
#     "bbox": {
#       "left": 76.28906,
#       "top": 234.35156,
#       "width": 25.523438,
#       "height": 10.828125
#     },
#     "text": "2017",
#     "confidence": 0.9985703
#   },
#   {
#     "bbox": {
#       "left": 153.63283,
#       "top": 234.35156,
#       "width": 20.882797,
#       "height": 11.6015625
#     },
#     "text": "570",
#     "confidence": 0.99976295
#   },
#   {
#     "bbox": {
#       "left": 231.75,
#       "top": 233.57812,
#       "width": 20.109375,
#       "height": 12.375
#     },
#     "text": "635",
#     "confidence": 0.9980234
#   },
#   {
#     "bbox": {
#       "left": 309.8672,
#       "top": 233.57812,
#       "width": 20.109406,
#       "height": 12.375
#     },
#     "text": "684",
#     "confidence": 0.9990146
#   },
#   {
#     "bbox": {
#       "left": 387.98438,
#       "top": 234.35156,
#       "width": 20.882843,
#       "height": 10.828125
#     },
#     "text": "397",
#     "confidence": 0.9918258
#   },
#   {
#     "bbox": {
#       "left": 465.32812,
#       "top": 233.57812,
#       "width": 20.882812,
#       "height": 12.375
#     },
#     "text": "678",
#     "confidence": 0.99844164
#   },
#   {
#     "bbox": {
#       "left": 76.28906,
#       "top": 248.27344,
#       "width": 26.296875,
#       "height": 10.828125
#     },
#     "text": "2018",
#     "confidence": 0.9985013
#   },
#   {
#     "bbox": {
#       "left": 154.40625,
#       "top": 248.27344,
#       "width": 20.109375,
#       "height": 11.6015625
#     },
#     "text": "647",
#     "confidence": 0.9995241
#   },
#   {
#     "bbox": {
#       "left": 232.52345,
#       "top": 247.5,
#       "width": 19.335922,
#       "height": 12.375
#     },
#     "text": "325",
#     "confidence": 0.99904805
#   },
#   {
#     "bbox": {
#       "left": 309.8672,
#       "top": 248.27344,
#       "width": 20.882812,
#       "height": 11.6015625
#     },
#     "text": "319",
#     "confidence": 0.99951863
#   },
#   {
#     "bbox": {
#       "left": 387.2109,
#       "top": 248.27344,
#       "width": 20.882843,
#       "height": 11.6015625
#     },
#     "text": "601",
#     "confidence": 0.9998559
#   },
#   {
#     "bbox": {
#       "left": 465.32812,
#       "top": 248.27344,
#       "width": 20.882812,
#       "height": 11.6015625
#     },
#     "text": "520",
#     "confidence": 0.8118042
#   },
#   {
#     "bbox": {
#       "left": 76.28906,
#       "top": 262.1953,
#       "width": 25.523438,
#       "height": 10.828125
#     },
#     "text": "2019",
#     "confidence": 0.9996662
#   },
#   {
#     "bbox": {
#       "left": 153.63283,
#       "top": 262.1953,
#       "width": 20.882797,
#       "height": 11.6015625
#     },
#     "text": "343",
#     "confidence": 0.98267853
#   },
#   {
#     "bbox": {
#       "left": 231.75,
#       "top": 262.1953,
#       "width": 20.882812,
#       "height": 11.6015625
#     },
#     "text": "680",
#     "confidence": 0.9993749
#   },
#   {
#     "bbox": {
#       "left": 309.8672,
#       "top": 262.1953,
#       "width": 20.109406,
#       "height": 11.6015625
#     },
#     "text": "687",
#     "confidence": 0.9980623
#   },
#   {
#     "bbox": {
#       "left": 387.2109,
#       "top": 262.1953,
#       "width": 21.656311,
#       "height": 10.828125
#     },
#     "text": "447",
#     "confidence": 0.94608366
#   },
#   {
#     "bbox": {
#       "left": 465.32812,
#       "top": 262.1953,
#       "width": 20.882812,
#       "height": 11.6015625
#     },
#     "text": "674",
#     "confidence": 0.99929357
#   },
#   {
#     "bbox": {
#       "left": 76.28906,
#       "top": 276.1172,
#       "width": 25.523438,
#       "height": 10.828125
#     },
#     "text": "2020",
#     "confidence": 0.6010238
#   },
#   {
#     "bbox": {
#       "left": 153.63283,
#       "top": 276.1172,
#       "width": 20.882797,
#       "height": 11.6015625
#     },
#     "text": "425",
#     "confidence": 0.99970216
#   },
#   {
#     "bbox": {
#       "left": 231.75,
#       "top": 276.1172,
#       "width": 20.109375,
#       "height": 11.6015625
#     },
#     "text": "542",
#     "confidence": 0.84416807
#   },
#   {
#     "bbox": {
#       "left": 309.8672,
#       "top": 276.1172,
#       "width": 20.109406,
#       "height": 11.6015625
#     },
#     "text": "553",
#     "confidence": 0.9997799
#   },
#   {
#     "bbox": {
#       "left": 387.2109,
#       "top": 276.1172,
#       "width": 20.882843,
#       "height": 11.6015625
#     },
#     "text": "477",
#     "confidence": 0.99894935
#   },
#   {
#     "bbox": {
#       "left": 465.32812,
#       "top": 276.1172,
#       "width": 20.882812,
#       "height": 11.6015625
#     },
#     "text": "648",
#     "confidence": 0.9928328
#   },
#   {
#     "bbox": {
#       "left": 71.64844,
#       "top": 311.6953,
#       "width": 47.179688,
#       "height": 13.921875
#     },
#     "text": "Complex",
#     "confidence": 0.73864263
#   },
#   {
#     "bbox": {
#       "left": 119.60156,
#       "top": 310.92188,
#       "width": 36.351547,
#       "height": 13.921875
#     },
#     "text": "Tables",
#     "confidence": 0.9749371
#   },
#   {
#     "bbox": {
#       "left": 71.64844,
#       "top": 327.9375,
#       "width": 51.820312,
#       "height": 10.828125
#     },
#     "text": "Enrollment",
#     "confidence": 0.5721793
#   },
#   {
#     "bbox": {
#       "left": 122.69531,
#       "top": 327.16406,
#       "width": 14.6953125,
#       "height": 13.921875
#     },
#     "text": "by",
#     "confidence": 0.99629956
#   },
#   {
#     "bbox": {
#       "left": 135.84375,
#       "top": 327.9375,
#       "width": 38.671875,
#       "height": 10.828125
#     },
#     "text": "Student",
#     "confidence": 0.9968425
#   },
#   {
#     "bbox": {
#       "left": 174.51562,
#       "top": 327.9375,
#       "width": 25.523438,
#       "height": 10.828125
#     },
#     "text": "Level",
#     "confidence": 0.97821426
#   },
#   {
#     "bbox": {
#       "left": 199.26562,
#       "top": 327.9375,
#       "width": 20.10939,
#       "height": 11.6015625
#     },
#     "text": "and",
#     "confidence": 0.89253336
#   },
#   {
#     "bbox": {
#       "left": 219.37502,
#       "top": 328.71094,
#       "width": 37.898422,
#       "height": 11.6015625
#     },
#     "text": "Campus",
#     "confidence": 0.99828243
#   },
#   {
#     "bbox": {
#       "left": 75.515625,
#       "top": 351.14062,
#       "width": 27.84375,
#       "height": 10.828125
#     },
#     "text": "Term",
#     "confidence": 0.8593718
#   },
#   {
#     "bbox": {
#       "left": 170.64844,
#       "top": 351.91406,
#       "width": 35.57814,
#       "height": 11.6015625
#     },
#     "text": "Eugene",
#     "confidence": 0.9980902
#   },
#   {
#     "bbox": {
#       "left": 357.04688,
#       "top": 351.14062,
#       "width": 41.765625,
#       "height": 10.828125
#     },
#     "text": "Portland",
#     "confidence": 0.98920715
#   },
#   {
#     "bbox": {
#       "left": 169.875,
#       "top": 365.0625,
#       "width": 72.703125,
#       "height": 13.1484375
#     },
#     "text": "Undergraduate",
#     "confidence": 0.9984653
#   },
#   {
#     "bbox": {
#       "left": 263.46094,
#       "top": 365.0625,
#       "width": 45.632812,
#       "height": 10.828125
#     },
#     "text": "Graduate",
#     "confidence": 0.98748827
#   },
#   {
#     "bbox": {
#       "left": 357.04688,
#       "top": 365.0625,
#       "width": 71.92969,
#       "height": 13.1484375
#     },
#     "text": "Undergraduate",
#     "confidence": 0.99784636
#   },
#   {
#     "bbox": {
#       "left": 450.6328,
#       "top": 365.0625,
#       "width": 45.632812,
#       "height": 10.828125
#     },
#     "text": "Graduate",
#     "confidence": 0.99949026
#   },
#   {
#     "bbox": {
#       "left": 75.515625,
#       "top": 378.21094,
#       "width": 20.109375,
#       "height": 12.375
#     },
#     "text": "Fall",
#     "confidence": 0.9834049
#   },
#   {
#     "bbox": {
#       "left": 94.85156,
#       "top": 378.98438,
#       "width": 25.523438,
#       "height": 10.828125
#     },
#     "text": "2019",
#     "confidence": 0.99976987
#   },
#   {
#     "bbox": {
#       "left": 170.64844,
#       "top": 378.98438,
#       "width": 30.164062,
#       "height": 11.6015625
#     },
#     "text": "19886",
#     "confidence": 0.99895763
#   },
#   {
#     "bbox": {
#       "left": 263.46094,
#       "top": 378.98438,
#       "width": 25.523438,
#       "height": 11.6015625
#     },
#     "text": "3441",
#     "confidence": 0.999474
#   },
#   {
#     "bbox": {
#       "left": 358.59375,
#       "top": 380.53125,
#       "width": 23.203125,
#       "height": 8.5078125
#     },
#     "text": "1024",
#     "confidence": 0.9470996
#   },
#   {
#     "bbox": {
#       "left": 450.6328,
#       "top": 378.98438,
#       "width": 20.109375,
#       "height": 11.6015625
#     },
#     "text": "208",
#     "confidence": 0.99947566
#   },
#   {
#     "bbox": {
#       "left": 76.28906,
#       "top": 392.90625,
#       "width": 34.804688,
#       "height": 10.828125
#     },
#     "text": "Winter",
#     "confidence": 0.96061194
#   },
#   {
#     "bbox": {
#       "left": 110.32031,
#       "top": 392.90625,
#       "width": 26.296875,
#       "height": 10.828125
#     },
#     "text": "2020",
#     "confidence": 0.9979855
#   },
#   {
#     "bbox": {
#       "left": 170.64844,
#       "top": 393.6797,
#       "width": 30.164062,
#       "height": 10.828125
#     },
#     "text": "19660",
#     "confidence": 0.9710489
#   },
#   {
#     "bbox": {
#       "left": 263.46094,
#       "top": 392.90625,
#       "width": 25.523438,
#       "height": 11.6015625
#     },
#     "text": "3499",
#     "confidence": 0.9993044
#   },
#   {
#     "bbox": {
#       "left": 357.8203,
#       "top": 394.45312,
#       "width": 23.976562,
#       "height": 8.5078125
#     },
#     "text": "1026",
#     "confidence": 0.984504
#   },
#   {
#     "bbox": {
#       "left": 450.6328,
#       "top": 392.90625,
#       "width": 20.109375,
#       "height": 11.6015625
#     },
#     "text": "200",
#     "confidence": 0.9974618
#   },
#   {
#     "bbox": {
#       "left": 75.515625,
#       "top": 406.0547,
#       "width": 33.257812,
#       "height": 14.6953125
#     },
#     "text": "Spring",
#     "confidence": 0.97478706
#   },
#   {
#     "bbox": {
#       "left": 108.0,
#       "top": 406.82812,
#       "width": 25.523438,
#       "height": 11.6015625
#     },
#     "text": "2020",
#     "confidence": 0.99962187
#   },
#   {
#     "bbox": {
#       "left": 170.64844,
#       "top": 407.60156,
#       "width": 30.164062,
#       "height": 10.828125
#     },
#     "text": "19593",
#     "confidence": 0.9981654
#   },
#   {
#     "bbox": {
#       "left": 263.46094,
#       "top": 406.82812,
#       "width": 25.523438,
#       "height": 11.6015625
#     },
#     "text": "3520",
#     "confidence": 0.9972957
#   },
#   {
#     "bbox": {
#       "left": 356.27347,
#       "top": 406.82812,
#       "width": 20.882782,
#       "height": 11.6015625
#     },
#     "text": "998",
#     "confidence": 0.9999634
#   },
#   {
#     "bbox": {
#       "left": 449.85938,
#       "top": 406.82812,
#       "width": 20.109375,
#       "height": 11.6015625
#     },
#     "text": "211",
#     "confidence": 0.9989343
#   },
#   {
#     "bbox": {
#       "left": 71.64844,
#       "top": 443.95312,
#       "width": 40.21875,
#       "height": 13.921875
#     },
#     "text": "Eugene",
#     "confidence": 0.947399
#   },
#   {
#     "bbox": {
#       "left": 112.64062,
#       "top": 443.1797,
#       "width": 60.328133,
#       "height": 13.1484375
#     },
#     "text": "Enrollment",
#     "confidence": 0.59022754
#   },
#   {
#     "bbox": {
#       "left": 173.74219,
#       "top": 443.1797,
#       "width": 15.46875,
#       "height": 14.6953125
#     },
#     "text": "by",
#     "confidence": 0.9990839
#   },
#   {
#     "bbox": {
#       "left": 189.21094,
#       "top": 443.95312,
#       "width": 43.312515,
#       "height": 11.6015625
#     },
#     "text": "Student",
#     "confidence": 0.977022
#   },
#   {
#     "bbox": {
#       "left": 234.07031,
#       "top": 443.95312,
#       "width": 27.84375,
#       "height": 11.6015625
#     },
#     "text": "Level",
#     "confidence": 0.99917275
#   },
#   {
#     "bbox": {
#       "left": 262.6875,
#       "top": 443.1797,
#       "width": 21.65625,
#       "height": 13.1484375
#     },
#     "text": "and",
#     "confidence": 0.99966836
#   },
#   {
#     "bbox": {
#       "left": 285.1172,
#       "top": 443.95312,
#       "width": 44.859406,
#       "height": 13.921875
#     },
#     "text": "Campus",
#     "confidence": 0.9980867
#   },
#   {
#     "bbox": {
#       "left": 75.515625,
#       "top": 460.96875,
#       "width": 27.84375,
#       "height": 10.828125
#     },
#     "text": "Term",
#     "confidence": 0.85645354
#   },
#   {
#     "bbox": {
#       "left": 169.875,
#       "top": 460.96875,
#       "width": 72.703125,
#       "height": 13.1484375
#     },
#     "text": "Undergraduate",
#     "confidence": 0.99846256
#   },
#   {
#     "bbox": {
#       "left": 263.46094,
#       "top": 460.96875,
#       "width": 45.632812,
#       "height": 10.828125
#     },
#     "text": "Graduate",
#     "confidence": 0.9875328
#   },
#   {
#     "bbox": {
#       "left": 75.515625,
#       "top": 474.1172,
#       "width": 20.109375,
#       "height": 12.375
#     },
#     "text": "Fall",
#     "confidence": 0.9927366
#   },
#   {
#     "bbox": {
#       "left": 94.85156,
#       "top": 474.89062,
#       "width": 25.523438,
#       "height": 10.828125
#     },
#     "text": "2019",
#     "confidence": 0.9996063
#   },
#   {
#     "bbox": {
#       "left": 169.875,
#       "top": 474.89062,
#       "width": 31.710938,
#       "height": 11.6015625
#     },
#     "text": "19886",
#     "confidence": 0.99740005
#   },
#   {
#     "bbox": {
#       "left": 263.46094,
#       "top": 474.89062,
#       "width": 25.523438,
#       "height": 11.6015625
#     },
#     "text": "3441",
#     "confidence": 0.9997465
#   },
#   {
#     "bbox": {
#       "left": 75.515625,
#       "top": 488.8125,
#       "width": 36.351562,
#       "height": 11.6015625
#     },
#     "text": "Winter",
#     "confidence": 0.5510917
#   },
#   {
#     "bbox": {
#       "left": 110.32031,
#       "top": 488.8125,
#       "width": 26.296875,
#       "height": 11.6015625
#     },
#     "text": "2020",
#     "confidence": 0.99790466
#   },
#   {
#     "bbox": {
#       "left": 170.64844,
#       "top": 489.58594,
#       "width": 30.9375,
#       "height": 10.828125
#     },
#     "text": "19660",
#     "confidence": 0.997701
#   },
#   {
#     "bbox": {
#       "left": 263.46094,
#       "top": 488.8125,
#       "width": 25.523438,
#       "height": 11.6015625
#     },
#     "text": "3499",
#     "confidence": 0.998442
#   },
#   {
#     "bbox": {
#       "left": 74.74219,
#       "top": 501.96094,
#       "width": 34.804688,
#       "height": 14.6953125
#     },
#     "text": "Spring",
#     "confidence": 0.97799575
#   },
#   {
#     "bbox": {
#       "left": 107.22656,
#       "top": 502.73438,
#       "width": 26.296875,
#       "height": 11.6015625
#     },
#     "text": "2020",
#     "confidence": 0.99730945
#   },
#   {
#     "bbox": {
#       "left": 170.64844,
#       "top": 503.5078,
#       "width": 30.164062,
#       "height": 10.828125
#     },
#     "text": "19593",
#     "confidence": 0.99840254
#   },
#   {
#     "bbox": {
#       "left": 263.46094,
#       "top": 502.73438,
#       "width": 25.523438,
#       "height": 11.6015625
#     },
#     "text": "3520",
#     "confidence": 0.9986492
#   },
#   {
#     "bbox": {
#       "left": 72.421875,
#       "top": 539.8594,
#       "width": 44.859375,
#       "height": 11.6015625
#     },
#     "text": "Portland",
#     "confidence": 0.9986205
#   },
#   {
#     "bbox": {
#       "left": 119.60156,
#       "top": 540.6328,
#       "width": 58.78125,
#       "height": 10.828125
#     },
#     "text": "Enrollment",
#     "confidence": 0.65937376
#   },
#   {
#     "bbox": {
#       "left": 179.9297,
#       "top": 539.08594,
#       "width": 15.4687195,
#       "height": 14.6953125
#     },
#     "text": "by",
#     "confidence": 0.8503273
#   },
#   {
#     "bbox": {
#       "left": 195.39842,
#       "top": 539.8594,
#       "width": 42.539078,
#       "height": 11.6015625
#     },
#     "text": "Student",
#     "confidence": 0.9994079
#   },
#   {
#     "bbox": {
#       "left": 239.48438,
#       "top": 539.8594,
#       "width": 28.617188,
#       "height": 11.6015625
#     },
#     "text": "Level",
#     "confidence": 0.8241065
#   },
#   {
#     "bbox": {
#       "left": 268.10156,
#       "top": 539.08594,
#       "width": 22.429688,
#       "height": 13.1484375
#     },
#     "text": "and",
#     "confidence": 0.99856657
#   },
#   {
#     "bbox": {
#       "left": 291.3047,
#       "top": 540.6328,
#       "width": 44.859375,
#       "height": 13.921875
#     },
#     "text": "Campus",
#     "confidence": 0.995214
#   },
#   {
#     "bbox": {
#       "left": 75.515625,
#       "top": 556.875,
#       "width": 28.617188,
#       "height": 11.6015625
#     },
#     "text": "Term",
#     "confidence": 0.8373514
#   },
#   {
#     "bbox": {
#       "left": 169.875,
#       "top": 556.875,
#       "width": 72.703125,
#       "height": 13.1484375
#     },
#     "text": "Undergraduate",
#     "confidence": 0.9986313
#   },
#   {
#     "bbox": {
#       "left": 263.46094,
#       "top": 556.875,
#       "width": 45.632812,
#       "height": 11.6015625
#     },
#     "text": "Graduate",
#     "confidence": 0.90139395
#   },
#   {
#     "bbox": {
#       "left": 75.515625,
#       "top": 570.02344,
#       "width": 20.109375,
#       "height": 12.375
#     },
#     "text": "Fall",
#     "confidence": 0.9917374
#   },
#   {
#     "bbox": {
#       "left": 94.85156,
#       "top": 570.7969,
#       "width": 25.523438,
#       "height": 11.6015625
#     },
#     "text": "2019",
#     "confidence": 0.99967825
#   },
#   {
#     "bbox": {
#       "left": 169.875,
#       "top": 570.7969,
#       "width": 25.523422,
#       "height": 11.6015625
#     },
#     "text": "1024",
#     "confidence": 0.99830097
#   },
#   {
#     "bbox": {
#       "left": 262.6875,
#       "top": 570.7969,
#       "width": 20.882812,
#       "height": 11.6015625
#     },
#     "text": "208",
#     "confidence": 0.99536186
#   },
#   {
#     "bbox": {
#       "left": 76.28906,
#       "top": 585.4922,
#       "width": 35.578125,
#       "height": 10.828125
#     },
#     "text": "Winter:",
#     "confidence": 0.5434162
#   },
#   {
#     "bbox": {
#       "left": 110.32031,
#       "top": 584.71875,
#       "width": 26.296875,
#       "height": 11.6015625
#     },
#     "text": "2020",
#     "confidence": 0.99808073
#   },
#   {
#     "bbox": {
#       "left": 170.64844,
#       "top": 584.71875,
#       "width": 24.749985,
#       "height": 11.6015625
#     },
#     "text": "1026",
#     "confidence": 0.9994442
#   },
#   {
#     "bbox": {
#       "left": 262.6875,
#       "top": 584.71875,
#       "width": 20.882812,
#       "height": 11.6015625
#     },
#     "text": "200",
#     "confidence": 0.98960376
#   },
#   {
#     "bbox": {
#       "left": 75.515625,
#       "top": 597.09375,
#       "width": 33.257812,
#       "height": 14.6953125
#     },
#     "text": "Spring",
#     "confidence": 0.98026454
#   },
#   {
#     "bbox": {
#       "left": 108.0,
#       "top": 597.8672,
#       "width": 25.523438,
#       "height": 11.6015625
#     },
#     "text": "2020",
#     "confidence": 0.9998276
#   },
#   {
#     "bbox": {
#       "left": 169.875,
#       "top": 597.8672,
#       "width": 20.109375,
#       "height": 11.6015625
#     },
#     "text": "998",
#     "confidence": 0.9179111
#   },
#   {
#     "bbox": {
#       "left": 263.46094,
#       "top": 597.8672,
#       "width": 20.109375,
#       "height": 11.6015625
#     },
#     "text": "211",
#     "confidence": 0.9486863
#   }
# ]

    return json.dumps(ocr_words)


def visualize_predictions(images, predictions, subfolder_path):
    class_labels = [
        "Caption", "Footnote", "Formula", "ListItem", "PageFooter",
        "PageHeader", "Picture", "SectionHeader", "Table", "Text", "Title"
    ]

    for i, (image, pred_dict) in enumerate(zip(images, predictions)):
        pred_dict = pred_dict.get("instances", {})

        image_resized = image.resize((image.width * 2, image.height * 2))
        draw = ImageDraw.Draw(image_resized)

        boxes = pred_dict.get("boxes", [])
        scores = pred_dict.get("scores", [])
        classes = pred_dict.get("classes", [])

        try:
            for order, (box, score, cls_idx) in enumerate(zip(boxes, scores, classes), start=1):
                if score <= 0:
                    continue
                scaled_box = [
                    box["left"] * 2,
                    box["top"] * 2,
                    (box["left"] + box["width"]) * 2,
                    (box["top"] + box["height"]) * 2
                ]
                draw.rectangle(scaled_box, outline="red", width=3)

                class_label = class_labels[cls_idx] if cls_idx < len(class_labels) else "Unknown"
                label_text = f"{order}: {score:.2f} ({class_label})"

                text_position = (scaled_box[0], max(0, scaled_box[1] - 20))
                text_width = len(label_text) * 6
                text_height = 15
                label_bbox = [
                    text_position[0],
                    text_position[1],
                    text_position[0] + text_width,
                    text_position[1] + text_height
                ]
                draw.rectangle(label_bbox, fill="red")

                try:
                    font = ImageFont.truetype("DejaVuSans", 50)
                except OSError:
                    font = ImageFont.load_default()

                draw.text(
                    (text_position[0] + 2, text_position[1] + 2),
                    label_text,
                    fill="black",
                    font=font
                )
        except Exception as ex:
            print(f"Error drawing predictions on image {i}: {ex}")

        annotated_name = subfolder_path / f"annotated_page_{i}.jpg"
        image_resized.save(annotated_name)


def post_image_to_async(server_url, img_bytes, ocr_data_json):
    start_time = time.time()
    response = requests.post(
        server_url,
        files=[("file", ("image.jpg", img_bytes, "image/jpeg"))],
        data={"ocr_data": json.dumps({"data": json.loads(ocr_data_json)})}
    )
    elapsed = time.time() - start_time
    return response, elapsed


if __name__ == "__main__":

    pdf_path = "figures/test_batch5.pdf"
    server_url = "http://localhost:8001/batch_async"

    for use_tesseract_ocr in [True, False]:
        ocr_mode = "with_ocr" if use_tesseract_ocr else "without_ocr"
        subfolder_path = ANNOTATED_IMAGES_DIR / ocr_mode
        subfolder_path.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        print(f"Converting PDF to images for OCR mode: {ocr_mode}...")
        pdf_images = convert_from_path(str(pdf_path), dpi=150, fmt="jpg")
        end_time = time.time()
        print(f"Conversion completed in {end_time - start_time:.2f} seconds")

        request_data_list = []
        for image_idx, pil_img in enumerate(pdf_images):
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()

            if use_tesseract_ocr:
                ocr_data_json = get_tesseract_ocr_data(pil_img)
                # ocr_data_json = json.dumps([json.loads(ocr_data_json), json.loads(ocr_data_json)])
            else:
                ocr_data_json = "[]"

            request_data_list.append((img_bytes, ocr_data_json))

        all_predictions = []
        request_times = []
        print(f"Sending {len(request_data_list)} requests to: {server_url}")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(post_image_to_async, server_url, data[0], data[1])
                for data in request_data_list
            ]

        for i, fut in enumerate(futures):
            try:
                response, req_time = fut.result()
                request_times.append(req_time)

                if response.status_code == 200:
                    predictions = response.json()
                    all_predictions.append(predictions)
                else:
                    print(f"Error processing image {i}: HTTP {response.status_code}")
                    print(f"Response: {response.text}")
                    all_predictions.append({"instances": {}})
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                all_predictions.append({"instances": {}})

        total_duration = max(request_times)
        avg_request_time = total_duration / len(request_times)
        pages_per_sec = len(request_times) / total_duration
        print(f"Image size: {pdf_images[0].size}")

        print(f"Average request time: {avg_request_time:.2f} seconds")
        print(f"Min request time: {min(request_times):.2f} seconds")
        print(f"Max request time: {max(request_times):.2f} seconds")
        print(f"Total time for all requests: {total_duration:.2f} seconds")
        print(f"Pages per second: {pages_per_sec:.2f}")

        table_data = [
            ["OCR Mode", "Total Requests", "Total Time (s)", "Avg Time per Request (s)", "Pages per Second"],
            [ocr_mode, len(request_times), f"{total_duration:.2f}", f"{avg_request_time:.2f}", f"{pages_per_sec:.2f}"]
        ]

        print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

        if all_predictions:
            print(f"Visualizing predictions for all successful responses in {ocr_mode} mode...")
            visualize_predictions(pdf_images, all_predictions, subfolder_path)
