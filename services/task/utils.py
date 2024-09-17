import subprocess
import mimetypes
from pathlib import Path
import os
import cv2
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR, draw_ocr

def check_imagemagick_installed():
    try:
        subprocess.run(['magick', '-version'], check=True, capture_output=True)
        print("ImageMagick is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("ImageMagick is not installed or not in the system PATH")

def needs_conversion(file: Path) -> bool:
    mime_type, _ = mimetypes.guess_type(file)
    return mime_type in [
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-powerpoint',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation'
    ]

def save_ocr(img_path, out_path, result, font):
    os.makedirs(out_path, exist_ok=True)
    save_path = os.path.join(out_path, img_path.split('/')[-1] + 'output')
    
    image = cv2.imread(img_path)
    
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    
    im_show = draw_ocr(image, boxes, txts, scores, font_path=font)
    
    cv2.imwrite(save_path, im_show)
 
    img = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
    plt.imshow(img)