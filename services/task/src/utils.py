import subprocess
import mimetypes
from pathlib import Path
import os
import cv2
import matplotlib.pyplot as plt
from paddleocr import draw_ocr
import shutil
import xml.etree.ElementTree as ET

def check_imagemagick_installed():
    try:
        subprocess.run(['convert', '-version'], check=True, capture_output=True)
        print("ImageMagick is installed")
        
        # Check and update ImageMagick policy
        policy_file = '/etc/ImageMagick-6/policy.xml'
        tree = ET.parse(policy_file)
        root = tree.getroot()
        
        pdf_policy = root.find(".//policy[@pattern='PDF']")
        if pdf_policy is not None and pdf_policy.get('rights') == 'none':
            print("Updating ImageMagick policy to allow PDF operations")
            pdf_policy.set('rights', 'read|write')
            
            # Create a backup of the original file
            shutil.copy2(policy_file, f"{policy_file}.bak")
            
            # Write the updated policy
            tree.write(policy_file)
            print("ImageMagick policy updated successfully")
        else:
            print("ImageMagick policy already allows PDF operations")
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("ImageMagick is not installed or not in the system PATH")
    except Exception as e:
        print(f"Error updating ImageMagick policy: {str(e)}")
        raise RuntimeError("Failed to update ImageMagick policy")
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