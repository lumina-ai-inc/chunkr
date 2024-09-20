import subprocess
import mimetypes
from pathlib import Path
import os
import cv2
import matplotlib.pyplot as plt
from paddleocr import draw_ocr
import shutil
import xml.etree.ElementTree as ET
import re

def clean_policy_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove malformed XML comments
    cleaned_content = re.sub(r'<!--(?!.*-->).*', '', content, flags=re.DOTALL)
    
    with open(file_path, 'w') as f:
        f.write(cleaned_content)

def check_imagemagick_installed():
    try:
        subprocess.run(['convert', '-version'], check=True, capture_output=True)
        print("ImageMagick is installed")
        
        # Check and update ImageMagick policy
        policy_file = '/etc/ImageMagick-6/policy.xml'
        
        # Clean up the policy file before parsing
        clean_policy_file(policy_file)
        
        try:
            tree = ET.parse(policy_file)
        except ET.ParseError as parse_error:
            print(f"Error parsing ImageMagick policy file: {str(parse_error)}")
            print("Attempting to read the file contents:")
            with open(policy_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    print(f"{i:3d}: {line.rstrip()}")
            raise RuntimeError("Failed to parse ImageMagick policy file") from parse_error
        
        root = tree.getroot()
        
        patterns_to_remove = ['PS', 'PS2', 'PS3', 'EPS', 'PDF', 'XPS']
        elements_removed = False

        for pattern in patterns_to_remove:
            policy = root.find(f".//policy[@pattern='{pattern}']")
            if policy is not None and policy.get('rights') == 'none':
                root.remove(policy)
                elements_removed = True
                print(f"Removed policy for {pattern}")

        if elements_removed:
            # Create a backup of the original file
            shutil.copy2(policy_file, f"{policy_file}.bak")
            
            # Write the updated policy
            tree.write(policy_file)
            print("ImageMagick policy updated successfully")
        else:
            print("No changes were needed in the ImageMagick policy")
        
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