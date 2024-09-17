import requests
import json
import base64
from pathlib import Path

def convert_pdf_to_images(pdf_path, output_dir, density=150):
    # Prepare the URL and files for the request
    url = 'http://localhost:3000/convert_to_img'
    files = {
        'file': open(pdf_path, 'rb')
    }
    data = {
        'density': str(density)
    }

    # Send the POST request
    response = requests.post(url, files=files, data=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()
        
        # Create the output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save each page as a PNG image
        for page_number, base64_image in result.items():
            # Decode the base64 image
            image_data = base64.b64decode(base64_image)
            
            # Save the image
            output_path = Path(output_dir) / f"page_{page_number}.png"
            with open(output_path, 'wb') as f:
                f.write(image_data)
            
            print(f"Saved page {page_number} as {output_path}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Usage
pdf_path = '/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/CIM-04-Alcatel-Lucent.pdf'
output_dir = 'output'
convert_pdf_to_images(pdf_path, output_dir)
