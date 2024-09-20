import requests
import base64
from pathlib import Path
import os
import time

def convert_file_to_images(file_path, output_dir, density=150, extension="png"):
    # Prepare the URL and files for the request
    url = 'http://35.236.179.125:3000/convert_to_img'
    files = {
        'file': open(file_path, 'rb')
    }
    data = {
        'density': str(density),
        'extension': extension
    }

    # Send the POST request
    response = requests.post(url, files=files, data=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()
        
        # Create the output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save each page as a image
        for page_number, base64_image in result.items():
            # Decode the base64 image
            image_data = base64.b64decode(base64_image)
            
            # Save the image
            output_path = Path(output_dir) / f"page_{page_number}.{extension}"
            with open(output_path, 'wb') as f:
                f.write(image_data)
            
            print(f"Saved page {page_number} as {output_path}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Usage
file_path = '/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/1c749c35-cc15-56b2-ade5-010fbf1a9778.pdf'
base_output_dir = 'output'

# Extract the filename without extension
file_name = Path(file_path).stem

# Create the output directory using the filename
output_dir = Path(base_output_dir) / file_name
os.makedirs(output_dir, exist_ok=True)

# Start timing
start_time = time.time()

convert_file_to_images(file_path, output_dir, 300, "png")

# End timing
end_time = time.time()

# Calculate and print execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")