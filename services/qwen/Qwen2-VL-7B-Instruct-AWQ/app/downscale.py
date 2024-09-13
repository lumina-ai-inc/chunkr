import os
from PIL import Image

def downscale_images(input_folder, output_folder, max_size=(800, 800)):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image
            with Image.open(input_path) as img:
                # Downscale the image
                img.thumbnail(max_size, Image.LANCZOS)
                
                # Save the downscaled image
                img.save(output_path, 'PNG')

    print(f"All PNG images in {input_folder} have been downscaled and saved to {output_folder}")

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    input_folder = os.path.join(script_dir, "todown")
    output_folder = os.path.join(script_dir, "downscaled_images2")
    
    downscale_images(input_folder, output_folder)
