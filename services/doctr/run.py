from doctr.io import DocumentFile
from doctr.models import ocr_predictor, kie_predictor
from functools import partial
import json
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import os
from PIL import Image, ImageDraw
import time
import torch
from shutil import copy2

torch.multiprocessing.set_start_method('spawn', force=True) 

def ocr_pdf():
    input_file = "input/test.pdf"
    output_dir = f"output/{os.path.basename(input_file)}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    doc = DocumentFile.from_pdf(input_file)
    start_time = time.time()
    predictor = ocr_predictor('fast_base', 'crnn_vgg16_bn', pretrained=True, export_as_straight_boxes=True, det_bs=16, reco_bs=8192).cuda()
    end_time = time.time()
    print(f"Time taken to load model: {end_time - start_time} seconds")
    start_time = time.time()
    result = predictor(doc)
    end_time = time.time()
    print(f"Time taken to run inference: {end_time - start_time} seconds")
    json_output = result.export()
    with open(f"{output_dir}/result.json", "w") as f:
        json.dump(json_output, f)
    synthetic_pages = result.synthesize()
    for i, page in enumerate(synthetic_pages):
        plt.figure()  
        plt.imshow(page)
        plt.axis('off')
        plt.savefig(f"{output_dir}/synthetic_page_{i}.png", 
            dpi=500,            
            bbox_inches='tight', 
            pad_inches=0
        )       
        plt.close()
    if i > 0:
        print(f"Pages per second: {i / (end_time - start_time)}")
        
def ocr_images_with_loop():
    input_dir = "input/ocr"
    output_dir = f"output/ocr"
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    start_time = time.time()
    predictor = ocr_predictor('fast_base', 'crnn_vgg16_bn', pretrained=True, export_as_straight_boxes=True, det_bs=16, reco_bs=8192).cuda()
    end_time = time.time()
    print(f"Time taken to load model: {end_time - start_time} seconds")
    for image_path in image_paths:
        output_file = f"{output_dir}/{os.path.basename(image_path)}"
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        doc = DocumentFile.from_images([image_path])
        start_time = time.time()
        result = predictor(doc)
        end_time = time.time()
        print(f"Time taken to run inference: {end_time - start_time} seconds")
        json_output = result.export()
        with open(f"{output_file}/result.json", "w") as f:
            json.dump(json_output, f)
        synthetic_pages = result.synthesize()
        for i, page in enumerate(synthetic_pages):
            plt.figure()  
            plt.imshow(page)
            plt.axis('off')
            plt.savefig(f"{output_file}/synthetic_page_{i}.png", 
                dpi=500,            
                bbox_inches='tight', 
                pad_inches=0
            )       
            plt.close()
        if i > 0:
            print(f"Pages per second: {i / (end_time - start_time)}")

def ocr_images():
    input_dir = "input/ocr"
    output_dir = f"output/ocr"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_paths = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir))]
    doc = DocumentFile.from_images(image_paths)
    start_time = time.time()
    predictor = ocr_predictor('fast_base', 'crnn_vgg16_bn', pretrained=True, export_as_straight_boxes=True, det_bs=16, reco_bs=8192).cuda()
    end_time = time.time()
    print(f"Time taken to load model: {end_time - start_time} seconds")
    start_time = time.time()
    result = predictor(doc)
    end_time = time.time()
    print(f"Time taken to run inference: {end_time - start_time} seconds")
    start_time = time.time()
    json_output = result.export()
    end_time = time.time()
    print(f"Time taken to export JSON: {end_time - start_time} seconds")
    with open(f"{output_dir}/result.json", "w") as f:
        json.dump(json_output, f)
    synthetic_pages = result.synthesize()
    for i, page in enumerate(synthetic_pages):
        plt.figure()  
        plt.imshow(page)
        plt.axis('off')
        plt.savefig(f"{output_dir}/{os.path.basename(image_paths[i])}.png", 
            dpi=500,            
            bbox_inches='tight', 
            pad_inches=0
        )       
        plt.close()
    if i > 0:
        print(f"Pages per second: {i / (end_time - start_time)}")
        


def process_single_image(image_path, predictor, output_dir):
    output_file = f"{output_dir}/{os.path.basename(image_path)}"
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    
    doc = DocumentFile.from_images([image_path])
    start_time = time.time()
    result = predictor(doc)
    end_time = time.time()
    print(f"Time taken to run inference for {os.path.basename(image_path)}: {end_time - start_time} seconds")
    
    json_output = result.export()
    with open(f"{output_file}/result.json", "w") as f:
        json.dump(json_output, f)
    
    synthetic_pages = result.synthesize()
    for i, page in enumerate(synthetic_pages):
        plt.figure()  
        plt.imshow(page)
        plt.axis('off')
        plt.savefig(f"{output_file}/synthetic_page_{i}.png", 
            dpi=500,            
            bbox_inches='tight', 
            pad_inches=0
        )       
        plt.close()

def ocr_images_parallel():
    input_dir = "input/ocr"
    output_dir = f"output/ocr"
    chunk_size = 2
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    
    # Initialize predictor
    start_time = time.time()
    predictor = ocr_predictor('fast_base', 'crnn_vgg16_bn', pretrained=True, 
                            export_as_straight_boxes=True).cuda()
    end_time = time.time()
    print(f"Time taken to load model: {end_time - start_time} seconds")
    
    num_processes = max(1, cpu_count() - 1)
    print(f"Running with {num_processes} processes")
    
    process_func = partial(process_single_image, predictor=predictor, output_dir=output_dir)
    
    start_time = time.time()
    with Pool(num_processes) as pool:
        pool.map(process_func, image_paths, chunk_size)
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"Total time taken: {total_time} seconds")
    print(f"Images per second: {len(image_paths) / total_time}")

def draw_boxes(image_path, idx, result, output_dir):
    image = Image.open(image_path)
    base_filename = os.path.basename(image_path)

    # Draw blocks (red)
    block_image = image.copy()
    draw = ImageDraw.Draw(block_image)
    for block in result['pages'][idx]['blocks']:
        coords = block['geometry']
        left = coords[0][0] * image.width
        top = coords[0][1] * image.height
        right = coords[1][0] * image.width
        bottom = coords[1][1] * image.height
        draw.rectangle([left, top, right, bottom], outline="red", width=2)
    block_image.save(f"{output_dir}/{base_filename}_blocks.png")

    # Draw lines (green)
    line_image = image.copy()
    draw = ImageDraw.Draw(line_image)
    for block in result['pages'][idx]['blocks']:
        for line in block['lines']:
            coords = line['geometry']
            left = coords[0][0] * image.width
            top = coords[0][1] * image.height
            right = coords[1][0] * image.width
            bottom = coords[1][1] * image.height
            draw.rectangle([left, top, right, bottom], outline="green", width=2)
    line_image.save(f"{output_dir}/{base_filename}_lines.png")

    # Draw words (blue)
    word_image = image.copy()
    draw = ImageDraw.Draw(word_image)
    for block in result['pages'][idx]['blocks']:
        for line in block['lines']:
            for word in line['words']:
                coords = word['geometry']
                left = coords[0][0] * image.width
                top = coords[0][1] * image.height
                right = coords[1][0] * image.width
                bottom = coords[1][1] * image.height
                draw.rectangle([left, top, right, bottom], outline="blue", width=2)
    word_image.save(f"{output_dir}/{base_filename}_words.png")
    
def ocr_and_draw_boxes():
    # input_dir = "input/ocr"
    # input_dir = "/data/Chunkr/dataset/pages/TMM.2018.2872898"
    input_dir = "input/TMM.2018.2872898"
    output_dir = f"output/geometry"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    
    predictor = ocr_predictor('fast_base', 'crnn_vgg16_bn', pretrained=True, 
                            export_as_straight_boxes=True).cuda()
    doc = DocumentFile.from_images(image_paths)
    result = predictor(doc)
    
    json_output = result.export()
    for idx, image_path in enumerate(image_paths):
        draw_boxes(image_path, idx, json_output, output_dir)

def kie_process():
    input_dir = "input/form"
    output_dir = "output/kie"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize KIE predictor
    start_time = time.time()
    model = kie_predictor(det_arch='db_resnet50', 
                         reco_arch='crnn_vgg16_bn', 
                         pretrained=True).cuda()
    end_time = time.time()
    print(f"Time taken to load KIE model: {end_time - start_time} seconds")
    
    # Process images
    image_paths = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir))]
    doc = DocumentFile.from_images(image_paths)
    
    start_time = time.time()
    result = model(doc)
    end_time = time.time()
    print(f"Time taken for KIE inference: {end_time - start_time} seconds")
    json_output = result.export()
    with open(f"{output_dir}/result.json", "w") as f:
        json.dump(json_output, f)
    
    # Export results
    for page_idx, page in enumerate(result.pages):
        predictions = page.predictions

        # Save predictions to JSON
        output_file = f"{output_dir}/{os.path.basename(image_paths[page_idx])}.json"
        with open(output_file, "w") as f:
            json.dump(predictions, f, indent=2)
        
        # Visualize predictions
        image = Image.open(image_paths[page_idx])
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        # Generate colors dynamically using HSV color space
        def generate_distinct_colors(n):
            colors = {}
            for i in range(n):
                # Use HSV to generate evenly spaced hues
                hue = i / n
                # Convert HSV to RGB (using fixed saturation and value)
                rgb = plt.cm.hsv(hue)[:3]  # Get RGB values (exclude alpha)
                # Convert RGB to hex color
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0] * 255),
                    int(rgb[1] * 255),
                    int(rgb[2] * 255)
                )
                colors[list(predictions.keys())[i]] = hex_color
            return colors

        # Generate colors dynamically based on unique classes
        colors = generate_distinct_colors(len(predictions.keys()))
        
        for class_name, pred_list in predictions.items():
            color = colors[class_name]
            for pred in pred_list:
                coords = pred.geometry
                left = coords[0][0] * image.width
                top = coords[0][1] * image.height
                right = coords[1][0] * image.width
                bottom = coords[1][1] * image.height
                draw.rectangle([left, top, right, bottom], outline=color, width=2)
                # Add label above the box
                draw.text((left, top-15), f"{class_name}: {pred.value}", fill=color)
        
        draw_image.save(f"{output_dir}/{os.path.basename(image_paths[page_idx])}_kie.png")

def ocr_images_organized_batch(input_dir, output_dir):
    """Process all images in batch and save results in separate folders per image."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get sorted list of image paths
    image_paths = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir))]
    
    # Create output directories for each image
    for image_path in image_paths:
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, file_name)
        os.makedirs(image_output_dir, exist_ok=True)
        # Copy original images
        copy2(image_path, os.path.join(image_output_dir, "base.jpg"))
    
    # Process all images in batch
    start_time = time.time()
    predictor = ocr_predictor('fast_base', 'crnn_vgg16_bn', pretrained=True, 
                            export_as_straight_boxes=True, det_bs=16, reco_bs=8192).cuda()
    doc = DocumentFile.from_images(image_paths)
    result = predictor(doc)
    end_time = time.time()
    print(f"Time taken to process {len(image_paths)} images: {end_time - start_time} seconds")
    print(f"Images per second: {len(image_paths) / (end_time - start_time)}")
    
    # Save results for each image
    json_output = result.export()
    synthetic_pages = result.synthesize()
    
    for i, image_path in enumerate(image_paths):
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, file_name)
        
        # Save JSON results for this page
        page_result = {
            "pages": [json_output["pages"][i]]  # Extract just this page's results
        }
        with open(os.path.join(image_output_dir, "results.json"), "w") as f:
            json.dump(page_result, f, indent=2)
        
        # Save visualization
        plt.figure()
        plt.imshow(synthetic_pages[i])
        plt.axis('off')
        plt.savefig(os.path.join(image_output_dir, "visualization.png"),
            dpi=500,
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close()

if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # ocr_images_parallel()
    # ocr_images()
    # ocr_and_draw_boxes()
    # kie_process()
    ocr_images_organized_batch("input/TMM.2018.2872898", "output/organized")