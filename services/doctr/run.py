from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from functools import partial
import json
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import os
import time
import torch

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


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # ocr_images_parallel()
    ocr_images()
