import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import requests
from multiprocessing import Pool
import functools

def load_ocr_results(image_path):
    """Load and parse OCR results from a JSON file."""
    with open(os.path.join(image_path, "results.json")) as f:
        return json.load(f)

def extract_words_and_bboxes(results, by_line=False):
    """Extract words/lines and bounding boxes from OCR results as pairs.
    
    Args:
        results: OCR results dictionary
        by_line: If True, extract line-level text and boxes. If False, extract word-level.
    """
    pairs = []
    for page in results["pages"]:
        for block in page["blocks"]:
            for line in block["lines"]:
                if by_line:
                    # Combine all words in the line
                    text = " ".join(word["value"] for word in line["words"])
                    [[x1, y1], [x2, y2]] = line["geometry"]
                    pairs.append({
                        "text": text,
                        "bbox": [
                            int(x1 * 1000),
                            int(y1 * 1000),
                            int(x2 * 1000),
                            int(y2 * 1000)
                        ]
                    })
                else:
                    # Process individual words
                    for word in line["words"]:
                        [[x1, y1], [x2, y2]] = word["geometry"]
                        pairs.append({
                            "text": word["value"],
                            "bbox": [
                                int(x1 * 1000),
                                int(y1 * 1000),
                                int(x2 * 1000),
                                int(y2 * 1000)
                            ]
                        })
    return pairs

def get_reading_order(pairs, url):
    """Get reading order from layout service."""
    response = requests.post(
        f"{url}/predict",
        json={"pairs": pairs},
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    return response.json()["reading_order"]

def create_visualization(image, pairs, indices, output_path):
    """Create and save visualization of word boxes with indices."""
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(image)
    
    for idx, word_idx in enumerate(indices):
        pair = pairs[word_idx]
        bbox = pair["bbox"]
        # Convert to relative coordinates
        x1, y1, x2, y2 = [
            bbox[0] / 1000 * image.width,
            bbox[1] / 1000 * image.height,
            bbox[2] / 1000 * image.width,
            bbox[3] / 1000 * image.height
        ]
        
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1, str(idx),
            color='red', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_single_image(image_dir, input_dir, url):
    """Process a single image directory."""
    ocr_results_path = os.path.join(input_dir, image_dir)
    if not os.path.isdir(ocr_results_path):
        return
        
    try:
        # Process the image
        results = load_ocr_results(ocr_results_path)
        
        # Process both lines and words
        for level in ['line', 'word']:
            by_line = (level == 'line')
            pairs = extract_words_and_bboxes(results, by_line=by_line)
            reading_order = get_reading_order(pairs, url)
            
            # Load image and create visualizations
            image = Image.open(os.path.join(ocr_results_path, f"base.jpg"))
            # Create both original and ordered visualizations
            for is_ordered in [False, True]:
                indices = reading_order if is_ordered else range(len(pairs))
                suffix = f"{level}_{('ordered' if is_ordered else 'original')}"
                output_path = os.path.join(ocr_results_path, f"reading_order_{suffix}.png")
                create_visualization(image, pairs, indices, output_path)
            
    except Exception as e:
        print(f"Error processing {image_dir}: {e}")

def visualize_reading_order(input_dir, max_pages=None, url="http://localhost:8000"):
    """Create visualizations of reading order for OCR results at both line and word levels."""
    image_dirs = sorted(os.listdir(input_dir))[:max_pages]
    
    # Create a partial function with fixed arguments
    process_func = functools.partial(process_single_image, input_dir=input_dir, url=url)
    
    # Process images in parallel
    with Pool() as pool:
        pool.map(process_func, image_dirs)

if __name__ == "__main__":
    input_dir = "../doctr/output/organized"
    visualize_reading_order(input_dir)
