import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import requests

def load_ocr_results(image_path):
    """Load and parse OCR results from a JSON file."""
    with open(os.path.join(image_path, "results.json")) as f:
        return json.load(f)

def extract_words_and_bboxes(results):
    """Extract words and bounding boxes from OCR results."""
    words, bboxes = [], []
    for page in results["pages"]:
        for block in page["blocks"]:
            for line in block["lines"]:
                for word in line["words"]:
                    words.append(word["value"])
                    [[x1, y1], [x2, y2]] = word["geometry"]
                    bboxes.append([
                        int(x1 * 1000),
                        int(y1 * 1000),
                        int(x2 * 1000),
                        int(y2 * 1000)
                    ])
    return words, bboxes

def get_reading_order(text, bboxes, url):
    """Get reading order from layout service."""
    payload = {"text": text, "bboxes": bboxes}
    response = requests.post(
        f"{url}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    return response.json()["reading_order"]

def create_visualization(image, words, bboxes, indices, output_path):
    """Create and save visualization of word boxes with indices."""
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(image)
    
    for idx, word_idx in enumerate(indices):
        bbox = bboxes[word_idx]
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

def visualize_reading_order(input_dir, max_pages=None, url="http://localhost:8000"):
    """Create visualizations of reading order for OCR results."""
    for image_dir in sorted(os.listdir(input_dir))[:max_pages]:
        ocr_results_path = os.path.join(input_dir, image_dir)
        if not os.path.isdir(ocr_results_path):
            continue
            
        try:
            # Process the image
            results = load_ocr_results(ocr_results_path)
            words, bboxes = extract_words_and_bboxes(results)
            reading_order = get_reading_order(" ".join(words), bboxes, url)
            
            # Load image and create visualizations
            image = Image.open(os.path.join(ocr_results_path, f"base.jpg"))
            # Create both original and ordered visualizations
            for is_ordered in [False, True]:
                indices = reading_order if is_ordered else range(len(words))
                suffix = "ordered" if is_ordered else "original"
                output_path = os.path.join(ocr_results_path, f"reading_order_{suffix}.png")
                create_visualization(image, words, bboxes, indices, output_path)
                
        except Exception as e:
            print(f"Error processing {image_dir}: {e}")
            continue
        
if __name__ == "__main__":
    input_dir = "../doctr/output/organized"
    visualize_reading_order(input_dir)
