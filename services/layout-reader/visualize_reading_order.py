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
        results = load_ocr_results(ocr_results_path)
        
        for level in ['line', 'word']:
            by_line = (level == 'line')
            pairs = extract_words_and_bboxes(results, by_line=by_line)
            reading_order = get_reading_order(pairs, url)
            
            image = Image.open(os.path.join(ocr_results_path, f"base.jpg"))
            for is_ordered in [False, True]:
                indices = reading_order if is_ordered else range(len(pairs))
                suffix = f"{level}_{('ordered' if is_ordered else 'original')}"
                output_path = os.path.join(ocr_results_path, f"reading_order_{suffix}.png")
                create_visualization(image, pairs, indices, output_path)
            print(f"Processed {image_dir} with {level} level")
    except Exception as e:
        print(f"Error processing {image_dir}: {e}")

def extract_segments_and_bboxes(results, max_chars=50):
    """Extract segments and bounding boxes from Chunkr results, grouped by page."""
    segments_by_page = {}
    
    for chunk in results.get("chunks", []):
        for segment in chunk.get("segments", []):
            page_num = segment.get("page_number")
            if not page_num:
                continue
                
            if page_num not in segments_by_page:
                segments_by_page[page_num] = []
                
            # Extract bbox and normalize coordinates
            bbox = segment.get("bbox", {})
            # Handle empty text segments with empty string
            text = segment.get("content") or " "
            
            # Process text to show first and last sentences
            if text:
                sentences = text.split('. ')
                if len(sentences) > 2:
                    if len(text) > max_chars:
                        # Show first sentence and last sentence with ellipsis
                        truncated_text = f"{sentences[0]}. ... {sentences[-1]}"
                        if len(truncated_text) > max_chars:
                            # If still too long, truncate each end
                            half_chars = (max_chars - 5) // 2  # 5 chars for " ... "
                            truncated_text = f"{text[:half_chars]}...{text[-half_chars:]}"
                    else:
                        truncated_text = text
                else:
                    # If text is short or has 1-2 sentences, keep it as is
                    truncated_text = text if len(text) <= max_chars else f"{text[:max_chars]}..."
            else:
                truncated_text = ""
            
            # Only skip if there's no bbox
            if not bbox:
                continue
                
            segments_by_page[page_num].append({
                "text": truncated_text,
                "bbox": [
                    int(bbox.get("left", 0) * 1000 / segment["page_width"]),
                    int(bbox.get("top", 0) * 1000 / segment["page_height"]),
                    int((bbox.get("left", 0) + bbox.get("width", 0)) * 1000 / segment["page_width"]),
                    int((bbox.get("top", 0) + bbox.get("height", 0)) * 1000 / segment["page_height"])
                ]
            })
    
    return segments_by_page

def process_single_image_chunkr(image_dir, input_dir, url):
    """Process a single image directory using Chunkr format."""
    results_path = os.path.join(input_dir, image_dir, "results.json")
    if not os.path.isfile(results_path):
        return
        
    try:
        # Load and process results
        with open(results_path) as f:
            results = json.load(f)
            
        segments_by_page = extract_segments_and_bboxes(results)
        
        # Process each page
        for page_num, segments in segments_by_page.items():
            if not segments:
                continue
                
            # Get reading order from service
            reading_order = get_reading_order(segments, url)
            
            # Load image and create visualizations
            image_path = os.path.join(input_dir, image_dir, f"page_{page_num}.jpg")
            if not os.path.exists(image_path):
                image_path = os.path.join(input_dir, image_dir, "base.jpg")
            
            image = Image.open(image_path)
            
            # Create both original and ordered visualizations
            for is_ordered in [False, True]:
                indices = reading_order if is_ordered else range(len(segments))
                suffix = f"segment_p{page_num}_{'ordered' if is_ordered else 'original'}"
                output_path = os.path.join(input_dir, image_dir, f"reading_order_{suffix}.png")
                create_visualization(image, segments, indices, output_path)
            print(f"Processed {page_num} with segment level")
    except Exception as e:
        print(f"Error processing Chunkr format for {image_dir}: {e}")

def visualize_reading_order(input_dir, max_pages=None, url="http://localhost:8000", format="doctr"):
    """Create visualizations of reading order for OCR results at line, word, and segment levels.
    
    Args:
        input_dir: Directory containing the input files
        max_pages: Maximum number of pages to process
        url: URL of the layout service
        format: Either "doctr" or "chunkr" to specify input format
    """
    image_dirs = sorted(os.listdir(input_dir))[:max_pages]
    
    if format == "doctr":
        process_func = functools.partial(process_single_image, input_dir=input_dir, url=url)
    else:
        process_func = functools.partial(process_single_image_chunkr, input_dir=input_dir, url=url)
    
    with Pool() as pool:
        pool.map(process_func, image_dirs)


if __name__ == "__main__":
    # input_dir = "../doctr/output/organized"
    input_dir = "input"
    visualize_reading_order(input_dir, format="chunkr")
