#!/usr/bin/env python3
"""
Bounding Box Visualizer for Chunkr JSON Output

This script reads a Chunkr JSON file and creates visualizations of the OCR bounding boxes
for each segment by cropping the page images using segment bounding boxes.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any
import colorsys
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import os

def generate_colors(n: int) -> List[str]:
    """Generate n distinct colors for different OCR boxes."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.8
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append('#%02x%02x%02x' % tuple(int(c * 255) for c in rgb))
    return colors

def load_chunkr_data(json_path: str) -> Dict[str, Any]:
    """Load and parse the Chunkr JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def download_image(image_url: str) -> Image.Image:
    """Download an image from URL."""
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Warning: Could not download image from {image_url}: {e}")
        return None

def get_page_images(data: Dict[str, Any]) -> List[str]:
    """Extract page image URLs from the Chunkr JSON."""
    if 'output' in data and 'page_images' in data['output']:
        return data['output']['page_images']
    return []

def extract_segments(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all segments from the Chunkr JSON."""
    segments = []
    
    # Navigate to the segments data
    if 'output' in data and 'chunks' in data['output']:
        for chunk in data['output']['chunks']:
            if 'segments' in chunk:
                for segment in chunk['segments']:
                    segments.append(segment)
    
    return segments

def crop_segment_from_page(page_image: Image.Image, segment_bbox: Dict[str, Any]) -> Image.Image:
    """Crop a segment from the page image using the segment's bbox."""
    left = int(segment_bbox.get('left', 0))
    top = int(segment_bbox.get('top', 0))
    width = int(segment_bbox.get('width', 0))
    height = int(segment_bbox.get('height', 0))
    
    # Calculate crop box (left, top, right, bottom)
    crop_box = (left, top, left + width, top + height)
    
    # Ensure crop box is within image bounds
    img_width, img_height = page_image.size
    crop_box = (
        max(0, crop_box[0]),
        max(0, crop_box[1]),
        min(img_width, crop_box[2]),
        min(img_height, crop_box[3])
    )
    
    # Crop the image
    cropped_image = page_image.crop(crop_box)
    return cropped_image

def create_segment_image_with_ocr(segment: Dict[str, Any], segment_index: int, 
                                 output_dir: Path, page_images_cache: Dict[int, Image.Image]) -> None:
    """Create a visualization of OCR results for a single segment."""
    
    # Get segment information
    segment_type = segment.get('segment_type', 'Unknown')
    segment_id = segment.get('segment_id', f'segment_{segment_index}')
    page_num = segment.get('page_number', 1)
    
    # Get segment bbox
    segment_bbox = segment.get('bbox', {})
    if not segment_bbox or not all(k in segment_bbox for k in ['left', 'top', 'width', 'height']):
        print(f"  No valid bbox for segment {segment_index}")
        return
    
    # Get OCR results
    ocr_results = segment.get('ocr', [])
    if not ocr_results:
        print(f"  No OCR results for segment {segment_index}")
        return
    
    # Get the page image
    if page_num not in page_images_cache:
        print(f"  No page image available for page {page_num}")
        return
    
    page_image = page_images_cache[page_num]
    
    # Crop the segment from the page image
    segment_image = crop_segment_from_page(page_image, segment_bbox)
    
    if segment_image.size[0] == 0 or segment_image.size[1] == 0:
        print(f"  Invalid segment crop for segment {segment_index}")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Display the segment image
    ax.imshow(segment_image, extent=[0, segment_image.size[0], segment_image.size[1], 0], aspect='auto')
    ax.set_xlim(0, segment_image.size[0])
    ax.set_ylim(segment_image.size[1], 0)
    ax.set_aspect('equal')
    
    # Generate colors for OCR boxes
    colors = generate_colors(len(ocr_results))
    
    # Draw OCR bounding boxes (these are relative to the segment)
    text_boxes_count = 0
    for i, ocr_item in enumerate(ocr_results):
        bbox = ocr_item.get('bbox', {})
        text = ocr_item.get('text', '')
        confidence = ocr_item.get('confidence', 0)
        
        left = bbox.get('left', 0)
        top = bbox.get('top', 0)
        width = bbox.get('width', 0)
        height = bbox.get('height', 0)
        
        # Skip empty text boxes
        if not text.strip():
            continue
        
        text_boxes_count += 1
        
        # Create rectangle patch
        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            (left, top), width, height,
            linewidth=2, edgecolor=color, facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add text annotation
        text_x = left + width / 2
        text_y = top - 5  # Position text above the box
        
        # Truncate long text for display
        display_text = text[:50] + "..." if len(text) > 50 else text
        
        ax.text(text_x, text_y, display_text, 
               ha='center', va='bottom', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor=color),
               rotation=0)
    
    # Add title
    ax.set_title(f'Segment {segment_index} (Page {page_num}) - {segment_type}\n'
                f'Size: {segment_image.size[0]}x{segment_image.size[1]} | OCR Results: {text_boxes_count} text boxes', 
                fontsize=14, fontweight='bold')
    
    # Remove axis labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save the plot
    safe_segment_id = segment_id.replace('/', '_').replace('\\', '_')
    output_file = output_dir / f'segment_{segment_index:04d}_{safe_segment_id}.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved segment visualization: {output_file}")

def create_segment_summary(segments: List[Dict[str, Any]], output_dir: Path) -> None:
    """Create a summary of all segments processed."""
    summary_file = output_dir / 'segments_summary.txt'
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Segment Processing Summary\n")
        f.write("=" * 50 + "\n\n")
        
        total_segments = len(segments)
        segments_with_bbox = sum(1 for s in segments if s.get('bbox'))
        segments_with_ocr = sum(1 for s in segments if s.get('ocr'))
        
        f.write(f"Total segments: {total_segments}\n")
        f.write(f"Segments with bounding boxes: {segments_with_bbox}\n")
        f.write(f"Segments with OCR results: {segments_with_ocr}\n\n")
        
        # Group by segment type
        segment_types = {}
        for i, segment in enumerate(segments):
            seg_type = segment.get('segment_type', 'Unknown')
            if seg_type not in segment_types:
                segment_types[seg_type] = []
            segment_types[seg_type].append(i)
        
        f.write("Segments by type:\n")
        for seg_type, indices in segment_types.items():
            f.write(f"  {seg_type}: {len(indices)} segments\n")
        
        f.write("\nSegment details:\n")
        for i, segment in enumerate(segments):
            seg_type = segment.get('segment_type', 'Unknown')
            seg_id = segment.get('segment_id', 'N/A')
            page_num = segment.get('page_number', 'N/A')
            has_bbox = bool(segment.get('bbox'))
            ocr_count = len(segment.get('ocr') or [])
            
            f.write(f"  {i:4d}: {seg_type:<15} | Page {page_num} | "
                   f"BBox: {'Yes' if has_bbox else 'No':<3} | OCR: {ocr_count:3d} items\n")
    
    print(f"Saved summary: {summary_file}")

def main():
    """Main function to process Chunkr JSON and create segment visualizations."""
    parser = argparse.ArgumentParser(description='Visualize OCR bounding boxes on cropped Chunkr segments')
    parser.add_argument('json_file', type=str, help='Path to the Chunkr JSON file')
    parser.add_argument('--output-dir', '-o', type=str, default='segment_visualizations', 
                       help='Output directory for visualizations (default: segment_visualizations)')
    parser.add_argument('--max-segments', '-m', type=int, help='Maximum number of segments to process (default: all)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create segments subdirectory
    segments_dir = output_dir / 'segments'
    segments_dir.mkdir(exist_ok=True)
    
    # Load data
    print(f"Loading Chunkr data from: {args.json_file}")
    data = load_chunkr_data(args.json_file)
    
    # Get page images
    page_image_urls = get_page_images(data)
    print(f"Found {len(page_image_urls)} page images")
    
    # Download and cache page images
    page_images_cache = {}
    for i, page_url in enumerate(page_image_urls):
        page_num = i + 1
        print(f"Downloading page {page_num} image: {page_url}")
        page_image = download_image(page_url)
        if page_image:
            page_images_cache[page_num] = page_image
            print(f"  Successfully loaded page {page_num} image ({page_image.size[0]}x{page_image.size[1]})")
        else:
            print(f"  Failed to load page {page_num} image")
    
    if not page_images_cache:
        print("No page images could be downloaded!")
        return
    
    # Extract segments
    segments = extract_segments(data)
    
    if not segments:
        print("No segments found in the JSON file!")
        return
    
    print(f"Found {len(segments)} segments")
    
    # Limit segments if requested
    if args.max_segments:
        segments = segments[:args.max_segments]
        print(f"Processing first {len(segments)} segments")
    
    # Process each segment
    processed_count = 0
    for i, segment in enumerate(segments):
        print(f"Processing segment {i+1}/{len(segments)}...")
        
        # Only process segments with valid bounding boxes
        if segment.get('bbox') and segment.get('ocr'):
            create_segment_image_with_ocr(segment, i, segments_dir, page_images_cache)
            processed_count += 1
        else:
            bbox_status = "has bbox" if segment.get('bbox') else "no bbox"
            ocr_status = "has OCR" if segment.get('ocr') else "no OCR"
            print(f"  Skipping segment {i} - {bbox_status}, {ocr_status}")
    
    # Create summary
    create_segment_summary(segments, output_dir)
    
    print(f"\nProcessing complete!")
    print(f"Processed {processed_count} segments with bounding boxes and OCR")
    print(f"Check the '{segments_dir}' directory for segment visualizations")
    print(f"Check the '{output_dir}' directory for summary")

if __name__ == "__main__":
    main() 