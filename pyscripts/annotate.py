import json
import fitz
import os

def draw_bounding_boxes(pdf_path, json_data, output_path, draw_ocr=True):
    # Define colors for different types
    color_map = {
        "Caption": (1, 0, 0),  # Red
        "Footnote": (0, 1, 0),  # Green
        "Formula": (0, 0, 1),  # Blue
        "List item": (1, 1, 0),  # Yellow
        "Page footer": (1, 0.5, 0),  # Orange
        "Page header": (0.5, 0, 0.5),  # Purple
        "Picture": (1, 0.75, 0.8),  # Pink
        "Section header": (0.6, 0.3, 0),  # Brown
        "Table": (0.54, 0, 0),  # Dark red
        "Text": (0, 0, 0),  # Black
        "Title": (1, 0, 0),  # Red
    }

    # Load JSON data
    data = json_data

    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        # Check if 'segments' key exists, if not use the data directly
        if any('segments' in item for item in data):
            page_segments = [seg for item in data for seg in item.get("segments", []) if seg["page_number"] == page_num + 1]
        else:
            page_segments = [item for item in data if item["page_number"] == page_num + 1]

        # Get page dimensions
        page_width = page.rect.width
        page_height = page.rect.height

        # Draw rectangles for each segment
        for seg in page_segments:
            # Scale coordinates according to input PDF
            segment_rect = fitz.Rect(
                seg["bbox"]["top_left"][0] * page_width / seg["page_width"],
                seg["bbox"]["top_left"][1] * page_height / seg["page_height"],
                seg["bbox"]["bottom_right"][0] * page_width / seg["page_width"],
                seg["bbox"]["bottom_right"][1] * page_height / seg["page_height"]
            )
            color = color_map.get(seg["segment_type"], (0, 0, 0))  # Default to black if type not found
            page.draw_rect(segment_rect, color=color, width=2)

            # Draw OCR bbox if available and draw_ocr is True
            if draw_ocr and seg.get("ocr"):
                for ocr_result in seg["ocr"]:
                    # Calculate absolute coordinates for OCR bbox
                    ocr_rect = fitz.Rect(
                        segment_rect.x0 + ocr_result["bbox"]["top_left"][0] * segment_rect.width,
                        segment_rect.y0 + ocr_result["bbox"]["top_left"][1] * segment_rect.height,
                        segment_rect.x0 + ocr_result["bbox"]["bottom_right"][0] * segment_rect.width,
                        segment_rect.y0 + ocr_result["bbox"]["bottom_right"][1] * segment_rect.height
                    )
                    page.draw_rect(ocr_rect, color=(0, 0.5, 0.5), width=1)  # Teal color for OCR boxes

    # Save the modified PDF
    pdf_document.save(output_path)
    pdf_document.close()

if __name__ == "__main__":
    json_path = "output/De Beers Jewellers Ltd 2021_json.json"
    if not os.path.exists(json_path):
        print(f"Error: The file {json_path} does not exist.")
        print("Please ensure the JSON file has been generated before running this script.")
        exit(1)

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    draw_bounding_boxes(
        "input/De Beers Jewellers Ltd 2021.pdf",
        json_data,
        "output/De Beers Jewellers Ltd 2021_Annotated.pdf",
    )