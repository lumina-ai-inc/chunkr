import json
import fitz

import os

def draw_bounding_boxes(pdf_path, json_path, output_path):
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
    with open(json_path, "r") as f:
        data = json.load(f)

    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        # Filter objects for the current page
        page_objects = [obj for obj in data if obj["page_number"] == page_num + 1]

        # Group objects by markdown level
        markdown_groups = {}
        for obj in page_objects:
            markdown_level = obj.get("markdown_level", 0)
            if markdown_level not in markdown_groups:
                markdown_groups[markdown_level] = []
            markdown_groups[markdown_level].append(obj)

        # Draw rectangles for each object and markdown group
        for markdown_level, objects in markdown_groups.items():
            # Draw small boxes for individual segments
            for obj in objects:
                rect = fitz.Rect(
                    obj["left"],
                    obj["top"],
                    obj["left"] + obj["width"],
                    obj["top"] + obj["height"],
                )
                color = color_map.get(obj["type"], (0, 0, 0))  # Default to black if type not found
                page.draw_rect(rect, color=color, width=1)

            # Draw big box for the collected segments
            if objects:
                left = min(obj["left"] for obj in objects)
                top = min(obj["top"] for obj in objects)
                right = max(obj["left"] + obj["width"] for obj in objects)
                bottom = max(obj["top"] + obj["height"] for obj in objects)
                big_rect = fitz.Rect(left, top, right, bottom)
                page.draw_rect(big_rect, color=(0, 0.5, 0.5), width=2)  # Teal color for big box

    # Save the modified PDF
    pdf_document.save(output_path)
    pdf_document.close()



if __name__ == "__main__":
    json_path = "output/00c08086-9837-5551-8133-4e22ac28c6a5-HighQuality/bounding_boxes.json"
    if not os.path.exists(json_path):
        print(f"Error: The file {json_path} does not exist.")
        print("Please ensure the JSON file has been generated before running this script.")
        exit(1)

    draw_bounding_boxes(
        "input/00c08086-9837-5551-8133-4e22ac28c6a5-HighQuality.pdf",
        json_path,
        "output/00c08086-9837-5551-8133-4e22ac28c6a5-HighQuality/annotated.pdf",
    )