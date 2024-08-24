import json
import fitz


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

        # Draw rectangles for each object
        for obj in page_objects:
            rect = fitz.Rect(
                obj["left"],
                obj["top"],
                obj["left"] + obj["width"],
                obj["top"] + obj["height"],
            )
            color = color_map.get(
                obj["type"], (0, 0, 0)
            )  # Default to black if type not found
            page.draw_rect(rect, color=color, width=1)

    # Save the modified PDF
    pdf_document.save(output_path)
    pdf_document.close()
