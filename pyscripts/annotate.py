import json
import fitz
import os

def draw_bounding_boxes(pdf_path, json, output_path):
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
    data=json

    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        # Filter segments for the current page
        page_segments = [seg for item in data for seg in item.get("segments", []) if seg["page_number"] == page_num + 1]

        # Draw rectangles for each segment
        for seg in page_segments:
            rect = fitz.Rect(
                seg["left"],
                seg["top"],
                seg["left"] + seg["width"],
                seg["top"] + seg["height"],
            )
            color = color_map.get(seg["type"], (0, 0, 0))  # Default to black if type not found
            page.draw_rect(rect, color=color, width=1)

    # Save the modified PDF
    pdf_document.save(output_path)
    pdf_document.close()

if __name__ == "__main__":
    json_path = "output/De Beers Jewellers Ltd 2021_json.json"
    if not os.path.exists(json_path):
        print(f"Error: The file {json_path} does not exist.")
        print("Please ensure the JSON file has been generated before running this script.")
        exit(1)

    draw_bounding_boxes(
        "input/De Beers Jewellers Ltd 2021.pdf",
        json_path,
        "output/De Beers Jewellers Ltd 2021_Annotated.pdf",
    )