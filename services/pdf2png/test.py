import os
import json
import requests
import base64

def test_pdf_conversion():
    # Define the bounding boxes
    bounding_boxes = [
        {
            "left": 317.0,
            "top": 561.0,
            "width": 240.0,
            "height": 11.0,
            "page_number": 1
        },
        {
            "left": 317.0,
            "top": 583.0,
            "width": 248.0,
            "height": 116.0,
            "page_number": 1
        }
    ]

    # Define the PDF file path
    pdf_file_path = "/Users/ishaankapoor/chunk-my-docs/services/pdf2png/input/00c08086-9837-5551-8133-4e22ac28c6a5.pdf"

    # Convert PDF to PNG
    url = "http://localhost:8000/convert"
    files = {"file": ("test.pdf", open(pdf_file_path, "rb"), "application/pdf")}
    data = {"bounding_boxes": json.dumps(bounding_boxes)}
    
    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()  # Raise an error for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return

    # Check if the conversion was successful
    assert response.status_code == 200
    response_json = response.json()
    assert len(response_json["png_pages"]) == len(bounding_boxes)

    # Save the full JSON response and PNG files
    output_dir = "/Users/ishaankapoor/chunk-my-docs/services/pdf2png/output"
    os.makedirs(output_dir, exist_ok=True)

    # Save full JSON response
    with open(os.path.join(output_dir, "response.json"), "w") as f:
        json.dump(response_json, f, indent=2)

    # Save PNG files
    for i, png_page in enumerate(response_json["png_pages"]):
        # Save base64 PNG as actual PNG file
        png_data = base64.b64decode(png_page["base64_png"])
        with open(os.path.join(output_dir, f"snip_{i+1}.png"), "wb") as f:
            f.write(png_data)

    print("PDF conversion test passed successfully.")
    print(f"Output saved in {output_dir}")

if __name__ == "__main__":
    test_pdf_conversion()