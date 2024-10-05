import camelot
import concurrent.futures
import dotenv
import os
from pathlib import Path
import fitz
import requests
import sys

dotenv.load_dotenv(override=True)

SERVICE_URL=os.getenv("SERVICE_URL")


def get_ocr_results(image_path):
    with open(image_path, "rb") as image_file:
        response = requests.post(f"{SERVICE_URL}/paddle_ocr", files={"file": image_file})
    return response.json()

def create_searchable_pdf(input_file, ocr_results, output_file):
    pdf_document = fitz.open()
    
    img = fitz.open(input_file)
    page = pdf_document.new_page(width=img[0].rect.width, height=img[0].rect.height)
    
    page.insert_image(page.rect, filename=input_file)
    
    for result in ocr_results:
        # Extract all four corners
        tl_x, tl_y = result["bbox"]["top_left"]
        tr_x, tr_y = result["bbox"]["top_right"]
        br_x, br_y = result["bbox"]["bottom_right"]
        bl_x, bl_y = result["bbox"]["bottom_left"]

        # Calculate the maximum rectangle
        x0 = min(tl_x, bl_x)
        y0 = min(tl_y, tr_y)
        x1 = max(tr_x, br_x)
        y1 = max(bl_y, br_y)

        # Apply scaling
        scale = 72 / 150
        x0 *= scale
        y0 *= scale
        x1 *= scale
        y1 *= scale

        # Draw the bounding box
        rect = fitz.Rect(x0, y0, x1, y1)
        page.draw_rect(rect, color=(1, 0, 0, 0.3), fill=(1, 0, 0, 0.1), width=0.5)

        # Insert text box with center alignment
        rect = fitz.Rect(x0, y0, x1, y1)
        page.insert_textbox(rect, result["text"], color=(1, 1, 1, 0), align=fitz.TEXT_ALIGN_CENTER)

    pdf_document.save(output_file)
    pdf_document.close()


def process_pdf(input_file: Path, output_file: Path):
    if input_file.suffix.lower() in ('.png', '.jpg', '.jpeg'):
        ocr_results = get_ocr_results(input_file)
        temp_pdf = input_file.with_suffix('.pdf')
        create_searchable_pdf(input_file, ocr_results, temp_pdf)
        input_file = temp_pdf

    tables = camelot.read_pdf(str(input_file), pages="1")
    tables.export(str(output_file), f='html', compress=False)
    try:
        print(tables[0].parsing_report)
    except Exception as e:
        print(f"Error processing {input_file.name}: {e}")

    if input_file.name != temp_pdf.name:
        temp_pdf.unlink()


def process_files(input_files, output_dir):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for pdf_file in input_files:
            output_file = Path(output_dir) / f"{pdf_file.stem}.html"
            futures.append(executor.submit(process_pdf, pdf_file, output_file))

        for future in concurrent.futures.as_completed(futures):
            future.result()


def main(input_path: str, output_dir: str):
    input_path = Path(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        output_file = output_path / f"{input_path.stem}.html"
        process_pdf(input_path, output_file)
    elif input_path.is_dir():
        pdf_files = list(input_path.glob("*.pdf"))
        process_files(pdf_files, output_path)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(
            "Usage: python img_to_table.py <input_file_or_directory> [<output_directory>]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) == 3 else str(
        Path(input_path).parent / "output")
    main(input_path, output_dir)
