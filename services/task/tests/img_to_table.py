import camelot
import concurrent.futures
import dotenv
import fitz
import os
from pathlib import Path
from PIL import Image
import requests
import sys

dotenv.load_dotenv(override=True)

SERVICE_URL=os.getenv("SERVICE_URL")


def get_ocr_results(image_path):
    with open(image_path, "rb") as image_file:
        response = requests.post(f"{SERVICE_URL}/paddle_ocr", files={"file": image_file})
    return response.json()

def calculate_font_size(rect, text, max_font_size=60, min_font_size=4, padding=2):
    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = fitz.Font("helv")
        text_width = font.text_length(text, fontsize=font_size)
        text_height = font_size
        
        padded_width = rect.width - (2 * padding)
        padded_height = rect.height - (2 * padding)
        
        if text_width <= padded_width and text_height <= padded_height:
            return font_size
    return min_font_size

def create_searchable_pdf(input_file, ocr_results, output_file):
    pdf_document = fitz.open()
    
    # Use Pillow to get the correct image dimensions
    with Image.open(input_file) as img:
        img_width, img_height = img.size
    
    # Create a new page with the correct dimensions
    page = pdf_document.new_page(width=img_width, height=img_height)
    page_rect = page.rect
    page.insert_image(page_rect, filename=input_file)

    font_sizes = []
    for result in ocr_results:
        tl_x, tl_y = result["bbox"]["top_left"]
        tr_x, tr_y = result["bbox"]["top_right"]
        br_x, br_y = result["bbox"]["bottom_right"]
        bl_x, bl_y = result["bbox"]["bottom_left"]

        x0 = min(tl_x, bl_x)
        y0 = min(tl_y, tr_y)
        x1 = max(tr_x, br_x)
        y1 = max(bl_y, br_y)

        fill_rect = fitz.Rect(x0, y0, x1, y1)
        font_size = calculate_font_size(fill_rect, result["text"])
        font_sizes.append(font_size)
    
    smallest_font_size = min(font_sizes)
    
    font = fitz.Font(ordering=0)
    writer = fitz.TextWriter(page_rect, color=(0, 0, 0, 1))
    
    for result in ocr_results:
        tl_x, tl_y = result["bbox"]["top_left"]
        tr_x, tr_y = result["bbox"]["top_right"]
        br_x, br_y = result["bbox"]["bottom_right"]
        bl_x, bl_y = result["bbox"]["bottom_left"]

        x0 = min(tl_x, bl_x)
        y0 = min(tl_y, tr_y)
        x1 = max(tr_x, br_x)
        y1 = max(bl_y, br_y)

        fill_rect = fitz.Rect(x0, y0, x1, y1)
        writer.fill_textbox(fill_rect, result["text"], fontsize=smallest_font_size, align=fitz.TEXT_ALIGN_CENTER, warn=True, font=font)
    
    writer.write_text(page)

    pdf_document.save(output_file)
    pdf_document.close()


def process_img(input_file: Path, output_file: Path):
    try:
        if input_file.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            ocr_results = get_ocr_results(input_file)
            temp_pdf = input_file.with_suffix('.pdf')
            create_searchable_pdf(input_file, ocr_results, temp_pdf)
            input_file = temp_pdf

        tables = camelot.read_pdf(str(input_file), pages="1")
        tables.export(str(output_file), f='html', compress=False)
        print(input_file, tables[0].parsing_report)
    except Exception as e:
        print(f"Error processing {input_file.name}: {e}")


def process_files(input_files, output_dir):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for input_file in input_files:
            output_file = Path(output_dir) / f"{input_file.stem}.html"
            futures.append(executor.submit(process_img, input_file, output_file))

        for future in concurrent.futures.as_completed(futures):
            future.result()


def main(input_path: str, output_dir: str):
    input_path = Path(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        output_file = output_path / f"{input_path.stem}.html"
        process_img(input_path, output_file)
    elif input_path.is_dir():
        img_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))
        process_files(img_files, output_path)
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
