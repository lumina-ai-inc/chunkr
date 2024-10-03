import camelot
import concurrent.futures
import dotenv
import os
from pathlib import Path
from PIL import Image
from PyPDF2 import PdfFileWriter, PdfFileReader
import requests
from reportlab.pdfgen import canvas
import sys

dotenv.load_dotenv(override=True)

SERVICE_URL=os.getenv("SERVICE_URL")


def get_ocr_results(image_path):
    with open(image_path, "rb") as image_file:
        response = requests.post(f"{SERVICE_URL}/paddle_ocr", files={"file": image_file})
    return response.json()

def create_searchable_pdf(image_path, ocr_results, output_pdf_path):
    img = Image.open(image_path)
    pdf = canvas.Canvas(str(output_pdf_path), pagesize=img.size)
    pdf.drawImage(str(image_path), 0, 0)
    pdf.save()

    output = PdfFileWriter()
    input_pdf = PdfFileReader(open(str(output_pdf_path), "rb"))
    page = input_pdf.getPage(0)
    
    for result in ocr_results:
        bbox = result["bbox"]
        text = result["text"]
        page.add_text(text, x=bbox["top_left"][0], y=img.size[1] - bbox["top_left"][1], font_size=10)

    output.add_page(page)
    with open(str(output_pdf_path), "wb") as output_file:
        output.write(output_file)


def process_pdf(input_file: Path, output_file: Path):
    if input_file.suffix.lower() in ('.png', '.jpg', '.jpeg'):
        ocr_results = get_ocr_results(input_file)
        temp_pdf = input_file.with_suffix('.pdf')
        create_searchable_pdf(input_file, ocr_results, temp_pdf)
        input_file = temp_pdf

    tables = camelot.read_pdf(str(input_file), pages="1")
    tables.export(str(output_file), f='html', compress=False)
    try:
        print(f"Processing {input_file.name}:")
        print(tables[0].parsing_report)
    except Exception as e:
        print(f"Error processing {input_file.name}: {e}")

    # Remove temporary PDF if created
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
