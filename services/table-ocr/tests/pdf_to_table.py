import camelot
import concurrent.futures
from pathlib import Path
import sys


def process_pdf(input_file: Path, output_file: Path, pages: str = "1"):
    tables = camelot.read_pdf(str(input_file), pages=pages)
    tables.export(str(output_file), f='html', compress=False)
    try:
        print(f"Processing {input_file.name}:")
        print(f"output_file: {output_file}")
        print(tables[0].parsing_report)
    except Exception as e:
        print(f"Error processing {input_file.name}: {e}")


def process_files(input_files, output_dir, pages: str = "1"):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for pdf_file in input_files:
            output_file = Path(output_dir) / f"{pdf_file.stem}.html"
            futures.append(executor.submit(process_pdf, pdf_file, output_file, pages))

        for future in concurrent.futures.as_completed(futures):
            future.result()


def main(input_path: str, output_dir: str, pages: str = "1"):
    input_path = Path(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        output_file = output_path / f"{input_path.stem}.html"
        process_pdf(input_path, output_file, pages)
    elif input_path.is_dir():
        pdf_files = list(input_path.glob("*.pdf"))
        process_files(pdf_files, output_path, pages)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3, 4):
        print(
            "Usage: python pdf_to_table.py <input_file_or_directory> [<output_directory>] [<pages>]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) == 3 else str(
        Path(input_path).parent / "output")
    pages = sys.argv[3] if len(sys.argv) == 4 else "1"
    main(input_path, output_dir, pages)

