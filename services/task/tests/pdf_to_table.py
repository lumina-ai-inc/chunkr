import camelot
import concurrent.futures
from pathlib import Path
import sys


def process_pdf(input_file: Path, output_file: Path):
    tables = camelot.read_pdf(str(input_file), pages="1")
    tables.export(str(output_file), f='html', compress=False)
    try:
        print(f"Processing {input_file.name}:")
        print(tables[0].parsing_report)
    except Exception as e:
        print(f"Error processing {input_file.name}: {e}")


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
            "Usage: python pdf_to_table.py <input_file_or_directory> [<output_directory>]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) == 3 else str(
        Path(input_path).parent / "output")
    main(input_path, output_dir)
