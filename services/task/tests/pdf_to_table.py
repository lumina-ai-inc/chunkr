import camelot
    
def main(input_file: str, output_file: str):
    tables = camelot.read_pdf(input_file, pages="1")
    tables.export(output_file, f='html', compress=False)
    try:
        print(tables[0].parsing_report)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    input_file = "/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/Wellington-Altus Summer 2024 DDQ Update - Polar Long Short Fund.pdf"
    output_file = "/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/Wellington-Altus Summer 2024 DDQ Update - Polar Long Short Fund.html"
    main(input_file, output_file)