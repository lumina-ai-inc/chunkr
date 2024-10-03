import camelot
    
def main(input_file: str, output_file: str):
    tables = camelot.read_pdf(input_file)
    tables.export(output_file, f='html', compress=False)
    try:
        print(tables[0].parsing_report)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    input_file = "/Users/akhileshsharma/Downloads/CIM-05-Arion-Banki-hf.pdf"
    output_file = "/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/CIM-05-Arion-Banki-hf.html"
    main(input_file, output_file)
