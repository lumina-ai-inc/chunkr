from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Enum
from pathlib import Path
import tempfile
import camelot

app = FastAPI()


class PDFProcessResponse(BaseModel):
    html_content: str
    table_report: dict


class OutputFormat(str, Enum):
    json = "json"
    excel = "excel"
    html = "html"
    markdown = "markdown"

    def to_extension(self):
        return "md" if self == OutputFormat.markdown else self.value


def process_pdf(input_file: Path, output_file: Path, pages: str, format: OutputFormat):
    tables = camelot.read_pdf(str(input_file), pages=pages)
    tables.export(str(output_file), f=format.value, compress=False)
    parsing_report = tables[0].parsing_report
    print(parsing_report)
    return output_file, parsing_report


@app.post("/pdf_to_table", response_model=PDFProcessResponse)
async def process_pdf_endpoint(
    file: UploadFile = File(...),
    pages: str = "1",
    format: OutputFormat = OutputFormat.html,
):
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / file.filename
        output_path = Path(temp_dir) / \
            f"{file.filename}_output.{format.to_extension()}"

        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        processed_file, parsing_report = process_pdf(
            input_path, output_path, pages, format)

        with open(processed_file, "r") as html_file:
            html_content = html_file.read()

    return PDFProcessResponse(html_content=html_content, table_report=parsing_report)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
