from fastapi import UploadFile

from pdf_layout_analysis.run_pdf_layout_analysis import analyze_pdf
from pdf_layout_analysis.run_pdf_layout_analysis_fast import analyze_pdf_fast
from toc.extract_table_of_contents import extract_table_of_contents


def get_toc(file: UploadFile, fast: bool):
    file_content = file.file.read()
    if fast:
        return extract_table_of_contents(file_content, analyze_pdf_fast(file_content))
    return extract_table_of_contents(file_content, analyze_pdf(file_content, ""))
