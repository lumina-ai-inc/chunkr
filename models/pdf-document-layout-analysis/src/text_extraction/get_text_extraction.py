from fastapi import UploadFile
from pdf_token_type_labels.TokenType import TokenType
from pdf_layout_analysis.run_pdf_layout_analysis import analyze_pdf
from pdf_layout_analysis.run_pdf_layout_analysis_fast import analyze_pdf_fast
from text_extraction.extract_text import extract_text


def get_text_extraction(file: UploadFile, fast: bool, types: str):
    file_content = file.file.read()
    if types == "all":
        token_types: list[TokenType] = [t for t in TokenType]
    else:
        token_types = list(set([TokenType.from_text(t.strip().replace(" ", "_")) for t in types.split(",")]))
    if fast:
        return extract_text(analyze_pdf_fast(file_content), token_types)
    return extract_text(analyze_pdf(file_content, ""), token_types)
