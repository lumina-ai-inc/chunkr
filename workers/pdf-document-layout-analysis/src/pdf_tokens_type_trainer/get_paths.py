from os.path import join
from pathlib import Path


def get_xml_path(pdf_labeled_data_project_path: str):
    return Path(join(pdf_labeled_data_project_path, "pdfs"))
