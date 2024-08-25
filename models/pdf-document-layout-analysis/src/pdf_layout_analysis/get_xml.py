import os
from os.path import join
from pathlib import Path

from configuration import XMLS_PATH


def get_xml(xml_file_name: str) -> str:
    xml_file_path = Path(join(XMLS_PATH, xml_file_name))

    with open(xml_file_path, mode="r") as file:
        content = file.read()
        os.remove(xml_file_path)
        return content
