import json
import os
import subprocess
import tempfile
from collections import Counter
from os.path import join, exists
from pathlib import Path
from statistics import mode

from lxml import etree
from lxml.etree import ElementBase, XMLSyntaxError

from pdf_features.PdfFont import PdfFont
from pdf_features.PdfModes import PdfModes
from pdf_features.PdfPage import PdfPage
from pdf_features.PdfToken import PdfToken
from pdf_token_type_labels.PdfLabels import PdfLabels
from pdf_token_type_labels.TokenType import TokenType
from pdf_tokens_type_trainer.config import (
    XML_NAME,
    LABELS_FILE_NAME,
    TOKEN_TYPE_RELATIVE_PATH,
)


class PdfFeatures:
    def __init__(
        self,
        pages: list[PdfPage],
        fonts: list[PdfFont],
        file_name="",
        file_type: str = "",
    ):
        self.pages = pages
        self.fonts = fonts
        self.file_name = file_name
        self.file_type = file_type
        self.pdf_modes: PdfModes = PdfModes()
        self.get_modes()
        self.get_mode_font()
        self.get_tokens_context()

    def loop_tokens(self):
        for page in self.pages:
            for token in page.tokens:
                yield page, token

    def set_token_types(self, labels: PdfLabels):
        if not labels.pages:
            return

        for page, token in self.loop_tokens():
            token.token_type = TokenType.from_index(labels.get_label_type(token.page_number, token.bounding_box))

    @staticmethod
    def from_poppler_etree(file_path: str | Path, file_name: str | None = None, dataset: str | None = None):
        try:
            file_content: str = open(file_path, errors="ignore").read()
        except (FileNotFoundError, UnicodeDecodeError, XMLSyntaxError):
            return None

        return PdfFeatures.from_poppler_etree_content(file_path, file_content, file_name, dataset)

    @staticmethod
    def from_poppler_etree_content(
        file_path: str | Path, file_content: str, file_name: str | None = None, dataset: str | None = None
    ):
        if not file_content:
            return PdfFeatures.get_empty()

        file_bytes: bytes = file_content.encode("utf-8")

        parser = etree.XMLParser(recover=True, encoding="utf-8")
        root: ElementBase = etree.fromstring(file_bytes, parser=parser)

        if root is None or not len(root):
            return PdfFeatures.get_empty()

        fonts: list[PdfFont] = [PdfFont.from_poppler_etree(style_tag) for style_tag in root.findall(".//fontspec")]
        fonts_by_font_id: dict[str, PdfFont] = {font.font_id: font for font in fonts}
        tree_pages: list[ElementBase] = [tree_page for tree_page in root.findall(".//page")]
        pages: list[PdfPage] = [
            PdfPage.from_poppler_etree(tree_page, fonts_by_font_id, file_name) for tree_page in tree_pages
        ]

        file_type: str = file_path.split("/")[-2] if not dataset else dataset
        file_name: str = Path(file_path).name if not file_name else file_name

        return PdfFeatures(pages, fonts, file_name, file_type)

    @staticmethod
    def contains_text(xml_path: str):
        try:
            file_content = open(xml_path).read()
            file_bytes = file_content.encode("utf-8")
            root: ElementBase = etree.fromstring(file_bytes)
            text_elements: list[ElementBase] = root.findall(".//text")
        except (FileNotFoundError, UnicodeDecodeError, XMLSyntaxError):
            return False
        return len(text_elements) > 0

    @staticmethod
    def from_pdf_path(pdf_path, xml_path: str = None):
        remove_xml = False if xml_path else True
        xml_path = xml_path if xml_path else join(tempfile.gettempdir(), "pdf_etree.xml")

        subprocess.run(["pdftohtml", "-i", "-xml", "-zoom", "1.0", pdf_path, xml_path])

        if not PdfFeatures.contains_text(xml_path):
            subprocess.run(["pdftohtml", "-i", "-hidden", "-xml", "-zoom", "1.0", pdf_path, xml_path])

        pdf_features = PdfFeatures.from_poppler_etree(xml_path, file_name=Path(pdf_path).name)

        if remove_xml and exists(xml_path):
            os.remove(xml_path)

        return pdf_features

    @staticmethod
    def from_labeled_data(pdf_labeled_data_root_path: str | Path, dataset: str, pdf_name: str):
        xml_path = join(pdf_labeled_data_root_path, "pdfs", pdf_name, XML_NAME)
        pdf_features = PdfFeatures.from_poppler_etree(xml_path, pdf_name, dataset)
        token_type_label_path: str = join(pdf_labeled_data_root_path, TOKEN_TYPE_RELATIVE_PATH)
        token_type_labels_path = join(token_type_label_path, dataset, pdf_name, LABELS_FILE_NAME)
        token_type_labels = PdfFeatures.load_labels(token_type_labels_path)
        pdf_features.set_token_types(token_type_labels)

        return pdf_features

    @staticmethod
    def load_labels(path: str) -> PdfLabels:
        if not exists(path):
            print(f"No labeled data for {path}")
            return PdfLabels(pages=[])

        labels_text = Path(path).read_text()
        labels_dict = json.loads(labels_text)
        return PdfLabels(**labels_dict)

    def get_modes(self):
        line_spaces, right_spaces = [0], [0]

        for page, token in self.loop_tokens():
            top, bottom = token.bounding_box.top, token.bounding_box.bottom
            left, right = token.bounding_box.left, token.bounding_box.right

            on_the_bottom = [page_token for page_token in page.tokens if bottom < page_token.bounding_box.top]

            on_the_right = [
                line_token
                for line_token in PdfToken.get_same_line_tokens(token, page.tokens)
                if right < line_token.bounding_box.left
            ]

            if len(on_the_bottom):
                line_spaces.append(min(map(lambda x: int(x.bounding_box.top - bottom), on_the_bottom)))

            if not on_the_right:
                right_spaces.append(int(right))

        self.pdf_modes.lines_space_mode = mode(line_spaces)
        self.pdf_modes.right_space_mode = int(self.pages[0].page_width - mode(right_spaces)) if self.pages else 0

    def get_mode_font(self):
        fonts_counter: Counter = Counter()
        for page, token in self.loop_tokens():
            fonts_counter.update([token.font.font_id])

        if len(fonts_counter.most_common()) == 0:
            return

        font_mode_id = fonts_counter.most_common()[0][0]
        font_mode_token = [font for font in self.fonts if font.font_id == font_mode_id]
        if font_mode_token:
            self.pdf_modes.font_size_mode = float(font_mode_token[0].font_size)

    def get_tokens_context(self):
        for page, token in self.loop_tokens():
            token.get_context(page.tokens)

    @staticmethod
    def get_empty():
        return PdfFeatures([], [])
