import string
import roman
import numpy as np
from fast_trainer.PdfSegment import PdfSegment
from pdf_features.PdfToken import PdfToken
from pdf_features.Rectangle import Rectangle
from data_model.SegmentBox import SegmentBox
from toc.data.TOCItem import TOCItem
from toc.methods.two_models_v3_segments_context_2.Modes import Modes
from toc.PdfSegmentation import PdfSegmentation


class TitleFeatures:
    SPECIAL_MARKERS = [".", "(", ")", "\\", "/", ":", ";", "-", "_", "[", "]", "•", "◦", "*", ","]
    ALPHABET = list(string.ascii_lowercase)
    ALPHABET_UPPERCASE = list(string.ascii_uppercase)
    ROMAN_NUMBERS = [roman.toRoman(i) for i in range(1, 151)]
    ROMAN_NUMBERS_LOWERCASE = [x.lower() for x in ROMAN_NUMBERS]
    BULLET_POINTS = [ALPHABET, ALPHABET_UPPERCASE, ROMAN_NUMBERS, ROMAN_NUMBERS_LOWERCASE]

    def __init__(self, pdf_segment: PdfSegment, segment_tokens: list[PdfToken], pdf_features, modes: Modes):
        self.modes = modes
        self.pdf_segment = pdf_segment
        self.pdf_features = pdf_features

        self.segment_tokens: list[PdfToken] = segment_tokens
        self.first_characters: str = ""
        self.first_characters_special_markers_count: int = 0
        self.font_size: float = 0.0
        self.text_content: str = ""
        self.width: float = 0
        self.font_family: str = ""
        self.font_color: str = ""
        self.line_height: float = 0.0
        self.uppercase: bool = False
        self.bold: float = False
        self.italics: float = False
        self.first_characters_type = 0
        self.bullet_points_type = 0
        self.text_centered: int = 0
        self.is_left: bool = False
        self.indentation: int = -1
        self.left: int = self.pdf_segment.bounding_box.left
        self.top: int = self.pdf_segment.bounding_box.top
        self.right: int = self.pdf_segment.bounding_box.right
        self.bottom: int = self.pdf_segment.bounding_box.bottom

        self.initialize_text_properties()
        self.process_first_characters()
        self.process_font_properties()
        self.process_positional_properties()

    def initialize_text_properties(self):
        words = [token.content for token in self.segment_tokens]
        self.text_content = " ".join(words)

    def process_first_characters(self):
        self.first_characters = self.text_content.split(" ")[0].split("\n")[0].split("\t")[0]
        clean_first_characters = [x for x in self.first_characters if x not in self.SPECIAL_MARKERS]
        characters_checker = {
            1: lambda x_list: len(x_list) == len([letter for letter in x_list if letter in "IVXL"]),
            2: lambda x_list: len(x_list) == len([letter for letter in x_list if letter in "IVXL".lower()]),
            3: lambda x_list: len(x_list) == len([letter for letter in x_list if letter in "1234567890"]),
            4: lambda x_list: len(x_list) == len([letter for letter in x_list if letter == letter.upper()]),
        }

        self.first_characters_type = next(
            (index for index, type_checker in characters_checker.items() if type_checker(clean_first_characters)), 0
        )

        self.bullet_points_type = (
            self.SPECIAL_MARKERS.index(self.first_characters[-1]) + 1
            if self.first_characters[-1] in self.SPECIAL_MARKERS
            else 0
        )
        self.first_characters_special_markers_count = len(
            [x for x in self.first_characters[:-1] if x in self.SPECIAL_MARKERS]
        )

    def process_font_properties(self):
        self.font_family = self.segment_tokens[0].font.font_id
        self.font_color = self.segment_tokens[0].font.color
        self.bold = sum(token.font.bold for token in self.segment_tokens) / len(self.segment_tokens)
        self.italics = sum(token.font.italics for token in self.segment_tokens) / len(self.segment_tokens)
        self.uppercase = self.text_content.upper() == self.text_content
        font_sizes = [token.font.font_size for token in self.segment_tokens]
        self.font_size = np.mean(font_sizes)

    def process_positional_properties(self):
        self.line_height = self.segment_tokens[0].font.font_size
        page_width = self.pdf_features.pages[self.pdf_segment.page_number - 1].page_width
        self.text_centered = 1 if abs(self.left - (page_width - self.right)) < 10 else 0
        self.is_left = self.left < page_width - self.right if not self.text_centered else False
        self.indentation = int((self.left - self.modes.left_space_mode) / 15) if self.is_left else -1

    def get_features_to_merge(self) -> np.array:
        return (
            1 if self.bold else 0,
            1 if self.italics else 0,
        )

    def get_features_toc(self) -> np.array:
        return (
            1 if self.bold else 0,
            1 if self.italics else 0,
            self.first_characters_type,
            self.first_characters_special_markers_count,
            self.bullet_points_type,
        )

    def get_possible_previous_point(self) -> list[str]:
        previous_characters = self.first_characters
        final_special_markers = ""
        last_part = ""
        for letter in list(reversed(previous_characters)):
            if not last_part and letter in self.SPECIAL_MARKERS:
                final_special_markers = previous_characters[-1] + final_special_markers
                previous_characters = previous_characters[:-1]
                continue

            if last_part and letter in self.SPECIAL_MARKERS:
                break

            last_part = letter + last_part
            previous_characters = previous_characters[:-1]

        previous_items = self.get_previous_items(last_part)

        if not previous_items and len(self.first_characters) >= 4:
            return [self.first_characters]

        return [previous_characters + x + final_special_markers for x in previous_items]

    def get_previous_items(self, item: str):
        previous_items = []

        for bullet_points in self.BULLET_POINTS:
            if item in bullet_points and bullet_points.index(item):
                previous_items.append(bullet_points[bullet_points.index(item) - 1])

        if item.isnumeric():
            previous_items.append(str(int(item) - 1))

        return previous_items

    @staticmethod
    def from_pdf_segmentation(pdf_segmentation: PdfSegmentation) -> list["TitleFeatures"]:
        titles_features = list()
        modes = Modes(pdf_features=pdf_segmentation.pdf_features)
        for pdf_segment in pdf_segmentation.pdf_segments:
            segment_tokens = pdf_segmentation.tokens_by_segments[pdf_segment]
            titles_features.append(TitleFeatures(pdf_segment, segment_tokens, pdf_segmentation.pdf_features, modes))

        return titles_features

    def to_toc_item(self, indentation):
        return TOCItem(
            indentation=indentation,
            label=self.text_content,
            selection_rectangle=SegmentBox.from_pdf_segment(self.pdf_segment, self.pdf_features.pages),
        )

    def append(self, other_title_features: "TitleFeatures"):
        other_segment = other_title_features.pdf_segment
        merged_bounding_box = Rectangle.merge_rectangles([self.pdf_segment.bounding_box, other_segment.bounding_box])
        merged_content = self.pdf_segment.text_content + other_segment.text_content
        merged_segment = PdfSegment(
            self.pdf_segment.page_number, merged_bounding_box, merged_content, self.pdf_segment.segment_type
        )
        segment_tokens = self.segment_tokens + other_title_features.segment_tokens
        return TitleFeatures(merged_segment, segment_tokens, pdf_features=self.pdf_features, modes=self.modes)
