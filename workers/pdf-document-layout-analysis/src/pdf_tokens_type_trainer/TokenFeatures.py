import string
import unicodedata

from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfToken import PdfToken
from pdf_tokens_type_trainer.config import CHARACTER_TYPE


class TokenFeatures:
    def __init__(self, pdfs_features: PdfFeatures):
        self.pdfs_features = pdfs_features

    def get_features(self, token_1: PdfToken, token_2: PdfToken, page_tokens: list[PdfToken]):
        same_font = True if token_1.font.font_id == token_2.font.font_id else False

        return (
            [
                same_font,
                self.pdfs_features.pdf_modes.font_size_mode / 100,
                len(token_1.content),
                len(token_2.content),
                token_1.content.count(" "),
                token_2.content.count(" "),
                sum(character in string.punctuation for character in token_1.content),
                sum(character in string.punctuation for character in token_2.content),
            ]
            + self.get_position_features(token_1, token_2, page_tokens)
            + self.get_unicode_categories(token_1)
            + self.get_unicode_categories(token_2)
        )

    def get_position_features(self, token_1: PdfToken, token_2: PdfToken, page_tokens):
        left_1 = token_1.bounding_box.left
        right_1 = token_1.bounding_box.right
        height_1 = token_1.bounding_box.height
        width_1 = token_1.bounding_box.width

        left_2 = token_2.bounding_box.left
        right_2 = token_2.bounding_box.right
        height_2 = token_2.bounding_box.height
        width_2 = token_2.bounding_box.width

        right_gap_1, left_gap_2 = (
            token_1.pdf_token_context.left_of_token_on_the_right - right_1,
            left_2 - token_2.pdf_token_context.right_of_token_on_the_left,
        )

        absolute_right_1 = max(right_1, token_1.pdf_token_context.right_of_token_on_the_right)
        absolute_right_2 = max(right_2, token_2.pdf_token_context.right_of_token_on_the_right)

        absolute_left_1 = min(left_1, token_1.pdf_token_context.left_of_token_on_the_left)
        absolute_left_2 = min(left_2, token_2.pdf_token_context.left_of_token_on_the_left)

        right_distance, left_distance, height_difference = left_2 - left_1 - width_1, left_1 - left_2, height_1 - height_2

        top_distance = token_2.bounding_box.top - token_1.bounding_box.top - height_1
        top_distance_gaps = self.get_top_distance_gap(token_1, token_2, page_tokens)

        start_lines_differences = absolute_left_1 - absolute_left_2
        end_lines_difference = abs(absolute_right_1 - absolute_right_2)

        return [
            absolute_right_1,
            token_1.bounding_box.top,
            right_1,
            width_1,
            height_1,
            token_2.bounding_box.top,
            right_2,
            width_2,
            height_2,
            right_distance,
            left_distance,
            right_gap_1,
            left_gap_2,
            height_difference,
            top_distance,
            top_distance - self.pdfs_features.pdf_modes.lines_space_mode,
            top_distance_gaps,
            top_distance - height_1,
            end_lines_difference,
            start_lines_differences,
            self.pdfs_features.pdf_modes.lines_space_mode - top_distance_gaps,
            self.pdfs_features.pdf_modes.right_space_mode - absolute_right_1,
        ]

    @staticmethod
    def get_top_distance_gap(token_1: PdfToken, token_2: PdfToken, page_tokens):
        top_distance = token_2.bounding_box.top - token_1.bounding_box.top - token_1.bounding_box.height
        tokens_in_the_middle = [
            token
            for token in page_tokens
            if token_1.bounding_box.bottom <= token.bounding_box.top < token_2.bounding_box.top
        ]

        gap_middle_bottom = 0
        gap_middle_top = 0

        if tokens_in_the_middle:
            tokens_in_the_middle_top = min([token.bounding_box.top for token in tokens_in_the_middle])
            tokens_in_the_middle_bottom = max([token.bounding_box.bottom for token in tokens_in_the_middle])
            gap_middle_top = tokens_in_the_middle_top - token_1.bounding_box.top - token_1.bounding_box.height
            gap_middle_bottom = token_2.bounding_box.top - tokens_in_the_middle_bottom

        top_distance_gaps = top_distance - (gap_middle_bottom - gap_middle_top)
        return top_distance_gaps

    @staticmethod
    def get_unicode_categories(token: PdfToken):
        if token.id == "pad_token":
            return [-1] * len(CHARACTER_TYPE) * 4

        categories = [unicodedata.category(letter) for letter in token.content[:2] + token.content[-2:]]
        categories += ["no_category"] * (4 - len(categories))

        categories_one_hot_encoding = list()

        for category in categories:
            categories_one_hot_encoding.extend([0] * len(CHARACTER_TYPE))
            if category not in CHARACTER_TYPE:
                continue

            category_index = len(categories_one_hot_encoding) - len(CHARACTER_TYPE) + CHARACTER_TYPE.index(category)
            categories_one_hot_encoding[category_index] = 1

        return categories_one_hot_encoding
