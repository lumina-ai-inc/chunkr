import dataclasses
import hashlib
from statistics import mode

from pdf_features.PdfFeatures import PdfFeatures


@dataclasses.dataclass
class Modes:
    lines_space_mode: float
    left_space_mode: float
    right_space_mode: float
    font_size_mode: float
    font_family_name_mode: str
    font_family_mode: int
    font_family_mode_normalized: float
    pdf_features: PdfFeatures

    def __init__(self, pdf_features: PdfFeatures):
        self.pdf_features = pdf_features
        self.set_modes()

    def set_modes(self):
        line_spaces, right_spaces, left_spaces = [0], [0], [0]
        for page, token in self.pdf_features.loop_tokens():
            right_spaces.append(self.pdf_features.pages[0].page_width - token.bounding_box.right)
            left_spaces.append(token.bounding_box.left)
            line_spaces.append(token.bounding_box.bottom)

        self.lines_space_mode = mode(line_spaces)
        self.left_space_mode = mode(left_spaces)
        self.right_space_mode = mode(right_spaces)

        font_sizes = [token.font.font_size for page, token in self.pdf_features.loop_tokens() if token.font]
        self.font_size_mode = mode(font_sizes) if font_sizes else 0
        font_ids = [token.font.font_id for page, token in self.pdf_features.loop_tokens() if token.font]
        self.font_family_name_mode = mode(font_ids) if font_ids else ""
        self.font_family_mode = abs(
            int(
                str(hashlib.sha256(self.font_family_name_mode.encode("utf-8")).hexdigest())[:8],
                16,
            )
        )
        self.font_family_mode_normalized = float(f"{str(self.font_family_mode)[0]}.{str(self.font_family_mode)[1:]}")
