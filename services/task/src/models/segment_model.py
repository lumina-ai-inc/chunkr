from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
from src.models.ocr_model import OCRResponse

class SegmentType(str, Enum):
    Title = "Title"
    SectionHeader = "Section header"
    Text = "Text"
    ListItem = "List item"
    Table = "Table"
    Picture = "Picture"
    Caption = "Caption"
    Formula = "Formula"
    Footnote = "Footnote"
    PageHeader = "Page header"
    PageFooter = "Page footer"

class Segment(BaseModel):
    segment_id: str
    left: float
    top: float
    width: float
    height: float
    page_number: int
    page_width: float
    page_height: float
    text: str
    text_ocr: Optional[str] = None
    segment_type: SegmentType = Field(..., alias="type")
    ocr: Optional[OCRResponse] = None
    image: Optional[str] = None

    def update_text_ocr(self):
        """
        Extract all text from OCR results and update the text_ocr field.
        """
        if self.ocr and self.ocr.results:
            ocr_texts = [result.text for result in self.ocr.results]
            self.text_ocr = " ".join(ocr_texts)