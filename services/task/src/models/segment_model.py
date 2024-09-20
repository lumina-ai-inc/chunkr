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
    left: float
    top: float
    width: float
    height: float
    page_number: int
    page_width: float
    page_height: float
    text: str
    segment_type: SegmentType = Field(..., alias="type")
    segment_id: str
    ocr: Optional[OCRResponse] = None
    image: Optional[str] = None

