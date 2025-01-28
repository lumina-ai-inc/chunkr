from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from typing import Any, List, Optional

class GenerationStrategy(str, Enum):
    LLM = "LLM"
    AUTO = "Auto"

class CroppingStrategy(str, Enum):
    ALL = "All"
    AUTO = "Auto"

class GenerationConfig(BaseModel):
    html: Optional[GenerationStrategy] = None
    llm: Optional[str] = None
    markdown: Optional[GenerationStrategy] = None
    crop_image: Optional[CroppingStrategy] = None

class SegmentProcessing(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=str.title)

    title: Optional[GenerationConfig] = Field(default=None, alias="Title")
    section_header: Optional[GenerationConfig] = Field(
        default=None, alias="SectionHeader"
    )
    text: Optional[GenerationConfig] = Field(default=None, alias="Text")
    list_item: Optional[GenerationConfig] = Field(default=None, alias="ListItem")
    table: Optional[GenerationConfig] = Field(default=None, alias="Table")
    picture: Optional[GenerationConfig] = Field(default=None, alias="Picture")
    caption: Optional[GenerationConfig] = Field(default=None, alias="Caption")
    formula: Optional[GenerationConfig] = Field(default=None, alias="Formula")
    footnote: Optional[GenerationConfig] = Field(default=None, alias="Footnote")
    page_header: Optional[GenerationConfig] = Field(default=None, alias="PageHeader")
    page_footer: Optional[GenerationConfig] = Field(default=None, alias="PageFooter")
    page: Optional[GenerationConfig] = Field(default=None, alias="Page")

class ChunkProcessing(BaseModel):
    ignore_headers_and_footers: Optional[bool] = None
    target_length: Optional[int] = None

class OcrStrategy(str, Enum):
    ALL = "All"
    AUTO = "Auto"

class SegmentationStrategy(str, Enum):
    LAYOUT_ANALYSIS = "LayoutAnalysis"
    PAGE = "Page"

class BoundingBox(BaseModel):
    left: float
    top: float
    width: float
    height: float

class OCRResult(BaseModel):
    bbox: BoundingBox
    text: str
    confidence: Optional[float]

class SegmentType(str, Enum):
    CAPTION = "Caption"
    FOOTNOTE = "Footnote"
    FORMULA = "Formula"
    LIST_ITEM = "ListItem"
    PAGE = "Page"
    PAGE_FOOTER = "PageFooter"
    PAGE_HEADER = "PageHeader"
    PICTURE = "Picture"
    SECTION_HEADER = "SectionHeader"
    TABLE = "Table"
    TEXT = "Text"
    TITLE = "Title"

class Segment(BaseModel):
    bbox: BoundingBox
    content: str
    page_height: float
    html: Optional[str] = None
    image: Optional[str] = None
    markdown: Optional[str] = None
    ocr: List[OCRResult]
    page_number: int
    page_width: float
    segment_id: str
    segment_type: SegmentType

class Chunk(BaseModel):
    chunk_id: str
    chunk_length: int
    segments: List[Segment]

class OutputResponse(BaseModel):
    chunks: List[Chunk]
    file_name: Optional[str]
    page_count: Optional[int]
    pdf_url: Optional[str]

class Model(str, Enum):
    FAST = "Fast"
    HIGH_QUALITY = "HighQuality"

class Pipeline(str, Enum):
    AZURE = "Azure"

class Configuration(BaseModel):
    chunk_processing: Optional[ChunkProcessing] = None
    expires_in: Optional[int] = None
    high_resolution: Optional[bool] = None
    ocr_strategy: Optional[OcrStrategy] = None
    segment_processing: Optional[SegmentProcessing] = None
    segmentation_strategy: Optional[SegmentationStrategy] = None
    pipeline: Optional[Pipeline] = None
    
class OutputConfiguration(Configuration):
    input_file_url: Optional[str] = None
    # Deprecated
    json_schema: Optional[Any] = None
    model: Optional[Model] = None
    target_chunk_length: Optional[int] = None
    
class Status(str, Enum):
    STARTING = "Starting"
    PROCESSING = "Processing"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
