from pydantic import BaseModel, Field, model_validator, ConfigDict
from enum import Enum
from typing import Optional, List, Dict

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
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=str.title
    )
    
    title: Optional[GenerationConfig] = Field(default=None, alias="Title")
    section_header: Optional[GenerationConfig] = Field(default=None, alias="SectionHeader")
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
    target_length: Optional[int] = None

class Property(BaseModel):
    name: str
    title: Optional[str] = None
    prop_type: str
    description: Optional[str] = None
    default: Optional[str] = None

class JsonSchema(BaseModel):
    title: str
    properties: List[Property]

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

class ExtractedJson(BaseModel):
    data: Dict

class OutputResponse(BaseModel):
    chunks: List[Chunk]
    extracted_json: Optional[ExtractedJson] = Field(default=None)

class Model(str, Enum):
    FAST = "Fast"
    HIGH_QUALITY = "HighQuality"

class Configuration(BaseModel):
    chunk_processing: Optional[ChunkProcessing] = Field(default=None)
    expires_in: Optional[int] = Field(default=None)
    high_resolution: Optional[bool] = Field(default=None)
    json_schema: Optional[JsonSchema] = Field(default=None)
    model: Optional[Model] = Field(default=None)
    ocr_strategy: Optional[OcrStrategy] = Field(default=None)
    segment_processing: Optional[SegmentProcessing] = Field(default=None)
    segmentation_strategy: Optional[SegmentationStrategy] = Field(default=None)

    @model_validator(mode='before')
    def map_deprecated_fields(cls, values: Dict) -> Dict:
        if isinstance(values, dict) and "target_chunk_length" in values:
            target_length = values.pop("target_chunk_length")
            if target_length is not None:
                values["chunk_processing"] = values.get("chunk_processing", {}) or {}
                values["chunk_processing"]["target_length"] = target_length
        return values

class Status(str, Enum):
    STARTING = "Starting"
    PROCESSING = "Processing"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    CANCELLED = "Cancelled"