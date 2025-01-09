from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List, Dict

class GenerationStrategy(str, Enum):
    LLM = "LLM"
    AUTO = "Auto"

class CroppingStrategy(str, Enum):
    ALL = "All" 
    AUTO = "Auto"

class LlmConfig(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.0

class AutoGenerationConfig(BaseModel):
    html: GenerationStrategy = GenerationStrategy.AUTO
    llm: Optional[LlmConfig] = None
    markdown: GenerationStrategy = GenerationStrategy.AUTO
    crop_image: CroppingStrategy = CroppingStrategy.AUTO

class LlmGenerationConfig(BaseModel):
    html: GenerationStrategy = GenerationStrategy.LLM
    llm: Optional[LlmConfig] = None
    markdown: GenerationStrategy = GenerationStrategy.LLM
    crop_image: CroppingStrategy = CroppingStrategy.AUTO

class SegmentProcessing(BaseModel):
    title: AutoGenerationConfig = Field(default_factory=AutoGenerationConfig)
    section_header: AutoGenerationConfig = Field(default_factory=AutoGenerationConfig)
    text: AutoGenerationConfig = Field(default_factory=AutoGenerationConfig)
    list_item: AutoGenerationConfig = Field(default_factory=AutoGenerationConfig)
    table: LlmGenerationConfig = Field(default_factory=LlmGenerationConfig)
    picture: AutoGenerationConfig = Field(default_factory=AutoGenerationConfig)
    caption: AutoGenerationConfig = Field(default_factory=AutoGenerationConfig)
    formula: LlmGenerationConfig = Field(default_factory=LlmGenerationConfig)
    footnote: AutoGenerationConfig = Field(default_factory=AutoGenerationConfig)
    page_header: AutoGenerationConfig = Field(default_factory=AutoGenerationConfig)
    page_footer: AutoGenerationConfig = Field(default_factory=AutoGenerationConfig)
    page: AutoGenerationConfig = Field(default_factory=AutoGenerationConfig)

class ChunkProcessing(BaseModel):
    target_length: int = 512

class Property(BaseModel):
    name: str
    title: Optional[str]
    prop_type: str
    description: Optional[str]
    default: Optional[str]

class JsonSchema(BaseModel):
    title: str
    properties: List[Property]
    schema_type: Optional[str]

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
    html: Optional[str]
    image: Optional[str]
    markdown: Optional[str]
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
    chunks: List[Chunk] = []
    extracted_json: Optional[ExtractedJson]

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
    target_chunk_length: Optional[int] = Field(default=None)

class Status(str, Enum):
    STARTING = "Starting"
    PROCESSING = "Processing"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"