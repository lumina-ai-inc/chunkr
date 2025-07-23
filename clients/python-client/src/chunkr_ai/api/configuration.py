from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from typing import Any, List, Optional, Union
from pydantic import field_validator, field_serializer

class CroppingStrategy(str, Enum):
    ALL = "All"
    AUTO = "Auto"

class SegmentFormat(str, Enum):
    HTML = "Html"
    MARKDOWN = "Markdown"

class EmbedSource(str, Enum):
    CONTENT = "Content"
    HTML = "HTML"  # Deprecated
    MARKDOWN = "Markdown"  # Deprecated
    LLM = "LLM"

class GenerationStrategy(str, Enum):
    LLM = "LLM"
    AUTO = "Auto"

class GenerationConfig(BaseModel):
    format: Optional[SegmentFormat] = None
    strategy: Optional[GenerationStrategy] = None
    llm: Optional[str] = None
    crop_image: Optional[CroppingStrategy] = None
    embed_sources: Optional[List[EmbedSource]] = None
    extended_context: Optional[bool] = None
    # Deprecated fields for backwards compatibility
    html: Optional[GenerationStrategy] = None  # Deprecated: Use format=SegmentFormat.HTML and strategy instead
    markdown: Optional[GenerationStrategy] = None  # Deprecated: Use format=SegmentFormat.MARKDOWN and strategy instead

class SegmentProcessing(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=str.title)

    caption: Optional[GenerationConfig] = Field(default=None, alias="Caption")
    footnote: Optional[GenerationConfig] = Field(default=None, alias="Footnote")
    formula: Optional[GenerationConfig] = Field(default=None, alias="Formula")
    list_item: Optional[GenerationConfig] = Field(default=None, alias="ListItem")
    page: Optional[GenerationConfig] = Field(default=None, alias="Page")
    page_footer: Optional[GenerationConfig] = Field(default=None, alias="PageFooter")
    page_header: Optional[GenerationConfig] = Field(default=None, alias="PageHeader")
    picture: Optional[GenerationConfig] = Field(default=None, alias="Picture")
    section_header: Optional[GenerationConfig] = Field(default=None, alias="SectionHeader")
    table: Optional[GenerationConfig] = Field(default=None, alias="Table")
    text: Optional[GenerationConfig] = Field(default=None, alias="Text")
    title: Optional[GenerationConfig] = Field(default=None, alias="Title")

class Tokenizer(str, Enum):
    WORD = "Word"
    CL100K_BASE = "Cl100kBase"
    XLM_ROBERTA_BASE = "XlmRobertaBase"
    BERT_BASE_UNCASED = "BertBaseUncased"

class TokenizerType(BaseModel):
    enum_value: Optional[Tokenizer] = None
    string_value: Optional[str] = None

    @classmethod
    def from_enum(cls, enum_value: Tokenizer) -> "TokenizerType":
        return cls(enum_value=enum_value)
    
    @classmethod
    def from_string(cls, string_value: str) -> "TokenizerType":
        return cls(string_value=string_value)
    
    def __str__(self) -> str:
        if self.enum_value is not None:
            return f"enum:{self.enum_value.value}"
        elif self.string_value is not None:
            return f"string:{self.string_value}"
        return ""
    
    model_config = ConfigDict()
    
    def model_dump(self, **kwargs):
        if self.enum_value is not None:
            return {"Enum": self.enum_value.value}
        elif self.string_value is not None:
            return {"String": self.string_value}
        return {}

class ChunkProcessing(BaseModel):
    ignore_headers_and_footers: Optional[bool] = True
    target_length: Optional[int] = None
    tokenizer: Optional[Union[TokenizerType, Tokenizer, str]] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    @field_serializer('tokenizer')
    def serialize_tokenizer(self, tokenizer: Optional[TokenizerType], _info):
        if tokenizer is None:
            return None
        return tokenizer.model_dump()

    @field_validator('tokenizer', mode='before')
    def validate_tokenizer(cls, v):
        if v is None:
            return None
        
        if isinstance(v, TokenizerType):
            return v
        
        if isinstance(v, Tokenizer):
            return TokenizerType(enum_value=v)
        
        if isinstance(v, dict):
            if "Enum" in v:
                try:
                    return TokenizerType(enum_value=Tokenizer(v["Enum"]))
                except ValueError:
                    return TokenizerType(string_value=v["Enum"])
            elif "String" in v:
                return TokenizerType(string_value=v["String"])
            
        if isinstance(v, str):
            try:
                return TokenizerType(enum_value=Tokenizer(v))
            except ValueError:
                return TokenizerType(string_value=v)
                
        raise ValueError(f"Cannot convert {v} to TokenizerType")

class OcrStrategy(str, Enum):
    ALL = "All"
    AUTO = "Auto"

class SegmentationStrategy(str, Enum):
    LAYOUT_ANALYSIS = "LayoutAnalysis"
    PAGE = "Page"

class ErrorHandlingStrategy(str, Enum):
    FAIL = "Fail"
    CONTINUE = "Continue"

class FallbackStrategy(BaseModel):
    type: str
    model_id: Optional[str] = None
    
    @classmethod
    def none(cls) -> "FallbackStrategy":
        return cls(type="None")
    
    @classmethod
    def default(cls) -> "FallbackStrategy":
        return cls(type="Default")
    
    @classmethod
    def model(cls, model_id: str) -> "FallbackStrategy":
        return cls(type="Model", model_id=model_id)
    
    def __str__(self) -> str:
        if self.type == "Model":
            return f"Model({self.model_id})"
        return self.type
    
    def model_dump(self, **kwargs):
        if self.type == "Model":
            return {"Model": self.model_id}
        return self.type
    
    @field_validator('type')
    def validate_type(cls, v):
        if v not in ["None", "Default", "Model"]:
            raise ValueError(f"Invalid fallback strategy: {v}")
        return v
    
    model_config = ConfigDict()
    
    @classmethod
    def model_validate(cls, obj):
        # Handle string values like "None" or "Default"
        if isinstance(obj, str):
            if obj in ["None", "Default"]:
                return cls(type=obj)
            # Try to parse as Enum value if it's not a direct match
            try:
                return cls(type=obj)
            except ValueError:
                pass  # Let it fall through to normal validation
                
        # Handle dictionary format like {"Model": "model-id"}
        elif isinstance(obj, dict) and len(obj) == 1:
            if "Model" in obj:
                return cls(type="Model", model_id=obj["Model"])
        
        # Fall back to normal validation
        return super().model_validate(obj)

class LlmProcessing(BaseModel):
    model_id: Optional[str] = None
    fallback_strategy: FallbackStrategy = Field(default_factory=FallbackStrategy.default)
    max_completion_tokens: Optional[int] = None
    temperature: float = 0.0
    
    model_config = ConfigDict()
    
    @field_serializer('fallback_strategy')
    def serialize_fallback_strategy(self, fallback_strategy: FallbackStrategy, _info):
        return fallback_strategy.model_dump()

    @field_validator('fallback_strategy', mode='before')
    def validate_fallback_strategy(cls, v):
        if isinstance(v, str):
            if v == "None":
                return FallbackStrategy.none()
            elif v == "Default":
                return FallbackStrategy.default()
            # Try to parse as a model ID if it's not None or Default
            try:
                return FallbackStrategy.model(v)
            except ValueError:
                pass  # Let it fall through to normal validation
        # Handle dictionary format like {"Model": "model-id"}
        elif isinstance(v, dict) and len(v) == 1:
            if "Model" in v:
                return FallbackStrategy.model(v["Model"])
            elif "None" in v or v.get("None") is None:
                return FallbackStrategy.none()
            elif "Default" in v or v.get("Default") is None:
                return FallbackStrategy.default()
                
        return v

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

class Alignment(str, Enum):
    LEFT = "Left"
    CENTER = "Center"
    RIGHT = "Right"
    JUSTIFY = "Justify"

class VerticalAlignment(str, Enum):
    TOP = "Top"
    MIDDLE = "Middle"
    BOTTOM = "Bottom"
    BASELINE = "Baseline"

class CellStyle(BaseModel):
    bg_color: Optional[str] = None
    text_color: Optional[str] = None
    font_face: Optional[str] = None
    is_bold: Optional[bool] = None
    align: Optional[Alignment] = None
    valign: Optional[VerticalAlignment] = None

class Cell(BaseModel):
    cell_id: str
    text: str
    range: str
    formula: Optional[str] = None
    value: Optional[str] = None
    hyperlink: Optional[str] = None
    style: Optional[CellStyle] = None

class Page(BaseModel):
    image: str
    page_number: int
    page_height: float
    page_width: float
    ss_sheet_name: Optional[str] = None

class Segment(BaseModel):
    bbox: BoundingBox
    content: str = ""
    page_height: float
    llm: Optional[str] = None
    html: Optional[str] = None
    image: Optional[str] = None
    markdown: Optional[str] = None
    ocr: Optional[List[OCRResult]] = Field(default_factory=list)
    page_number: int
    page_width: float
    segment_id: str
    segment_type: SegmentType
    confidence: Optional[float]
    text: str = ""
    segment_length: Optional[int] = None
    # Spreadsheet-specific fields
    ss_cells: Optional[List[Cell]] = None
    ss_header_bbox: Optional[BoundingBox] = None
    ss_header_ocr: Optional[List[OCRResult]] = None
    ss_header_text: Optional[str] = None
    ss_header_range: Optional[str] = None
    ss_range: Optional[str] = None
    ss_sheet_name: Optional[str] = None

class Chunk(BaseModel):
    chunk_id: str
    chunk_length: int
    segments: List[Segment]
    embed: Optional[str] = None

class OutputResponse(BaseModel):
    chunks: List[Chunk]
    file_name: Optional[str]
    mime_type: Optional[str] = None
    pages: Optional[List[Page]] = None
    page_count: Optional[int]
    pdf_url: Optional[str]

class Model(str, Enum):
    FAST = "Fast"
    HIGH_QUALITY = "HighQuality"

class Pipeline(str, Enum):
    AZURE = "Azure"
    CHUNKR = "Chunkr"

class Configuration(BaseModel):
    chunk_processing: Optional[ChunkProcessing] = None
    expires_in: Optional[int] = None
    error_handling: Optional[ErrorHandlingStrategy] = None
    high_resolution: Optional[bool] = None
    ocr_strategy: Optional[OcrStrategy] = None
    segment_processing: Optional[SegmentProcessing] = None
    segmentation_strategy: Optional[SegmentationStrategy] = None
    pipeline: Optional[Pipeline] = None
    llm_processing: Optional[LlmProcessing] = None
    
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
