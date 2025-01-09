from .auth import HeadersMixin
import asyncio
from datetime import datetime
from enum import Enum
import httpx
from pydantic import BaseModel, Field, PrivateAttr
import requests
import time
from typing import Optional, List, Dict, Union

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
    crop_image: CroppingStrategy = CroppingStrategy.ALL

class LlmGenerationConfig(BaseModel):
    html: GenerationStrategy = GenerationStrategy.LLM
    llm: Optional[LlmConfig] = None
    markdown: GenerationStrategy = GenerationStrategy.LLM
    crop_image: CroppingStrategy = CroppingStrategy.ALL

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
    chunk_processing: ChunkProcessing = Field(default_factory=ChunkProcessing)
    expires_in: Optional[int] = None
    high_resolution: bool = False
    json_schema: Optional[JsonSchema] = None
    model: Optional[Model] = Field(None, deprecated=True)
    ocr_strategy: OcrStrategy = OcrStrategy.AUTO
    segment_processing: SegmentProcessing = Field(default_factory=SegmentProcessing)
    segmentation_strategy: SegmentationStrategy = SegmentationStrategy.LAYOUT_ANALYSIS
    target_chunk_length: Optional[int] = Field(None, deprecated=True)


class Status(str, Enum):
    STARTING = "Starting"
    PROCESSING = "Processing"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"

class TaskResponse(BaseModel, HeadersMixin):
    configuration: Configuration
    created_at: datetime
    expires_at: Optional[datetime]
    file_name: Optional[str]
    finished_at: Optional[datetime]
    input_file_url: Optional[str]
    message: str
    output: Optional[OutputResponse]
    page_count: Optional[int]
    pdf_url: Optional[str]
    status: Status
    task_id: str
    task_url: Optional[str]
    _api_key: Optional[str] = PrivateAttr(default=None)

    def with_api_key(self, api_key: str) -> 'TaskResponse':
        """Helper function to set api key on a TaskResponse after creation"""
        self._api_key = api_key
        return self

    def poll(self) -> 'TaskResponse':
        """Poll the task for completion"""
        if not self.task_url:
            raise ValueError("Task URL not found in response")
        
        while True:
            r = requests.get(self.task_url, headers=self._headers())
            r.raise_for_status()
            self.__dict__.update(r.json())
            if self.status == "Failed":
                raise ValueError(self.message)
            if self.status not in ("Starting", "Processing"):
                return self
            time.sleep(0.5)

    async def poll_async(self) -> 'TaskResponse':
        """Async poll the task for completion"""
        if not self.task_url:
            raise ValueError("Task URL not found in response")
        
        async with httpx.AsyncClient() as client:
            while True:
                r = await client.get(self.task_url, headers=self._headers())
                r.raise_for_status()
                self.__dict__.update(r.json())
                if self.status == "Failed":
                    raise ValueError(self.message)
                if self.status not in ("Starting", "Processing"):
                    return self
                await asyncio.sleep(0.5)


    def _get_content(self, content_type: str) -> str:
        """Helper method to get either HTML, Markdown, or raw content."""
        if not self.output:
            return ""
        parts = []
        for c in self.output.chunks:
            for s in c.segments:
                content = getattr(s, content_type)
                if content:
                    parts.append(content)
        return "\n".join(parts)

    def html(self) -> str:
        """Get full HTML for the task"""
        return self._get_content("html")

    def markdown(self) -> str:
        """Get full markdown for the task"""
        return self._get_content("markdown")
        
    def content(self) -> str:
        """Get full text for the task"""
        return self._get_content("content")

class TaskPayload(BaseModel):
    current_configuration: Configuration
    file_name: str
    image_folder_location: str
    input_location: str
    output_location: str
    pdf_location: str
    previous_configuration: Optional[Configuration]
    task_id: str
    user_id: str
