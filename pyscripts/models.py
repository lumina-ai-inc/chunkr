from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
import dotenv
dotenv.load_dotenv(override=True)

class TableOcr(str, Enum):
    HTML = "HTML"
    JSON = "JSON"
    
    

class Model(str, Enum):
    Research = "Research"
    Fast = "Fast"
    HighQuality = "HighQuality"


class Status(str, Enum):
    Starting = "Starting"
    Processing = "Processing"
    Succeeded = "Succeeded"
    Failed = "Failed"
    Canceled = "Canceled"


class Configuration(BaseModel):
    model: Model
    target_chunk_length: int

class TaskResponse(BaseModel):
    task_id: str
    status: Status
    created_at: datetime
    finished_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    message: Optional[str] = None
    input_file_url: Optional[str] = None
    output: Optional[list[dict]] = None
    task_url: Optional[str] = None
    configuration: Configuration
    file_name: Optional[str] = None
    page_count: Optional[int] = None
    pdf_location: Optional[str] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class OcrStrategy(str, Enum):
    Auto = "Auto"
    Off = "Off"
    All = "All"


class UploadForm(BaseModel):
    file: str
    model: Model
    target_chunk_length: Optional[int] = None
    ocr_strategy: Optional[OcrStrategy] = OcrStrategy.Auto

