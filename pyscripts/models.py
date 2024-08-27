from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
import dotenv

dotenv.load_dotenv(override=True)


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


class TaskResponse(BaseModel):
    task_id: str
    status: Status
    created_at: datetime
    finished_at: Optional[str] = None
    expiration_time: Optional[datetime] = None
    message: Optional[str] = None
    input_file_url: Optional[str] = None
    output_file_url: Optional[str] = None
    task_url: Optional[str] = None
    model: Model

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
