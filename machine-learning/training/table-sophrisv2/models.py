from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union
from PIL import Image


class ContentItem(BaseModel):
    type: str
    text: str | None = None
    image: Image.Image | None = None

    class Config:
        arbitrary_types_allowed = True


class Message(BaseModel):
    role: str
    content: List[ContentItem]


class Conversation(BaseModel):
    messages: List[Message]


class TableTrainingSample(BaseModel):
    """Represents a single training sample with image and HTML."""
    image: Image.Image
    html: str
    table_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

