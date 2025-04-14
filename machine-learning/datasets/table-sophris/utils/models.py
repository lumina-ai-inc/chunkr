from pydantic import BaseModel
from typing import Optional, List


class QAPair(BaseModel):
    """Model to store question-answer pairs for VLM training."""
    image_url: str
    html: str
    prompt: str
    response: Optional[str] = None
    metadata: Optional[dict] = None


class TrainingDataset(BaseModel):
    """Collection of QA pairs for a VLM training run."""
    qa_pairs: List[QAPair]
    name: str
    description: Optional[str] = None
    version: Optional[str] = None
    created_at: Optional[str] = None

