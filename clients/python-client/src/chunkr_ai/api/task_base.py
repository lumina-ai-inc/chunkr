from .config import Configuration, OutputConfiguration, OutputResponse, Status
from .protocol import ChunkrClientProtocol
from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Generic
from pydantic import BaseModel, PrivateAttr
from datetime import datetime

T = TypeVar("T", bound="TaskBase")

class TaskBase(BaseModel, ABC, Generic[T]):
    configuration: OutputConfiguration
    created_at: datetime
    expires_at: Optional[datetime]
    finished_at: Optional[datetime]
    message: str
    output: Optional[OutputResponse]
    started_at: Optional[datetime]
    status: Status
    task_id: str
    task_url: Optional[str]
    _client: Optional[ChunkrClientProtocol] = PrivateAttr(default=None)

    @abstractmethod
    def _poll_request(self) -> dict:
        """Helper method to make polling request with retry logic (synchronous)"""
        pass

    @abstractmethod
    def poll(self) -> T:
        """Poll the task for completion."""
        pass

    @abstractmethod
    def update(self, config: Configuration) -> T:
        """Update the task configuration."""
        pass

    @abstractmethod
    def cancel(self) -> T:
        """Cancel the task."""
        pass

    @abstractmethod
    def delete(self) -> T:
        """Delete the task."""
        pass

    def with_client(self, client: ChunkrClientProtocol) -> T:
        self._client = client
        return self

    def _check_status(self) -> Optional[T]:
        """Helper method to check task status and handle completion/failure"""
        if self.status == "Failed":
            raise ValueError(self.message)
        if self.status not in ("Starting", "Processing"):
            return self
        return None

    def html(self) -> str:
        """Get the full HTML of the task"""
        return self._get_content("html")

    def markdown(self) -> str:
        """Get the full markdown of the task"""
        return self._get_content("markdown")

    def content(self) -> str:
        """Get the full content of the task"""
        return self._get_content("content")

    def _get_content(self, t: str) -> str:
        if not self.output:
            return ""
        parts = []
        for c in self.output.chunks:
            for s in c.segments:
                v = getattr(s, t)
                if v:
                    parts.append(v)
        return "\n".join(parts)
