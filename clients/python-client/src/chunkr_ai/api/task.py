from .protocol import ChunkrClientProtocol
from .config import Configuration, OutputResponse
import asyncio
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, PrivateAttr
import time
from typing import Optional, Union

class Status(str, Enum):
    STARTING = "Starting"
    PROCESSING = "Processing"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    
class TaskResponse(BaseModel):
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
    started_at: Optional[datetime]
    status: Status
    task_id: str
    task_url: Optional[str]
    _client: Optional[Union[ChunkrClientProtocol]] = PrivateAttr(default=None)

    def with_client(self, client: Union[ChunkrClientProtocol]) -> 'TaskResponse':
        self._client = client
        return self
    
    def _poll_request_sync(self) -> dict:
        """Helper method to make polling request with retry logic (synchronous)"""
        if not self.task_url:
            raise ValueError("Task URL not found in response")

        while True:
            try:
                r = self._client._session.get(self.task_url, headers=self._client._headers())
                r.raise_for_status()
                return r.json()
            except (ConnectionError, TimeoutError) as _:
                print("Connection error while polling the task, retrying...")
                time.sleep(0.5)
            except Exception as e:
                raise

    async def _poll_request_async(self) -> dict:
        """Helper method to make polling request with retry logic (asynchronous)"""
        if not self.task_url:
            raise ValueError("Task URL not found in response")

        while True:
            try:
                r = await self._client._client.get(self.task_url, headers=self._client._headers())
                r.raise_for_status()
                response = r.json()
                return response
            except (ConnectionError, TimeoutError) as _:
                print("Connection error while polling the task, retrying...")
                await asyncio.sleep(0.5)
            except Exception as e:
                raise

    def _check_status(self) -> Optional['TaskResponse']:
        """Helper method to check task status and handle completion/failure"""
        if self.status == "Failed":
            raise ValueError(self.message)
        if self.status not in ("Starting", "Processing"):
            return self
        return None

    def poll(self) -> 'TaskResponse':
        """Poll the task for completion."""
        while True:
            response = self._poll_request_sync()
            self.__dict__.update(response)
            
            if result := self._check_status():
                return result
            
            time.sleep(0.5)

    async def poll_async(self) -> 'TaskResponse':
        """Poll the task for completion asynchronously."""
        while True:
            response = await self._poll_request_async()
            self.__dict__.update(response)
            
            if result := self._check_status():
                return result
            
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