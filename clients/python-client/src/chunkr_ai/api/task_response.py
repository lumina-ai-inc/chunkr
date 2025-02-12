from datetime import datetime
from typing import TypeVar, Optional, Generic
from pydantic import BaseModel, PrivateAttr
import asyncio
import json
import os

from .configuration import Configuration, OutputConfiguration, OutputResponse, Status
from .protocol import ChunkrClientProtocol
from .misc import prepare_upload_data
from .decorators import anywhere, require_task, retry_on_429

T = TypeVar("T", bound="TaskResponse")

class TaskResponse(BaseModel, Generic[T]):
    configuration: OutputConfiguration
    created_at: datetime
    expires_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    message: Optional[str] = None
    output: Optional[OutputResponse] = None
    started_at: Optional[datetime] = None
    status: Status
    task_id: str
    task_url: Optional[str] = None
    include_chunks: bool = False
    _base64_urls: bool = False
    _client: Optional[ChunkrClientProtocol] = PrivateAttr(default=None)

    def with_client(self, client: ChunkrClientProtocol, include_chunks: bool = False, base64_urls: bool = False) -> T:
        self._client = client
        self.include_chunks = include_chunks
        self._base64_urls = base64_urls
        return self

    def _check_status(self) -> Optional[T]:
        """Helper method to check task status and handle completion/failure"""
        if self.status == "Failed":
            if getattr(self._client, 'raise_on_failure', True):
                raise ValueError(self.message)
            return self
        if self.status not in ("Starting", "Processing"):
            return self
        return None

    @require_task()
    async def _poll_request(self) -> dict:
        try:
            r = await self._client._client.get(
                self.task_url, headers=self._client._headers()
            )
            r.raise_for_status()
            return r.json()
        except (ConnectionError, TimeoutError, OSError) as e:
            print(f"Connection error while polling the task: {str(e)}, retrying...")
            await asyncio.sleep(0.5)
            return await self._poll_request() 
        except Exception as e:
            raise e

    @anywhere()
    async def poll(self) -> T:
        """Poll the task for completion."""
        while True:
            j = await self._poll_request()
            updated = TaskResponse(**j).with_client(self._client)
            self.__dict__.update(updated.__dict__)
            if res := self._check_status():
                return res
            await asyncio.sleep(0.5)

    @anywhere()
    @require_task()
    @retry_on_429()
    async def update(self, config: Configuration) -> T:
        """Update the task configuration."""
        data = await prepare_upload_data(None, None, config)
        r = await self._client._client.patch(
            f"{self.task_url}/parse",
            json=data,
            headers=self._client._headers()
        )
        r.raise_for_status()
        updated = TaskResponse(**r.json()).with_client(self._client)
        self.__dict__.update(updated.__dict__)
        return await self.poll()

    @anywhere()
    @require_task()
    async def delete(self) -> T:
        """Delete the task."""
        r = await self._client._client.delete(
            self.task_url, headers=self._client._headers()
        )
        r.raise_for_status()
        return self

    @anywhere()
    @require_task()
    async def cancel(self) -> T:
        """Cancel the task."""
        r = await self._client._client.get(
            f"{self.task_url}/cancel", headers=self._client._headers()
        )
        r.raise_for_status()
        return await self.poll()

    def _write_to_file(self, content: str | dict, output_file: str, is_json: bool = False) -> None:
        """Helper method to write content to a file
        
        Args:
            content: Content to write (string or dict for JSON)
            output_file: Path to save the content
            is_json: Whether the content should be written as JSON
        """
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)
        if output_file:
            directory = os.path.dirname(output_file)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                if is_json:
                    json.dump(content, f, cls=DateTimeEncoder, indent=2)
                else:
                    f.write(content)

    def html(self, output_file: str = None) -> str:
        """Get the full HTML of the task
        
        Args:
            output_file (str, optional): Path to save the HTML content. Defaults to None.
        """
        content = self._get_content("html")
        self._write_to_file(content, output_file)
        return content

    def markdown(self, output_file: str = None) -> str:
        """Get the full markdown of the task
        
        Args:
            output_file (str, optional): Path to save the markdown content. Defaults to None.
        """
        content = self._get_content("markdown", separator="\n\n")
        self._write_to_file(content, output_file)
        return content

    def content(self, output_file: str = None) -> str:
        """Get the full content of the task
        
        Args:
            output_file (str, optional): Path to save the content. Defaults to None.
        """
        content = self._get_content("content")
        self._write_to_file(content, output_file)
        return content
    
    def json(self, output_file: str = None) -> dict:
        """Get the full task data as JSON
        
        Args:
            output_file (str, optional): Path to save the task data as JSON. Defaults to None.
        """
        data = self.model_dump()
        self._write_to_file(data, output_file, is_json=True)
        return data

    def _get_content(self, t: str, separator: str = "\n") -> str:
        if not self.output:
            return ""
        parts = []
        for c in self.output.chunks:
            for s in c.segments:
                v = getattr(s, t)
                if v:
                    parts.append(v)
        return separator.join(parts)
