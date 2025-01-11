from .base import ChunkrBase
from .task import TaskResponse
from .config import Configuration
import httpx
from pathlib import Path
from PIL import Image
from typing import Union, BinaryIO

class ChunkrAsync(ChunkrBase):
    """Asynchronous Chunkr API client"""
    
    def __init__(self, url: str = None, api_key: str = None):
        super().__init__(url, api_key)
        self._client = httpx.AsyncClient()

    async def upload(self, file: Union[str, Path, BinaryIO, Image.Image], config: Configuration = None) -> TaskResponse:
        """Upload a file and wait for processing to complete.

        Args:
            file: The file to upload. 
            config: Configuration options for processing. Optional.

        Examples:
        ```python
        # Upload from file path
        await chunkr.upload("document.pdf")

        # Upload from opened file
        with open("document.pdf", "rb") as f:
            await chunkr.upload(f)
        
        # Upload from URL
        await chunkr.upload("https://example.com/document.pdf")

        # Upload from base64 string (must include MIME type header)
        await chunkr.upload("data:application/pdf;base64,JVBERi0xLjcKCjEgMCBvYmo...")

        # Upload an image
        from PIL import Image
        img = Image.open("photo.jpg")
        await chunkr.upload(img)
        ```
        Returns:
            TaskResponse: The completed task response
        """
        task = await self.start_upload(file, config)
        return await task.poll_async()

    async def start_upload(self, file: Union[str, Path, BinaryIO, Image.Image], config: Configuration = None) -> TaskResponse:
        """Upload a file for processing and immediately return the task response. It will not wait for processing to complete. To wait for the full processing to complete, use `task.poll_async()`.

        Args:
            file: The file to upload.
            config: Configuration options for processing. Optional.

        Examples:
        ```
        # Upload from file path
        task = await chunkr.start_upload("document.pdf")

        # Upload from opened file
        with open("document.pdf", "rb") as f:
            task = await chunkr.start_upload(f)
    
        # Upload from URL
        task = await chunkr.start_upload("https://example.com/document.pdf")

        # Upload from base64 string (must include MIME type header)
        task = await chunkr.start_upload("data:application/pdf;base64,JVBERi0xLjcKCjEgMCBvYmo...")

        # Upload an image
        from PIL import Image
        img = Image.open("photo.jpg")
        task = await chunkr.start_upload(img)

        # Wait for the task to complete - this can be done when needed
        await task.poll_async()
        ```

        Returns:
            TaskResponse: The initial task response
        """
        files, data = self._prepare_upload_data(file, config)
        r = await self._client.post(
            f"{self.url}/api/v1/task",
            files=files,
            json=config.model_dump() if config else {},
            headers=self._headers()
        )
        r.raise_for_status()
        return TaskResponse(**r.json()).with_client(self)

    async def get_task(self, task_id: str) -> TaskResponse:
        r = await self._client.get(
            f"{self.url}/api/v1/task/{task_id}",
            headers=self._headers()
        )
        r.raise_for_status()
        return TaskResponse(**r.json()).with_client(self)
    
    async def delete_task(self, task_id: str) -> None:
        r = await self._client.delete(
            f"{self.url}/api/v1/task/{task_id}",
            headers=self._headers()
        )
        r.raise_for_status()
    
    async def cancel_task(self, task_id: str) -> None:
        r = await self._client.post(
            f"{self.url}/api/v1/task/{task_id}/cancel",
            headers=self._headers()
        )
        r.raise_for_status()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()