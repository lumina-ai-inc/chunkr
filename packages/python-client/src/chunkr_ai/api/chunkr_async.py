from .chunkr import Chunkr
from .models import TaskResponse, Configuration
import httpx
from pathlib import Path
from PIL import Image
from typing import Union, BinaryIO

class ChunkrAsync(Chunkr):
    """Async client for interacting with the Chunkr API.
    
    This class inherits from the Chunkr class but works with async HTTP requests.
    """

    async def upload(self, file: Union[str, Path, BinaryIO, Image.Image], config: Configuration = None) -> TaskResponse:
        """Upload a file and wait for processing to complete.

        Args:
            file: The file to upload. 
            config: Configuration options for processing. Optional.

        Examples:
        ```python
        # Upload from file path
        task = await chunkr.upload("document.pdf")

        # Upload from opened file
        with open("document.pdf", "rb") as f:
            task = await chunkr.upload(f)

        # Upload an image
        from PIL import Image
        img = Image.open("photo.jpg")
        task = await chunkr.upload(img)
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
        url = f"{self.url}/api/v1/task"
        filename, file_obj = self._prepare_file(file)
        async with httpx.AsyncClient() as client:
            files = {"file": (filename, file_obj)}
            r = await client.post(
                url, 
                files=files, 
                json=config.dict() if config else {},
                headers=self._headers()
            )
            r.raise_for_status()
            return TaskResponse(**r.json()).with_api_key(self._api_key)

    async def get_task(self, task_id: str) -> TaskResponse:
        url = f"{self.url}/api/v1/task/{task_id}"
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=self._headers())
            r.raise_for_status()
            return TaskResponse(**r.json()).with_api_key(self._api_key)

 