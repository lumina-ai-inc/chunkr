from .chunkr import Chunkr
from .models import TaskResponse, Configuration
import httpx
import io
from PIL import Image
from typing import Union, BinaryIO

class ChunkrAsync(Chunkr):
    """Async client for interacting with the Chunkr API.
    
    This class inherits from the Chunkr class but works with async HTTP requests.
    """

    async def upload(self, file: Union[str, BinaryIO, Image.Image, bytes, io.BytesIO], config: Configuration = None) -> TaskResponse:
        task = await self.start_upload(file, config)
        return await task.poll_async()

    async def start_upload(self, file: Union[str, BinaryIO, Image.Image, bytes, io.BytesIO], config: Configuration = None) -> TaskResponse:
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

 