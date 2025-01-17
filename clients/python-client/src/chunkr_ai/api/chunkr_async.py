from .chunkr_base import ChunkrBase
from .config import Configuration
from .misc import prepare_upload_data
from .task_async import TaskResponseAsync
import httpx
from pathlib import Path
from PIL import Image
from typing import Union, BinaryIO

class ChunkrAsync(ChunkrBase):
    """Asynchronous Chunkr API client"""
    
    def __init__(self, url: str = None, api_key: str = None):
        super().__init__(url, api_key)
        self._client = httpx.AsyncClient()

    async def upload(self, file: Union[str, Path, BinaryIO, Image.Image], config: Configuration = None) -> TaskResponseAsync:
        if not self._client or self._client.is_closed:
            self._client = httpx.AsyncClient()
        try:
            task = await self.create_task(file, config)
            return await task.poll()
        except Exception as e:
            await self._client.aclose()
            raise e
    
    async def update(self, task_id: str, config: Configuration) -> TaskResponseAsync:
        if not self._client or self._client.is_closed:
            self._client = httpx.AsyncClient()
        try:
            task = await self.update_task(task_id, config)
            return await task.poll()
        except Exception as e:
            await self._client.aclose()
            raise e

    async def create_task(self, file: Union[str, Path, BinaryIO, Image.Image], config: Configuration = None) -> TaskResponseAsync:
        if not self._client or self._client.is_closed:
            self._client = httpx.AsyncClient()
        try:
            files = prepare_upload_data(file, config)
            r = await self._client.post(
                f"{self.url}/api/v1/task",
                files=files,
                headers=self._headers()
            )
            r.raise_for_status()
            return TaskResponseAsync(**r.json()).with_client(self)
        except Exception as e:
            await self._client.aclose()
            raise e

    async def update_task(self, task_id: str, config: Configuration) -> TaskResponseAsync:
        if not self._client or self._client.is_closed:
            self._client = httpx.AsyncClient()
        try:
            files = prepare_upload_data(None, config)
            r = await self._client.patch(
                f"{self.url}/api/v1/task/{task_id}",
                files=files,
                headers=self._headers()
            )
     
            r.raise_for_status()
            return TaskResponseAsync(**r.json()).with_client(self)
        except Exception as e:
            await self._client.aclose()
            raise e
    
    async def get_task(self, task_id: str) -> TaskResponseAsync:
        if not self._client or self._client.is_closed:
            self._client = httpx.AsyncClient()
        try:
            r = await self._client.get(
                f"{self.url}/api/v1/task/{task_id}",
                headers=self._headers()
            )
            r.raise_for_status()
            return TaskResponseAsync(**r.json()).with_client(self)
        except Exception as e:
            await self._client.aclose()
            raise e
    
    async def delete_task(self, task_id: str) -> None:
        if not self._client or self._client.is_closed:
            self._client = httpx.AsyncClient()
        try:
            r = await self._client.delete(
                f"{self.url}/api/v1/task/{task_id}",
                headers=self._headers()
            )
            r.raise_for_status()
        except Exception as e:
            await self._client.aclose()
            raise e
    
    async def cancel_task(self, task_id: str) -> None:
        if not self._client or self._client.is_closed:
            self._client = httpx.AsyncClient()
        try:
            r = await self._client.get(
                f"{self.url}/api/v1/task/{task_id}/cancel",
                headers=self._headers()
            )
            r.raise_for_status()
        except Exception as e:
            await self._client.aclose()
            raise e

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()