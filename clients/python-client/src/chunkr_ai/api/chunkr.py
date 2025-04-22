from pathlib import Path
from PIL import Image
from typing import Union, BinaryIO, Optional, cast, Awaitable

from .configuration import Configuration
from .decorators import anywhere, ensure_client, retry_on_429
from .misc import prepare_upload_data
from .task_response import TaskResponse
from .chunkr_base import ChunkrBase
from .protocol import ChunkrClientProtocol

class Chunkr(ChunkrBase):
    """Chunkr API client that works in both sync and async contexts"""
    
    @anywhere()
    @ensure_client()
    async def upload(
        self,
        file: Union[str, Path, BinaryIO, Image.Image, bytes, bytearray, memoryview],
        config: Optional[Configuration] = None,
        filename: Optional[str] = None,
    ) -> TaskResponse:
        task = await cast(Awaitable[TaskResponse], self.create_task(file, config, filename))
        return await cast(Awaitable[TaskResponse], task.poll())

    @anywhere()
    @ensure_client()
    async def update(self, task_id: str, config: Configuration) -> TaskResponse:
        task = await cast(Awaitable[TaskResponse], self.update_task(task_id, config))
        return await cast(Awaitable[TaskResponse], task.poll())

    @anywhere()
    @ensure_client()
    @retry_on_429()
    async def create_task(
        self,
        file: Union[str, Path, BinaryIO, Image.Image, bytes, bytearray, memoryview],
        config: Optional[Configuration] = None,
        filename: Optional[str] = None,
    ) -> TaskResponse:
        """Create a new task with the given file and configuration."""
        data = await prepare_upload_data(file, filename, config)
        assert self._client is not None
        r = await self._client.post(
            f"{self.url}/api/v1/task/parse", json=data, headers=self._headers()
        )
        r.raise_for_status()
        return TaskResponse(**r.json()).with_client(cast(ChunkrClientProtocol, self), True, False)

    @anywhere()
    @ensure_client()
    @retry_on_429()
    async def update_task(self, task_id: str, config: Optional[Configuration] = None) -> TaskResponse:
        """Update an existing task with new configuration."""
        data = await prepare_upload_data(None, None, config)
        assert self._client is not None
        r = await self._client.patch(
            f"{self.url}/api/v1/task/{task_id}/parse",
            json=data,
            headers=self._headers(),
        )
        r.raise_for_status()
        return TaskResponse(**r.json()).with_client(cast(ChunkrClientProtocol, self), True, False)

    @anywhere()
    @ensure_client()
    async def get_task(self, task_id: str, include_chunks: bool = True, base64_urls: bool = False) -> TaskResponse:
        params = {
            "base64_urls": str(base64_urls).lower(),
            "include_chunks": str(include_chunks).lower()
        }
        assert self._client is not None
        r = await self._client.get(
            f"{self.url}/api/v1/task/{task_id}",
            params=params,
            headers=self._headers()
        )
        r.raise_for_status()
        return TaskResponse(**r.json()).with_client(cast(ChunkrClientProtocol, self), include_chunks, base64_urls)

    @anywhere()
    @ensure_client()
    async def delete_task(self, task_id: str) -> None:
        assert self._client is not None
        r = await self._client.delete(
            f"{self.url}/api/v1/task/{task_id}", headers=self._headers()
        )
        r.raise_for_status()

    @anywhere()
    @ensure_client()
    async def cancel_task(self, task_id: str) -> None:
        assert self._client is not None
        r = await self._client.get(
            f"{self.url}/api/v1/task/{task_id}/cancel", headers=self._headers()
        )
        r.raise_for_status()

    @anywhere()
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None