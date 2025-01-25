from pathlib import Path
from PIL import Image
from typing import Union, BinaryIO

from .config import Configuration
from .decorators import anywhere, ensure_client
from .misc import prepare_upload_data
from .task_response import TaskResponse
from .chunkr_base import ChunkrBase

class Chunkr(ChunkrBase):
    """Chunkr API client that works in both sync and async contexts"""
    
    @anywhere()
    @ensure_client()
    async def upload(
        self,
        file: Union[str, Path, BinaryIO, Image.Image],
        config: Configuration = None,
    ) -> TaskResponse:
        task = await self.create_task(file, config)
        return await task.poll()

    @anywhere()
    @ensure_client()
    async def update(self, task_id: str, config: Configuration) -> TaskResponse:
        task = await self.update_task(task_id, config)
        return await task.poll()

    @anywhere()
    @ensure_client()
    async def create_task(
        self,
        file: Union[str, Path, BinaryIO, Image.Image],
        config: Configuration = None,
    ) -> TaskResponse:
        files = await prepare_upload_data(file, config, self._client)
        r = await self._client.post(
            f"{self.url}/api/v1/task", files=files, headers=self._headers()
        )
        r.raise_for_status()
        return TaskResponse(**r.json()).with_client(self)

    @anywhere()
    @ensure_client()
    async def update_task(self, task_id: str, config: Configuration) -> TaskResponse:
        files = await prepare_upload_data(None, config, self._client)
        r = await self._client.patch(
            f"{self.url}/api/v1/task/{task_id}",
            files=files,
            headers=self._headers(),
        )
        r.raise_for_status()
        return TaskResponse(**r.json()).with_client(self)

    @anywhere()
    @ensure_client()
    async def get_task(self, task_id: str) -> TaskResponse:
        r = await self._client.get(
            f"{self.url}/api/v1/task/{task_id}", headers=self._headers()
        )
        r.raise_for_status()
        return TaskResponse(**r.json()).with_client(self)

    @anywhere()
    @ensure_client()
    async def delete_task(self, task_id: str) -> None:
        r = await self._client.delete(
            f"{self.url}/api/v1/task/{task_id}", headers=self._headers()
        )
        r.raise_for_status()

    @anywhere()
    @ensure_client()
    async def cancel_task(self, task_id: str) -> None:
        r = await self._client.get(
            f"{self.url}/api/v1/task/{task_id}/cancel", headers=self._headers()
        )
        r.raise_for_status()

    @anywhere()
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None