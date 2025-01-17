import asyncio
from pydantic import BaseModel, PrivateAttr
from datetime import datetime
from typing import Optional, Union
from .task_base import TaskBase
from .protocol import ChunkrClientProtocol
from .config import Configuration, OutputResponse, Status
from .misc import prepare_upload_data

class TaskResponseAsync(BaseModel, TaskBase):
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

    def with_client(self, client: Union[ChunkrClientProtocol]) -> 'TaskResponseAsync':
        self._client = client
        return self

    async def poll(self) -> 'TaskResponseAsync':
        while True:
            j = await self._poll_request()
            updated = TaskResponseAsync(**j).with_client(self._client)
            self.__dict__.update(updated.__dict__)
            if res := self._check_status():
                return res
            await asyncio.sleep(0.5)

    async def _poll_request(self) -> dict:
        if not self.task_url:
            raise ValueError("Task URL not found")
        while True:
            try:
                r = await self._client._client.get(self.task_url, headers=self._client._headers())
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if self.status == Status.FAILED:
                    raise ValueError(self.message) from e
                await asyncio.sleep(0.5)

    def _check_status(self) -> Optional['TaskResponseAsync']:
        if self.status == Status.FAILED:
            raise ValueError(f"Task failed: {self.message}")
        if self.status == Status.CANCELLED:
            return self
        if self.status not in [Status.STARTING, Status.PROCESSING]:
            return self
        return None

    async def update(self, config: Configuration) -> 'TaskResponseAsync':
        if not self.task_url:
            raise ValueError("Task URL not found")
        f = prepare_upload_data(None, config)
        r = await self._client._client.patch(self.task_url, files=f, headers=self._client._headers())
        r.raise_for_status()
        updated = TaskResponseAsync(**r.json()).with_client(self._client)
        self.__dict__.update(updated.__dict__)
        return await self.poll()

    async def cancel(self):
        if not self.task_url:
            raise ValueError("Task URL not found")
        r = await self._client._client.get(f"{self.task_url}/cancel", headers=self._client._headers())
        r.raise_for_status()
        return await self.poll()

    async def delete(self):
        r = await self._client._client.delete(self.task_url, headers=self._client._headers())
        r.raise_for_status()

    def html(self) -> str:
        return self._get_content("html")

    def markdown(self) -> str:
        return self._get_content("markdown")

    def content(self) -> str:
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

    # Satisfying TaskBase abstract methods with stubs
