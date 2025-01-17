from .config import Configuration
from .misc import prepare_upload_data
from .task_base import TaskBase
import asyncio

class TaskResponseAsync(TaskBase):
    async def _poll_request(self) -> dict:
        try:
            r = await self._client._client.get(self.task_url, headers=self._client._headers())
            r.raise_for_status()
            return r.json()
        except (ConnectionError, TimeoutError) as _:
            print("Connection error while polling the task, retrying...")
            await asyncio.sleep(0.5)
        except Exception as e:
            raise

    async def poll(self) -> 'TaskResponseAsync':
        if not self.task_url:
            raise ValueError("Task URL not found")
        while True:
            j = await self._poll_request()
            updated = TaskResponseAsync(**j).with_client(self._client)
            self.__dict__.update(updated.__dict__)
            if res := self._check_status():
                return res
            await asyncio.sleep(0.5)

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
        if not self.task_url:
            raise ValueError("Task URL not found")
        r = await self._client._client.delete(self.task_url, headers=self._client._headers())
        r.raise_for_status()