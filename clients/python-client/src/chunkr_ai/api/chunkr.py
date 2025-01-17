from .chunkr_base import ChunkrBase
from .config import Configuration
from .task import TaskResponse
from pathlib import Path
from PIL import Image
import requests
from typing import Union, BinaryIO
from .misc import prepare_upload_data


class Chunkr(ChunkrBase):
    """Chunkr API client"""

    def __init__(self, url: str = None, api_key: str = None):
        super().__init__(url, api_key)
        self._session = requests.Session()

    def upload(
        self,
        file: Union[str, Path, BinaryIO, Image.Image],
        config: Configuration = None,
    ) -> TaskResponse:
        task = self.create_task(file, config)
        return task.poll()

    def update(self, task_id: str, config: Configuration) -> TaskResponse:
        task = self.update_task(task_id, config)
        return task.poll()

    def create_task(
        self,
        file: Union[str, Path, BinaryIO, Image.Image],
        config: Configuration = None,
    ) -> TaskResponse:
        files = prepare_upload_data(file, config)
        if not self._session:
            raise ValueError("Session not found")
        r = self._session.post(
            f"{self.url}/api/v1/task", files=files, headers=self._headers()
        )
        r.raise_for_status()
        return TaskResponse(**r.json()).with_client(self)

    def update_task(self, task_id: str, config: Configuration) -> TaskResponse:
        files = prepare_upload_data(None, config)
        if not self._session:
            raise ValueError("Session not found")
        r = self._session.patch(
            f"{self.url}/api/v1/task/{task_id}", files=files, headers=self._headers()
        )

        r.raise_for_status()
        return TaskResponse(**r.json()).with_client(self)

    def get_task(self, task_id: str) -> TaskResponse:
        if not self._session:
            raise ValueError("Session not found")
        r = self._session.get(
            f"{self.url}/api/v1/task/{task_id}", headers=self._headers()
        )
        r.raise_for_status()
        return TaskResponse(**r.json()).with_client(self)

    def delete_task(self, task_id: str) -> None:
        if not self._session:
            raise ValueError("Session not found")
        r = self._session.delete(
            f"{self.url}/api/v1/task/{task_id}", headers=self._headers()
        )
        r.raise_for_status()

    def cancel_task(self, task_id: str) -> None:
        if not self._session:
            raise ValueError("Session not found")
        r = self._session.get(
            f"{self.url}/api/v1/task/{task_id}/cancel", headers=self._headers()
        )
        r.raise_for_status()
