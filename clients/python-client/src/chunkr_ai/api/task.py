from .config import Configuration
from .misc import prepare_upload_data
from .task_base import TaskBase
import time

class TaskResponse(TaskBase):
    def _poll_request(self) -> dict:
        while True:
            try:
                r = self._client._session.get(self.task_url, headers=self._client._headers())
                r.raise_for_status()
                return r.json()
            except (ConnectionError, TimeoutError) as _:
                print("Connection error while polling the task, retrying...")
                time.sleep(0.5)
            except Exception as e:
                raise

    def poll(self) -> 'TaskResponse':
        if not self.task_url:
            raise ValueError("Task URL not found in response")
        while True:
            response = self._poll_request_sync()
            updated_task = TaskResponse(**response).with_client(self._client)
            self.__dict__.update(updated_task.__dict__)
            if result := self._check_status():
                return result
            time.sleep(0.5)
    
    def update(self, config: Configuration) -> 'TaskResponse':
        if not self.task_url:
            raise ValueError("Task URL not found")
        files = prepare_upload_data(None, config)
        r = self._client._session.patch(
            f"{self.task_url}",
            files=files,
            headers=self._client._headers()
        )
        r.raise_for_status()
        updated = TaskResponse(**r.json()).with_client(self._client)
        self.__dict__.update(updated.__dict__)
        return self.poll()
    
    def cancel(self):
        if not self.task_url:
            raise ValueError("Task URL not found")
        r = self._client._session.get(
            f"{self.task_url}/cancel",
            headers=self._client._headers()
        )
        r.raise_for_status()
        self.poll()

    def delete(self):
        if not self.task_url:
            raise ValueError("Task URL not found")
        r = self._client._session.delete(
            self.task_url,
            headers=self._client._headers()
        )
        r.raise_for_status()
