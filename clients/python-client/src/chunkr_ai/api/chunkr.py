from .base import ChunkrBase
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

    def upload(self, file: Union[str, Path, BinaryIO, Image.Image], config: Configuration = None) -> TaskResponse:
        """Upload a file and wait for processing to complete.

        Args:
            file: The file to upload. 
            config: Configuration options for processing. Optional.

        Examples:
        ```
        # Upload from file path
        chunkr.upload("document.pdf")

        # Upload from URL
        chunkr.upload("https://example.com/document.pdf")

        # Upload from base64 string (must include MIME type header)
        chunkr.upload("data:application/pdf;base64,JVBERi0xLjcKCjEgMCBvYmo...")

        # Upload from opened file
        with open("document.pdf", "rb") as f:
            chunkr.upload(f)

        # Upload an image
        from PIL import Image
        img = Image.open("photo.jpg")
        chunkr.upload(img)
        ```
        Returns:
            TaskResponse: The completed task response
        """
        task = self.create_task(file, config)
        return task.poll()

    def create_task(self, file: Union[str, Path, BinaryIO, Image.Image], config: Configuration = None) -> TaskResponse:
        """Upload a file for processing and immediately return the task response. It will not wait for processing to complete. To wait for the full processing to complete, use `task.poll()`

        Args:
            file: The file to upload.
            config: Configuration options for processing. Optional.

        Examples:
        ```
        # Upload from file path
        task = chunkr.start_upload("document.pdf")

        # Upload from opened file
        with open("document.pdf", "rb") as f:
            task = chunkr.start_upload(f)

        # Upload from URL
        task = chunkr.start_upload("https://example.com/document.pdf")

        # Upload from base64 string (must include MIME type header)
        task = chunkr.start_upload("data:application/pdf;base64,JVBERi0xLjcKCjEgMCBvYmo...")

        # Upload an image
        from PIL import Image
        img = Image.open("photo.jpg")
        task = chunkr.start_upload(img)

        # Wait for the task to complete - this can be done when needed
        task.poll()
        ```

        Returns:
            TaskResponse: The initial task response
        """
        files, data = prepare_upload_data(file, config)
        r = self._session.post(
            f"{self.url}/api/v1/task",
            files=files,
            data=data,  
            headers=self._headers()
        )
        r.raise_for_status()
        return TaskResponse(**r.json()).with_client(self)
    
    def update_task(self, task_id: str, config: Configuration) -> TaskResponse:
        """Update a task by its ID.
        
        Args:
            task_id: The ID of the task to update
            config: The new configuration to use

        Returns:
            TaskResponse: The updated task response
        """
        files, data = prepare_upload_data(None, config)
        if files:
            r = self._session.patch(
                f"{self.url}/api/v1/task/{task_id}",
                files=files,
                data=data,  
                headers=self._headers()
            )
        else:
            r = self._session.patch(
                f"{self.url}/api/v1/task/{task_id}",
                data=data,  
                headers=self._headers()
            )
        r.raise_for_status()
        return TaskResponse(**r.json()).with_client(self)
    
    def get_task(self, task_id: str) -> TaskResponse:
        """Get a task response by its ID.
        
        Args:
            task_id: The ID of the task to get

        Returns:
            TaskResponse: The task response
        """
        r = self._session.get(
            f"{self.url}/api/v1/task/{task_id}",
            headers=self._headers()
        )
        r.raise_for_status()
        return TaskResponse(**r.json()).with_client(self)


    def delete_task(self, task_id: str) -> None:
        """Delete a task by its ID.
        
        Args:
            task_id: The ID of the task to delete
        """
        r = self._session.delete(
            f"{self.url}/api/v1/task/{task_id}",
            headers=self._headers()
        )
        r.raise_for_status()

    def cancel_task(self, task_id: str) -> None:
        """Cancel a task by its ID.
        
        Args:
            task_id: The ID of the task to cancel
        """
        r = self._session.get(
            f"{self.url}/api/v1/task/{task_id}/cancel",
            headers=self._headers()
        )
        r.raise_for_status()

  