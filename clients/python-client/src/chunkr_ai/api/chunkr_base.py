from .configuration import Configuration
from .task_response import TaskResponse
from .auth import HeadersMixin
from abc import abstractmethod
from dotenv import load_dotenv
import httpx
import os
from pathlib import Path
from PIL import Image
from typing import BinaryIO, Union


class ChunkrBase(HeadersMixin):
    """Base class with shared functionality for Chunkr API clients.
    
    Args:
        url: The base URL of the Chunkr API. Defaults to the value of the CHUNKR_URL environment variable, or "https://api.chunkr.ai" if not set.
        api_key: The API key to use for authentication. Defaults to the value of the CHUNKR_API_KEY environment variable, or None if not set.
        raise_on_failure: Whether to raise an exception if the task fails. Defaults to False.
    """

    def __init__(self, url: str = None, api_key: str = None, raise_on_failure: bool = False):
        load_dotenv()
        self.url = url or os.getenv("CHUNKR_URL") or "https://api.chunkr.ai"
        self._api_key = api_key or os.getenv("CHUNKR_API_KEY")
        self.raise_on_failure = raise_on_failure
        
        if not self._api_key:
            raise ValueError(
                "API key must be provided either directly, in .env file, or as CHUNKR_API_KEY environment variable. You can get an api key at: https://www.chunkr.ai"
            )

        self.url = self.url.rstrip("/")
        self._client = httpx.AsyncClient()

    @abstractmethod
    def upload(
        self,
        file: Union[str, Path, BinaryIO, Image.Image],
        config: Configuration = None,
    ) -> TaskResponse:
        """Upload a file and wait for processing to complete.

        Args:
            file: The file to upload.
            config: Configuration options for processing. Optional.

        Examples:
        ```python
        # Upload from file path
        await chunkr.upload("document.pdf")

        # Upload from opened file
        with open("document.pdf", "rb") as f:
            await chunkr.upload(f)

        # Upload from URL
        await chunkr.upload("https://example.com/document.pdf")

        # Upload from base64 string (must include MIME type header)
        await chunkr.upload("data:application/pdf;base64,JVBERi0...")

        # Upload an image
        from PIL import Image
        img = Image.open("photo.jpg")
        await chunkr.upload(img)
        ```
        Returns:
            TaskResponse: The completed task response
        """
        pass

    @abstractmethod
    def update(
        self, task_id: str, config: Configuration
    ) -> TaskResponse:
        """Update a task by its ID and wait for processing to complete.

        Args:
            task_id: The ID of the task to update
            config: Configuration options for processing. Optional.

        Returns:
            TaskResponse: The updated task response
        """
        pass

    @abstractmethod
    def create_task(
        self,
        file: Union[str, Path, BinaryIO, Image.Image],
        config: Configuration = None,
    ) -> TaskResponse:
        """Upload a file for processing and immediately return the task response. It will not wait for processing to complete. To wait for the full processing to complete, use `task.poll()`.

        Args:
            file: The file to upload.
            config: Configuration options for processing. Optional.

        Examples:
        ```
        # Upload from file path
        task = await chunkr.create_task("document.pdf")

        # Upload from opened file
        with open("document.pdf", "rb") as f:
            task = await chunkr.create_task(f)

        # Upload from URL
        task = await chunkr.create_task("https://example.com/document.pdf")

        # Upload from base64 string (must include MIME type header)
        task = await chunkr.create_task("data:application/pdf;base64,JVBERi0xLjcKCjEgMCBvYmo...")

        # Upload an image
        from PIL import Image
        img = Image.open("photo.jpg")
        task = await chunkr.create_task(img)

        # Wait for the task to complete - this can be done when needed
        await task.poll()
        ```
        """
        pass

    @abstractmethod
    def update_task(
        self, task_id: str, config: Configuration
    ) -> TaskResponse:
        """Update a task by its ID and immediately return the task response. It will not wait for processing to complete. To wait for the full processing to complete, use `task.poll()`.

        Args:
            task_id: The ID of the task to update
            config: Configuration options for processing. Optional.

        Returns:
            TaskResponse: The updated task response
        """
        pass

    @abstractmethod
    def get_task(self, task_id: str, include_chunks: bool = True, base64_urls: bool = False) -> TaskResponse:
        """Get a task response by its ID.

        Args:
            task_id: The ID of the task to get
            include_chunks: Whether to include chunks in the output response. Defaults to True.
            base64_urls: Whether to return base64 encoded URLs. If false, the URLs will be returned as presigned URLs. Defaults to False.

        Returns:
            TaskResponse: The task response
        """
        pass

    @abstractmethod
    def delete_task(self, task_id: str) -> None:
        """Delete a task by its ID.

        Args:
            task_id: The ID of the task to delete
        """
        pass

    @abstractmethod
    def cancel_task(self, task_id: str) -> None:
        """Cancel a task by its ID.

        Args:
            task_id: The ID of the task to cancel
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the client connection. 
        This should be called when you're done using the client to properly clean up resources."""
        pass