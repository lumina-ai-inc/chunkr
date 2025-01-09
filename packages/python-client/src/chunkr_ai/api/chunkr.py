from .models import TaskResponse, Configuration
from dotenv import load_dotenv
import httpx
import io
import os
from pathlib import Path
from PIL import Image
import requests
from typing import Union, BinaryIO, Tuple

class Chunkr:
    """Client for interacting with the Chunkr API."""

    def __init__(self, url: str = None, api_key: str = None):
        load_dotenv()
        self.url = (
            url or 
            os.getenv('CHUNKR_URL') or 
            'https://api.chunkr.ai' 
        )
        self.api_key = (
            api_key or 
            os.getenv('CHUNKR_API_KEY')
        )
        if not self.api_key:
            raise ValueError("API key must be provided either directly, in .env file, or as CHUNKR_API_KEY environment variable. You can get an api key at: https://www.chunkr.ai")
            
        self.url = self.url.rstrip("/")

    def _headers(self):
        return {"Authorization": self.api_key}

    def _prepare_file(
        self,
        file: Union[str, BinaryIO, Image.Image, bytes, io.BytesIO]
    ) -> Tuple[str, BinaryIO]:
        """Convert various file types into a tuple of (filename, file-like object).

        Args:
            file: Input file in various formats

        Returns:
            Tuple[str, BinaryIO]: Filename and file-like object ready for upload
        """
        if isinstance(file, str):
            path = Path(file).resolve() 
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file}")
            return path.name, path.open("rb")
        elif isinstance(file, Image.Image):
            img_byte_arr = io.BytesIO()
            file.save(img_byte_arr, format=file.format or 'PNG')
            img_byte_arr.seek(0)
            return "image.png", img_byte_arr
        elif isinstance(file, bytes):
            return "document", io.BytesIO(file)
        elif isinstance(file, io.BytesIO):
            return "document", file
        else:
            return "document", file

    def upload(self, file: Union[str, BinaryIO, Image.Image, bytes, io.BytesIO], config: Configuration = None) -> TaskResponse:
        """Upload a file and wait for processing to complete.

        The file can be one of:
        - str: Path to a file on disk
        - BinaryIO: A file-like object (e.g., opened with 'rb' mode)
        - Image.Image: A PIL/Pillow Image object
        - bytes: Raw binary data
        - io.BytesIO: A binary stream in memory

        Args:
            file: The file to upload.
            config:
                Configuration options for processing. Optional.

        Returns:
            TaskResponse: The completed task response
        """
        return self.start_upload(file, config).poll()

    def start_upload(self, file: Union[str, BinaryIO, Image.Image, bytes, io.BytesIO], config: Configuration = None) -> TaskResponse:
        """Upload a file for processing and immediately return the task response. It will not wait for processing to complete. To wait for the full processing to complete, use `task.poll()`

        The file can be one of:
        - str: Path to a file on disk
        - BinaryIO: A file-like object (e.g., opened with 'rb' mode)
        - Image.Image: A PIL/Pillow Image object
        - bytes: Raw binary data
        - io.BytesIO: A binary stream in memory

        Args:
            file: The file to upload.
            config (Configuration, optional): Configuration options for processing

        Returns:
            TaskResponse: The initial task response

        Raises:
            requests.exceptions.HTTPError: If the API request fails
        """
        url = f"{self.url}/api/v1/task"
        filename, file_obj = self._prepare_file(file)
        files = {"file": (filename, file_obj)}   
        r = requests.post(url, files=files, json=config.dict() if config else {}, headers=self._headers())
        r.raise_for_status()
        return TaskResponse(**r.json())

    def get_task(self, task_id: str) -> TaskResponse:
        """Get a task response by its ID.
        
        Args:
            task_id (str): The ID of the task to get

        Returns:
            TaskResponse: The task response
        """
        url = f"{self.url}/api/v1/task/{task_id}"
        r = requests.get(url, headers=self._headers())
        r.raise_for_status()
        return TaskResponse(**r.json())


class ChunkrAsync(Chunkr):
    """Async client for interacting with the Chunkr API.
    
    This class inherits from the Chunkr class but works with async HTTP requests.
    """

    async def upload(self, file_path: str, config: Configuration = None) -> TaskResponse:
        task = await self.start_upload(file_path, config)
        return await task.poll_async()

    async def start_upload(self, file_path: str, config: Configuration = None) -> TaskResponse:
        url = f"{self.url}/api/v1/task"
        async with httpx.AsyncClient() as client:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "application/pdf")}
                r = await client.post(
                    url, 
                    files=files, 
                    json=config.dict() if config else None,
                    headers=self._headers()
                )
                r.raise_for_status()
                return TaskResponse(**r.json())

    async def get_task(self, task_id: str) -> TaskResponse:
        url = f"{self.url}/api/v1/task/{task_id}"
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=self._headers())
            r.raise_for_status()
            return TaskResponse(**r.json())

    