from .config import Configuration
from .task import TaskResponse
from .auth import HeadersMixin
from abc import abstractmethod
from dotenv import load_dotenv
import os
from pathlib import Path
from PIL import Image
from typing import BinaryIO, Union

class ChunkrBase(HeadersMixin):
    """Base class with shared functionality for Chunkr API clients."""

    def __init__(self, url: str = None, api_key: str = None):
        load_dotenv()
        self.url = (
            url or 
            os.getenv('CHUNKR_URL') or 
            'https://api.chunkr.ai' 
        )
        self._api_key = (
            api_key or 
            os.getenv('CHUNKR_API_KEY')
        )
        if not self._api_key:
            raise ValueError("API key must be provided either directly, in .env file, or as CHUNKR_API_KEY environment variable. You can get an api key at: https://www.chunkr.ai")
            
        self.url = self.url.rstrip("/")

    @abstractmethod
    def upload(self, file: Union[str, Path, BinaryIO, Image.Image], config: Configuration = None) -> TaskResponse:
        """Upload a file and wait for processing to complete.
        
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def create_task(self, file: Union[str, Path, BinaryIO, Image.Image], config: Configuration = None) -> TaskResponse:
        """Upload a file for processing and immediately return the task response.
        
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def update_task(self, task_id: str, config: Configuration) -> TaskResponse:
        """Update a task by its ID.
        
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def get_task(self, task_id: str) -> TaskResponse:
        """Get a task response by its ID.
        
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def delete_task(self, task_id: str) -> None:
        """Delete a task by its ID.
        
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def cancel_task(self, task_id: str) -> None:
        """Cancel a task by its ID.
        
        Must be implemented by subclasses.
        """
        pass

