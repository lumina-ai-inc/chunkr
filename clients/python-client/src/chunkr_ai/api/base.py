from .config import Configuration
from .task import TaskResponse
from .auth import HeadersMixin
from abc import abstractmethod
from dotenv import load_dotenv
import io
import json
import os
from pathlib import Path
from PIL import Image
import requests
from typing import BinaryIO, Tuple, Union

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

    def _prepare_file(
        self,
        file: Union[str, Path, BinaryIO, Image.Image]
    ) -> Tuple[str, BinaryIO]:
        """Convert various file types into a tuple of (filename, file-like object).

        Args:
            file: Input file, can be:
                - String or Path to a file
                - URL string starting with http:// or https://
                - Base64 string
                - Opened binary file (mode='rb')
                - PIL/Pillow Image object

        Returns:
            Tuple[str, BinaryIO]: (filename, file-like object) ready for upload

        Raises:
            FileNotFoundError: If the file path doesn't exist
            TypeError: If the file type is not supported
            ValueError: If the URL is invalid or unreachable
            ValueError: If the MIME type is unsupported
        """
        # Handle URLs
        if isinstance(file, str) and (file.startswith('http://') or file.startswith('https://')):
            response = requests.get(file)
            response.raise_for_status()
            file_obj = io.BytesIO(response.content)
            filename = Path(file.split('/')[-1]).name or 'downloaded_file'
            return filename, file_obj

        # Handle base64 strings
        if isinstance(file, str) and ',' in file and ';base64,' in file:
            try:
                # Split header and data
                header, base64_data = file.split(',', 1)
                import base64
                file_bytes = base64.b64decode(base64_data)
                file_obj = io.BytesIO(file_bytes)
                
                # Try to determine format from header
                format = 'bin'
                mime_type = header.split(':')[-1].split(';')[0].lower()
                
                # Map MIME types to file extensions
                mime_to_ext = {
                    'application/pdf': 'pdf',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                    'application/msword': 'doc',
                    'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
                    'application/vnd.ms-powerpoint': 'ppt',
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                    'application/vnd.ms-excel': 'xls',
                    'image/jpeg': 'jpg',
                    'image/png': 'png',
                    'image/jpg': 'jpg'
                }
                
                if mime_type in mime_to_ext:
                    format = mime_to_ext[mime_type]
                else:
                    raise ValueError(f"Unsupported MIME type: {mime_type}")

                return f"file.{format}", file_obj
            except Exception as e:
                raise ValueError(f"Invalid base64 string: {str(e)}")

        # Handle file paths
        if isinstance(file, (str, Path)):
            path = Path(file).resolve()
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file}")
            return path.name, open(path, 'rb')

        # Handle PIL Images
        if isinstance(file, Image.Image):
            img_byte_arr = io.BytesIO()
            format = file.format or 'PNG'
            file.save(img_byte_arr, format=format)
            img_byte_arr.seek(0)
            return f"image.{format.lower()}", img_byte_arr

        # Handle file-like objects
        if hasattr(file, 'read') and hasattr(file, 'seek'):
            # Try to get the filename from the file object if possible
            name = getattr(file, 'name', 'document') if hasattr(file, 'name') else 'document'
            return Path(name).name, file

        raise TypeError(f"Unsupported file type: {type(file)}")

    def _prepare_upload_data(
        self,
        file: Union[str, Path, BinaryIO, Image.Image],
        config: Configuration = None
    ) -> Tuple[dict, dict]:
        """Prepare files and data dictionaries for upload.
        
        Args:
            file: The file to upload
            config: Optional configuration settings
            
        Returns:
            Tuple[dict, dict]: (files dict, data dict) ready for upload
        """
        filename, file_obj = self._prepare_file(file)
        files = {"file": (filename, file_obj)}
        data = {}
        
        if config:
            config_dict = config.model_dump(mode="json", exclude_none=True)
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    files[key] = (None, json.dumps(value), 'application/json')
                else:
                    data[key] = value
                    
        return files, data
    
    @abstractmethod
    def upload(self, file: Union[str, Path, BinaryIO, Image.Image], config: Configuration = None) -> TaskResponse:
        """Upload a file and wait for processing to complete.
        
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def start_upload(self, file: Union[str, Path, BinaryIO, Image.Image], config: Configuration = None) -> TaskResponse:
        """Upload a file for processing and immediately return the task response.
        
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_task(self, task_id: str) -> TaskResponse:
        """Get a task response by its ID.
        
        Must be implemented by subclasses.
        """
        pass
