import io
import json
from pathlib import Path
from PIL import Image
import requests
from typing import Union, Tuple, BinaryIO, Optional
from .config import Configuration


def prepare_file(
    file: Union[str, Path, BinaryIO, Image.Image]
) -> Tuple[str, BinaryIO]:
    """Convert various file types into a tuple of (filename, file-like object)."""
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



def prepare_upload_data(
    file: Optional[Union[str, Path, BinaryIO, Image.Image]] = None,
    config: Optional[Configuration] = None
) -> dict:
    """Prepare files and data dictionaries for upload.
    
    Args:
        file: The file to upload
        config: Optional configuration settings
        
    Returns:
        dict: (files dict) ready for upload
    """
    files = {}
    if file:
        filename, file_obj = prepare_file(file)
        files = {"file": (filename, file_obj)}

    if config:
        config_dict = config.model_dump(mode="json", exclude_none=True)
        for key, value in config_dict.items():
            files[key] = (None, json.dumps(value), 'application/json')
                
    return files
