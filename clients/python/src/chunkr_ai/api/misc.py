from .configuration import Configuration
import base64
import io
from pathlib import Path
from PIL import Image
from typing import Union, Tuple, BinaryIO, Optional

async def prepare_file(file: Union[str, Path, BinaryIO, Image.Image]) -> Tuple[Optional[str], str]:
    """Convert various file types into a tuple of (filename, file content).

    Args:
        file: Input file, can be:
            - URL string starting with http:// or https://
            - Base64 string
            - Local file path (will be converted to base64)
            - Opened binary file (will be converted to base64)
            - PIL/Pillow Image object (will be converted to base64)

    Returns:
        Tuple[Optional[str], str]: (filename, content) where content is either a URL or base64 string
        The filename may be None for URLs, base64 strings, and PIL Images

    Raises:
        FileNotFoundError: If the file path doesn't exist
        TypeError: If the file type is not supported
        ValueError: If the URL is invalid or unreachable
        ValueError: If the MIME type is unsupported
    """
    # Handle strings
    if isinstance(file, str):
        if file.startswith(('http://', 'https://')):
            return None, file
        try:
            base64.b64decode(file)
            return None, file
        except:
            try:
                file = Path(file)
            except:
                raise ValueError("File must be a valid path, URL, or base64 string")

    # Handle file paths - convert to base64
    if isinstance(file, Path):
        path = Path(file).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file}")
        
        with open(path, "rb") as f:
            file_content = f.read()
            file_ext = path.suffix.lower().lstrip('.')
            if not file_ext:
                raise ValueError("File must have an extension")
            base64_str = base64.b64encode(file_content).decode()
            return path.name, base64_str

    # Handle PIL Images - convert to base64
    if isinstance(file, Image.Image):
        img_byte_arr = io.BytesIO()
        format = file.format or "PNG"
        file.save(img_byte_arr, format=format)
        img_byte_arr.seek(0)
        base64_str = base64.b64encode(img_byte_arr.getvalue()).decode()
        return None, base64_str

    # Handle file-like objects - convert to base64
    if hasattr(file, "read") and hasattr(file, "seek"):
        file.seek(0)
        file_content = file.read()
        name = getattr(file, "name", "document")
        file_ext = Path(name).suffix.lower().lstrip('.')
        if not file_ext:
            raise ValueError("File must have an extension")
        base64_str = base64.b64encode(file_content).decode()
        return Path(name).name, base64_str

    raise TypeError(f"Unsupported file type: {type(file)}")


async def prepare_upload_data(
    file: Optional[Union[str, Path, BinaryIO, Image.Image]] = None,
    filename: Optional[str] = None,
    config: Optional[Configuration] = None,
) -> dict:
    """Prepare data dictionary for upload.

    Args:
        file: The file to upload
        config: Optional configuration settings
        client: HTTP client for downloading remote files

    Returns:
        dict: JSON-serializable data dictionary ready for upload
    """
    data = {}
    if file:
        processed_filename, processed_file = await prepare_file(file)
        data["file"] = processed_file
        data["file_name"] = filename or processed_filename

    if config:
        data.update(config.model_dump(mode="json", exclude_none=True))
        
    return data
