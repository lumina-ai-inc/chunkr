from .configuration import Configuration
import base64
import io
from pathlib import Path
from PIL import Image
from typing import Union, Tuple, BinaryIO, Optional, Any

async def prepare_file(file: Union[str, Path, BinaryIO, Image.Image, bytes, bytearray, memoryview]) -> Tuple[Optional[str], str]:
    """Convert various file types into a tuple of (filename, file content).

    Args:
        file: Input file, can be:
            - URL string starting with http:// or https://
            - Base64 string
            - Local file path (will be converted to base64)
            - Opened binary file (will be converted to base64)
            - PIL/Pillow Image object (will be converted to base64)
            - Bytes object (will be converted to base64)

    Returns:
        Tuple[Optional[str], str]: (filename, content) where content is either a URL or base64 string
        The filename may be None for URLs, base64 strings, and PIL Images

    Raises:
        FileNotFoundError: If the file path doesn't exist
        TypeError: If the file type is not supported
        ValueError: If the URL is invalid or unreachable
        ValueError: If the MIME type is unsupported
    """
    # Handle bytes-like objects
    if isinstance(file, (bytes, bytearray, memoryview)):
        # Convert to bytes first if it's not already
        file_bytes = bytes(file)
        
        # Check if this might be an already-encoded base64 string in bytes form
        try:
            # Try to decode the bytes to a string and see if it's valid base64
            potential_base64 = file_bytes.decode('utf-8', errors='strict')
            base64.b64decode(potential_base64)
            # If we get here, it was a valid base64 string in bytes form
            return None, potential_base64
        except:
            # Not a base64 string in bytes form, encode it as base64
            base64_str = base64.b64encode(file_bytes).decode()
            return None, base64_str
        
    # Handle strings - urls or paths or base64
    if isinstance(file, str):
        # Handle URLs
        if file.startswith(('http://', 'https://')):
            return None, file
            
        # Handle data URLs
        if file.startswith('data:'):
            return None, file
            
        # Try to handle as a file path
        try:
            path = Path(file)
            if path.exists():
                # It's a valid file path, convert to Path object and continue processing
                file = path
            else:
                # If not a valid file path, try treating as base64
                try:
                    # Just test if it's valid base64, don't store the result
                    base64.b64decode(file)
                    return None, file
                except:
                    raise ValueError(f"File not found: {file} and it's not a valid base64 string")
        except Exception as e:
            # If string can't be converted to Path or decoded as base64, it might still be a base64 string
            try:
                base64.b64decode(file)
                return None, file
            except:
                raise ValueError(f"Unable to process file: {e}")

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
        if not name or not isinstance(name, str):
            name = None
        base64_str = base64.b64encode(file_content).decode()
        return name, base64_str

    raise TypeError(f"Unsupported file type: {type(file)}")


async def prepare_upload_data(
    file: Optional[Union[str, Path, BinaryIO, Image.Image, bytes, bytearray, memoryview]] = None,
    filename: Optional[str] = None,
    config: Optional[Configuration] = None,
) -> dict:
    """Prepare data dictionary for upload.

    Args:
        file: The file to upload
        filename: Optional filename to use (overrides any filename from the file)
        config: Optional configuration settings

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
